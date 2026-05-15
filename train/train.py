# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
import trl

from qwen_vl_utils import process_vision_info

from torch.utils.tensorboard import SummaryWriter
from model import Qwen3VLForConditionalGeneration

from transformers import (
    AutoProcessor,
    set_seed,
)
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
)

from utils.callbacks import get_callbacks
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig(trl.ModelConfig):
    output_model_local_path: str = field(
        default="test-output",
        metadata={"help": "Output model local path, do not set manually"},
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to push to a Hub revision/branch."}
    )


@dataclass
# pyre-fixme[11]: Annotation `ScriptArguments` is not defined as a type.
class SFTScriptArguments(ScriptArguments):
    dataset_class: str = field(
        default="dataset_pixel_depth_train",
        metadata={"help": "dataset class name"},
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "image folder path"},
    )
    freeze_mllm: bool = field(
        default=False,
        metadata={"help": "Stage 1a: freeze all, only train depth_head"},
    )
    freeze_vision: bool = field(
        default=False,
        metadata={"help": "Freeze Vision Encoder (recommended for stage 1b)"},
    )
    with_text_reply: bool = field(
        default=False,
        metadata={"help": "Include assistant text reply for LM loss (stage 1b)"},
    )
    depth_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for depth loss: total = lm_loss + weight * depth_loss"},
    )
    depth_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root dir of original depth maps (for pixel-level GT loading)"},
    )
    freeze_depth_head: bool = field(
        default=False,
        metadata={"help": "Freeze depth head (use together with --dataset_class dataset_qa_train to keep depth ability when SFT QA)."},
    )


processor = None


def convert_example(example):
    """Convert a data sample into Qwen VL `messages` format (generic version).

    Decide whether to append an assistant reply based on `_with_text_reply` and
    the `solution` field of the data:
    - `_with_text_reply=True` and `solution` non-empty: append assistant reply
      (used for LM loss).
    - Otherwise: only the user message (depth loss only).
    """
    messages = []
    if "system" in example:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": example["system"]}],
            }
        )

    problem = example.get("problem", "")
    image = example.get("image")
    video = example.get("video")
    user_content = []
    if video is not None:
        # Video sample: forward fps / max_frames / min_frames to qwen_vl_utils.process_vision_info
        video_item = {"type": "video", "video": video}
        if "video_fps" in example:
            video_item["fps"] = example["video_fps"]
        if "video_max_frames" in example:
            video_item["max_frames"] = example["video_max_frames"]
        if "video_min_frames" in example:
            video_item["min_frames"] = example["video_min_frames"]
        user_content.append(video_item)
    elif image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": problem})
    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    # If text reply is enabled and `solution` is non-empty, append an assistant reply
    solution = example.get("solution", "")
    if _with_text_reply and solution:
        messages.append(
            {
                "role": "assistant",
                "content": solution,
            }
        )

    example["messages"] = messages
    return example


# Global variable, set by main()
_with_text_reply = False


def collate_fn(examples):
    """Data collate function (generic version).

    `_with_text_reply=False` (stage 1a):
        - Only user message, labels all -100 (depth loss only).

    `_with_text_reply=True` (stage 1b):
        - Samples with solution: user + assistant; LM loss is computed on the
          assistant part.
        - Samples without solution: user only, labels all -100.
        - All samples with depth_labels contribute to depth loss.

    This supports mixed data: depth data (with depth_labels + solution) and
    VQA data (solution only).
    """
    import torch

    # Build messages
    converted = [convert_example(example) for example in examples]

    # Determine whether each sample has an assistant reply
    has_reply = []
    for example in examples:
        solution = example.get("solution", "")
        has_reply.append(_with_text_reply and bool(solution))

    # tokenize
    texts = [
        processor.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=not hr,  # add generation prompt when no assistant reply
        )
        for ex, hr in zip(examples, has_reply)
    ]

    flat_images = []
    flat_videos = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        if imgs:
            # process_vision_info returns either None or list[PIL.Image]
            flat_images.extend(imgs if isinstance(imgs, (list, tuple)) else [imgs])
        if vids:
            flat_videos.extend(vids if isinstance(vids, (list, tuple)) else [vids])

    processor_kwargs = dict(
        text=texts,
        return_tensors="pt",
        padding=True,
    )
    # Pass `images` only when present (passing None would hit an empty branch in image_processor and fail)
    if len(flat_images) > 0:
        processor_kwargs["images"] = flat_images
    if len(flat_videos) > 0:
        processor_kwargs["videos"] = flat_videos
    batch = processor(**processor_kwargs)

    # ===== Build LM labels =====
    # Strategy: compare token length of "with assistant reply" vs. "without assistant reply";
    #           the diff portion corresponds to the token positions of the assistant reply.
    labels = batch["input_ids"].clone()
    pad_token_id = processor.tokenizer.pad_token_id
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)

    # pad and image tokens never contribute to LM loss
    labels[labels == pad_token_id] = -100
    labels[labels == image_token_id] = -100

    for i, example in enumerate(examples):
        if has_reply[i]:
            # Build prompt-only messages without the assistant reply
            prompt_only_messages = [
                m for m in example["messages"] if m["role"] != "assistant"
            ]
            prompt_only_text = processor.apply_chat_template(
                prompt_only_messages, tokenize=False, add_generation_prompt=True,
            )
            prompt_only_ids = processor.tokenizer.encode(prompt_only_text, add_special_tokens=False)
            prompt_len = len(prompt_only_ids)

            # Set the prompt portion to -100, keep only the assistant reply portion
            labels[i, :prompt_len] = -100
        else:
            # No assistant reply: all -100
            labels[i, :] = -100

    batch["labels"] = labels

    # ===== Build pixel_depth_labels (pixel-level GT) =====
    pixel_depth_list = []
    has_pixel_depth = False
    for example in examples:
        pdl = example.get("pixel_depth_labels", None)
        if pdl is not None:
            # (H, W) np.float32 -> (1, H, W) torch.float32
            pixel_depth_list.append(torch.from_numpy(pdl).unsqueeze(0).float())
            has_pixel_depth = True
        else:
            pixel_depth_list.append(None)

    if has_pixel_depth:
        batch["pixel_depth_labels"] = pixel_depth_list

    # ===== Debug print: first sample of the first 3 batches =====
    global _collate_call_count
    if _collate_call_count < 3:
        _collate_call_count += 1
        i = 0
        input_ids = batch["input_ids"][i]
        lbl = batch["labels"][i]
        seq_len = (input_ids != pad_token_id).sum().item()
        n_img = (input_ids == image_token_id).sum().item()
        n_label_valid = (lbl != -100).sum().item()
        
        label_text = ""
        if n_label_valid > 0:
            valid_ids = lbl[lbl != -100]
            label_text = processor.tokenizer.decode(valid_ids, skip_special_tokens=True)
        
        print(f"\n[collate_fn debug #{_collate_call_count}]")
        print(f"  seq_len={seq_len}, image_tokens={n_img}, valid_labels={n_label_valid}")
        print(f"  label_text (assistant reply): '{label_text[:100]}'")
        print(f"  with_text_reply={_with_text_reply}")
        if has_pixel_depth and pixel_depth_list[0] is not None:
            pdl = pixel_depth_list[0]
            pdl_valid = pdl[pdl > 0]
            if pdl_valid.numel() > 0:
                range_str = f"[{pdl_valid.min():.2f}, {pdl_valid.max():.2f}]m"
            else:
                range_str = f"[NO_VALID_PIXELS, max={pdl.max():.2f}]m"
            print(f"  pixel_depth shape={pdl.shape}, "
                  f"valid={pdl_valid.numel()}/{pdl.numel()}, "
                  f"range={range_str}")
        else:
            print(f"  pixel_depth: None")
        print()
    
    return batch


_collate_call_count = 0


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    # Force epoch-based scheduling for this training entrypoint.
    if training_args.max_steps is not None and training_args.max_steps > 0:
        logger.warning(
            "Detected max_steps=%s. This script uses num_train_epochs; overriding max_steps to -1.",
            training_args.max_steps,
        )
        training_args.max_steps = -1

    if (
        training_args.num_train_epochs is None
        or float(training_args.num_train_epochs) <= 0
    ):
        raise ValueError(
            f"num_train_epochs must be > 0 when using epoch-based scheduling, got {training_args.num_train_epochs}."
        )

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Set global variable controlling whether collate_fn appends an assistant text reply
    global _with_text_reply
    _with_text_reply = script_args.with_text_reply
    logger.info(f"with_text_reply = {_with_text_reply}")


    print("script_args.image_folder = ", script_args.image_folder)
    training_args.output_dir = model_args.output_model_local_path

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    # Dynamic selection by --dataset_class:
    #   * dataset_pixel_depth_train (default): existing depth dataset (requires depth_root / canonical_size)
    #   * dataset_qa_train            : pure QA dataset (CV-Bench-3D single image + VSI-Bench video), no depth supervision
    import importlib
    ds_mod = importlib.import_module("utils.datasets")
    if not hasattr(ds_mod, script_args.dataset_class):
        raise ValueError(
            f"dataset_class '{script_args.dataset_class}' not found in utils.datasets. "
            f"Available: {[n for n in dir(ds_mod) if n.startswith('dataset_')]}"
        )
    dataset_cls = getattr(ds_mod, script_args.dataset_class)
    logger.info(f"Using dataset class: {script_args.dataset_class}")
    dataset = dataset_cls(
        script_args.dataset_name,
        script_args.image_folder,
        depth_root=script_args.depth_root,
    )

    print("[dataset] dataset_size = ", len(dataset))

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )
        logger.info("Using AutoProcessor for vision-language model.")
        if hasattr(processor, "pad_token") and processor.pad_token is None:
            processor.pad_token = processor.eos_token
        elif (
            hasattr(processor.tokenizer, "pad_token")
            and processor.tokenizer.pad_token is None
        ):
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
        logger.info("Using AutoProcessor.")

    # ###################
    # # Model init kwargs
    # ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=(
            get_kbit_device_map() if quantization_config is not None else None
        ),
        quantization_config=quantization_config,
    )

    qwen_kwargs = {k: v for k, v in model_kwargs.items() if k != "use_cache"}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **qwen_kwargs
    )

    # Set depth loss weight (passed to forward via config)
    model.config.depth_loss_weight = script_args.depth_loss_weight
    logger.info(f"depth_loss_weight = {script_args.depth_loss_weight}")

    # ===== Freezing strategy =====
    if script_args.freeze_mllm:
        # Stage 1a: freeze LLM + Vision Encoder + LM Head, train only the DPT Depth Head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.depth_head.parameters():
            param.requires_grad = True
        logger.info("Stage 1a: Frozen LLM + ViT, only training depth_head (DPT)")

    if script_args.freeze_vision:
        # Stage 1b: freeze Vision Encoder, train LLM + DPT Depth Head + LM Head
        for param in model.model.visual.parameters():
            param.requires_grad = False
        logger.info("Stage 1b: Frozen Vision Encoder")

    if script_args.freeze_depth_head:
        # QA-SFT scenario: freeze depth head to prevent LoRA/LM gradients from indirectly breaking depth ability
        if hasattr(model, "depth_head"):
            for param in model.depth_head.parameters():
                param.requires_grad = False
            logger.info("QA-SFT: Frozen depth_head (preserve depth ability)")
        else:
            logger.warning("freeze_depth_head=True but model has no attribute 'depth_head'; skip.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    ############################
    # Initialize the SFT Trainer
    ############################

    callbacks = get_callbacks(training_args, model_args)
    # # configure TensorboardCallback to upload to manifold
    callbacks.append(
        TensorBoardCallback(
            SummaryWriter(
                log_dir=os.path.join(
                    training_args.output_dir,
                    "tensorboard_logs",
                ),
                comment="",
                purge_step=None,
                max_queue=10,
                flush_secs=120,
                filename_suffix=str(uuid.uuid4()),
            )
        )
    )

    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False

    # ===== Custom Trainer: override compute_loss to pass depth_labels =====
    class DepthSFTTrainer(SFTTrainer):
        """Custom Trainer that ensures pixel_depth_labels is passed into model.forward()."""
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            pixel_depth_labels = inputs.pop("pixel_depth_labels", None)

            # pixel_depth_labels is a list of tensors; Trainer does not auto .to(device)
            if pixel_depth_labels is not None:
                device = next(model.parameters()).device
                pixel_depth_labels = [
                    t.to(device) if t is not None else None
                    for t in pixel_depth_labels
                ]

            outputs = model(
                **inputs,
                pixel_depth_labels=pixel_depth_labels,
            )

            loss = outputs.loss

            # Record individual losses for logging
            if outputs.depth_loss is not None:
                self._pixel_depth_loss = outputs.depth_loss.detach().item()
            else:
                self._pixel_depth_loss = 0.0

            # Compute lm_loss only when valid labels exist
            if (inputs.get("labels") is not None 
                and outputs.logits is not None
                and (inputs["labels"] != -100).any()):
                lm_loss = model.loss_function(
                    logits=outputs.logits,
                    labels=inputs["labels"],
                    vocab_size=model.config.text_config.vocab_size,
                )
                self._lm_loss = lm_loss.detach().item()
            else:
                self._lm_loss = 0.0

            if return_outputs:
                return loss, outputs
            return loss

        def log(self, logs, *args, **kwargs):
            """Add lm_loss and pixel_depth_loss to the log."""
            if hasattr(self, "_lm_loss"):
                logs["lm_loss"] = self._lm_loss
            if hasattr(self, "_pixel_depth_loss"):
                logs["pixel_depth_loss"] = self._pixel_depth_loss
            super().log(logs, *args, **kwargs)

    trainer = DepthSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=callbacks,
    )

    # ###############
    # # Training loop
    # ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ##################################
    # # Save model and create model card
    # ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    output_model_basename = os.path.basename(model_args.output_model_filename)
    model_args.output_model_local_path = training_args.output_dir
        
    os.makedirs(model_args.output_model_local_path, exist_ok=True)

    main(script_args, training_args, model_args)
