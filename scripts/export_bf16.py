"""Export a training checkpoint to a bf16 checkpoint for inference / release.

FSDP mixed-precision training keeps fp32 master weights, so checkpoints saved by
train/train.py are fp32 (stage 2: ~19 GB). This casts them to bf16 (~9 GB, same
as the released DepthVLM-4B) and copies the processor files.

Usage:
    python scripts/export_bf16.py <train_output_dir> <export_dir>
"""

import argparse
import os
import sys

import torch
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Qwen3VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="checkpoint directory written by train/train.py")
    parser.add_argument("dst", help="output directory for the bf16 checkpoint")
    args = parser.parse_args()

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.src, dtype=torch.bfloat16)
    dtypes = {p.dtype for p in model.parameters()}
    assert dtypes == {torch.bfloat16}, f"unexpected dtypes after cast: {dtypes}"
    model.config.dtype = "bfloat16"
    model.save_pretrained(args.dst)
    AutoProcessor.from_pretrained(args.src).save_pretrained(args.dst)

    size = sum(
        os.path.getsize(os.path.join(args.dst, f))
        for f in os.listdir(args.dst) if f.endswith(".safetensors")
    )
    print(f"exported {args.src} -> {args.dst} ({size / 2**30:.1f} GiB, bf16)")


if __name__ == "__main__":
    main()
