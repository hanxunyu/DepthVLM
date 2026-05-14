<h1 align="center" style="display: flex; align-items: flex-start; justify-content: center; gap: 12px;">
  <strong>
    Unlocking Dense Metric Depth Estimation in VLMs
  </strong>
</h1>

<p align="center">
    <a href="https://hanxunyu.github.io/" target="_blank">Hanxun Yu<sup>1,2*</sup></a>,
    <a href="https://github.com/Select-ing" target="_blank">Xuan Qu<sup>1,2*</sup></a>,
    <a href="https://w-ted.github.io/" target="_blank">Yuxin Wang<sup>2,3</sup></a>,
    <a href="https://person.zju.edu.cn/jkzhu" target="_blank">Jianke Zhu<sup>1</sup></a>,
    <a href="https://www.kelei.site/" target="_blank">Lei Ke<sup>2</sup></a>
    <br>
    <sup>1</sup>ZJU,
    <sup>2</sup>Tencent Hunyuan LLM,
    <sup>3</sup>HKUST
</p>

<div align="center">
    <a href='https://arxiv.org/abs/2512.16561' target="_blank"><img src='https://img.shields.io/badge/arXiv-XXXX-b31b1b?logo=arxiv&logoColor=red'></a>  
    <a href='' target="_blank"><img src='https://img.shields.io/badge/Project-Home%20Page-Green?logo=safari&logoColor=white'></a>  
    <a href='' target="_blank">
        <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-blue'>
    </a>
    <a href='' target="_blank">
        <img src='https://img.shields.io/badge/%F0%9F%93%A6%EF%B8%8F%20Hugging%20Face-Models-orange'>
    </a>
</div>




https://github.com/user-attachments/assets/66797dad-246b-47f6-be91-9f952fdd8b1a




## 🔍 Overview

<div align="left">
<img src="assets/teaser1.png" width="99%" alt="model">
</div>

**DepthVLM** serves as a unified foundation model for both low-level dense geometry prediction and high-level multimodal understanding, while achieving substantially faster inference compared with existing VLM-based approaches such as DepthLM and Youtu-VL.


## 📰 News
- [2026-05-20] 🔥 We release [DepthVLM-Bench](xxxx) in Hugging Face 🤗.
- [2026-05-20] 🔥 We release the checkpoints of [DepthVLM-4B](https://huggingface.co/JonnyYu828/DepthVLM-4B) and [DepthVLM-8B](https://huggingface.co/JonnyYu828/DepthVLM-8B) in Hugging Face 🤗.
- [2026-05-20] 🔥 We release the training and inference code.
- [2026-05-20] 🔥 We release the [paper](xxxxx) of DepthVLM.


## 🛠️ Installation

```
git clone https://github.com/hanxunyu/Stream3D-VLM.git
cd Stream3D-VLM

conda create -n stream3d-llm python=3.10 -y
conda activate stream3d-llm
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```
## 📊 Dataset and Benchmark
<div align="left">
<img src="assets/data_generation.png" width="99%" alt="data_generation">
</div>

**Illustration of our data generation pipeline.** Guided by a comprehensive task taxonomy spanning five cognitive competencies and three temporal interaction modes, the pipeline leverages detailed metadata from RGB-D video streams and a hybrid generation strategy to construct a large-scale spatio-temporal 3D QA dataset and the Stream3D-Bench for evaluating online 3D spatial understanding.

<!-- We provide **the training dataset** and **Stream3D-Bench** ([XXXX](https://huggingface.co/yuxinhk/N3D-VLM)) in Hugging Face 🤗.  -->

## 📦️ Pretrained models
We provide the pretrained models [🤗Stream3D-VLM-4B](https://huggingface.co/JonnyYu828/Stream3D-VLM) and [🤗Stream3D-VLM-8B](https://huggingface.co/JonnyYu828/Stream3D-VLM) in Hugging Face. 

<!-- ## 🚀 Training
```
# train 
python train.py
``` -->

## 🤖 Inference Examples 
Try our example inference script. 
```
# inference 
bash src/qwen_vl/eval/model_inference.sh
```

### Demo 1-Forward Response (Monitoring)



https://github.com/user-attachments/assets/a3af0dab-6c39-48b9-a43d-41596e4ceb57





### Demo 2-Realtime Perception (Observation)




https://github.com/user-attachments/assets/04a945b7-3c8d-4305-bd9a-0e7ca5fac45c




### Demo 3-Backward Tracing (Memory)




https://github.com/user-attachments/assets/091e1687-7e35-42f7-83c9-a53a378ccda3






## 👏 Acknowledgements
We are grateful for the open-source contributions of other projects:
- [StreamVGGT](https://github.com/wzzheng/streamvggt)
- [VG-LLM](https://github.com/LaVi-Lab/VG-LLM)
- [VLM-3R](https://github.com/VITA-Group/VLM-3R)


## 🖊️ Citation

```BibTeX

```
