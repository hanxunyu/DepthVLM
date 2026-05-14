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
<div align="left">
<img src="assets/teaser2.png" width="99%" alt="model">
</div>

**DepthVLM** serves as a unified foundation model for both low-level dense geometry prediction and high-level multimodal understanding, while achieving substantially faster inference compared with existing VLM-based approaches such as DepthLM and Youtu-VL.


## 📰 News
- [2026-05-20] 🔥 We release [DepthVLM-Bench](xxxx) in Hugging Face 🤗.
- [2026-05-20] 🔥 We release the checkpoints of [DepthVLM-4B](https://huggingface.co/JonnyYu828/DepthVLM-4B) and [DepthVLM-8B](https://huggingface.co/JonnyYu828/DepthVLM-8B) in Hugging Face 🤗.
- [2026-05-20] 🔥 We release the training and inference code.
- [2026-05-20] 🔥 We release the [paper](xxxxx) of DepthVLM.


## 🛠️ Installation

```
git clone https://github.com/hanxunyu/DepthVLM.git
cd DepthVLM

conda create -n depthvlm python=3.10 -y
conda activate depthvlm
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```
## 📊 Data Preparation
- Due to licensing restrictions, we are unable to directly release the curated data. Instead, we provide the full data curation pipeline for reproducibility. Please refer to [prepare_data.sh](./prepare_data.sh) for detailed dataset-specific preparation instructions.
- We provide example images from ScanNetpp in the [demo_images](./demo_images) folder.
- We also release the curated annotations of [DepthVLM-Bench](https://huggingface.co/yuxinhk/N3D-VLM) on Hugging Face 🤗.

## 📦️ Pretrained models
We provide the pretrained models [DepthVLM-4B](https://huggingface.co/JonnyYu828/Stream3D-VLM) and [DepthVLM-8B](https://huggingface.co/JonnyYu828/Stream3D-VLM) in Hugging Face 🤗. 


## 🤖 Inference Examples 
Try our example inference script. 
```
# inference 
bash src/qwen_vl/eval/model_inference.sh
```


## 🚀 Training
```
# train 
python train.py
```
DepthVLM-8B is trained for four days on 80 NVIDIA H20 GPUs (96GB), while DepthVLM-4B is trained for two days using the same computational resources.


## 🔬 Results

### Comparison with VLMs
<div align="left">
<img src="assets/table1.png" width="99%" alt="model">
</div>

### Comparison with pure vision models
<div align="left">
<img src="assets/table2.png" width="99%" alt="model">
</div>

### Visualization Comparison
<div align="left">
<img src="assets/visualization.png" width="99%" alt="model">
</div>

## 👏 Acknowledgements
We are grateful for the open-source contributions of other projects:
- [DepthLM](https://github.com/facebookresearch/DepthLM_Official)
- [Youtu-VL](https://github.com/TencentCloudADP/youtu-vl)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)


## 🖊️ Citation

```BibTeX

```
