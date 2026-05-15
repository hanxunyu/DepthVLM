# Data Process

## 1.Download Datasets

Argoverse2: https://www.argoverse.org/
Waymo: https://waymo.com/open/
nuScenes: https://www.nuscenes.org/
DDAD: https://github.com/TRI-ML/DDAD
ScanNet++: https://scannetpp.mlsg.cit.tum.de/scannetpp/
Taskonomy: https://github.com/StanfordVL/taskonomy/tree/master/data
HM3D: https://aihabitat.org/datasets/hm3d/
Matterport3D: https://niessner.github.io/Matterport/
sunRGBD: https://rgbd.cs.princeton.edu/
iBims-1: https://www.asg.ed.tum.de/lmf/ibims1/
NYUv2: Extract from sunRGBD
ETH3D: https://www.eth3d.net/

## 2.Extract Data

使用 extract_rgb_depth 下的脚本提取 RGB，Depthmap 和 相机内参。

## 3.Curate Datasets

使用 curate_datasets 下的脚本将数据集整理为训练可用的 json 文件。

## 4.Sample

1. 从验证集/测试集中抽图片。
2. 采点。
使用 curate_unifocal_v2 下的脚本对测试集图片进行采点，转换为测试可用的 json 文件。


# PS. 3.Curate Datasets 中需要提供分割文件路径的数据集：

- [x] scannetpp
- [x] hm3d
- [x] taskonomy

# Question

1.当前没有放 sample images 的代码，是否要放。
2.scannetpp 数据集我们用的是 s3d_train.txt 和 s3d_val.txt；当前代码中写的是 nvs_sem_train.txt 和 nvs_sem_val.txt （官方提供的，train 相同，val 少了 50 个 scenes）。

# 修改

1. Stage3 中强调是按照各个数据集的官方划分对训练集和验证集（测试集）都处理。Stage4 中强调是只从验证集（测试集）中抽图片。
2. 加入抽图片的代码。
3. 删除 欧式距离和 z-depth 的分支逻辑，只保留 z-depth。