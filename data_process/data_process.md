# Data Processing Pipeline

This directory contains the data processing scripts for preparing pixel-level depth estimation datasets. The pipeline consists of four stages: downloading raw datasets, extracting RGB/depth/intrinsics, curating metadata JSONL files, and sampling test images & points for evaluation.

## Directory Structure

```
data_process/
├── extract_rgb_depth/          # Stage 2: Extract RGB, depth maps, and intrinsics
│   ├── extract_rgb_depth_argoverse2.py
│   ├── extract_rgb_depth_ddad.py
│   ├── extract_rgb_depth_nuscenes.py
│   └── extract_rgb_depth_waymo.py
├── curate_datasets/            # Stage 3: Build per-dataset JSONL metadata
│   ├── create_data_pixel_level_argoverse2.py
│   ├── create_data_pixel_level_ddad.py
│   ├── create_data_pixel_level_eth3d.py
│   ├── create_data_pixel_level_hm3d.py
│   ├── create_data_pixel_level_ibims1.py
│   ├── create_data_pixel_level_matterport3d.py
│   ├── create_data_pixel_level_nuscenes.py
│   ├── create_data_pixel_level_nyuv2.py
│   ├── create_data_pixel_level_scannetpp.py
│   ├── create_data_pixel_level_sunrgbd.py
│   ├── create_data_pixel_level_taskonomy.py
│   ├── create_data_pixel_level_waymo.py
│   └── extract_nyuv2_from_sunrgbd.py
├── sample_test_images/         # Stage 4a: Sample test images from val/test splits
│   ├── depth_check.py
│   ├── sample_images_argoverse.py
│   ├── sample_images_ddad.py
│   ├── sample_images_generic.py
│   ├── sample_images_hm3d.py
│   ├── sample_images_matterport3d.py
│   ├── sample_images_nuscenes.py
│   ├── sample_images_scannet.py
│   ├── sample_images_sunrgbd.py
│   ├── sample_images_taskonomy.py
│   └── sample_images_waymo.py
├── sample_test_points/         # Stage 4b: Sample depth points on selected test images
│   └── sample_points.py
└── README.md
```

## Stage 1: Download Datasets

Download the following datasets from their official sources:

| Dataset | Source |
|---------|--------|
| Argoverse 2 | https://www.argoverse.org/ |
| Waymo Open | https://waymo.com/open/ |
| nuScenes | https://www.nuscenes.org/ |
| DDAD | https://github.com/TRI-ML/DDAD |
| ScanNet++ | https://scannetpp.mlsg.cit.tum.de/scannetpp/ |
| Taskonomy | https://github.com/StanfordVL/taskonomy/tree/master/data |
| HM3D | https://aihabitat.org/datasets/hm3d/ |
| Matterport3D | https://niessner.github.io/Matterport/ |
| SUN RGB-D | https://rgbd.cs.princeton.edu/ |
| iBims-1 | https://www.asg.ed.tum.de/lmf/ibims1/ |
| NYUv2 | Extracted from SUN RGB-D (see `extract_nyuv2_from_sunrgbd.py`) |
| ETH3D | https://www.eth3d.net/ |

## Stage 2: Extract RGB, Depth Maps, and Intrinsics

> **Applies to outdoor LiDAR datasets only:** Argoverse 2, Waymo, nuScenes, DDAD.

These scripts extract RGB images, project sparse LiDAR point clouds onto the image plane to generate depth maps, and save per-frame camera intrinsics. The output follows a unified directory layout:

```
out_folder/{split}/
├── rgb/{scene_id}/{timestamp}_{camera}.jpg
├── depth/{scene_id}/{timestamp}_{camera}.png      # uint16, depth_m × 256
├── intrinsics/{scene_id}/{timestamp}_{camera}.json # [fx, fy, cx, cy, W, H]
└── index.jsonl
```

**Example (Argoverse 2):**

```bash
python extract_rgb_depth/extract_rgb_depth_argoverse2.py \
    --root_folder /path/to/argoverse_raw \
    --out_folder /path/to/argoverse \
    --num_workers 32
```

Other indoor datasets (ScanNet++, Taskonomy, HM3D, etc.) already provide aligned RGB and depth data, so no extraction step is needed.

## Stage 3: Curate Dataset Metadata

Each script reads the extracted (or raw) data and produces JSONL files following the **official train/val/test splits** of each dataset. Both the training set and the validation (or test) set are processed, producing separate JSONL files for each split. Each JSONL record includes:

- `image`: relative path to the RGB image
- `depth_path`: relative path to the depth map
- `depth_scale`: scale factor to convert raw depth to meters
- `original_fx`, `original_fy`, `original_cx`, `original_cy`: camera intrinsics
- `original_width`, `original_height`: image dimensions
- Additional dataset-specific fields (e.g., `mask_valid_path`, `depth_format`)

**Example (Argoverse 2):**

```bash
python curate_datasets/create_data_pixel_level_argoverse2.py \
    --data_root /path/to/argoverse_v2 \
    --splits train,val,test \
    --output_dir ./annotations/argoverse2 \
    --num_workers 32
```

**NYUv2 note:** NYUv2 data is embedded within SUN RGB-D. Use `extract_nyuv2_from_sunrgbd.py` to extract it first:

```bash
python curate_datasets/extract_nyuv2_from_sunrgbd.py \
    --sunrgbd_root /path/to/sun_rgbd_official_link \
    --output_dir /path/to/nyuv2_extracted
```

## Stage 4: Prepare Evaluation Data

Evaluation data is prepared from the **validation/test split only**, in two steps.

### Stage 4a: Sample Test Images

Select a representative subset of images from the val/test JSONL for evaluation. Each dataset has a dedicated sampling script that ensures balanced coverage across scenes and cameras. Optional `--depth_root` enables depth validity checking during sampling.

- **Scene + camera grouped** (outdoor): `sample_images_argoverse.py`, `sample_images_ddad.py`, `sample_images_nuscenes.py`, `sample_images_waymo.py`
- **Scene grouped** (indoor with split file): `sample_images_hm3d.py`, `sample_images_matterport3d.py`, `sample_images_scannet.py`, `sample_images_taskonomy.py`
- **Stratified by sensor** (SUN RGB-D): `sample_images_sunrgbd.py`
- **Generic random** (no scene grouping): `sample_images_generic.py` — for NYUv2, ETH3D, iBims-1, etc.

**Example (Argoverse 2):**

```bash
python sample_test_images/sample_images_argoverse.py \
    --input_jsonl ./annotations/argoverse2/argoverse_pixel_depth_test.jsonl \
    --total_samples 1000 \
    --output_jsonl ./annotations/argoverse2/test/argoverse_pixel_depth_test_1000.jsonl \
    --depth_root /path/to/argoverse_v2
```

**Example (HM3D, with split file):**

```bash
python sample_test_images/sample_images_hm3d.py \
    --input_jsonl ./annotations/hm3d/hm3d_pixel_depth_val.jsonl \
    --split_file ./annotations/hm3d/split_metadata.csv \
    --target_split val \
    --total_samples 1000 \
    --output_jsonl ./annotations/hm3d/test/hm3d_pixel_depth_val_1000.jsonl \
    --depth_root /path/to/hm3d
```

### Stage 4b: Sample Depth Points

For each sampled test image, randomly sample valid pixels from the depth map at its original resolution and record their z-depth values. This produces the final evaluation JSONL.

Each output record appends the following fields:

```json
{
  "pixel_coords": [[col, row], ...],
  "depth": [d1, d2, ...],
  "depth_type": "z_depth"
}
```

**Example:**

```bash
python sample_test_points/sample_points.py \
    --input_jsonl ./annotations/argoverse2/test/argoverse_pixel_depth_test_1000.jsonl \
    --data_root /path/to/argoverse_v2 \
    --output_dir ./annotations/argoverse2/test \
    --points_per_image 10 \
    --seed 42
```

Use `--total_points` to specify an exact total number of sampled points (automatically distributed across all images):

```bash
python sample_test_points/sample_points.py \
    --input_jsonl ./annotations/eth3d/test/eth3d_pixel_depth_test_500.jsonl \
    --data_root /path/to/eth3d \
    --output_dir ./annotations/eth3d/test \
    --total_points 5000 \
    --seed 42
```
