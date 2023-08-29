# [IEEE TIM] Instance-Guided Point Cloud Single Object Tracking with Inception Transformer"

[Paper Link]() | [Project Page](https://github.com/ywu0912/TeamCode/tree/liujia99/PTIT) 

## Introduction

We propose a novel framework, the inception transformer-based point tracker for 3D point cloud tracking through four main stages: 1) feature extraction; 2) feature transform; 3) feature matching; and 4) feature offset.

<img src="docs/PTIT.png" align="center" width="100%">

## Installation
Create conda environment and install pytorch. Tested with pytorch 1.8.0 and CUDA 11.1. Might work with other versions as well, but not tested.
```sh
conda create -n ptit python=3.7
conda activate ptit
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge 
or [pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html]
```
Install dependencies
```sh
pip install -r requirements.txt
```

## Data Preparation
### KITTI Tracking Dataset
+ Download the data for [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).
+ Unzip the downloaded files.
+ Put the unzipped files under the same folder as following.
  ```
  [<Your KITTI dataset path>]
  ├── [calib]
  │    ├── {0000-0020}.txt
  ├── [label_02]
  │    ├── {0000-0020}.txt
  ├── [velodyne]
  │    ├── [0000-0020] folders with velodynes .bin files
  ```

### nuScenes Tracking Dataset
+ Download the dataset from the [download page](https://www.nuscenes.org/download).
+ Extract the downloaded files and make sure you have the following structure:
  ```
  [<Your nuScenes dataset path>]
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    maps	        -	Folder for all map files: rasterized .png images and vectorized .json files.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
  ```
>Note: We use the **train_track** split to train our model and test it with the **val** split. Both splits are officially provided by NuScenes. During testing, we ignore the sequences where there is no point in the first given bbox.


## Training and Testing
Train with the KITTI dataset (e.g., for the Car class)
```
python train_tracking.py --category_name Car --save_root_dir checkpoints/kitti/Car/I --model I
```
Test with the KITTI dataset. Checkpoints are provided at ```checkpoints/kitti/```
```
python test_tracking.py --category_name Car --save_root_dir results/kitti/Car/I --resume checkpoints/kitti/Car/I/netR_50.pth --model I
```

## Acknowledgement
This repo builds on top of [P2B](https://github.com/HaozheQi/P2B), [PTTR](https://github.com/Jasonkks/PTTR), and [Open3DSOT](https://github.com/Ghostish/Open3DSOT). We thank for their contributions.
