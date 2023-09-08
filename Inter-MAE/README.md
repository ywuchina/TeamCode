# [IEEE TMM 2023] Inter-Modal Masked Autoencoder for Self-Supervised Learning on Point Clouds

#### [Paper Link]() | [Project Page](https://github.com/ywu0912/TeamCode/tree/liujia99/Inter-MAE/) 

## Introduction

We propose Inter-MAE, a inter-modal MAE method for self-supervised learning on point clouds. Specifically, we first use Point-MAE as a baseline to partition point clouds into random low percentage of visible and high percentage of masked point patches. Then, a standard Transformer-based autoencoder is built by asymmetric design and shifting mask operations, and latent features are learned from the visible point patches aiming to recover the masked point patches. In addition, we generate image features based on ViT after point cloud rendering to form inter-modal contrastive learning with the decoded features of the completed point patches.

<img src="imgs/Inter-MAE.jpg" align="center" width="100%">


## 1. Requirements
PyTorch >= 1.7.0 < 1.11.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
pip install -r requirements.txt
```

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Datasets

We use ShapeNetRender, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. 

## 3. Inter-MAE Pre-training
To pretrain Inter-MAE on ShapeNetRender training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain.yaml --exp_name <output_file_name>
```
## 4. Inter-MAE Fine-tuning

Fine-tuning on ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_scan_hardest.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Fine-tuning on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_modelnet.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Few-shot learning, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/fewshot.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```
Part segmentation on ShapeNetPart, run:
```
cd segmentation
python main.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300
```

## 5. Visualization

Visulization of pre-trained model on ShapeNetRender validation set, run:

```
python main_vis.py --test --ckpts <path/to/pre-trained/model> --config cfgs/pretrain.yaml --exp_name <name>
```

## Citation

If you entrust our work with value, please consider giving a star ⭐ and citation.

```
@article{liu2023inter,
  title={Inter-Modal Masked Autoencoder for Self-Supervised Learning on Point Clouds},
  author={Liu, Jiaming and Wu, Yue and Gong, Maoguo and Liu, Zhixiao and Miao, Qiguang and Ma, Wenping},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgements

Our codes are built upon [Point-BERT](https://github.com/lulutang0608/Point-BERT) and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), we thank the authors for releasing their code. 