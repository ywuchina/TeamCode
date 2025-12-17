
# DcSplat
Official implementation of "DcSplat: Dual-Constraint Human Gaussian Splatting with Latent Multi-View Consistency".

## :fire: News
* **[2023.3.4]** We have created a code repository on [github](https://github.com/Xiaofei-CN/DcSplat) and will continue to update it in the future!
* **[2025.2.26]** Our paper [DcSplat: Dual-Constraint Human Gaussian Splatting with Latent Multi-View Consistency]() has been accepted at the The 40th Annual AAAI Conference on Artificial Intelligence!

## Method
<img src=figure/overview.png>

## Installation

To deploy and run DcSplat, run the following scripts:
```
conda create -n dcsplat python=3.8 -y
conda activate dcsplat

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/
pip install -e submodules/diff-gaussian-rasterization
cd ..

pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```
