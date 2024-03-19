# Noise2Score: Tweedie's Approach to Self-Supervised Image Denoising without Clean Images (NeurIPS 2021)
Official Pytorch implementation for [paper](https://papers.neurips.cc/paper/2021/file/077b83af57538aa183971a2fe0971ec1-Paper.pdf) presented on NeurIPS 2021

titled "_Noise2Score: Tweedie's Approach to Self-Supervised Image Denoising without Clean Images_".

<img src="./Network.png" width="70%" height="70%" alt="Network"></img>

## Abstract
In this work, we propose a novel self-supervised image denoising apporach without clean data, called Noise2Score. 
Our novel innovation came from the Tweedie’s formula, which provides explicit representation of denoise images through the score function. By combining with the score-function estimation using AR-DAE, our Noise2Score can be
applied to image denoising problem from any exponential family noises. Furthermore, an identical neural network training can be universally used regardless of the noise models, which leads to the
noise parameter estimation with minimal complexity. The links to SURE and existing Noise2X were also explained, which clearly showed why our method is a better generalization.

## Prerequisites
+python 3.8

+pytorch 1.10

+CUDA 11.1

+It is okay to use lower version of CUDA with proper pytorch version. 

ex) CUDA 10.2 with pytorch 1.7.0

## Getting started 
1) Clone the respository

```train
python main.py -m TA-VAAL -d cifar10 -c 5 # Other available datasets cifar100, fashionmnist, svhn
```

2) prepare Dataset
   
```train
python main.py -m TA-VAAL -d cifar10 -c 5 # Other available datasets cifar100, fashionmnist, svhn
```

3) Training

```train
python train.py -m TA-VAAL -d cifar10 -c 5 # Other available datasets cifar100, fashionmnist, svhn
```

5) Inference

```train
python test.py -m TA-VAAL -d cifar10 -c 5 # Other available datasets cifar100, fashionmnist, svhn
```

## Citation
if you find our work intersting, please consider citing

```
@article{kim2021noise2score,
  title={Noise2score: tweedie’s approach to self-supervised image denoising without clean images},
  author={Kim, Kwanyoung and Ye, Jong Chul},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={864--874},
  year={2021}
}
```

