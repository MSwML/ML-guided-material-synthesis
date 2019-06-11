# ML-guided-material-synthesis

By [Bijun Tang](https://scholar.google.com.sg/citations?user=qwXbP28AAAAJ&hl=en), Yuhao Lu, [Jiadong Zhou](https://scholar.google.com.sg/citations?user=tmVOLIcAAAAJ&hl=en), Han Wang, Prafful Golani, Manzhang Xu, Quan Xu, [Cuntai Guan](http://www.ntu.edu.sg/home/ctguan/), [Zheng Liu](http://www.ntu.edu.sg/home/z.liu/)

Nanyang Technological University.

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Environment setup](#environment-setup)
0. [Data](#data)
0. [Code](#code)


### Introduction

This repository contains the original models described in the paper "Machine learning-guided synthesis of advanced inorganic materials" (https://arxiv.org/abs/1905.03938). These models are those used for `MoS2 classification` task as well as `CQD regression` task.

**Note**

0. Re-implementations with **training code** and models from Facebook AI Research (FAIR): [blog](http://torch.ch/blog/2016/02/04/resnets.html), [code](https://github.com/facebook/fb.resnet.torch)
0. Code of improved **1K-layer ResNets** with 4.62% test error on CIFAR-10 in our new arXiv paper: https://github.com/KaimingHe/resnet-1k-layers

### Citation

If you use these models in your research, please cite:

	@article{Tang2019,
		author = {Bijun Tang, Yuhao Lu, Jiadong Zhou, Han Wang, Prafful Golani, Manzhang Xu, Quan Xu, Cuntai Guan, Zheng Liu},
		title = {Machine learning-guided synthesis of advanced inorganic materials},
		journal = {arXiv preprint arXiv:1905.03938},
		year = {2019}
	}

### Environment Setup

0. python environment setup:
	```
	python 3.6.6
	jupyter==1.0.0
	matplotlib==2.2.3
	numpy==1.15.1
	pandas==0.22.0
	scikit-learn==0.20.3
	scipy==1.1.0
	seaborn==0.9.0
	shap==0.24.0
	xgboost==0.80
	```	

0. In case of errors during setup, check out your installation of the following packages in Ubuntu or other Linux-based systems may help:
	```
	font-manager
	g++
	gcc
	python3-dev
	```	
    Or, upgrade your pip.	

	
### Data

0. Download all data files from [this link](https://entuedu-my.sharepoint.com/:f:/g/personal/yuhao001_e_ntu_edu_sg/EoOOorjtaEJBhZ6W-NIFPH4BcxM3yUAasf2C01Za2CewkQ) and put inside your local `data` folder. (Check out the [Code](#code) part for more information.)


### Code
0. Curves on ImageNet (solid lines: 1-crop val error; dashed lines: training error):
	![Training curves](https://cloud.githubusercontent.com/assets/11435359/13046277/e904c04c-d412-11e5-9260-efc5b8301e2f.jpg)

0. 1-crop validation error on ImageNet (center 224x224 crop from resized image with shorter side=256):

	model|top-1|top-5
	:---:|:---:|:---:
	[VGG-16](http://www.vlfeat.org/matconvnet/pretrained/)|[28.5%](http://www.vlfeat.org/matconvnet/pretrained/)|[9.9%](http://www.vlfeat.org/matconvnet/pretrained/)
	ResNet-50|24.7%|7.8%
	ResNet-101|23.6%|7.1%
	ResNet-152|23.0%|6.7%
	
0. 10-crop validation error on ImageNet (averaging softmax scores of 10 224x224 crops from resized image with shorter side=256), the same as those in the paper:

	model|top-1|top-5
	:---:|:---:|:---:
	ResNet-50|22.9%|6.7%
	ResNet-101|21.8%|6.1%
	ResNet-152|21.4%|5.7%
	
@ Yuhao Lu 2019
