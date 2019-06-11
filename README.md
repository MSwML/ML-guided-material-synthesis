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



### Citation

If you use these models in your research, please cite:

	@article{Tang2019,
		author = {Bijun Tang, Yuhao Lu, Jiadong Zhou, Han Wang, Prafful Golani, Manzhang Xu, Quan Xu, Cuntai Guan, Zheng Liu},
		title = {Machine learning-guided synthesis of advanced inorganic materials},
		journal = {arXiv preprint arXiv:1905.03938},
		year = {2019}
	}

### Environment Setup

0. Python environment setup:
	```
	python 3.6.6
	jupyter==1.0.0
	matplotlib==2.2.3
	numpy==1.16.0
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

0. Download all data files from [this link](https://entuedu-my.sharepoint.com/:f:/g/personal/yuhao001_e_ntu_edu_sg/EoOOorjtaEJBhZ6W-NIFPH4BcxM3yUAasf2C01Za2CewkQ) and put inside your local `data` folder. (See [Code](#code) below for more information.)

0. For more detailed description of the dataset, please check out our [paper](#introduction).


### Code
0. Code structure:
	- **scripts** 
		- run_ipynb.sh `: script to run all *.ipynb`
	- **results** `: folder to store all results and generated figures`
	- **utils** `: supporting functions`
	- **data** `: download data before running code (see [Data](#code))`
	- PAM_repeat1000times-\*\.py `: to repeat 1000 times of PAM with randomly selected initial training sets`
	- PAM_guidedSynthesis-\*\.ipynb `: to run 1 run of PAM, and plot the figures`
	- model_selection-\*\.ipynb `: to select best model with 10 repetitions of 10 X 10 cross validation; plus result interpretation`
	- data_overview.ipynb `: to plot feature correlation of dataset, and compute other descriptive statistics`
	- best_model_interpretation-\*\.ipynb `: to extract feature attribution values; and predict on generated input`	
	
	
**Note:**
*File names end with '-classification' are for classification or MoS2 dataset, while those end with '-regression' are for regression or CQD dataset.*


0. To run 

	model|top-1|top-5
	:---:|:---:|:---:
	[VGG-16](http://www.vlfeat.org/matconvnet/pretrained/)|[28.5%](http://www.vlfeat.org/matconvnet/pretrained/)|[9.9%](http://www.vlfeat.org/matconvnet/pretrained/)
	ResNet-50|24.7%|7.8%
	ResNet-101|23.6%|7.1%
	ResNet-152|23.0%|6.7%
	
	![Training curves](https://cloud.githubusercontent.com/assets/11435359/13046277/e904c04c-d412-11e5-9260-efc5b8301e2f.jpg)
	
