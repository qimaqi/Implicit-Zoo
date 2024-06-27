<h1 align="left">Implicit-Zoo 🦜: A Large-Scale Dataset of Neural Implicit Functions for 2D Images and 3D Scenes
 <a href="#Arxiv"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a> </h1> 

<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#download-link">Download Link</a> |
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#results-demo">Results Demo</a> |
  <a href="#news">News</a> |
  <!-- <a href="#statement">Statement</a> |
  <a href="#reference">Reference</a> -->
</p>




## Introduction
![Local Image](/asset/application.png)


This repository contains the download link, example code, test results for the paper Implicit-Zoo: A Large-Scale Dataset of Neural Implicit Functions for 2D Images and 3D Scenes. It showcase the effectiveness of optimizing monocular camera poses as a continuous function of time with neural network.

We have released the demo code, more details will be released soon, please check news for details.

## Download Link
Here we provide download link for [CIFAR-10-INRS](https://www.kaggle.com/datasets/alexanderqi/cifar10-inrs-dataset/data), [ImageNet-100-INRs](https://www.kaggle.com/datasets/alexanderqi/imagenet100-inrs-dataset) and [Omniobject3D from Kaggle](https://www.kaggle.com/datasets/alexanderqi/omniobject-inrs/data). For data size reason you can find ImageNet-1K-INRs in [google drive link](https://drive.google.com/drive/folders/1VJ9LMzFb1uiizhS9BzHUN4w-_R1-qyil?usp=drive_link). For CityScapes-INRs the cityscapes we will actively discuss this detail with the Cityscapes team and provide an update as soon as possible.


## Installation
```bash
conda env create --file environment.yml
conda activate implicit_zoo
```

## Quick Start
An introduction notebook for Dataset visualization: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qimaqi/Implicit-Zoo/blob/main/notebooks/data_visualization_demo.ipynb)



### Generate CIFAR-DATA
```bash
bash cifar_main_exps.sh
```
Note that you can customize config in **experiments/cifar_generate_configs/main.yaml**  Like customize network depth and width or training iteration times. Moreover the default CIFAR data installed place is in **./data**.  You can also change in code  **experiments/generate_cifar_dataset_siren.py** line 54.


## Results Demo
#### Time Cost
![Local Image](/asset/time-cost.png)
#### Visualize of queried Images 
![Local Image](/asset/quality_check_tight.jpg)
#### Diagram of learnable token
![Local Image](/asset/learnable_token.jpg)
#### CIFAR-10 Exps results
![Local Image](/asset/cifar-results.png)
![Local Image](/asset/cifar_token_loc.png)

# News
- [x] Create the repo
- [x] upload CIFAR-10 Dataset
<!-- - [x] upload CIFAR-10 Generate code
- [x] upload CIFAR-10 Experiments code -->
- [x] upload ImageNet-100 Dataset
- [x] upload ImageNet-1k Dataset
- [x] upload Omniobject3D Dataset

