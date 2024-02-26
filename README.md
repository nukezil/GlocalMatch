# GlocalMatch for Open-Domain SSL

## Introduction

This repository hosts the code for our paper accepted by IEEE Transactions on Knowledge and Data Engineering (TKDE):

>**Open-Domain Semi-Supervised Learning via Glocal Cluster Structure Exploitation** </br>
> Zekun Li, Lei Qi, Yawen Li, Yinghuan Shi*, Yang Gao</br>

[[`Preprint`](https://nukezil.github.io/files/TKDE_GlocalMatch_Preprint.pdf)][[`BibTeX`](#citation)]

## Preparation

### Required Packages

We suggest first creating a conda environment:

```sh
conda create --name glocalmatch python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

### Datasets

Please download the data files from [this link](https://www.dropbox.com/scl/fi/e9ndhafngjbxaefj18yqd/opendomain_data.zip?rlkey=kcs2zsxy7vy95cz48igwjyqxu&dl=0) and put them into the ``./data`` folder:

```
GlocalMatch
├── config
    └── ...
├── data
    ├── cifarstl
        └── cifar
        └── stl
    └── domainnet_balanced
        └── ...
    └── pacs
        └── ...
├── semilearn
    └── ...
└── ...  
```

## Usage
We implement [GlocalMatch](./semilearn/algorithms/glocalmatch/glocalmatch.py) using the codebase of [USB](https://github.com/microsoft/Semi-supervised-learning).

Here is an example to train GlocalMatch on the CIFAR-STL benchmark, with CIFAR as the labeled domain, and  45 labels available per class.
```sh
# seed = 1
CUDA_VISIBLE_DEVICES=0 python train.py --c config/opendomain_cv/glocalmatch/glocalmatch_cifarstl_c45_1.yaml
```
For other tasks, the config files have been released in ``./config/opendomain_cv``.

## Citation

```bibtex
@article{glocalmatch,
  title={Open-Domain Semi-Supervised Learning via Glocal Cluster Structure Exploitation},
  author={Li, Zekun and Qi, Lei and Li, Yawen and Shi, Yinghuan and Gao, Yang},
  journal={IEEE TKDE},
  year={2024}
}
```
