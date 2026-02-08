# ZePAD

The implementation of our ICLR 2026 paper "Zero-Sacrifice Persistent-Robustness Adversarial Defense for Pre-Trained Encoders".

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.8.0](https://img.shields.io/badge/pytorch-1.8.0-red.svg?style=plastic)

## Abstract

The widespread use of publicly available pre-trained encoders from self-supervised learning (SSL) has exposed a critical vulnerability: their susceptibility to downstream-agnostic adversarial examples (DAEs), which are crafted without knowledge of the downstream tasks but capable of misleading downstream models. While several defense methods have been explored recently, they rely primarily on task-specific adversarial fine-tuning, which inevitably limits generalizability and causes catastrophic forgetting and deteriorates benign performance.

Different from previous works, we propose a more rigorous defense goal that requires only a single tuning for diverse downstream tasks to defend against DAEs and preserve benign performance.

To achieve this defense goal, we introduce ***Zero-Sacrifice Persistent-Robustness Adversarial Defense (ZePAD)***, which is inspired by the inherent sensitivity of neural networks to data characteristics. Specifically, ZePAD is a dual-branch structure, which consists of a **Multi-Pattern Adversarial Enhancement Branch (MPAE-Branch)** that uses two adversarially fine-tuned encoders to strengthen adversarial resistance. The **Benign Memory Preservation Branch (BMP-Branch)** is trained on local data to ensure adversarial robustness does not compromise benign performance.

Surprisingly, we find that ZePAD can directly detect DAEs by evaluating branch confidence, without introducing any adversarial example identification task during training. Notably, by enriching feature diversity, our method enables a single adversarial fine-tuning to defend against DAEs across downstream tasks, thereby achieving persistent robustness.

Extensive experiments on **11 SSL methods** and **6 datasets** validate its effectiveness. In certain cases, it achieves a **29.20% improvement in benign performance** and a **73.86% gain in adversarial robustness**, highlighting its zero-sacrifice property.



## Setup
- **Get code**
```shell 
git clone https://github.com/Lawliet0o/ZePAD.git
```

- **Build environment**
```shell
cd ZePAD
# use anaconda to build environment 
conda create -n ZePAD python=3.8
conda activate ZePAD
# install packages
pip install -r requirements.txt
```
- **The final project should be like this:**
    ```shell
    dataset
    └- cifar10
    └- stl10
    └- ...
    ZePAD
    └- utils
        └- predict
        └- confidence
        └- data_loder
        └- drc
        └- gr
        └- load_model
    └- models
        └- linear
        └- resnet
        └- triple_model
    └- pretrained_encoders
        └- cifar10 (pre-training dataset)
          └- wmse
             └- wmse-cifar10-6z3m2p9o-ep=999.ckpt
    └- output
    └- atk_advencoder
        └- model
            └- adv_gan
            └- ...
        └- utils
            └- load_data
            └- ...
        └- gan_per_attack
    └- ...
    ```
    
- **Download pre-trained encoders**
  
  - All of our pre-trained encoders were obtained from the [solo-learn](https://github.com/vturrisi/solo-learn)  repository, and some missing pre-trained encoders were trained by us based on their code.
  - Please move the downloaded pre-trained encoder into  /pretrained_encoders/[pre-dataset]/[method].


### CIFAR-10

| Method       | Backbone | Epochs | Acc@1 | Acc@5 | Checkpoint |
|--------------|:--------:|:------:|:--------------:|:--------------:|:----------:|
| BYOL         | ResNet18 |  1000  |  92.58     |     99.79      | [Link](https://drive.google.com/drive/folders/1KxeYAEE7Ev9kdFFhXWkPZhG-ya3_UwGP?usp=sharing) |
| DINO         | ResNet18 |  1000  |  89.52     |     99.71      | [Link](https://drive.google.com/drive/folders/1vyqZKUyP8sQyEyf2cqonxlGMbQC-D1Gi?usp=sharing) |
| MoCo V2+     | ResNet18 |  1000  |  92.94     |     99.79      | [Link](https://drive.google.com/drive/folders/1ruNFEB3F-Otxv2Y0p62wrjA4v5Fr2cKC?usp=sharing) |
| MoCo V3      | ResNet18 |  1000  |  93.10     |     99.80      | [Link](https://drive.google.com/drive/folders/1KwZTshNEpmqnYJcmyYPvfIJ_DNwqtAVj?usp=sharing) |
| NNCLR        | ResNet18 |  1000  |  91.88     |     99.78      | [Link](https://drive.google.com/drive/folders/1xdCzhvRehPmxinphuiZqFlfBwfwWDcLh?usp=sharing) |
| ReSSL        | ResNet18 |  1000  |  90.63     |     99.62      | [Link](https://drive.google.com/drive/folders/1jrFcztY2eO_fG98xPshqOD15pDIhLXp-?usp=sharing) |
| SimCLR       | ResNet18 |  1000  |  90.74     |     99.75      | [Link](https://drive.google.com/drive/folders/1mcvWr8P2WNJZ7TVpdLHA_Q91q4VK3y8O?usp=sharing) |
| SwAV         | ResNet18 |  1000  |  89.17     |     99.68      | [Link](https://drive.google.com/drive/folders/1nlJH4Ljm8-5fOIeAaKppQT6gtsmmW1T0?usp=sharing) |
| VIbCReg      | ResNet18 |  1000  |  91.18     |     99.74      | [Link](https://drive.google.com/drive/folders/1XvxUOnLPZlC_-OkeuO7VqXT7z9_tNVk7?usp=sharing) |
| W-MSE        | ResNet18 |  1000  |  88.67     |     99.68      | [Link](https://drive.google.com/drive/folders/1xPCiULzQ4JCmhrTsbxBp9S2jRZ01KiVM?usp=sharing) |


### ImageNet-100

| Method                  | Backbone | Epochs | Acc@1 | Acc@5| Checkpoint |
|-------------------------|:--------:|:------:|:--------------:|:---------------:|:----------:|
| BYOL        | ResNet18 |   400  | 80.16     |     95.02       |  [Link](https://drive.google.com/drive/folders/1riOLjMawD_znO4HYj8LBN2e1X4jXpDE1?usp=sharing) |
| DINO                    | ResNet18 |   400  | 74.84     |     92.92       | [Link](https://drive.google.com/drive/folders/1NtVvRj-tQJvrMxRlMtCJSAecQnYZYkqs?usp=sharing) |
| MoCo V2+    | ResNet18 |   400  | 78.20     |     95.50       |  [Link](https://drive.google.com/drive/folders/1ItYBtMJ23Yh-Rhrvwjm4w1waFfUGSoKX?usp=sharing) |
| MoCo V3     | ResNet18 |   400  | 80.36     |     95.18       |  [Link](https://drive.google.com/drive/folders/15J0JiZsQAsrQler8mbbio-desb_nVoD1?usp=sharing) |
| NNCLR       | ResNet18 |   400  | 79.80     |     95.28       |  [Link](https://drive.google.com/drive/folders/1QMkq8w3UsdcZmoNUIUPgfSCAZl_LSNjZ?usp=sharing) |
| ReSSL                   | ResNet18 |   400  | 76.92     |     94.20       |   [Link](https://drive.google.com/drive/folders/1urWIFACLont4GAduis6l0jcEbl080c9U?usp=sharing) |
| SimCLR      | ResNet18 |   400  | 77.64     |     94.06        |    [Link](https://drive.google.com/drive/folders/1yxAVKnc8Vf0tDfkixSB5mXe7dsA8Ll37?usp=sharing) |
| SwAV                    | ResNet18 |   400  | 74.04     |     92.70       |   [Link](https://drive.google.com/drive/folders/1VWCMM69sokzjVoPzPSLIsUy5S2Rrm1xJ?usp=sharing) |
| VIbCReg                 | ResNet18 |   400  | 79.86     |     94.98       |   [Link](https://drive.google.com/drive/folders/1Q06hH18usvRwj2P0bsmoCkjNUX_0syCK?usp=sharing) |
| W-MSE                   | ResNet18 |   400  | 67.60     |     90.94       |    [Link](https://drive.google.com/drive/folders/1TxubagNV4z5Qs7SqbBcyRHWGKevtFO5l?usp=sharing) |



## Quick Start

This section illustrates a case that use ZePAD to protect an encoder.

* **Settings**:
* MAPE-Branch
  
  * Victim: W-MSE(pre-trained on CIFAR10)
  
  * Adv-AU model: BYOL(pre-trained on ImageNet)
  
* BMP-Branch
  
  * SimCLR(pre-trained on CIFAR10)
  
* Fine-tuning dataset: CIFAR10
  
* Attack: AdvEncoder([CGCL-codes/AdvEncoder: The implementation of our ICCV 2023 paper "Downstream-agnostic Adversarial Examples" (github.com)](https://github.com/CGCL-codes/AdvEncoder))
  
* Attacker's dataset: CIFAR10
  
* Downstream dataset: CIFAR10

- **Adversarial Fine-tuning**
  - Here, we select the W-MSE which is pre-trained on CIFAR10 as the victim model
  - The Adv-AU model is selected as BYOL which is pre-trained on ImageNet
  - Adversarial fine-tuning dataset is selected as CIFAR10
```shell 
python adversarial_fine-tuning.py --dataset cifar10 --pre_dataset cifar10 --ssl_method wmse
python adversarial_fine-tuning.py --dataset cifar10 --pre_dataset imagenet --ssl_method byol
```
- **Train Downstream Models**
  - We can use the "train_down_with_pretrained_encoder.py " to train the downstream models for clean(without adversarial fine-tuning) encoders
  - We can use the "train_down_singleE.py" to train the downstream models for the fine-tuned encoders
  - Here, the BMP-Branch is selected as SimCLR which is pre-trained on CIFAR10
  - So, in this case, we need to train the downstream models for these three encoders
```shell 
python train_down_with_pretrained_encoder.py --dataset cifar10 --pre_dataset cifar10 --ssl_method simclr
python train_down_singleE.py --dataset cifar10 --pre_dataset cifar10 --ft_dataset cifar10 --ssl_method wmse
python train_down_singleE.py --dataset cifar10 --pre_dataset imagenet --ft_dataset cifar10 --ssl_method byol
```

After these, we can get three downstream models which are stored in ./aft_downsteam_se and ./clean_downstream

* **Train Attacker's Perturbations** 

  ```shell
  python ./atk_advencoder/gan_per_attack.py --dataset cifar10 --victim wmse --pre_dataset cifar10
  ```

  After this, we can get the perturbations which is stored in ./advencoder, we have completed the training phrase.

* **Test with Baseline**

  * Here, we will test the performance on baseline(without any defense)

  ```shell
  python test_down_withoutDefense.py --dataset cifar10 --pre_dataset cifar10 --sup_dataset cifar10 --ssl_method wmse
  ```

  After running this command, we can get three metrics:

  * Benign Accuracy(BA): the accuracy on benign data
  * Adv Accuracy(we call RA in our paper): the accuracy on perturbed data
  * Attack Success Rate(ASR): a metric on attacker's view

* **Test with ZePAD**

```shell
python zepad_test.py --dataset cifar10 --pre_dataset cifar10 --ssl_method wmse --sup_dataset cifar10 --ft_dataset cifar10 --helper simclr
```

We can also get the same three metrics. 

## Pretrained Checkpoints

We provide a small set of representative pretrained checkpoints (via Git LFS)  to facilitate reproducibility and further research.

Due to storage considerations, only selected models are included. 

