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
git lfs pull
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
  - Please move the downloaded pre-trained encoder into  ./pretrained_encoders/[pre-dataset]/[method].


### 

## Quick Start

This section provides an example of how to apply **ZePAD** to protect a pre-trained encoder against downstream-agnostic adversarial examples.

* **Settings**:
  * MPAE-Branch
    * Victim: W-MSE (pre-trained on CIFAR10)

    * Adv-AU model: BYOL (pre-trained on ImageNet)

  * BMP-Branch
    * SimCLR (pre-trained on CIFAR10)

  * Fine-tuning dataset: CIFAR10
  * Attack: AdvEncoder(implementation from the ICCV 2023 paper *“Downstream-agnostic Adversarial Examples”*)  
      https://github.com/CGCL-codes/AdvEncoder
  * Attacker's dataset: CIFAR10
  * Downstream dataset: CIFAR10


- **Adversarial Fine-tuning**
  
  In this example, we perform adversarial fine-tuning under the following configuration:
  - The **victim encoder** is W-MSE pre-trained on CIFAR-10.
  - The **Adv-AU encoder** is BYOL pre-trained on ImageNet.
  - The **adversarial fine-tuning dataset** is CIFAR-10.
  
  Run the following commands to perform adversarial fine-tuning for each branch:
```shell 
python adversarial_fine-tuning.py --dataset cifar10 --pre_dataset cifar10 --ssl_method wmse
python adversarial_fine-tuning.py --dataset cifar10 --pre_dataset imagenet --ssl_method byol
```
- **Train Downstream Models**
  
  We train downstream classifiers for both clean (non-adversarially fine-tuned) encoders and adversarially fine-tuned encoders as follows:
  
  - `train_down_with_pretrained_encoder.py` is used to train downstream models on **clean encoders** (without adversarial fine-tuning).
  - `train_down_singleE.py` is used to train downstream models on **adversarially fine-tuned encoders**.
  - In this example, the **BMP-Branch encoder** is SimCLR pre-trained on CIFAR-10.
  - For comparison, we also train a downstream model using the victim encoder.
  - Consequently, we train downstream models for the following three encoders:
    1. SimCLR (clean, pre-trained on CIFAR-10)
    2. W-MSE (adversarially fine-tuned on CIFAR-10)
    3. BYOL (adversarially fine-tuned, pre-trained on ImageNet)
    4. W-MSE (clean, pre-trained on CIFAR-10)
  
  Run the following commands to train the downstream models:
```shell 
python train_down_with_pretrained_encoder.py --dataset cifar10 --pre_dataset cifar10 --ssl_method simclr
python train_down_with_pretrained_encoder.py --dataset cifar10 --pre_dataset cifar10 --ssl_method wmse
python train_down_singleE.py --dataset cifar10 --pre_dataset cifar10 --ft_dataset cifar10 --ssl_method wmse
python train_down_singleE.py --dataset cifar10 --pre_dataset imagenet --ft_dataset cifar10 --ssl_method byol
```

​	After training, the resulting downstream models are saved in:

​	`./clean_downstream`

​	`./aft_downstream_se`

* **Train Attacker's Perturbations** 

  We first train the attacker to generate downstream-agnostic adversarial perturbations using AdvEncoder:

  ```shell
  python ./atk_advencoder/gan_per_attack.py --dataset cifar10 --victim wmse --pre_dataset cifar10
  ```

  After training, the generated perturbations are saved in `./advencoder`.
  At this point, the training phase is completed.

* **Evaluation on Baseline (No Defense)**

  We evaluate the downstream model **without any defense mechanism** as a baseline.

  ```shell
  python test_down_withoutDefense.py --dataset cifar10 --pre_dataset cifar10 --sup_dataset cifar10 --ssl_method wmse
  ```

  * This command reports the following three metrics:
    - **Benign Accuracy (BA)**: accuracy on clean (benign) samples.
    - **Robust Accuracy (RA)**: accuracy on adversarially perturbed samples (referred to as *Adv Accuracy* in some scripts).
    - **Attack Success Rate (ASR)**: the success rate of the attacker from the adversarial perspective.
  
* **Test with ZePAD**

​	We then evaluate the downstream model protected by **ZePAD**:

```shell
python zepad_test.py --dataset cifar10 --pre_dataset cifar10 --ssl_method wmse --sup_dataset cifar10 --ft_dataset cifar10 --helper simclr
```

​	The same three metrics (**BA**, **RA**, and **ASR**) are reported for comparison.



## BibTeX

If you find **ZePAD** useful for your research, please consider citing our paper:

```bibtex
@inproceedings{Lei_2026_ZePAD,
  title     = {Zero-Sacrifice Persistent-Robustness Adversarial Defense for Pre-Trained Encoders},
  author    = {Zhuxin Lei, Ziyuan Yang and Yi Zhang},
  booktitle = {Proceedings of the International Conference on Learning Representations},
  year      = {2026}
}
