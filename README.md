# MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm

<p align="left">
<!--     <a href='https://arxiv.org/abs/2210.09729'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://silvester.wang/HUMANISE/paper.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a> -->
    <a href='https://motionlab-anonymous.github.io/motionlab.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>


## Folder Structure
```
├── checkpoints
│   ├── clip-vit-large-patch14
│   ├── glove
│   ├── mdm-ldm
│   │   ├── vae7.ckpt
│   │   ├── motionclip.pth.tar
│   ├── smpl
│   │   ├── SMPL_NEUTRAL.pkl
│   ├── smplh
│   │   ├── SMPLH_NEUTRAL.npz
│   ├── t2m
│   │   ├── Comp_v6_KLD01
├── datasets
│   ├── all
│   │   ├── new_joint_vecs
│   │   ├── new_joints
│   │   ├── texts
│   │   ├── train_humanml.txt
│   │   ├── train_motionfix.txt
│   │   ├── val_humanml.txt
│   │   ├── val_motionfix.txt
│   │   ├── test_humanml.txt
│   │   ├── test_motionfix.txt
├── experiments
│   ├── rfmotion
│   │   ├── SPECIFIED NAME OF EXPERIMENTS
│   │   │   ├── checkpoints
```


### 1. Setup Conda:
```
conda env create -f environment.yml
conda activate rfmotion
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```
