# MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm

<p align="left">
    <a href='https://motionlab-anonymous.github.io/motionlab.github.io/'><img src='https://img.shields.io/badge/video-video-purple' alt='youtube video'></a>
    <a href='https://www.youtube.com/watch?v=X5yFMSJLNcE'><img src='https://img.shields.io/badge/project-project-blue' alt='project page'></a>
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
python: 3.9.20; torch: 2.1.1; pytorch-lightning: 1.9.4; cuda: 11.8.0;

```
conda env create -f environment.yml
conda activate rfmotion
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

### 2. Download Dependencies:
The results should be placed as shown in Folder Structure, including glove,t2m and smplx.
```
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

### 3.Prepare Datasets:
Download the [AMASS](https://amass.is.tue.mpg.de/) dataset and [MotionFix](https://github.com/atnikos/motionfix) dataset.

Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) to process the AMASS data into HumanML3D format, then copy the results into "all" as shown in Folder Structure.

Follow the instructions in [MotionFix-Retarget](https://github.com/MotionLab-Anonymous/MotionFix-Retarget) to process the MotionFix data into HumanML3D format, then copy the results into "all" as shown in Folder Structure.

### 4. Download Checkpoint for mcm-ldm:
The results should be placed as shown in Folder Structure, including [vae7.ckpt and motionclip.pth.tar](https://drive.google.com/drive/folders/1r6aDXpv_72whHxkJnSfOaJixavoer0Yf).

## Demo the MotionLab
You should first check the configure in ./configs/config_rfmotion.yaml

Importantly, the checkpoint and tasks of demo are assigned by:

      DEMO:
        TYPE: "text" # for text-based motion generation; sourcetext, sourcehint, hint, inbetween, style
        CHECKPOINTS: "./experiments/rfmotion/baseline/checkpoints/epoch=2199.ckpt"  # Pretrained model path
        
```
cd ./script
bash demo.sh
```

Notably, rendering the video directly here may result in poor export results, which may cause the video clarity to decrease and the lighting to be unclear. It is recommended to export the mesh and then render the video in professional 3D software.

## Train the MotionLab
You should first check the configure in ./configs/config_rfmotion.yaml
```
cd ./script
bash train_rfmotion.sh
```

## Evaluate the MotionLab
You should first check the configure in ./configs/config_rfmotion.yaml

Importantly, the checkpoint and evaluate metrics are assigned by: 

      TEST:
            CHECKPOINTS: ""  # Pretrained model path
              
      METRIC:
            TYPE: ["MaskedMetrics", "TM2TMetrics", "SourceTextMetrics", "SourceHintMetrics", "SourceTextHintMetrics", "InbetweenMetrics", "TextInbetweenMetrics","TextHintMetrics", "HintMetrics", "StyleMetrics", ]
```
cd ./script
bash test_rfmotion.sh
```

## Acknowledgements

Some codes are borrowed from [MLD](https://github.com/ChenFengYe/motion-latent-diffusion), [MotionFix](https://github.com/atnikos/motionfix), [MCM-LDM](https://github.com/XingliangJin/MCM-LDM), [diffusers](https://github.com/huggingface/diffusers).

## Citation
If you find MotionLab useful for your work please cite:
```
@article{       ,
  author    = {},
  title     = {MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm},
  journal   = {},
  year      = {2025},
}
```


