source activate
conda activate rfmotion
export CUDA_VISIBLE_DEVICES=0 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

cd ..
python demo.py \
--cfg ./configs/config_rfmotion.yaml \
--cfg_assets ./configs/assets.yaml \