#!/bin/bash
source activate base
cd /13994058190/WYH/TOP-ReID
pip install timm
python train_net.py --config_file /13994058190/WYH/TOP-ReID/configs/RGBNT201/TOP-ReID.yml
