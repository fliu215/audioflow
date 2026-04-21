#!/bin/bash

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/omni/omni.yaml"