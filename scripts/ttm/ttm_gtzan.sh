#!/bin/bash

# 1. Compute latents
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=0 python -m compute_latents.gtzan \
		--dataset_root="./datasets/gtzan" \
		--split=${SPLIT} \
		--out_dir="./latents/gtzan/${SPLIT}/audio"
done

# 2. Create jsonls
for SPLIT in "train" "test"; do
	python -m create_jsonl.ttm.gtzan \
		--vae_dir="./latents/gtzan/${SPLIT}/audio" \
		--out_path="./jsonls/ttm/${SPLIT}/gtzan.jsonl"
done

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/ttm/ttm_gtzan.yaml"