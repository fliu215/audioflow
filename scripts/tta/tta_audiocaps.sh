#!/bin/bash

# 1. Compute latents
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.audiocaps \
		--dataset_root="./datasets/audiocaps2.0" \
		--split=${SPLIT} \
		--out_dir="./latents/audiocaps2.0/${SPLIT}/audio"
done

# 2. Create jsonls
for SPLIT in "train" "test"; do
	python -m create_jsonl.tta.audiocaps \
		--dataset_root="./datasets/audiocaps2.0" \
		--split=${SPLIT} \
		--vae_dir="./latents/audiocaps2.0/${SPLIT}/audio" \
		--out_path="./jsonls/tta/${SPLIT}/audiocaps2.0.jsonl"
done

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/tta/tta_audiocaps.yaml"