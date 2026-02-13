#!/bin/bash

# 1. Compute latents
# Mixture latents
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.musdb18hq stereo \
		--dataset_root="./datasets/musdb18hq/" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture"
done

# Vocals latents
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.musdb18hq stereo \
		--dataset_root="./datasets/musdb18hq" \
		--stem="vocals" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/vocals"
done

# 2. Create jsonls
for SPLIT in "train" "test"; do
	python -m create_jsonl.mono2stereo.musdb18hq \
		--input_vae_dir="./latents/musdb18hq/${SPLIT}/vocals" \
		--target_vae_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--out_path="./jsonls/vocals2mixture/${SPLIT}/musdb18hq.jsonl"
done

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/vocals2mixture/vocals2mixture_musdb18hq.yaml"