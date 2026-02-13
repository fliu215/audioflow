#!/bin/bash

# 1. Compute latents
# Mixture latents
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=0 python -m compute_latents.musdb18hq stereo \
		--dataset_root="./datasets/musdb18hq/" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture"
done

# DAC latents
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=4 python -m compute_latents.musdb18hq dac \
		--dataset_root="./datasets/musdb18hq" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture_dac"
done

# 2. Create jsonls
for SPLIT in "train" "test"; do
	python -m create_jsonl.codec2music.musdb18hq \
		--input_vae_dir="./latents/musdb18hq/${SPLIT}/mixture_dac" \
		--target_vae_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--out_path="./jsonls/dac2stereo/${SPLIT}/musdb18hq.jsonl"
done

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/codec2audio/dac2stereo_musdb18hq.yaml"