#!/bin/bash

# Compute latents
# Stereo latents
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.musdb18hq stereo \
		--dataset_root="./datasets/musdb18hq/" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture"
done

# Mono latents
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.musdb18hq mono \
		--dataset_root="./datasets/musdb18hq" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture_mono"
done

# 2. Create jsonls
for SPLIT in "train" "test"; do
	python -m create_jsonl.mono2stereo.musdb18hq \
		--input_vae_dir="./latents/musdb18hq/${SPLIT}/mixture_mono" \
		--target_vae_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--out_path="./jsonls/mono2stereo/${SPLIT}/musdb18hq.jsonl"
done

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/mono2stereo/mono2stereo_musdb18hq.yaml"