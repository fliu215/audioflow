#!/bin/bash

# 1. Compute latents
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.musdb18hq stereo \
		--dataset_root="./datasets/musdb18hq/" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./datasets/musdb18hq_vae/mixture/${SPLIT}"
done

for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.musdb18hq stereo \
		--dataset_root="./datasets/musdb18hq" \
		--stem="vocals" \
		--split=${SPLIT} \
		--out_dir="./datasets/musdb18hq_vae/vocals/${SPLIT}"
done

# 2. Create jsonls
for SPLIT in "train" "test"; do
	python -m create_jsonl.mono2stereo.musdb18hq \
		--input_vae_dir="./datasets/musdb18hq_vae/vocals/${SPLIT}" \
		--target_vae_dir="./datasets/musdb18hq_vae/mixture/${SPLIT}" \
		--out_path="./jsonls/vocals2mixture/${SPLIT}/musdb18hq.jsonl"
done

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/vocals2mixture/vocals2mixture_musdb18hq.yaml"