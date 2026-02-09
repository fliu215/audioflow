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
	CUDA_VISIBLE_DEVICES=4 python -m compute_latents.musdb18hq dac \
		--dataset_root="./datasets/musdb18hq" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./datasets/musdb18hq_vae/mixture_dac/${SPLIT}"
done

# 2. Create jsonls
for SPLIT in "train" "test"; do
	python -m create_jsonl.codec2music.musdb18hq \
		--input_vae_dir="./datasets/musdb18hq_vae/mixture_dac/${SPLIT}" \
		--target_vae_dir="./datasets/musdb18hq_vae/mixture/${SPLIT}" \
		--out_path="./jsonls/dac2stereo/${SPLIT}/musdb18hq.jsonl"
done

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/dac2stereo/dac2stereo_musdb18hq.yaml"