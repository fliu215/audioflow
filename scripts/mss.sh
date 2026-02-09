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
	INPUT_STEM="mixture"
	TARGET_STEM="vocals"
	python -m create_jsonl.mss.musdb18hq \
		--input_stem=${INPUT_STEM} \
		--target_stem=${TARGET_STEM} \
		--input_vae_dir="./datasets/musdb18hq_vae/${INPUT_STEM}/${SPLIT}" \
		--target_vae_dir="./datasets/musdb18hq_vae/${TARGET_STEM}/${SPLIT}" \
		--out_path="./jsonls/mss/${INPUT_STEM}2${TARGET_STEM}/${SPLIT}/musdb18hq.jsonl"
done

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/mss/mixture2vocals_musdb18hq.yaml"