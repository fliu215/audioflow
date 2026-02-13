#!/bin/bash

# 1. Compute latents
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=3 python -m compute_latents.ljspeech \
		--dataset_root="./datasets/LJSpeech-1.1" \
		--split=${SPLIT} \
		--out_dir="./latents/ljspeech/${SPLIT}/audio"
done

# 2. Create jsonls
for SPLIT in "train" "test"; do
	python -m create_jsonl.tts.ljspeech \
		--dataset_root="./datasets/LJSpeech-1.1" \
		--vae_dir="./latents/ljspeech/${SPLIT}/audio" \
		--out_path="./jsonls/tts/${SPLIT}/ljspeech.jsonl"
done

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/tts/tts_ljspeech.yaml"