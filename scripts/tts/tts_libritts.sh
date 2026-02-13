#!/bin/bash

# 1. Compute latents
SPLIT="train"
for SUBDIR in "train-clean-100" "train-clean-360" "train-other-500"; do
	CUDA_VISIBLE_DEVICES=3 python -m compute_latents.libritts \
		--audios_dir="./datasets/LibriTTS/${SUBDIR}" \
		--out_dir="./latents/libritts/${SPLIT}/${SUBDIR}/audio" \
		--csv_path="./latents/libritts/${SPLIT}/${SUBDIR}/text.csv"
done

SPLIT="test"
for SUBDIR in "test-clean" "test-other"; do
	CUDA_VISIBLE_DEVICES=3 python -m compute_latents.libritts \
		--audios_dir="./datasets/LibriTTS/${SUBDIR}" \
		--out_dir="./latents/libritts/${SPLIT}/${SUBDIR}/audio" \
		--csv_path="./latents/libritts/${SPLIT}/${SUBDIR}/text.csv"
done

# 2. Create jsonls
SPLIT="train"
for SUBDIR in "train-clean-100" "train-clean-360" "train-other-500"; do
	python -m create_jsonl.tts.libritts \
		--vae_dir="./latents/libritts/${SPLIT}/${SUBDIR}/audio" \
		--csv_path="./latents/libritts/${SPLIT}/${SUBDIR}/text.csv" \
		--out_path="./jsonls/tts/${SPLIT}/libritts_${SUBDIR}.jsonl"
done

SPLIT="test"
for SUBDIR in "test-clean" "test-other"; do
	python -m create_jsonl.tts.libritts \
		--vae_dir="./latents/libritts/${SPLIT}/${SUBDIR}/audio" \
		--csv_path="./latents/libritts/${SPLIT}/${SUBDIR}/text.csv" \
		--out_path="./jsonls/tts/${SPLIT}/libritts_${SUBDIR}.jsonl"
done

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/tts/tts_libritts.yaml"