#!/bin/bash

# 1. Compute latents
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=3 python -m compute_latents.clotho \
		--dataset_root="./datasets/clotho2.1" \
		--split=${SPLIT} \
		--out_dir="./datasets/clotho2.1/${SPLIT}"
done

# 2. Create jsonls
for SPLIT in "train" "test"; do
	python -m create_jsonl.tta.clotho \
		--dataset_root="./datasets/clotho2.1" \
		--split=${SPLIT} \
		--vae_dir="./datasets/clotho2.1/${SPLIT}" \
		--out_path="./jsonls/tta/${SPLIT}/clotho2.1.jsonl"
done

# 3. Train
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/tta/tta_clotho.yaml"