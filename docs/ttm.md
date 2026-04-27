## 1. Training a text to music system

## 1.1 Download datasets

Download the dataset corresponding to the task. 

GTZAN (1.3 GB, 8 hours):

```bash
bash ./scripts/download_gtzan.sh
```

## 1.2 Pre-extract VAE latents

```bash
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.gtzan \
        --dataset_root="./datasets/gtzan" \
        --split=${SPLIT} \
        --out_dir="./latents/gtzan/${SPLIT}/audio"
done
```

## 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m create_jsonls.ttm.gtzan \
        --latent_dir="./latents/gtzan/${SPLIT}/audio" \
        --out_path="./jsonls/ttm/${SPLIT}/gtzan.jsonl"
done
```

## 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/ttm/ttm_gtzan.yaml"
```

## 1.5 Sample
```python
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config="./kqq_configs/ttm/ttm_gtzan.yaml" \
    --ckpt_path="checkpoints/train/ttm_gtzan/step=200000_ema.pth" \
    --task="text to music" \
    --prompt="blues" \
    --out_path="out_ttm.wav"
```