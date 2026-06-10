## 1. Training a Text to Audio System

### 1.1 Download datasets

Download the AudioCaps dataset (131 GB) from the official website.

The dataset structure after extraction is as follows:

<pre>
audiocaps (131 GB)
├── train (49274 files)
├── val (494 files)
├── test (957 files)
├── train.csv
├── val.csv
├── test.csv
├── LICENSE.txt
└── README.md
</pre>

### 1.2 Pre-extract VAE latents

```bash
# Extract audio latent
ENCODER="mmaudio_vae"
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m scripts.extract_features.audiocaps audio \
        --dataset_root="./datasets/audiocaps2.0" \
        --split=${SPLIT} \
        --encoder_name=${ENCODER} \
        --out_dir="./features/audiocaps2.0/${SPLIT}/audio/${ENCODER}"
done

# Extract text
for SPLIT in "train" "test"; do
    python -m scripts.extract_features.audiocaps text \
        --dataset_root="./datasets/audiocaps2.0" \
        --split=${SPLIT} \
        --out_dir="./features/audiocaps2.0/${SPLIT}/text/raw"
done
```

### 1.3 Prepare JSONL files

```bash
ENCODER="mmaudio_vae"
for SPLIT in "train" "test"; do
    python -m tools.create_jsonls.tta.audiocaps \
        --input_texts_dir="./features/audiocaps2.0/${SPLIT}/text/raw" \
        --target_latents_dir="./features/audiocaps2.0/${SPLIT}/audio/${ENCODER}" \
        --out_path="./jsonls/tta/${SPLIT}/audiocaps2.0.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config="./configs/tta/tta_audiocaps.yaml"
```

### 1.5 Sample
```python
CUDA_VISIBLE_DEVICES=0 python -m scripts.sample \
    --config="./configs/ttm/ttm_gtzan.yaml" \
    --ckpt_path="checkpoints/train/ttm_gtzan/step=200000_ema.pth" \
    --prompt="<audio>a dog barking</audio>" \
    --out_path="gen_ttm.wav"
```