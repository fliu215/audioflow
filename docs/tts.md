## 1. Training a Text to Speech System

### 1.1 Download datasets

Download the LJSpeech dataset corresponding to the task. 

GTZAN (1.3 GB, 8 hours):

```bash
bash ./scripts/download_datasets/download_gtzan.sh
```

The dataset structure after extraction is as follows:

<pre>
LJSpeech-1.1 (3.6 GB)
├── wavs (13,100 .wavs)
│   ├── LJ001-0001.wav
│   └── ...
├── metadata.csv
├── README
├── train.txt
├── valid.txt
└── test.txt
</pre>

### 1.2 Pre-extract VAE latents

```bash
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.ljspeech \
        --dataset_root="./datasets/LJSpeech-1.1" \
        --split=${SPLIT} \
        --out_dir="./latents/ljspeech/${SPLIT}/audio"
done
```

### 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m create_jsonls.tts.ljspeech \
        --dataset_root="./datasets/LJSpeech-1.1" \
        --latent_dir="./latents/ljspeech/${SPLIT}/audio" \
        --out_path="./jsonls/tts/${SPLIT}/ljspeech.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/tts/tts_ljspeech.yaml"
```

### 1.5 Sample
```python
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config="./configs/tts/tts_ljspeech.yaml" \
    --ckpt_path="checkpoints/train/tts_ljspeech/step=200000_ema.pth" \
    --task="text to speech" \
    --prompt="Today is a sunny day." \
    --out_path="out_tts.wav"
```