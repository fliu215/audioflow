# AudioFlow: Audio Generation with Flow Matching for the CCF-AATC 2026 Challenge

This repository provides a baseline for the CCF-AATC 2026 Challenge [Track 1](https://ccf-aatc.org.cn/), which aims to do music restoration under multiple distortions:
- Distortion from poor-quality amplification equipment
- Multi-speaker amplification and reverberation

## 0. Install dependencies

```bash
# Install Python environment
conda create --name audio_flow python=3.10

# Activate environment
conda activate audio_flow
cd audio_flow

# Install Python packages dependencies
bash env.sh
```

## 1. Baseline Training

### 1.1 Download datasets

Download the officially provided [dataset](https://ccf-aatc.org.cn/) (10 hours):

The dataset structure is as follows:

<pre>
dataset
├── train (202 files)
│   ├── 安静街道__笔记本_低_1M
│   │   ├── 原始.wav
│   │   ├── pcm01.wav
│   │   ├── pcm02.wav
│   │   ├── pcm03.wav
│   │   └── phone.wav
│   ... 
│   └── ...
└── valid (25 files)
    ├── 户外公园__平板_高_5M
    │   ├── 原始.wav
    │   ├── pcm01.wav
    │   ├── pcm02.wav
    │   ├── pcm03.wav
    │   └── phone.wav
    ... 
    └── ...
</pre>
The `原始.wav` file is the target audio. `phone.wav` is the degraded audio recorded with a mobile phone and requires restoration. `pcm01/02/03.wav` are degraded recordings captured by three microphones. The three microphones are arranged in a linear array, with microphone spacings of 40 mm and 120 mm. The microphone recordings are sampled at 48 kHz, while all other audio files are sampled at 16 kHz. In the baseline setting, `phone.wav` is used as the input audio, and the `原始.wav` file is used as the target.

### 1.2 Pre-extract VAE latents

```bash
# Mixture
for SPLIT in "train" "valid"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.musdb18hq stereo \
        --dataset_root="./datasets" \
        --stem="phone" \
        --split=${SPLIT} \
        --latent_type="mmaudio_vae" \
        --out_dir="./latents/aatc/${SPLIT}/mixture"
done

# Target
for SPLIT in "train" "valid"; do
    CUDA_VISIBLE_DEVICES=0 python -m compute_latents.musdb18hq stereo \
        --dataset_root="./datasets" \
        --stem="原始" \
        --split=${SPLIT} \
        --latent_type="mmaudio_vae" \
        --out_dir="./latents/aatc/${SPLIT}/target"
done
```

### 1.3 Prepare JSONL files

```bash
for SPLIT in "train" "valid"; do
    python -m create_jsonls.mss.musdb18hq \
        --input_latent_dir="./latents/aatc/${SPLIT}/mixture" \
        --target_latent_dir="./latents/aatc/${SPLIT}/target" \
        --out_path="./jsonls/mss/${SPLIT}/aatc.jsonl"
done
```

### 1.4 Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/mss/mss_musdb18hq.yaml"
```

### 1.5 Sample
```python
# Single
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --config="./configs/mss/mss_musdb18hq.yaml" \
    --ckpt_path="checkpoints/train/mss_musdb18hq/step=1000000_ema.pth" \
    --task="music source separation" \
    --duration=10 \
    --input_path="./assets/music_10s.wav" \
    --out_path="out_mss.wav"
# Batch
CUDA_VISIBLE_DEVICES=0 python batch_sample_chunked.py \
  --config "./configs/mss/mss_musdb18hq.yaml" \
  --ckpt_path "checkpoints/train/mss_musdb18hq/step=1000000_ema.pth" \
  --dataset_root "datasets/test" \
  --input_filename "phone" \
  --out_dir "batch_results" \
  --chunk_duration "10" \
  --overlap_duration "2" \
  --output_sr "16000" \
  --mono_output \
  --skip_existing
```

### 1.6 Evaluate
```
python evaluate_mss_metrics.py --compute_fad --compute_visqol --compute_input_metrics
```

## 2. Results
The validation results of the baseline system are shown below.

| Method | SI-SNR (dB) ↑ | LSD ↓ | FAD ↓ | ViSQOL ↑ |
|:---:|:---:|:---:|:---:|:---:|
| Input | -46.91 | 1.91 | 22.70 | 3.37 |
| Baseline | -50.50 | 1.71 | 8.90 | 3.45 |

## External links

[1] Conditional flow matching: https://github.com/atong01/conditional-flow-matching

[2] DiT: https://github.com/facebookresearch/DiT
