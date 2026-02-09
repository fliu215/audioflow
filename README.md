# AudioFlow: Audio Generation with Flow Matching

This repository contains a tutorial on audio generation using conditional flow matching implemented in PyTorch. Signals from any modality, including text, audio, MIDI, images, and video, can be converted to audio using conditional flow matching. The figure below shows the framework.

The supported tasks include:

| Tasks                   | Supported    | Dataset    | Config yaml                                                  |
|-------------------------|--------------|------------|--------------------------------------------------------------|
| Text to music           | ✅           | GTZAN      | [scripts/ttm.sh](scripts/ttm.sh)                             |
| Text to speech          | ✅           | GTZAN      | [scripts/tts.sh](scripts/tts.sh)                             |
| Text to audio           | ✅           | GTZAN      | [scripts/tta.sh](scripts/tta.sh)                             |
| MIDI to music           | ✅           | MAESTRO    | To appear                                                    |
| Codec to audio          | ✅           | MUSDB18HQ  | [scripts/dac2stereo.sh](scripts/dac2stereo.sh)               |
| Mono to stereo          | ✅           | MUSDB18HQ  | [scripts/mono2stereo.sh](scripts/mono2stereo.sh)             |
| Super resolution        | ✅           | MUSDB18HQ  | [scripts/superresolution.sh](scripts/superresolution.sh)     |
| Music source separation | ✅           | MUSDB18HQ  | [scripts/mss.sh](scripts/mss.sh)                             |
| Vocals to music         | ✅           | MUSDB18HQ  | [scripts/vocals2mixture.sh](scripts/vocals2mixture.sh)       |


## 0. Install dependencies

```bash
# Clone the repo
git clone https://github.com/qiuqiangkong/audio_flow
cd audio_flow

# Install Python environment
conda create --name audio_flow python=3.10

# Activate environment
conda activate audio_flow

# Install Python packages dependencies
bash env.sh
```

We start from a text to music example as follows.

## 0. Download datasets

Download the dataset corresponding to the task. 

GTZAN (1.3 GB, 8 hours):

```bash
bash ./scripts/download_gtzan.sh
```

MUSDB18HQ (30 GB, 10 hours):

```bash
bash ./scripts/download_musdb18hq.sh
```

To download more datasets please see [scripts](scripts).

## 1. Pre-extract VAE latents

```bash
for SPLIT in "train" "test"; do
    CUDA_VISIBLE_DEVICES=6 python -m compute_latents.gtzan \
        --dataset_root="./datasets/gtzan" \
        --split=${SPLIT} \
        --out_dir="./datasets/gtzan_vae/${SPLIT}"
done
```

## 2 Prepare JSONL files

```bash
for SPLIT in "train" "test"; do
    python -m create_jsonl.ttm.gtzan \
        --split=${SPLIT} \
        --vae_dir="./datasets/gtzan_vae/${SPLIT}" \
        --out_path="./jsonls/ttm/${SPLIT}/gtzan.jsonl"
done
```

## 3. Train
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/ttm/ttm_gtzan.yaml"
```

To run more examples please see [configs](configs).

## External links

[1] Conditional flow matching: https://github.com/atong01/conditional-flow-matching

[2] DiT: https://github.com/facebookresearch/DiT