# AudioFlow: Audio Generation with Flow Matching

This repository contains a tutorial on audio generation using conditional flow matching implemented in PyTorch. Signals from any modality, including text, audio, MIDI, images, and video, can be converted to audio using conditional flow matching. The figure below shows the framework.

The supported tasks include:

| Tasks                   | Supported    | Dataset    | Config yaml                                                  |
|-------------------------|--------------|------------|--------------------------------------------------------------|
| Text to music           | ✅           | GTZAN      | [configs/text2music.yaml](configs/text2music.yaml)           |
| MIDI to music           | ✅           | MAESTRO    | [configs/midi2music.yaml](configs/midi2music.yaml)           |
| Codec to audio          | ✅           | MUSDB18HQ  | [configs/codec2audio.yaml](configs/codec2audio.yaml)         |
| Mono to stereo          | ✅           | MUSDB18HQ  | [configs/mono2stereo.yaml](configs/mono2stereo.yaml)         |
| Super resolution        | ✅           | MUSDB18HQ  | [configs/superresolution.yaml](configs/superresolution.yaml) |
| Music source separation | ✅           | MUSDB18HQ  | [configs/mss.yaml](configs/mss.yaml)                         |
| Vocal to music          | ✅           | MUSDB18HQ  | [configs/vocal2music.yaml](configs/vocal2music.yaml)         |


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

## 1. Download datasets

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

## 2. Train

### 2.0 Pre-extract VAE latent

In training, uses can use (1) online VAE extraction, or (2) offline VAE extraction. We adopt (2) to speed up the training of flow matching and to save RAM. 

```python
CUDA_VISIBLE_DEVICES=0 python -m compute_latents.gtzan_vae \
  --dataset_root="./datasets/gtzan" \
  --out_dir="./datasets/gtzan_vae" \
  --augmentation_repeats=10
```

### 2.1 Train with single GPU

Here is an example of training a text to music generation system. Users can train different tasks viewing more config yaml files at [configs](configs).

```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/text2music.yaml" --no_log
```

### 2.2 Finetune

Extract VAE latent:

```python
CUDA_VISIBLE_DEVICES=6 python -m compute_latents.musdb18hq_vae stems \
  --dataset_root="./datasets/musdb18hq" \
  --out_dir="./datasets/musdb18hq_vae" \
  --augmentation_repeats=10
```

Train:

```python
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --config="./configs/mss.yaml" \
  --ckpt_path="checkpoints/train/text2music/step=300000_ema.pth" \
  --no_log
```

To run more examples please see [configs](configs).

## External links

[1] Conditional flow matching: https://github.com/atong01/conditional-flow-matching

[2] DiT: https://github.com/facebookresearch/DiT