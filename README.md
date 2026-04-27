# AudioFlow: Audio Generation with Flow Matching

This repository contains a tutorial on audio generation using conditional flow matching implemented in PyTorch. Signals from any modality, including text, audio, MIDI, images, and video, can be converted to audio using conditional flow matching. The figure below shows the framework.

The supported tasks include:

| Tasks                   | Supported    | Dataset    | Config yaml                                                  |
|-------------------------|--------------|------------|--------------------------------------------------------------|
| Text to music           | ✅           | GTZAN      | [scripts/ttm/ttm_gtzan.sh](scripts/ttm/ttm_gtzan.sh)                             |
| Text to speech          | ✅           | LJSpeech <br> LibriTTS      | [scripts/tts/tts_ljspeech.sh](scripts/tts/tts_ljspeech.sh) <br> [scripts/tts/tts_libritts.sh](scripts/tts/tts_libritts.sh)                            |
| Text to audio           | ✅           | Clotho <br> AudioCaps      | [scripts/tta/tta_clotho.sh](scripts/tta/tta_clotho.sh) <br> [scripts/tta/tta_audiocaps.sh](scripts/tta/tta_audiocaps.sh)                            |
| MIDI to music           | ✅           | MAESTRO    | To appear                                                    |
| Codec to audio          | ✅           | MUSDB18HQ  | [scripts/codec2audio/dac2stereo.sh](scripts/codec2audio/dac2stereo.sh)               |
| Mono to stereo          | ✅           | MUSDB18HQ  | [scripts/mono2stereo/mono2stereo_musdb18hq.sh](scripts/mono2stereo/mono2stereo_musdb18hq.sh)             |
| Super resolution        | ✅           | MUSDB18HQ  | [scripts/superresolution/superresolution_musdb18hq.sh](scripts/superresolution/superresolution_musdb18hq.sh)     |
| Music source separation | ✅           | MUSDB18HQ  | [scripts/mss/mixture2vocals_musdb18hq.sh](scripts/mss/mixture2vocals_musdb18hq.sh)                             |
| Vocals to music         | ✅           | MUSDB18HQ  | [scripts/vocals2audio/vocals2mixture.sh](scripts/vocals2audio/vocals2mixture.sh.sh)       |


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

## 1. Training a text to music system

### 1.1 Download datasets

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

## External links

[1] Conditional flow matching: https://github.com/atong01/conditional-flow-matching

[2] DiT: https://github.com/facebookresearch/DiT