
# ttm (done)
# mss (done)
# vocal-to-music
# mono-to-stereo
# super-resolution
# codec-to-music
# tta (done)
# tts

# midi-to-music(todo)
# image-to-music(todo)
# video-to-music(todo)

########## Compute latents
# GTZAN
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.gtzan \
		--dataset_root="./datasets/gtzan" \
		--split=${SPLIT} \
		--out_dir="./latents/gtzan/${SPLIT}/audio"
done

# MUSDB18HQ
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.musdb18hq stereo \
		--dataset_root="./datasets/musdb18hq/" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture"
done


for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.musdb18hq stereo \
		--dataset_root="./datasets/musdb18hq" \
		--stem="vocals" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/vocals"
done

for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.musdb18hq mono \
		--dataset_root="./datasets/musdb18hq" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture_mono"
done

for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=4 python -m compute_latents.musdb18hq dac \
		--dataset_root="./datasets/musdb18hq" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture_dac"
done

for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=0 python -m compute_latents.musdb18hq lowres \
		--dataset_root="./datasets/musdb18hq" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture_lowres"
done

# LJSpeech
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=3 python -m compute_latents.ljspeech \
		--dataset_root="./datasets/LJSpeech-1.1" \
		--split=${SPLIT} \
		--out_dir="./latents/ljspeech/${SPLIT}/audio"
done

# LibriTTS
for SUBDIR in "train-clean-100" "train-clean-360" "train-other-500"; do
	CUDA_VISIBLE_DEVICES=3 python -m compute_latents.libritts \
		--audios_dir="./datasets/LibriTTS/${SUBDIR}" \
		--out_dir="./latents/libritts/train/${SUBDIR}/audio" \
		--csv_path="./latents/libritts/train/${SUBDIR}/text.csv"
done

for SUBDIR in "test-clean" "test-other"; do
	CUDA_VISIBLE_DEVICES=3 python -m compute_latents.libritts \
		--audios_dir="./datasets/LibriTTS/${SUBDIR}" \
		--out_dir="./latents/libritts/test/${SUBDIR}/audio" \
		--csv_path="./latents/libritts/test/${SUBDIR}/text.csv"
done


for SPLIT in "train-clean-100" "train-clean-360" "train-other-500" "test-clean" "test-other"; do
	CUDA_VISIBLE_DEVICES=3 python -m compute_latents.libritts \
		--dataset_root="./datasets/LibriTTS" \
		--split=${SPLIT} \
		--out_dir="./latents/libritts/${SPLIT}/audio" \
		--csv_path="./latents/libritts/${SPLIT}/text.csv"
done


# Clotho
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.clotho \
		--dataset_root="./datasets/clotho2.1" \
		--split=${SPLIT} \
		--out_dir="./latents/clotho2.1/${SPLIT}/audio"
done

# AudioCaps
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.audiocaps \
		--dataset_root="./datasets/audiocaps2.0" \
		--split=${SPLIT} \
		--out_dir="./latents/audiocaps2.0/${SPLIT}/audio"
done

# AVE
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.ave audio \
		--dataset_root="./datasets/AVE_Dataset" \
		--split=${SPLIT} \
		--out_dir="./latents/ave/${SPLIT}/audio"
done

for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.ave video \
		--dataset_root="./datasets/AVE_Dataset" \
		--split=${SPLIT} \
		--out_dir="./latents/ave/${SPLIT}/video"
done

####### Create jsonls
# ttm
for SPLIT in "train" "test"; do
	python -m create_jsonl.ttm.gtzan \
		--vae_dir="./latents/gtzan/${SPLIT}/audio" \
		--out_path="./jsonls/ttm/${SPLIT}/gtzan.jsonl"
done

# mss
for SPLIT in "train" "test"; do
	python -m create_jsonl.mss.musdb18hq \
		--input_vae_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--target_vae_dir="./latents/musdb18hq/${SPLIT}/vocals" \
		--out_path="./jsonls/mixture2vocals/${SPLIT}/musdb18hq.jsonl"
done

# vocals to mix
for SPLIT in "train" "test"; do
	python -m create_jsonl.mono2stereo.musdb18hq \
		--input_vae_dir="./latents/musdb18hq/${SPLIT}/vocals" \
		--target_vae_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--out_path="./jsonls/vocals2mixture/${SPLIT}/musdb18hq.jsonl"
done


# mono to stereo
for SPLIT in "train" "test"; do
	python -m create_jsonl.mono2stereo.musdb18hq \
		--input_vae_dir="./latents/musdb18hq/${SPLIT}/mixture_mono" \
		--target_vae_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--out_path="./jsonls/mono2stereo/${SPLIT}/musdb18hq.jsonl"
done

# super-resolution
for SPLIT in "train" "test"; do
	python -m create_jsonl.superresolution.musdb18hq \
		--input_vae_dir="./latents/musdb18hq/${SPLIT}/mixture_lowres" \
		--target_vae_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--out_path="./jsonls/superresolution/${SPLIT}/musdb18hq.jsonl"
done

# codec to audio
for SPLIT in "train" "test"; do
	python -m create_jsonl.codec2music.musdb18hq \
		--input_vae_dir="./latents/musdb18hq/${SPLIT}/mixture_dac" \
		--target_vae_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--out_path="./jsonls/dac2stereo/${SPLIT}/musdb18hq.jsonl"
done

# tts
for SPLIT in "train" "test"; do
	python -m create_jsonl.tts.ljspeech \
		--dataset_root="./datasets/LJSpeech-1.1" \
		--vae_dir="./latents/ljspeech/${SPLIT}/audio" \
		--out_path="./jsonls/tts/${SPLIT}/ljspeech.jsonl"
done

# LibriTTS
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

# tta
for SPLIT in "train" "test"; do
	python -m create_jsonl.tta.clotho \
		--dataset_root="./datasets/clotho2.1" \
		--split=${SPLIT} \
		--vae_dir="./latents/clotho2.1/${SPLIT}/audio" \
		--out_path="./jsonls/tta/${SPLIT}/clotho2.1.jsonl"
done

# tta
for SPLIT in "train" "test"; do
	python -m create_jsonl.tta.audiocaps \
		--dataset_root="./datasets/audiocaps2.0" \
		--split=${SPLIT} \
		--vae_dir="./latents/audiocaps2.0/${SPLIT}/audio" \
		--out_path="./jsonls/tta/${SPLIT}/audiocaps2.0.jsonl"
done


#### Train
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/ttm/ttm_gtzan.yaml" --no_log
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/mss/mixture2vocals_musdb18hq.yaml" --no_log
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/tts/tts_ljspeech.yaml" --no_log
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/tta/tta_clotho.yaml" --no_log
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/tta/tta_audiocaps.yaml" --no_log
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/vocals2mixture/vocals2mixture_musdb18hq.yaml" --no_log
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/mono2stereo/mono2stereo_musdb18hq.yaml" --no_log
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/superresolution/superresolution_musdb18hq.yaml" --no_log
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/dac2stereo/dac2stereo_musdb18hq.yaml" --no_log


#### Sample
# CUDA_VISIBLE_DEVICES=4 python sample.py \
# 	--config="./configs/tts/tts_ljspeech.yaml" \
# 	--ckpt_path="checkpoints/train/tts_ljspeech/step=200000_ema.pth"

CUDA_VISIBLE_DEVICES=4 python sample.py \
	--config="./configs/tta/tta_audiocaps.yaml" \
	--ckpt_path="./checkpoints/train/tta_audiocaps/step=300000_ema.pth" \
	--task="text_to_audio" \
	--prompt="dog barking" \
	--out_path="_zz.wav"


CUDA_VISIBLE_DEVICES=4 python sample.py \
	--config="./configs/tts/tts_libritts.yaml" \
	--ckpt_path="/public/qq_to_zhuangyi/audio_flow4/audio_flow4/checkpoints/train/tts_libritts_ft/step=500000_ema.pth" \
	--task="text_to_speech" \
	--prompt="Happy new year! My name is Peter. " \
	--out_path="_zz.wav"

CUDA_VISIBLE_DEVICES=4 python sample.py \
	--config="./configs/codec2audio/dac2stereo_musdb18hq.yaml" \
	--ckpt_path="/public/qq_to_jiahe/audio_flow4/checkpoints/train/dac2stereo_musdb18hq/step=200000_ema.pt" \
	--task="codec_to_music" \
	--prompt="" \
	--instruction="generate music from DAC codec" \
	--audio_path="/public/qq_to_jiahe/audio_flow4/test_audios/dac2stereo/test,idx=0,input.wav" \
	--out_path="_zz.wav"

CUDA_VISIBLE_DEVICES=4 python sample.py \
	--config="./configs/mss/mixture2vocals_musdb18hq.yaml" \
	--ckpt_path="./checkpoints/train/mixture2vocals_musdb18hq/step=200000_ema.pt" \
	--task="music_source_separation" \
	--prompt="" \
	--audio_path="_zz.wav" \
	--out_path="_zz.wav"

	