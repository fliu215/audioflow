
# ttm
# tts
# tta 
# mss
# vocals-to-music
# mono-to-stereo
# super-resolution
# codec-to-music (todo)
# editing 
# midi-to-music

# fma
# image-to-music(todo)
# video-to-music(todo)

pip install "git+https://github.com/hkchengrex/MMAudio.git"


########## Compute latents
# GTZAN
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=7 python -m compute_latents.gtzan \
		--dataset_root="./datasets/gtzan" \
		--split=${SPLIT} \
		--out_dir="./latents/gtzan/${SPLIT}/audio"
done

# FMA
for SPLIT in "train" "test"; do
	for SUBSET in "small" "medium" "large"; do
		CUDA_VISIBLE_DEVICES=6 python -m compute_latents.fma audio \
			--dataset_root="./datasets/fma/fma_full" \
			--metadata_root="./datasets/fma/fma_metadata" \
			--split=${SPLIT} \
			--subset=${SUBSET} \
			--out_dir="./latents/fma/${SPLIT}/${SUBSET}/audio"
	done
done

# LJSpeech
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.ljspeech \
		--dataset_root="./datasets/LJSpeech-1.1" \
		--split=${SPLIT} \
		--latent_type="levo_vae" \
		--out_dir="./latents/ljspeech/${SPLIT}/audio/levo_vae"
done


for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.ljspeech \
		--dataset_root="./datasets/LJSpeech-1.1" \
		--split=${SPLIT} \
		--latent_type="architts_vae" \
		--out_dir="./latents/ljspeech/${SPLIT}/audio/architts_vae"
done

# LibriTTS
for SUBDIR in "train-clean-100" "train-clean-360" "train-other-500"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.libritts \
		--audios_dir="./datasets/LibriTTS/${SUBDIR}" \
		--out_dir="./latents/libritts/train/${SUBDIR}/audio" \
		--csv_path="./latents/libritts/train/${SUBDIR}/text.csv"
done

for SUBDIR in "test-clean" "test-other"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.libritts \
		--audios_dir="./datasets/LibriTTS/${SUBDIR}" \
		--out_dir="./latents/libritts/test/${SUBDIR}/audio" \
		--csv_path="./latents/libritts/test/${SUBDIR}/text.csv"
done

# Clotho
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.clotho \
		--dataset_root="./datasets/clotho2.1" \
		--split=${SPLIT} \
		--out_dir="./latents/clotho2.1/${SPLIT}/audio"
done

# AudioCaps
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=7 python -m compute_latents.audiocaps \
		--dataset_root="./datasets/audiocaps2.0" \
		--split=${SPLIT} \
		--out_dir="./latents/audiocaps2.0/${SPLIT}/audio"
done

for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.audiocaps \
		--dataset_root="./datasets/audiocaps2.0" \
		--split=${SPLIT} \
		--latent_type="mmaudio_vae" \
		--out_dir="./latents/audiocaps2.0/${SPLIT}/audio/mmaudio_vae"
done


# AudioSet
CUDA_VISIBLE_DEVICES=3 python -m compute_latents.audioset \
	--audios_dir="./datasets/audioset/audios/balanced_train_segments" \
	--out_dir="./latents/audioset/train/balanced_train_segments/audio"

for IDX in {00..40}; do
	CUDA_VISIBLE_DEVICES=3 python -m compute_latents.audioset \
		--audios_dir="./datasets/audioset/audios/unbalanced_train_segments/unbalanced_train_segments_part${IDX}" \
		--out_dir="./latents/audioset/train/unbalanced_train_segments/unbalanced_train_segments_part${IDX}/audio"	
done

# MUSDB18HQ, mixture
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.musdb18hq stereo \
		--dataset_root="./datasets/musdb18hq" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture"
done

# MUSDB18HQ, vocals
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.musdb18hq stereo \
		--dataset_root="./datasets/musdb18hq" \
		--stem="vocals" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/vocals"
done

for STEM in "bass" "drums" "other" do
	for SPLIT in "train" "test"; do
		CUDA_VISIBLE_DEVICES=5 python -m compute_latents.musdb18hq stereo \
			--dataset_root="./datasets/musdb18hq" \
			--stem=${STEM} \
			--split=${SPLIT} \
			--out_dir="./latents/musdb18hq/${SPLIT}/${STEM}"
	done
done

# MUSDB18HQ, mono
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.musdb18hq mono \
		--dataset_root="./datasets/musdb18hq" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture_mono"
done

# MUSDB18HQ, DAC
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=4 python -m compute_latents.musdb18hq dac \
		--dataset_root="./datasets/musdb18hq" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture_dac"
done

# MUSDB18HQ, lowres
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.musdb18hq lowres \
		--dataset_root="./datasets/musdb18hq" \
		--stem="mixture" \
		--split=${SPLIT} \
		--out_dir="./latents/musdb18hq/${SPLIT}/mixture_lowres"
done

# MAESTRO
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=7 python -m compute_latents.maestro audio \
		--dataset_root="./datasets/maestro-v3.0.0" \
		--split=${SPLIT} \
		--out_dir="./latents/maestro-v3.0.0/${SPLIT}/audio"
done

for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=5 python -m compute_latents.maestro midi \
		--dataset_root="./datasets/maestro-v3.0.0" \
		--split=${SPLIT} \
		--out_dir="./latents/maestro-v3.0.0/${SPLIT}/midi"
done

# video2audio, AVE
for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.ave audio \
		--dataset_root="./datasets/AVE" \
		--split=${SPLIT} \
		--out_dir="./latents/ave/${SPLIT}/audio/levo_vae"
done

for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.ave audio \
		--dataset_root="./datasets/AVE" \
		--split=${SPLIT} \
		--latent_type="levo_vae" \
		--out_dir="./latents/ave/${SPLIT}/audio/mmaudio_vae"
done

for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=3 python -m compute_latents.ave video \
		--dataset_root="./datasets/AVE" \
		--split=${SPLIT} \
		--latent_type="clip" \
		--out_dir="./latents/ave/${SPLIT}/video/clip"
done


for SPLIT in "train" "test"; do
	CUDA_VISIBLE_DEVICES=6 python -m compute_latents.ave video_mae \
		--dataset_root="./datasets/AVE" \
		--split=${SPLIT} \
		--out_dir="./latents/ave/${SPLIT}/video_mae"
done


for SPLIT in "train" "test"; do
	python -m compute_latents.ave text \
		--dataset_root="./datasets/AVE" \
		--split=${SPLIT} \
		--out_dir="./latents/ave/${SPLIT}/text"
done


############################ Create jsonls
# ttm, GTZAN
for SPLIT in "train" "test"; do
	python -m create_jsonls.ttm.gtzan \
		--latent_dir="./latents/gtzan/${SPLIT}/audio" \
		--out_path="./jsonls/ttm/${SPLIT}/gtzan.jsonl"
done

# ttm, FMA
for SPLIT in "train" "test"; do
	for SUBSET in "small" "medium" "large"; do
		python -m create_jsonls.ttm.fma \
			--latent_dir="./latents/fma/${SPLIT}/${SUBSET}/audio" \
			--parquet_path="./datasets/fma/train-00000-of-00001.parquet" \
			--out_path="./jsonls/ttm/${SPLIT}/fma_${SUBSET}.jsonl"
	done
done

for SPLIT in "train" "test"; do
	for SUBSET in "small" "medium" "large"; do
		python -m create_jsonls.ttm.fma_tags \
			--latent_dir="./latents/fma/${SPLIT}/${SUBSET}/audio" \
			--parquet_path="./datasets/fma/train-00000-of-00001.parquet" \
			--out_path="./jsonls/ttm/${SPLIT}/fma_tags_${SUBSET}.jsonl"
	done
done


# tts, LJSpeech
for SPLIT in "train" "test"; do
	python -m create_jsonls.tts.ljspeech \
		--dataset_root="./datasets/LJSpeech-1.1" \
		--latent_dir="./latents/ljspeech/${SPLIT}/audio" \
		--out_path="./jsonls/tts/${SPLIT}/ljspeech.jsonl"
done

for SPLIT in "train" "test"; do
	python -m create_jsonls.tts.ljspeech \
		--dataset_root="./datasets/LJSpeech-1.1" \
		--latent_dir="./latents/ljspeech/${SPLIT}/audio/architts_vae" \
		--out_path="./jsonls/tts/${SPLIT}/ljspeech_architts_vae.jsonl"
done


# tts, LibriTTS
SPLIT="train"
for SUBDIR in "train-clean-100" "train-clean-360" "train-other-500"; do
	python -m create_jsonls.tts.libritts \
		--latent_dir="./latents/libritts/${SPLIT}/${SUBDIR}/audio" \
		--csv_path="./latents/libritts/${SPLIT}/${SUBDIR}/text.csv" \
		--out_path="./jsonls/tts/${SPLIT}/libritts_${SUBDIR}.jsonl" \
		--multi_jsonls \
		--chunk_size=10000
done

SPLIT="test"
for SUBDIR in "test-clean" "test-other"; do
	python -m create_jsonls.tts.libritts \
		--latent_dir="./latents/libritts/${SPLIT}/${SUBDIR}/audio" \
		--csv_path="./latents/libritts/${SPLIT}/${SUBDIR}/text.csv" \
		--out_path="./jsonls/tts/${SPLIT}/libritts_${SUBDIR}.jsonl"
done


# tta, Clotho
for SPLIT in "train" "test"; do
	python -m create_jsonls.tta.clotho \
		--dataset_root="./datasets/clotho2.1" \
		--split=${SPLIT} \
		--latent_dir="./latents/clotho2.1/${SPLIT}/audio" \
		--out_path="./jsonls/tta/${SPLIT}/clotho2.1.jsonl"
done

# tta, AudioCaps
for SPLIT in "train" "test"; do
	python -m create_jsonls.tta.audiocaps \
		--dataset_root="./datasets/audiocaps2.0" \
		--split=${SPLIT} \
		--latent_dir="./latents/audiocaps2.0/${SPLIT}/audio" \
		--out_path="./jsonls/tta/${SPLIT}/audiocaps2.0.jsonl"
done

for SPLIT in "train" "test"; do
	python -m create_jsonls.tta.audiocaps \
		--dataset_root="./datasets/audiocaps2.0" \
		--split=${SPLIT} \
		--latent_dir="./latents/audiocaps2.0/${SPLIT}/audio/mmaudio_vae" \
		--out_path="./jsonls/tta/${SPLIT}/audiocaps2.0_mmaudio_vae.jsonl"
done


# tta, AudioSet
python -m create_jsonls.tta.audioset \
	--csv_path="./datasets/audioset/AudioSetCaps_caption.csv" \
	--latent_dir="./latents/audioset/train/balanced_train_segments/audio" \
	--out_path="./jsonls/tta/train/audioset_balance.jsonl"

for IDX in {00..40}; do
	python -m create_jsonls.tta.audioset \
		--csv_path="./datasets/audioset/AudioSetCaps_caption.csv" \
		--latent_dir="./latents/audioset/train/unbalanced_train_segments/unbalanced_train_segments_part${IDX}/audio" \
		--out_path="./jsonls/tta/train/audioset_unbalance_${IDX}.jsonl"
done


# CUDA_VISIBLE_DEVICES=3 python -m compute_latents.audioset \
# 	--audios_dir="./datasets/audioset/audios/balanced_train_segments" \
# 	--out_dir="./latents/audioset/train/balanced_train_segments/audio"

# for IDX in {00..40}; do
# 	CUDA_VISIBLE_DEVICES=3 python -m compute_latents.audioset \
# 		--audios_dir="./datasets/audioset/audios/unbalanced_train_segments/unbalanced_train_segments_part${IDX}" \
# 		--out_dir="./latents/audioset/train/unbalanced_train_segments/unbalanced_train_segments_part${IDX}/audio"	
# done



# mss, MUSDB18HQ
for SPLIT in "train" "test"; do
	python -m create_jsonls.mss.musdb18hq \
		--input_latent_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--target_latent_dir="./latents/musdb18hq/${SPLIT}/vocals" \
		--out_path="./jsonls/mss/${SPLIT}/musdb18hq.jsonl"
done

# vocals2music, MUSDB18HQ
for SPLIT in "train" "test"; do
	python -m create_jsonls.vocals2music.musdb18hq \
		--input_latent_dir="./latents/musdb18hq/${SPLIT}/vocals" \
		--target_latent_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--out_path="./jsonls/vocals2music/${SPLIT}/musdb18hq.jsonl"
done

# mono2stereo, MUSDB18HQ
for SPLIT in "train" "test"; do
	python -m create_jsonls.mono2stereo.musdb18hq \
		--input_latent_dir="./latents/musdb18hq/${SPLIT}/mixture_mono" \
		--target_latent_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--out_path="./jsonls/mono2stereo/${SPLIT}/musdb18hq.jsonl"
done

# superresolution, MUSDB18HQ
for SPLIT in "train" "test"; do
	python -m create_jsonls.superresolution.musdb18hq \
		--input_latent_dir="./latents/musdb18hq/${SPLIT}/mixture_lowres" \
		--target_latent_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--out_path="./jsonls/superresolution/${SPLIT}/musdb18hq.jsonl"
done

# codec2audio, MUSDB18HQ
for SPLIT in "train" "test"; do
	python -m create_jsonls.codec2audio.musdb18hq \
		--input_latent_dir="./latents/musdb18hq/${SPLIT}/mixture_dac" \
		--target_latent_dir="./latents/musdb18hq/${SPLIT}/mixture" \
		--out_path="./jsonls/codec2audio/${SPLIT}/musdb18hq.jsonl"
done

# midi2audio, Maestro
for SPLIT in "train" "test"; do
	python -m create_jsonls.midi2audio.maestro \
		--input_latent_dir="./latents/maestro-v3.0.0/${SPLIT}/midi" \
		--target_latent_dir="./latents/maestro-v3.0.0/${SPLIT}/audio" \
		--out_path="./jsonls/midi2audio/${SPLIT}/maestro-v3.0.0.jsonl"
done

# editing, MUSDB18HQ
for SPLIT in "train" "test"; do
	python -m create_jsonls.editing.musdb18hq \
		--latent_dir="./latents/musdb18hq" \
		--split=${SPLIT} \
		--out_path="./jsonls/editing/${SPLIT}/musdb18hq.jsonl"
done

# video2audio, AVE
for SPLIT in "train" "test"; do
	python -m create_jsonls.video2audio.ave \
		--text_dir="./latents/ave/${SPLIT}/text" \
		--video_latent_dir="./latents/ave/${SPLIT}/video" \
		--audio_latent_dir="./latents/ave/${SPLIT}/audio" \
		--out_path="./jsonls/video2audio/${SPLIT}/ave.jsonl"
done

# video2audio, AVE
for SPLIT in "train" "test"; do
	python -m create_jsonls.video2audio.ave \
		--text_dir="./latents/ave/${SPLIT}/text" \
		--video_latent_dir="./latents/ave/${SPLIT}/video" \
		--audio_latent_dir="./latents/ave/${SPLIT}/audio/mmaudio_vae" \
		--out_path="./jsonls/video2audio/${SPLIT}/ave.jsonl"
done

for SPLIT in "train" "test"; do
	python -m create_jsonls.video2audio.ave \
		--text_dir="./latents/ave/${SPLIT}/text" \
		--video_latent_dir="./latents/ave/${SPLIT}/video_mae" \
		--audio_latent_dir="./latents/ave/${SPLIT}/audio" \
		--out_path="./jsonls/video2audio/${SPLIT}/ave_audio_mae.jsonl"
done

################ Train
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/ttm/ttm_gtzan.yaml" --no_log
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/tts/tts_ljspeech.yaml" --no_log
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/tts/tts_libritts.yaml" --no_log
CUDA_VISIBLE_DEVICES=4 python train.py --config="./configs/tta/tta_audiocaps.yaml" --no_log

CUDA_VISIBLE_DEVICES=6 python train.py --config="./configs/mss/mss_musdb18hq.yaml" --no_log
CUDA_VISIBLE_DEVICES=6 python train.py --config="./configs/vocals2music/vocals2music_musdb18hq.yaml" --no_log
CUDA_VISIBLE_DEVICES=7 python train.py --config="./configs/mono2stereo/mono2stereo_musdb18hq.yaml" --no_log
CUDA_VISIBLE_DEVICES=7 python train.py --config="./configs/superresolution/superresolution_musdb18hq.yaml" --no_log
CUDA_VISIBLE_DEVICES=5 python train.py --config="./configs/codec2audio/codec2audio_musdb18hq.yaml" --no_log
CUDA_VISIBLE_DEVICES=5 python train.py --config="./configs/midi2audio/midi2audio_maestro.yaml" --no_log
CUDA_VISIBLE_DEVICES=5 python train.py --config="./configs/editing/editing_musdb18hq.yaml" --no_log
CUDA_VISIBLE_DEVICES=5 python train.py --config="./configs/video2audio/video2audio_ave.yaml" --no_log

############## Sample
# TTM
CUDA_VISIBLE_DEVICES=5 python sample.py \
	--config="./configs/ttm/ttm_gtzan.yaml" \
	--ckpt_path="checkpoints/train/ttm_gtzan/step=200000_ema.pth" \
	--task="text to music" \
	--prompt="blues" \
	--out_path="_zz_ttm.wav"

# TTS
CUDA_VISIBLE_DEVICES=5 python sample.py \
	--config="./configs/tts/tts_ljspeech.yaml" \
	--ckpt_path="checkpoints/train/tts_ljspeech/step=200000_ema.pth" \
	--task="text to speech" \
	--prompt="Today is a sunny day." \
	--out_path="_zz_tts.wav"

# TTA
CUDA_VISIBLE_DEVICES=5 python sample.py \
	--config="./configs/tta/tta_audiocaps.yaml" \
	--ckpt_path="checkpoints/train/tta_audiocaps/step=200000_ema.pth" \
	--task="text to audio" \
	--prompt="a dog barking and a children speaking." \
	--out_path="_zz_tta.wav"

# MSS
CUDA_VISIBLE_DEVICES=5 python sample.py \
	--config="./configs/mss/mss_musdb18hq.yaml" \
	--ckpt_path="checkpoints/train/mss_musdb18hq/step=200000_ema.pth" \
	--task="music source separation" \
	--input_path="./assets/music_10s.wav" \
	--out_path="_zz_mss.wav"

# Vocals to music
CUDA_VISIBLE_DEVICES=5 python sample.py \
	--config="./configs/vocals2music/vocals2music_musdb18hq.yaml" \
	--ckpt_path="checkpoints/train/vocals2music_musdb18hq/step=200000_ema.pth" \
	--task="vocals to music" \
	--input_path="./assets/music_10s_vocals.wav" \
	--out_path="_zz_vocals2music.wav"

# Mono to stereo
CUDA_VISIBLE_DEVICES=5 python sample.py \
	--config="./configs/mono2stereo/mono2stereo_musdb18hq.yaml" \
	--ckpt_path="checkpoints/train/mono2stereo_musdb18hq/step=200000_ema.pth" \
	--task="mono to stereo" \
	--input_path="./assets/music_10s_mono.wav" \
	--out_path="_zz_mono2stereo.wav"

# Super-resolution
CUDA_VISIBLE_DEVICES=5 python sample.py \
	--config="./configs/superresolution/superresolution_musdb18hq.yaml" \
	--ckpt_path="checkpoints/train/superresolution_musdb18hq/step=200000_ema.pth" \
	--task="super-resolution" \
	--input_path="./assets/music_10s_lowres.wav" \
	--out_path="_zz_superresolution.wav"

# Editing
for STEM in "vocals" "bass" "drums" "other"; do
	CUDA_VISIBLE_DEVICES=5 python sample.py \
		--config="./configs/editing/editing_musdb18hq.yaml" \
		--ckpt_path="checkpoints/train/editing_musdb18hq/step=500000_ema.pth" \
		--task="audio editing" \
		--prompt="separate mixture into ${STEM}" \
		--input_path="./assets/music_10s.wav" \
		--out_path="_zz_editing_${STEM}.wav"
done

# MIDI to audio
CUDA_VISIBLE_DEVICES=5 python sample.py \
	--config="./configs/midi2audio/midi2audio_maestro.yaml" \
	--ckpt_path="checkpoints/train/midi2audio_maestro/step=200000_ema.pth" \
	--task="midi to audio" \
	--input_path="./assets/piano.mid" \
	--out_path="_zz_midi2audio.wav"

# MIDI to audio
# CUDA_VISIBLE_DEVICES=5 python sample.py \
# 	--config="./configs/midi2audio/midi2audio_maestro.yaml" \
# 	--ckpt_path="checkpoints/train/midi2audio_maestro/step=200000_ema.pth" \
# 	--task="video to audio" \
# 	--input_path="./assets/piano.mid" \
# 	--out_path="_zz_midi2audio.wav"


##################### Eval tts
CUDA_VISIBLE_DEVICES=4 python eval_tts.py \
	
python evaluate.py \
    --gt_audio "_gt" \
    --gt_cache "_gt_cache" \
    --pred_audio "_pred" \
    --pred_cache "_pred_cache" \
    # --audio_length=10 \
    # --recompute_pred_cache \
    # --skip_video_related \
    # --output_metrics_dir="_out_metrics" \
    # --debug

# python evaluate.py  --gt_audio <gt audio path> \
# 	--gt_cache <gt cache path> 
# 	--pred_audio <pred audio path> --pred_cache <pred cache path> --audio_length=<length of audio wanted in seconds> 

# python evaluate.py  --gt_audio <gt audio path> --gt_cache <gt cache path> --pred_audio <pred audio path> --pred_cache <pred cache path> --audio_length=<length of audio wanted in seconds> 


# Eval TTA
CUDA_VISIBLE_DEVICES=3 python -m evaluate.eval_tta sample \
	--config="./kqq_configs/tta/tta_audiocaps_02a.yaml" \
	--ckpt_path="./checkpoints/train/tta_audiocaps_02a/step=500000_ema.pth" \
	--gt_dir="./_results/tta_gt" \
	--out_dir="./_results/tta_out"


# Eval TTA (xiquan)
CUDA_VISIBLE_DEVICES=6 python -m evaluate.eval_tta sample2 \
	--config="./kqq_configs/tta/tta_audiocaps_mmaudio_vae.yaml" \
	--ckpt_path="./checkpoints/train/tta_audiocaps_mmaudio_vae/step=900000_ema.pth" \
	--gt_dir="./_results/tta_gt" \
	--out_dir="./_results/tta_out"


# AV-benchmark
python evaluate.py \
    --gt_audio "_gt" \
    --gt_cache "_gt_cache" \
    --pred_audio "_pred" \
    --pred_cache "_pred_cache" \
    --audio_length=10
    # --recompute_pred_cache \
    # --skip_video_related \
    # --output_metrics_dir="_out_metrics"
    # --debug

# train_cfg.py    cfg
# train_cfg2.py    cfg, register in adapter

# kqq_configs/tta/tta_audiocaps_mmaudio_vae_02a.yaml  FlanT5
# kqq_configs/tta/tta_audiocaps_mmaudio_vae_03a.yaml  FlanT5, cfg, fd=19
# kqq_configs/video2audio/video2audio_ave_audiomae_03a.yaml		v2a, cfg

# Transformer03a     cfg
# models/transformer_cfg.py  