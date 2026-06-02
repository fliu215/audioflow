import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pickle
import librosa
import os
import soundfile
import torchaudio
from audio_flow.encoders.audio.levo_vae import LevoVAE


def add():

    device = "cuda"

    if False:
        z = torch.zeros((250, 64)).to(device)
    else:
        import pickle
        z = pickle.load(open("_zz.pkl", "rb"))
        z = torch.Tensor(z[None, :, :]).to(device)

    vae = LevoVAE().to(device)
    y = vae.decode(z).data.cpu().numpy()[0]  # (c, l)
    soundfile.write(file="_zz.wav", data=y[0], samplerate=48000)

    from IPython import embed; embed(using=False); os._exit(0)


def add2():

    device = "cuda"

    audio_path = "latents/maestro-v3.0.0/train/audio/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.h5"
    midi_path = "latents/maestro-v3.0.0/train/midi/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.h5"

    with h5py.File(midi_path, 'r') as hf:
        midi_latent = hf["latent"][:]  # (l, d)

    with h5py.File(audio_path, 'r') as hf:
        vae_latent = hf["latent"][:]  # (l, d)

    plt.matshow(midi_latent[0:3000].T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")

    vae = LevoVAE().to(device)
    z = torch.Tensor(vae_latent[None, 0:750, :]).to(device)
    y = vae.decode(z).data.cpu().numpy()[0]  # (c, l)
    soundfile.write(file="_zz.wav", data=y[0], samplerate=48000)

    from IPython import embed; embed(using=False); os._exit(0)


def add3():
    audio_path = "./assets/music_10s.wav"
    audio, fs = librosa.load(path=audio_path, sr=None, mono=True)
    # from IPython import embed; embed(using=False); os._exit(0)    

    soundfile.write(file="_zz.wav", data=audio, samplerate=fs)


def add4():
    audio_path = "./assets/music_10s.wav"
    audio, fs = librosa.load(path=audio_path, sr=None, mono=False)
    audio = librosa.resample(y=audio, orig_sr=fs, target_sr=8000)
    audio = librosa.resample(y=audio, orig_sr=8000, target_sr=fs)

    soundfile.write(file="_zz.wav", data=audio.T, samplerate=fs)


def add5():
    from torchvision.io import read_video
    from audio_flow.encoders.image.clip import CLIPEncoder

    device = "cuda"
    encoder = CLIPEncoder().to(device)

    path = "/datasets/AVE/AVE_Dataset/AVE/_QQP43H56TA.mp4"
    out_path = "_tmp.mp4"
    cmd = f"ffmpeg -i {path} -r 25 {out_path}"
    os.system(cmd)

    video, _, info = read_video(out_path, output_format="TCHW", pts_unit="sec")
    latent = extract_images_latents_in_chunks(encoder, video.numpy(), chunk_size=16)


def add5b():
    from torchvision.io import read_video
    from audio_flow.encoders.text.clip import CLIPEncoder

    device = "cuda"
    encoder = CLIPEncoder().to(device)

    out = encoder(["aasdf", 'asdf, evie'])
    from IPython import embed; embed(using=False); os._exit(0)


def add5c():
    from torchvision.io import read_video
    from audio_flow.encoders.image.clip import CLIPEncoder

    device = "cuda"
    encoder = CLIPEncoder().to(device)

    path = "/datasets/AVE/AVE_Dataset/AVE/_QQP43H56TA.mp4"
    out_path = "_tmp.mp4"
    # cmd = f"ffmpeg -y -i {path} -r 24 {out_path}"
    # os.system(cmd)

    from transformers import VideoMAEImageProcessor, VideoMAEModel
    from decord import VideoReader

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    model.eval()

    vr = VideoReader(out_path)
    indices = np.arange(0, 192, 12)
    frames = vr.get_batch(indices).asnumpy()

    inputs = processor(list(frames), return_tensors="pt")
    pixel_values = (inputs["pixel_values"].to(device))

    with torch.no_grad():
        outputs = model(pixel_values)

    feat = outputs.last_hidden_state

    B = feat.shape[0]
    D = feat.shape[2]
    feat = feat.reshape(B, 8, 196, D)  # (B, 8, 196, 768)
    feat = feat.mean(dim=2)  # (b, t, d)
    feat = F.pad(feat, pad=(0, 0, 0, 2), mode="replicate")

    from IPython import embed; embed(using=False); os._exit(0)


def extract_images_latents_in_chunks(
    model: nn.Module, 
    images: np.array, 
    chunk_size: int,
) -> np.array:
    r"""Convert audio into latents.
    
    Args:
        model (nn.Module)
        x (np.ndarray): (b, c, h, w)

    Returns:
        out: (d, t)
    """
    device = next(model.parameters()).device
    latents = []
    i = 0
    
    while i < images.shape[0]:
        x = torch.from_numpy(images[i : i + chunk_size, ...]).to(device)  # (b, c, h, w)

        with torch.no_grad():
            model.eval()
            latent = model(x).data.cpu().numpy()  # (d, t)

        latents.append(latent)
        i += chunk_size

    return np.concatenate(latents, axis=0)


def add6():

    import whisper

    # 加载模型（可选: tiny, base, small, medium, large）
    model = whisper.load_model("large")

    # 识别音频
    # path = "./results/train/tts_ljspeech/steps=990000_ema/test,idx=0,task=text to speech,prompt=Shortly before the day fixed for execution, Bishop made a full confession, the bulk of which bore the impress of truth,,gen.wav"
    path = "./results/train/tts_ljspeech/steps=990000_ema/test,idx=2,task=text to speech,prompt=A number of people who resembled some of those in the photographs were placed under surveillance at the Trade Mart.,gen.wav"
    
    result = model.transcribe(path)

    print(result["text"])


def add7():
    a1 = "genre: a1, a2; xxx"
    a2 = a1.split(";")
    print(a2)


def add8():

    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download

    from mmaudio.model.utils.features_utils import FeaturesUtils

    device = "cuda"

    vae_path = hf_hub_download(
        repo_id="hkchengrex/MMAudio",
        filename="ext_weights/v1-44.pth",
    )

    fu = FeaturesUtils(
        tod_vae_ckpt=vae_path,
        bigvgan_vocoder_ckpt=None,
        synchformer_ckpt=None,
        enable_conditions=False,
        mode="44k",
        need_vae_encoder=True,
    ).to(device).eval()

    wav, sr = torchaudio.load("input.wav")
    wav = wav.mean(0, keepdim=True)

    if sr != 44100:
        wav = torchaudio.functional.resample(wav, sr, 44100)

    wav = wav.to(device)

    with torch.inference_mode():
        posterior = fu.encode_audio(wav)
        z = posterior.sample()
        mel = fu.decode(z)

    print(z.shape, mel.shape)


def add9():

    from architts_vae12_5hz import ArchiTTSVAE12Hz

    vae = ArchiTTSVAE12Hz(device="cuda")
    audio = vae.load_audio("./assets/LJ001-0001.wav")      # (1, 1, samples), 24 kHz
    latent = vae.encode(audio)               # (1, T, 64)
    recon = vae.decode(latent)               # (1, 1, samples)
    vae.save_audio(recon, "_zz.wav")

    print(latent.shape)
    from IPython import embed; embed(using=False); os._exit(0)


def add10():

    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download
    from mmaudio.model.utils.features_utils import FeaturesUtils

    device = "cuda"

    vae_path = hf_hub_download(
        repo_id="hkchengrex/MMAudio",
        filename="ext_weights/v1-44.pth",
    )

    fu = FeaturesUtils(
        tod_vae_ckpt=vae_path,
        bigvgan_vocoder_ckpt=None,
        synchformer_ckpt=None,
        enable_conditions=False,
        mode="44k",
        need_vae_encoder=True,
    ).to(device).eval()

    wav, sr = torchaudio.load("./assets/music_10s.wav")   # (C, L)
    wav = wav.mean(0, keepdim=True)          # mono

    if sr != 44100:
        wav = torchaudio.functional.resample(wav, sr, 44100)

    wav = wav.to(device)

    with torch.inference_mode():
        posterior = fu.encode_audio(wav)
        z = posterior.sample()       # latent
        from IPython import embed; embed(using=False); os._exit(0)
        mel_rec = fu.decode(z.transpose(1, 2))       # reconstructed mel
        out = fu.vocode(mel_rec)

    soundfile.write(file="_zz.wav", data=out[0, 0].cpu().numpy(), samplerate=44100)

    print(z.shape, mel_rec.shape)


def add10b():
    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download

    from mmaudio.model.utils.features_utils import FeaturesUtils

    device = "cuda"

    # 16kHz VAE
    vae_path = hf_hub_download(
        repo_id="hkchengrex/MMAudio",
        filename="ext_weights/v1-16.pth",
    )

    # 16kHz BigVGAN
    bigvgan_path = hf_hub_download(
        repo_id="hkchengrex/MMAudio",
        filename="ext_weights/best_netG.pt",
    )

    fu = FeaturesUtils(
        tod_vae_ckpt=vae_path,
        bigvgan_vocoder_ckpt=bigvgan_path,
        synchformer_ckpt=None,
        enable_conditions=False,
        mode="16k",
        need_vae_encoder=True,
    ).to(device).eval()

    wav, sr = torchaudio.load("./assets/music_10s.wav")   # (C, L)
    wav = wav.mean(0, keepdim=True)

    # resample to 16k
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    wav = wav.to(device)

    with torch.inference_mode():

        # encode
        posterior = fu.encode_audio(wav)

        z = posterior.sample()
        mel = fu.decode(z.transpose(1, 2))

        print("latent:", z.shape)
        print("mel:", mel.shape)
        out = fu.vocode(mel)

    print(out.shape)
    soundfile.write(file="_zz.wav", data=out[0, 0].cpu().numpy(), samplerate=16000)

    from IPython import embed; embed(using=False); os._exit(0)


def add11():

    from audio_flow.encoders.audio.mmaudio_vae import MMAudioVAE

    model = MMAudioVAE()

    wav, sr = torchaudio.load("./assets/music_10s.wav")   # (C, L)
    wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = Tensor(wav)[None, :, :]
    

    z = model.encode(wav)
    out = model.decode(z)
    soundfile.write(file="_zz.wav", data=out[0, 0].cpu().numpy(), samplerate=16000)

    from IPython import embed; embed(using=False); os._exit(0)


def add12():

    from transformers import AutoTokenizer, T5EncoderModel

    model_name = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = T5EncoderModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).cuda()

    text = [
        "A dog barking in a room",
        "A piano is playing"
    ]

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    feat = outputs.last_hidden_state

    print(feat.shape)
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    add12() 