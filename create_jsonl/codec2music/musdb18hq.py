import argparse

from create_jsonl.mss.musdb18hq import create_jsonl

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_vae_dir", type=str, required=True)
    parser.add_argument("--target_vae_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    args.task = "codec_to_music"
    args.instruction = "generate music from DAC codec"

    create_jsonl(args)