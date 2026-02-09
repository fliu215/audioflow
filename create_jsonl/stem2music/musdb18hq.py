import argparse

from create_jsonl.mss.musdb18hq import create_jsonl

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_stem", type=str, required=True)
    parser.add_argument("--target_stem", type=str, required=True)
    parser.add_argument("--input_vae_dir", type=str, required=True)
    parser.add_argument("--target_vae_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    args.task = "music_generation_from_stem"
    args.instruction = f"generate {args.target_stem} from {args.input_stem}"

    create_jsonl(args)