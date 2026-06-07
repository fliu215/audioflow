import argparse
from pathlib import Path

import h5py

from audioflow.utils.text import read_lines
from audioflow.utils.json import write_jsonl


def create_jsonl(args):

    # Arguments
    captions_dir = Path(args.captions_dir)
    latents_dir = Path(args.latents_dir)
    out_path = Path(args.out_path)

    latent_paths = sorted(latents_dir.glob("*.h5"))
    metas = []

    for n, latent_path in enumerate(latent_paths):

        if n % 100 == 0: 
            print(f"{n}/{len(latent_paths)}")

        txt_path = captions_dir / (latent_path.stem + ".txt")
        captions = read_lines(txt_path)

        hf = h5py.File(latent_path, "r")

        meta = {
            "input": {
                "text": {
                    "music": {
                        "caption": captions[0],
                        "genre": "",
                        "bpm": "",
                        "instruments": "",
                        "mood": "",
                        "mixing": ""
                    }
                }
            },
            "target": {
                "audio": {
                    "path": str(latent_path),
                    "type": hf.attrs["type"],
                    "fps": hf.attrs["fps"],
                    "duration": hf.attrs["duration"]
                }
            }
        }

        hf.close()
        metas.append(meta)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, metas)
    print(f"Write out to {out_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_dir", type=str)
    parser.add_argument("--latents_dir", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    create_jsonl(args)