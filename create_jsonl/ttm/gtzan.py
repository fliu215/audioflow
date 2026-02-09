import argparse
from pathlib import Path

import h5py

from audio_flow.utils import write_jsonl


def create_jsonl(args):

    # Arguments
    vae_dir = args.vae_dir
    out_path = args.out_path
    task = "text_to_music"

    paths = list(Path(vae_dir).glob("*.h5"))
    metas = []

    for n, path in enumerate(paths):
        if n % 100 == 0: 
            print(f"{n}/{len(paths)}")

        label = Path(path).stem.split(".")[0]
        attrs = get_attrs_from_hdf5(path)

        meta = {
            "task": task,
            "input": {
                "text": {
                    "prompt": [label],
                    "language": "en"
                }
            },
            "target": {
                "audio": {
                    "path": str(path),
                    "latent_type": attrs["latent_type"], 
                    "fps": attrs["fps"],
                    "num_frames": attrs["n_frames"],
                    "duration": attrs["duration"]
                }
            }
        }
        metas.append(meta)
           
    write_jsonl(metas, out_path)
    print(f"Write out to {out_path}")


def get_attrs_from_hdf5(path: str) -> dict:
    with h5py.File(path, 'r') as hf:
        attrs = {
            "fps": hf.attrs["fps"],
            "duration": hf.attrs["duration"],
            "latent_type": hf.attrs["latent_type"],
            "n_frames": hf["latent"].shape[-1]
        }
        
    return attrs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_dir", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    create_jsonl(args)