import argparse
from pathlib import Path

import h5py

from audioflow.utils.json import write_jsonl
from audioflow.utils.text import read_lines


def create_jsonl(args):

    # Arguments
    txt_dir = Path(args.input_texts_dir)
    tgt_dir = Path(args.target_latents_dir)
    out_path = Path(args.out_path)
    chunk_size = args.chunk_size

    tgt_paths = sorted(tgt_dir.glob("*.h5"))
    metas = []

    for n, tgt_path in enumerate(tgt_paths):

        if n % 100 == 0: 
            print(f"{n}/{len(tgt_paths)}")

        # Text
        txt_path = txt_dir / f"{tgt_path.stem}.txt"
        text = read_lines(txt_path)[0]

        # Target latent meta
        tgt_meta = read_hdf5_attrs(tgt_path)
        
        meta = {
            "input": {
                "text": f"<speech>{text}</speech>",
            },
            "target": {
                "audio": {
                    "path": tgt_meta["path"],
                    "type": tgt_meta["type"],
                    "fps": tgt_meta["fps"],
                    "duration": tgt_meta["duration"]
                }
            }
        }
        metas.append(meta)

    # Sort metas by duration
    metas.sort(key=lambda meta: meta["target"]["audio"]["duration"])

    # Remove long audios
    metas = [m for m in metas if m["target"]["audio"]["duration"] < 30.]

    # Create directory
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to jsonls
    if len(metas) < chunk_size:
        write_jsonl(metas, out_path)
        print(f"Write out to {out_path}")

    else:
        for i in range(0, len(metas), chunk_size):
            end = min(i + chunk_size, len(metas))
            path = Path(out_path)
            path = path.parent / f"{path.stem}_{i}-{end}{path.suffix}"
            write_jsonl(metas[i : end], path)
            print(f"Write out to {path}")


def read_hdf5_attrs(path) -> dict:
    with h5py.File(path, "r") as hf:
        return {
            "path": str(path),
            "type": hf.attrs["type"],
            "fps": hf.attrs["fps"],
            "duration": hf.attrs["duration"]
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_texts_dir", type=str)
    parser.add_argument("--target_latents_dir", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--chunk_size", type=int, default=10000)
    args = parser.parse_args()

    create_jsonl(args)