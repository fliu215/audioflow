import argparse
from pathlib import Path

import h5py

from audio_flow.utils import write_jsonl


def create_jsonl(args):

    # Arguments
    input_vae_dir = args.input_vae_dir
    target_vae_dir = args.target_vae_dir
    out_path = args.out_path
    task = args.task
    instruction = args.instruction

    input_paths = list(Path(input_vae_dir).glob("*.h5"))
    target_paths = list(Path(target_vae_dir).glob("*.h5"))

    input_names = [path.name for path in input_paths]
    target_names = [path.name for path in target_paths]
    names = intersect_lists(input_names, target_names)

    input_paths = [Path(input_vae_dir, name) for name in names]
    target_paths = [Path(target_vae_dir, name) for name in names]
    metas = []

    for n in range(len(names)):
        if n % 100 == 0: 
            print(f"{n}/{len(names)}")

        input_attrs = get_attrs_from_hdf5(input_paths[n])
        target_attrs = get_attrs_from_hdf5(target_paths[n])

        meta = {
            "task": task,
            "input": {
                "text": {
                    "instruction": instruction,
                    "language": "en"
                },
                "audio": {
                    "path": str(input_paths[n]),
                    "latent_type": input_attrs["latent_type"], 
                    "fps": input_attrs["fps"],
                    "num_frames": input_attrs["n_frames"],
                    "duration": input_attrs["duration"]
                }
            },
            "target": {
                "audio": {
                    "path": str(target_paths[n]),
                    "latent_type": target_attrs["latent_type"], 
                    "fps": target_attrs["fps"],
                    "num_frames": target_attrs["n_frames"],
                    "duration": target_attrs["duration"]
                }
            }
        }
        metas.append(meta)

    write_jsonl(metas, out_path)
    print(f"Write out to {out_path}")


def intersect_lists(list1, list2) -> list:
    set2 = set(list2)
    return [x for x in list1 if x in set2]


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
    parser.add_argument("--input_stem", type=str, required=True)
    parser.add_argument("--target_stem", type=str, required=True)
    parser.add_argument("--input_vae_dir", type=str, required=True)
    parser.add_argument("--target_vae_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    args.task = "music_source_separation"
    args.instruction = f"separate {args.target_stem} from {args.input_stem}"

    create_jsonl(args)