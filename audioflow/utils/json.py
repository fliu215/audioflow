import json


def write_jsonl(path, metas: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for meta in metas:
            f.write(json.dumps(meta, ensure_ascii=False))
            f.write("\n")


def read_jsonl(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]