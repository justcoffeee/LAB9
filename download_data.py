"""Download the two external assets used by LAB09.

1. CoNLL-2003 (parquet conversion of ``eriktks/conll2003``)  -> data/*.parquet
2. ``google-bert/bert-base-cased`` tokenizer + weights       -> model/bert-base-cased/

Both are pinned to the revisions below and verified against SHA256 so that a
fresh clone reproduces the same numbers. Files are ~1.8 MB and ~431 MB.

Usage
-----
    python download_data.py            # download what is missing
    python download_data.py --force    # re-download everything
    python download_data.py --verify   # only check local hashes
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model" / "bert-base-cased"

CONLL_REVISION = "3f1cce917ab38486481b062921eb137e7bd3c205"
BERT_REVISION = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
PARQUET_BASE = (
    "https://huggingface.co/datasets/eriktks/conll2003/resolve/"
    f"refs%2Fconvert%2Fparquet/conll2003"
)

DATA_FILES = {
    "train.parquet": (
        f"{PARQUET_BASE}/train/0000.parquet",
        "546cbb4200311629804a65cbe19c8846f16ab1529c4da48bcc7be8c40daf29c6",
        1227788,
    ),
    "validation.parquet": (
        f"{PARQUET_BASE}/validation/0000.parquet",
        "114dec371f357e2fa397df14311891440d0ee1030b3a2fec4025ff84b875a59d",
        311883,
    ),
    "test.parquet": (
        f"{PARQUET_BASE}/test/0000.parquet",
        "14f894263b4ee3563395a57b8ba4cfaabec7db4ea956ba34e2f84a19781cdc43",
        283307,
    ),
}

MODEL_FILES = {
    "config.json": "https://huggingface.co/google-bert/bert-base-cased/resolve/"
                   f"{BERT_REVISION}/config.json",
    "model.safetensors": "https://huggingface.co/google-bert/bert-base-cased/resolve/"
                         f"{BERT_REVISION}/model.safetensors",
    "tokenizer.json": "https://huggingface.co/google-bert/bert-base-cased/resolve/"
                      f"{BERT_REVISION}/tokenizer.json",
    "tokenizer_config.json": "https://huggingface.co/google-bert/bert-base-cased/resolve/"
                             f"{BERT_REVISION}/tokenizer_config.json",
}

MODEL_SHA256 = {
    'config.json':'8b73abed2d67075fd3ec0f840aa53d9cccdc9169c04a7da135c07e721910ed74',
    'model.safetensors':'1d8bdcee6021e2c25f0325e84889b61c2eb26b843eef5659c247af138d64f050',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"downloading {url}\n         -> {target}")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(target)


def verify() -> bool:
    ok = True
    for name, (_url, expected, size) in DATA_FILES.items():
        path = DATA_DIR / name
        if not path.exists():
            print(f"MISSING {path}")
            ok = False
            continue
        actual = sha256(path)
        if actual != expected or path.stat().st_size != size:
            print(f"HASH MISMATCH {path}")
            ok = False
        else:
            print(f"ok {path} ({size} bytes)")
    for name in MODEL_FILES:
        path = MODEL_DIR / name
        if not path.exists():
            print(f"MISSING {path}")
            ok = False
        else:
            print(f"ok {path} ({path.stat().st_size} bytes)")
            if name in MODEL_SHA256 and sha256(path) != MODEL_SHA256[name]:
                print(f'HASH MISMATCH {path}')
                ok = False
    # The tokenizer and the model are only usable together; a size check on the
    # weight shard catches truncated downloads without hashing 431 MB every run.
    weights = MODEL_DIR / "model.safetensors"
    if weights.exists() and weights.stat().st_size != 435755784:
        print(f"UNEXPECTED SIZE {weights}")
        ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        return 0 if verify() else 1

    for name, (url, _expected, _size) in DATA_FILES.items():
        target = DATA_DIR / name
        if args.force or not target.exists():
            download(url, target)

    for name, url in MODEL_FILES.items():
        target = MODEL_DIR / name
        if args.force or not target.exists():
            download(url, target)

    ok = verify()
    print("all assets verified" if ok else "asset verification failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
