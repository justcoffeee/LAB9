"""Reproduce the whole LAB09 pipeline end to end.

    python run_experiments.py                 # everything (needs the GPU, ~25 min)
    python run_experiments.py --skip-baseline --skip-training   # analysis only

Each step is a separate process so a failure stops the pipeline with a clear
message. Logs are appended to history/run.log.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / "history"
REQUIRED = [
    ROOT / "data" / "train.parquet",
    ROOT / "model" / "bert-base-cased" / "model.safetensors",
]


def run(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===\n$ {' '.join(command)}", flush=True)
    started = time.perf_counter()
    log = HISTORY / "run.log"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {name} ===\n$ {' '.join(command)}\n")
        fh.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")
        fh.write(result.stdout)
        fh.write(f"[exit {result.returncode}]\n")
    print(result.stdout[-4000:])
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")
    print(f"=== {name} done in {time.perf_counter() - started:.1f}s ===", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-reports", action="store_true")
    parser.add_argument("--skip-ie", action="store_true")
    parser.add_argument("--ie-sentences", type=int, default=30)
    args = parser.parse_args()

    HISTORY.mkdir(exist_ok=True)
    missing = [path for path in REQUIRED if not path.exists()]
    if missing:
        print("missing assets:")
        for path in missing:
            print(f"  {path}")
        print("run `python download_data.py` first")
        return 1

    python = sys.executable
    if not args.skip_baseline:
        run("baseline", [python, "-u", "baseline_ner.py", "--max-iter", "200"])
    if not args.skip_training:
        run("transformer (Strategy A)", [python, "-u", "ner_finetuning.py",
                                         "--tag", "main", "--epochs", "3"])
    if not args.skip_ablation:
        run("transformer (Strategy B ablation)",
            [python, "-u", "ner_finetuning.py", "--tag", "propagate",
             "--strategy", "propagate", "--epochs", "3"])
    if not args.skip_reports:
        run("reports", [python, "-u", "make_reports.py"])
    if not args.skip_ie:
        run("information extraction", [python, "-u", "information_extraction.py",
                                       "--evaluate", str(args.ie_sentences)])
    run("documentation", [python, "-u", "write_documentation.py"])
    run("executed notebooks", [python, "-u", "build_notebooks.py"])
    run("validate artifacts", [python, "-u", "validate_artifacts.py"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
