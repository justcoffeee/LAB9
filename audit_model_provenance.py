"""Fetch the pinned upstream BERT assets separately for provenance verification."""
from pathlib import Path
import hashlib
import json
import requests

ROOT=Path(__file__).resolve().parent
REV='cd5ef92a9fb2f889e972770a36d4ed042daf221e'
BASE=f'https://huggingface.co/google-bert/bert-base-cased/resolve/{REV}'


def main():
    directory=ROOT/'model/upstream'
    directory.mkdir(parents=True,exist_ok=True)
    records={}
    for name in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json']:
        target=directory/name
        url=f'{BASE}/{name}'
        if not target.exists():
            with requests.get(url,stream=True,timeout=180) as response:
                response.raise_for_status()
                with target.open('wb') as f:
                    for chunk in response.iter_content(1024*1024):f.write(chunk)
        records[name]={'url':url,'bytes':target.stat().st_size,
                       'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        print(name,records[name],flush=True)
    (directory/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
