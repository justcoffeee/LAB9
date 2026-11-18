"""Re-run held-out evaluation from the existing checkpoints without retraining."""
from datetime import datetime, timezone
import hashlib
import torch
from transformers import AutoTokenizer
from information_extraction import load_model
from ner_finetuning import encode, build_loader, evaluate
from ner_utils import ROOT, PRED_DIR, MODEL_DIR, load_conll, read_json, write_json


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    tok=AutoTokenizer.from_pretrained(str(MODEL_DIR),local_files_only=True)
    sentences=load_conll()['test']
    checks=[]
    for tag,strategy in [('main','first'),('propagate','propagate')]:
        checkpoint=ROOT/'model'/f'best_ner_{tag}.pt'
        _,model,device=load_model(checkpoint)
        encoded,_=encode(tok,sentences,128,strategy)
        for item,sentence in zip(encoded,sentences):item['gold_ids']=sentence['ner_tags']
        batches=build_loader(tok,encoded,32,device,False,42)
        actual=evaluate(model,batches,device,device.type=='cuda')
        expected=read_json(PRED_DIR/f'transformer_{tag}_test_predictions.json')
        mismatch=sum(g!=p for old,new in zip(expected,actual['pred_tags'])
                     for g,p in zip(old['pred_tags'],new))
        saved=read_json(PRED_DIR/f'transformer_{tag}_metrics.json')
        digest=hashlib.sha256()
        with checkpoint.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        check={'tag':tag,'checkpoint_sha256':digest.hexdigest(),'test_sentences':len(sentences),
               'word_prediction_mismatches':mismatch,'entity_f1_recomputed':actual['entity']['f1'],
               'entity_f1_recorded':saved['splits']['test']['entity']['f1'],
               'exact_match':mismatch==0 and actual['entity']==saved['splits']['test']['entity']}
        checks.append(check);print(check,flush=True)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result={'verified_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,
            'all_passed':all(c['exact_match'] for c in checks)}
    write_json(PRED_DIR/'checkpoint_verification.json',result)
    if not result['all_passed']:raise RuntimeError('Checkpoint verification differs from stored predictions')


if __name__=='__main__':main()
