"""세 모델 고정 전체 2K 질의 CPU 검산. 모델 로드/forward/파일수정 없음."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from official.evaluation.zsre_query_parity import compare_queries


def run(stream, hub, locks):
    data=Path(stream).read_bytes()
    records=json.loads(data)
    assert len(records)==2000
    locks=json.loads(Path(locks).read_text())['audits']
    result={'requests':2000,'stream_file_sha256':hashlib.sha256(data).hexdigest(),
            'model_forward_calls':0,'models':{}}
    folders={'llama3':'meta-llama--Meta-Llama-3-8B-Instruct',
             'gptj':'EleutherAI--gpt-j-6b','qwen25':'Qwen--Qwen2.5-7B-Instruct'}
    for name,folder in folders.items():
        expected=locks[name+'-zsre']['tokenizer_files_sha256']
        snapshots=(Path(hub)/('models--'+folder)/'snapshots').glob('*')
        path=next(p for p in snapshots if all((p/f).exists() and
            hashlib.sha256((p/f).read_bytes()).hexdigest()==sha for f,sha in expected.items()))
        tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        tok.padding_side='right'
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        receipt=compare_queries(tok,records,model_family=name)
        receipt['tokenizer_files_sha256']=expected
        receipt['tokenizer_snapshot_revision']=path.name
        result['models'][name]=receipt
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stream',required=True);p.add_argument('--hub',required=True)
    p.add_argument('--locks',default='official/hparams/tokenizers.lock.json')
    a=p.parse_args()
    print(json.dumps(run(a.stream,a.hub,a.locks),indent=2,ensure_ascii=False))
