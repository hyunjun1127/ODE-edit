"""실패 G1 토큰 assertion만 재현. 모델/GPU/평가/수리/Slurm 호출 없음."""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU_ONLY_REPRO_REQUIRES_EMPTY_CUDA_VISIBLE_DEVICES'
    lock=json.loads((root/'control/execution.lock.json').read_text())
    assert lock['source_commit']=='a95876f8e5c4cf59df9cd9d7f824d1ac99f8bc77'
    old=json.loads((root/'inputs/design/evidence/audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-execution.lock.json').read_text())
    sys.path[:0]=[lock['dependencies'],lock['repo'],old['blue_root']]
    os.chdir(old['blue_root'])  # Same native import cwd as frozen Runtime:48.
    import transformers
    from transformers import AutoTokenizer
    from project.run_scripts.alpha_key_concentration_causal import technical
    representation=importlib.import_module('rome.repr_tools')
    assert transformers.__version__=='4.44.2'
    assert sha(technical.__file__)=='3f67e994be3dd2d137c32895275944b9d08919b9ec6fd0a3de3a6783b80b6dd6'
    # Original runtime.py:28-29 and current runtime.py:55-56 have this same setup.
    original=AutoTokenizer.from_pretrained(old['snapshot'],local_files_only=True)
    original.add_bos_token=False;original.pad_token_id=original.eos_token_id
    current=AutoTokenizer.from_pretrained(old['snapshot'],local_files_only=True)
    before=current.backend_tokenizer.post_processor.__getstate__()
    current.add_bos_token=False;current.pad_token_id=current.eos_token_id
    after=current.backend_tokenizer.post_processor.__getstate__()
    # Small retained JSON only; no edited tensor or 12-checkpoint scan.
    retained=Path('/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B050')
    contexts=json.loads((retained/'contexts.json').read_text())
    commit=json.loads((retained/'commit.json').read_text())
    assert digest(contexts)==commit['context_hash']
    records=json.loads(Path(old['dataset']).read_text())[5000:5002]
    requests=[r['requested_rewrite'] for r in records]
    templates=[c.format(r['prompt']) for r in requests for group in contexts for c in group]
    words=[r['subject'] for r in requests for group in contexts for _ in group]
    texts=[p.format(w) for p,w in zip(templates,words,strict=True)]
    original_pack=original(texts,padding=True,return_tensors='pt')
    current_pack=current(texts,padding=True,return_tensors='pt')
    indices=representation.get_words_idxs_in_templates(current,templates,words,'last')
    exact={k:bool((original_pack[k]==current_pack[k]).all()) for k in original_pack}
    rows=[]
    for i,(ids,mask,idx) in enumerate(zip(current_pack['input_ids'],current_pack['attention_mask'],indices,strict=True)):
        n=int(mask.sum());valid=ids[:n].tolist()
        rows.append(dict(case_id=int(records[i//6]['case_id']),context_index=i%6,valid_tokens=n,
                         first_token_id=valid[0],bos_present=valid[0]==current.bos_token_id,
                         subject_last=idx[0],lookup_in_range=0<=idx[0]<n,
                         input_token_ids_sha256=digest(valid),right_padding=bool(mask[:n].all()) and not bool(mask[n:].any())))
    try:
        technical.token_fixtures(SimpleNamespace(tok=current,repr=representation,contexts=contexts),records)
    except technical.TechnicalFailure as exc:
        failure=dict(type=type(exc).__name__,code=exc.code,details=exc.details,message=str(exc))
    else:
        raise AssertionError('EXPECTED_FROZEN_GATE_FAILURE_NOT_REPRODUCED')
    synthetic='Alpha example'
    synthetic_default=current(synthetic)['input_ids']
    synthetic_no_special=current(synthetic,add_special_tokens=False)['input_ids']
    assert failure['message']=='WRITER_UNEXPECTED_BOS: 0'
    assert all(exact.values()) and all(r['bos_present'] and r['right_padding'] and r['lookup_in_range'] for r in rows)
    assert before==after and current.add_bos_token is False
    post=json.loads(after)
    value=dict(status='FIRST_EXCEPTION_REPRODUCED_CPU_ONLY',method_modified=False,model_loads=0,model_forwards=0,
               GPU_allocations=0,slurm_calls=0,checkpoints_scanned=0,
               transformers_version=transformers.__version__,transformers_file=transformers.__file__,
               tokenizer_class=type(current).__name__,tokenizer_class_defines_add_bos_token=inspect.getattr_static(type(current),'add_bos_token',None) is not None,
               assigned_add_bos_token=current.add_bos_token,bos_token_id=current.bos_token_id,
               backend_postprocessor_unchanged=before==after,postprocessor_sha256=hashlib.sha256(after).hexdigest(),
               postprocessor_type=post['type'],postprocessor_structure=post,
               original_current_token_pack_exact=exact,actual_case_ids=[int(r['case_id']) for r in records],
               actual_sequences=len(rows),bos_sequences=sum(r['bos_present'] for r in rows),rows=rows,
               failure=failure,synthetic=dict(default_first=synthetic_default[0],explicit_no_special_first=synthetic_no_special[0],
                                            default_equals_bos_plus_no_special=synthetic_default==[current.bos_token_id]+synthetic_no_special,
                                            note='explicit no-special is diagnostic only, not a proposed runtime change'),
               context_sha256=sha(retained/'contexts.json'),context_semantic_sha256=digest(contexts),
               original_commit_sha256=sha(retained/'commit.json'),
               source_files=[dict(path=str(q),sha256=sha(q),bytes=Path(q).stat().st_size) for q in
                 [Path(technical.__file__),Path(lock['repo'])/'project/run_scripts/alpha_key_concentration_causal/runtime.py',
                  Path('/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/native/runtime.py'),Path(representation.__file__),
                  Path(old['snapshot'])/'tokenizer_config.json',Path(old['snapshot'])/'tokenizer.json']])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('rows','source_files')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
