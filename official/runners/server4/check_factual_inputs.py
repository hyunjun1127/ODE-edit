"""Exact published evaluator token-plan preflight; CPU tokenizer only, no LM."""
import argparse
from pathlib import Path
from official.experiments.prepare import read, file_sha, digest, write_new
from official.evaluation import factual
from .observe import verify_api, external_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assets=read(args.assets)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(assets['model_snapshot'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    result=dict(schema='server4-official-factual-input-plan-v1',GPU=0,model_forward=0,
                api=verify_api(),external_identity=external_identity(assets),datasets={})
    for dataset,member in assets['streams'].items():
        if file_sha(member['path'])!=member['sha256']:raise ValueError('STREAM_SHA')
        records=read(member['path'])
        if len(records)!=2000:raise ValueError('EXACT_FIRST2000_REQUIRED')
        try:
            cases,queries,signatures=factual._plan(records,dataset,tokenizer)
            counts={kind:sum(q['kind']==kind for q in queries)
                    for kind in ('rewrite','paraphrase','neighborhood')}
            result['datasets'][dataset]=dict(status='TOKEN_PLAN_PASS',requests=len(cases),
                candidate_sequences=len(queries),candidate_sequences_by_kind=counts,
                cohort_sha256=digest(signatures),stream_sha256=member['sha256'])
        except factual.FactualError as exc:
            result['datasets'][dataset]=dict(status='TOKEN_PLAN_BLOCKED',error=str(exc),
                                            stream_sha256=member['sha256'])
    result['status']='PASS' if all(v['status']=='TOKEN_PLAN_PASS' for v in result['datasets'].values()) else 'BLOCKED'
    result['actual_model_native_parity']='NOT_RUN'
    write_new(args.output,result)
    print(result)
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
