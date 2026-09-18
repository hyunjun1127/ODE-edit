"""CPU syntax, focused fixtures, exact tokenizer/input binding; no model load."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from .common import ROOT,write,member,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);a=p.parse_args()
    output=Path(a.receipt);result=dict(new_GPU=0,model_load=0)
    try:
        package=Path(__file__).parent
        for f in package.rglob('*.py'):ast.parse(f.read_bytes())
        result['source']=[member(f) for f in sorted(package.rglob('*.py'))]
        run=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(package/'tests'),'-v'],capture_output=True,text=True)
        result['unit_exit']=run.returncode;result['unit_output']=run.stdout+run.stderr
        if run.returncode:raise ValueError('CPU_TEST_FAILURE')
        from transformers import AutoTokenizer
        from scripts.fixed_counterfact import load_prefix
        from .binding import protected_sequences
        lock=json.loads((ROOT/'inputs/base-binding.json').read_text())
        t=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);t.add_bos_token=False;t.pad_token_id=t.eos_token_id
        e=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);e.pad_token_id=e.eos_token_id
        cold=json.loads(Path(lock['cold_capsule']['path']).read_text());records=load_prefix(lock['dataset_root'],1000)
        if digest(records)!=lock['records_digest']:raise ValueError('FIXED_PREFIX')
        if [[t(x)['input_ids'] for x in g] for g in cold['contexts']]!=cold['context_tokens']:raise ValueError('CONTEXT_TOKENS')
        r=[dict(x['requested_rewrite'],case_id=x['case_id']) for x in records[:8]]
        packs,rows,unique,meta=protected_sequences(t,e,r,cold['contexts'])
        if len(rows)!=8*2*7 or {x['kind'] for x in rows}!={'native','canonical'}:raise ValueError('PROTECTED_SCOPE')
        result['token_fixture']=dict(sequence_rows=len(rows),inputs=len(packs),logical_prefix_columns=len(unique),
            sequence_identity=digest(rows),context_token_identity=digest(cold['context_tokens']),missing_old=meta['missing_old'])
        result['status']='CPU_PASS_ACTUAL_T_NOT_RUN'
    except BaseException as exc:
        result.update(status='CPU_FAIL',error=repr(exc),traceback=traceback.format_exc())
        write(output,result);raise
    print(json.dumps(write(output,result)))

if __name__=='__main__':main()
