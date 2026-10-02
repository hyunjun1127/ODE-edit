"""Bounded CPU owner audit; never a GPU qualification or submission."""
import argparse
import ast
import io
import json
from pathlib import Path
import subprocess
import unittest
from .common import ROOT,LOCAL,INSTRUCTION,member,write,require,digest

def leaves(value):
    if isinstance(value,dict):
        return sum((leaves(v) for v in value.values()),[])
    if isinstance(value,list):
        return sum((leaves(v) for v in value),[])
    return [value]

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='preflight-v1');a=p.parse_args()
    out=LOCAL/a.name;require(not out.exists(),'PREFLIGHT_CREATE_ONCE')
    config=json.loads((LOCAL/'preparation-v1/configuration.json').read_text())
    native=[]
    for row in config['native_BLUE']['members']:
        path=Path(row['path']);rel=path.relative_to('/data/janghj/BLUE')
        sealed=subprocess.check_output(['git','-C','/data/janghj/BLUE','show',config['native_BLUE']['commit']+':'+str(rel)])
        require(sealed==path.read_bytes(),'NATIVE_BLUE_WORKING_SOURCE_DIFF:'+str(rel))
        native.append(row)
    archived=[]
    for version in ('native-writer-v8','realization-v9'):
        path=ROOT/('plans/global/2026-10-03-jlz-'+version+'/math/results.json')
        data=json.loads(path.read_text());all_fields=leaves(data)
        archived.append(dict(file=member(path),top_level=list(data),parsed_scalar_fields=len(all_fields),
            parsed_content_sha=digest(data),interpretation='원 CPU 증거 전체 JSON 파싱; 새 GPU 검증 아님'))
    source=ROOT/'project/run_scripts/jlz_realization';sources=[]
    for path in sorted(source.glob('*.py')):
        ast.parse(path.read_text());sources.append(member(path))
    from . import test_core
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_core))
    write(out/'tests.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        output=stream.getvalue(),CPU_only=True,actual_LM_GPU='NOT_RUN'))
    require(result.wasSuccessful(),'CPU_REGRESSION_FAILED')
    write(out/'receipt.json',dict(instruction_id=INSTRUCTION,owner='SH4',review='OWNER_AUDIT',independent_reviewer=False,
        native_source_matches_exact_commit=native,archived_math_full_parse=archived,sources=sources,
        CPU_tests=result.testsRun,actual_LM_GPU='NOT_RUN',job_ids=[],
        input_receipt=member(LOCAL/'preparation-v1/full-read.json'),archive_receipt=member(LOCAL/'inputs/design-package/receiver-proof.json'),
        host_plan_GiB=config['resources']['host_plan_total_GiB'],host_limit_GiB=59,
        scope=dict(main_fits=10,main_candidates=250,main_updates=240,Q1_candidates=100,Q1_updates=96,
            Q2_in_main_A_B1=True,additional_B100_fits=0,new_baseline=0,replay=False,no_B6=True,noCP=True),
        notice='CPU surrogate 및 source 검산만 완료. 실제 Q1과 Q2는 별도 receipt 필요.'))
    print(json.dumps(dict(receipt=str(out/'receipt.json'),CPU_tests=result.testsRun,status='CPU_PASS_GPU_NOT_RUN')))

if __name__=='__main__':main()
