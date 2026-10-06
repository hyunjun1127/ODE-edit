"""Static/import/config confirmation only: no synthetic or toy execution."""
import argparse,importlib,json,time
from pathlib import Path
from . import ROOT,TASK,NONCE,DESIGN,member,sha,write,require

def run(out):
    out=Path(out);require(not out.exists(),'CREATE_ONCE_PREFLIGHT')
    folder=ROOT/'project/run_scripts/jlz_interference_l1'
    files=sorted(folder.glob('*.py'))
    expected={'__init__','profile','price','projection','controller','optimizer','engine','fit',
              'telemetry','storage','observer_io','run','collect','prepare','submit','preflight'}
    require(expected<=set(p.stem for p in files),'MISSING_IMPLEMENTATION_MODULE')
    started=time.monotonic()
    for path in files:
        compile(path.read_text(),str(path),'exec')
        if path.stem!='__init__':importlib.import_module('project.run_scripts.jlz_interference_l1.'+path.stem)
    manifest=ROOT/DESIGN/'artifact-manifest.json'
    require(sha(manifest)=='b7c7e7644bd1878143ce6df43f58b384978b10e01a235617d29081f5052fb78b','CANONICAL_MANIFEST')
    for row in json.loads(manifest.read_text())['files']:
        p=ROOT/DESIGN/row['path'];require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'CANONICAL_MEMBER')
    from .profile import ARMS,arm_profile,resource_order
    require(ARMS==('PRICE','FLAT','REVERSE'),'THREE_ARMS')
    for arm in ARMS:
        p=arm_profile({},arm);require(p['eligible_layers']==[4,5,6,7,8] and p['K_eval']==25
            and p['max_updates']==24 and p['c']==.75 and not p['blind'],'PROFILE_IDENTITY')
    require(resource_order(2)=={'PRICE':[],'FLAT':['PRICE'],'REVERSE':['PRICE'],'collector':list(ARMS)},'DAG_RESOURCE_ONLY')
    files.append(ROOT/'project/run_scripts/jlz_native_writer_aware/builder.py')
    result=dict(instruction_id=NONCE,task_id=TASK,status='PASS',passed=True,source=[member(p) for p in files],
        scope='source syntax/import/config/canonical identity only',seconds=time.monotonic()-started,
        synthetic_tests=0,toy_tests=0,numerical_model_tests=0,model_load=0,model_forward=0,
        new_GPU=0,Slurm_write=0,actual_model_qualified=False,old_PASS_inherited=False,
        actual_checks='Same main B1 cached c0/first proposal; no extra solve/forward/fit',
        owner_check_not_independent_red=True)
    write(out,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    print(json.dumps(run(p.parse_args().out)))
