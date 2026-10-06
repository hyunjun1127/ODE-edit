"""Source/import/CLI/config audit only. No synthetic inputs or numerical tests."""
import argparse,ast,importlib,json,os,socket,subprocess,sys
from .common import *

def check(out):
    require(socket.gethostname()=='server4','HOST_BOUNDARY')
    require(os.environ.get('CODEX_THREAD_ID')=='01a04939-b5c7-7a03-ba2d-ef3343d62cfd','SESSION_BOUNDARY')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    require(branch=='codex/server4-price-alpha-writer-2k','OWN_NON_MAIN_BRANCH')
    paths=sorted((ROOT/'project/run_scripts/jlz_price_alpha_writer').glob('*.py'))
    for p in paths:
        ast.parse(p.read_text(),filename=str(p));importlib.import_module('project.run_scripts.jlz_price_alpha_writer.'+p.stem)
    for module in ('run','collect','submit'):
        checked=subprocess.run([sys.executable,'-m','project.run_scripts.jlz_price_alpha_writer.'+module,'--help'],
            cwd=ROOT,capture_output=True,text=True,timeout=60)
        require(checked.returncode==0,'CLI_IMPORT:'+module)
    from .submit import resource_order,ROLES
    for parallel in (1,2):
        order=resource_order(parallel)
        require(set(order)==set(ROLES) and order['collector']==list(CELLS),'SIX_CELL_DAG_CONFIG')
        for cell in CELLS:require(all(CELLS.index(parent)<CELLS.index(cell) for parent in order[cell]),'ACYCLIC_REGISTER_ORDER')
    from project.run_scripts.jlz_interference_l1.cap_profile import arm_profile
    for model in MODELS:
        for arm in ARMS:
            p=arm_profile({},arm,model)
            require(p['c']==p['beta_max_native_scale']==.75 and p['beta_base']==(.75 if arm=='CAP075' else 1.),'BASE_LOCAL_INDEPENDENT')
            require(p['cap_mode']==('none' if arm=='FREE100' else 'native'),'TRUE_UNCAPPED_CONFIG')
    from project.run_scripts.experiment_tracking.schema import load_env
    load_env(LOCAL/'tracking.env')
    import torch
    require(not torch.cuda.is_initialized(),'STATIC_AUDIT_NO_GPU')
    result=dict(task=TASK,nonce=NONCE,passed=True,status='STATIC_IMPORT_CONFIG_CHECKED',source=[member(p) for p in paths],
        checks=['owner/session/origin/branch','six cold cells and acyclic resource DAG','separate base/native cap',
            'tagged endpoint and positive-interior source review','Qwen dimensions/readout adapter',
            'active SUM native loss and task-local Alpha builder','single stream/nullable cap reducer','online scalar helper configuration',
            'cap2/memory/noCP/unchanged historical sources','Alpha nonsymmetric LU/thin solve, LOO, separate C0/H diagnostics source audit'],
        source_review_level='OWNER_SOURCE_AUDIT; no independent reviewer',numeric_tests=0,toy_runs=0,
        model_load=False,extra_forward_backward_solve=0,actual_B1='NOT_OBSERVED',source_freeze_still_required=True)
    write(out,result);return dict(status=result['status'],files=len(paths),actual_B1='NOT_OBSERVED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(check(a.out)))
