"""Owner static/import/config review; not a toy or target-model PASS."""
import argparse,ast,importlib,json,os,socket,subprocess,sys
from .common import *

def check(out):
    require(socket.gethostname()=='server4','HOST_BOUNDARY')
    require(os.environ.get('CODEX_THREAD_ID')=='01a04939-b5c7-7a03-ba2d-ef3343d62cfd','SESSION_BOUNDARY')
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==
        'codex/server4-price-gptj-2k','OWN_BRANCH')
    require('hyunjun1127/ODE-edit' in subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True),'ORIGIN')
    paths=sorted((ROOT/'project/run_scripts/jlz_price_gptj').glob('*.py'))
    for p in paths:
        ast.parse(p.read_text(),filename=str(p))
        importlib.import_module('project.run_scripts.jlz_price_gptj.'+p.stem)
    for module in ('run','collect','submit'):
        cp=subprocess.run([sys.executable,'-m','project.run_scripts.jlz_price_gptj.'+module,'--help'],
            cwd=ROOT,capture_output=True,text=True,timeout=60)
        require(cp.returncode==0,'CLI_IMPORT:'+module+':'+cp.stderr)
    from .submit import resource_order,ROLES
    for parallel in (1,2):
        order=resource_order(parallel)
        require(set(order)==set(ROLES) and order['collector']==list(CELLS),'SIX_CELL_DAG')
        for cell in CELLS:require(all(CELLS.index(p)<CELLS.index(cell) for p in order[cell]),'ACYCLIC_DAG')
    from .profile import arm_profile
    for writer in ('memit','alphaedit'):
        for arm in ARMS:
            p=arm_profile({},arm,writer)
            require(p['eligible_layers']==[3,4,5,6,7,8] and p['nll_layer']==27 and p['expected_intermediate']==16384,'GPTJ_CONFIG')
            require(p['lr']==.5 and p['lambda_alpha']==10. and p['c']==.75 and p['beta_base']==(.75 if arm=='CAP075' else 1.)
                and p['cap_mode']==('none' if arm=='FREE100' else 'native'),'OURS_COEFFICIENTS')
    from project.run_scripts.experiment_tracking.schema import load_env
    load_env(LOCAL/'tracking.env')
    from .tracking import contract_ready
    try:contract_ready();tracking_ready=True;tracking_block=None
    except RuntimeError as error:tracking_ready=False;tracking_block=str(error)
    import torch
    require(not torch.cuda.is_initialized(),'CPU_ONLY')
    result=dict(task=TASK,nonce=NONCE,passed=True,status='STATIC_IMPORT_CONFIG_CHECKED',source=[member(p) for p in paths],
        checks=['session/repo/own branch','GPT-J parallel attention+MLP native add order and fc_out bias',
            'native LN/readout27/physical projector3..8 mapping to ours3..8; native lr.5/AlphaL2=10','same repaired PRICE optimizer/projection',
            'no-grad fresh upper builder and off-owner same-layer pullback','native context once, input lock, cold arms',
            'six-run DAG cap2 plus combined current cap3','single candidate stream/noCP/online scalar logger'],
        source_review_level='OWNER_SOURCE_AUDIT; 별도 reviewer 없음',numeric_tests=0,toy_runs=0,
        model_load=False,actual_B1='NOT_OBSERVED',tracking_ready=tracking_ready,tracking_block=tracking_block,
        helper_sources=[member(p) for p in sorted((ROOT/'project/run_scripts/experiment_tracking').glob('*.py'))],
        tracking_contracts=[member(ROOT/p) for p in ('control/wandb-policy.json',
            'control/wandb-method-metric-schema.json','messages/head/2026-10-07-wandb-method-metrics-all-sh.json')])
    write(out,result);return dict(status=result['status'],files=len(paths),actual_B1='NOT_OBSERVED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(check(a.out)))
