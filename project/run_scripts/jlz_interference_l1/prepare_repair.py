"""Bind a USER-requested PRICE repair and existing exact W0 scalar observations."""
import argparse,json
from pathlib import Path
from . import ROOT,LOCAL,member,write,require
from .prepare import prepare
from .w0_reuse import seal_w0_reuse

def prepare_repair(out,attempt,preflight):
    out,attempt,preflight=map(lambda p:Path(p).resolve(),(out,attempt,preflight))
    require(attempt==LOCAL/'repair-59721' and out==LOCAL/'repair-59721-preparation','EXACT_USER_REPAIR_PATHS')
    require(not (out/'configuration.json').exists(),'CREATE_ONCE_REPAIR_CONFIG')
    prepare(out/'fresh-bindings',attempt,preflight,cpus=8)
    c=json.loads((out/'fresh-bindings/configuration.json').read_text())
    c['execution_arms']=['PRICE']
    c['repair']=dict(failed_job=59721,request='59721 fail되었으니 교정해',
        W0_request='기존 데이터 활용이 가능한 것 같으면 제외해',
        controls_request='flat이랑 reverse는 취소시키자 ours가 중요하니 ours 먼저 돌려',
        regression=member(LOCAL/'repair-59721-regression.json'),
        control_cancellation=member(ROOT/'audits/servers/server4/jlz-interference-priced-l1-2k/user-cancel-controls.json'),
        no_checkpoint_resume=True,method_or_tolerance_change=False,automatic_retry=False)
    c['W0_reuse']=seal_w0_reuse(LOCAL/'attempt',c)
    write(out/'w0-reuse.json',c['W0_reuse']);write(out/'configuration.json',c)
    return dict(configuration=str(out/'configuration.json'),execution_arms=c['execution_arms'],
        W0_status=c['W0_reuse']['status'],W0_rows=c['W0_reuse']['row_count'],W0_new_forward=0,
        W0_original_seconds=c['W0_reuse']['original_evaluation_seconds'],new_model_load=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--preflight',required=True);a=p.parse_args();print(json.dumps(prepare_repair(a.out,a.attempt,a.preflight)))
