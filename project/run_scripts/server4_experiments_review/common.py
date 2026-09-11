from pathlib import Path
import numpy as np
from project.run_scripts.blue_fivearm_analysis.common import sha,digest,read,save,csvread,csvwrite,stats
from project.run_scripts.blue_lifelong_analysis.common import table,SCHEDULE,MULT,SAMPLE_ROOT
from project.run_scripts.blue_lifelong_analysis.metrics import reduce_eval,reduce_rows
from project.run_scripts.blue_lifelong_analysis.aggregate import transition

REPO=Path(__file__).resolve().parents[3]
BASE=Path('/data/janghj/ODE-edit')
LOCAL=BASE/'local/server4-experiments-review/20260911-v1'
OUT=REPO/'experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1'
OLD=REPO/'experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v3'
INHERITED=OLD/'inherited-v2'
INSTRUCTION='ODEEDIT-S06-SERVER4-COMPLETED-EXPERIMENTS-DETAILED-REVIEW-SH4-V1'
OLDARMS=['MEMIT_ORIGINAL','AlphaEdit_ORIGINAL','MEMIT_L4_ONLY','AlphaEdit_L4_ONLY','MEMIT_L8_ONLY','AlphaEdit_L8_ONLY']
NEWARMS=['MEMIT_L5_ONLY','AlphaEdit_L5_ONLY','MEMIT_L6_ONLY','AlphaEdit_L6_ONLY','MEMIT_L7_ONLY','AlphaEdit_L7_ONLY','BASE_ALPHAEDIT','BASE_MEMIT']
ARMS=OLDARMS+NEWARMS
JOBS=['39307','39283_1','39283_2','39283_3','39283_4','39283_5','40441','40442','40443','40444','40445','40446','42657','42658']
def family(arm):return 'AlphaEdit' if 'AlphaEdit' in arm or 'ALPHAEDIT' in arm else 'MEMIT'
def label(arm):
    if arm.startswith('BASE_'):return arm+' (blue=False; L4–L8)'
    return arm.replace('_ORIGINAL','_BLUE (L4+L8)').replace('_L','_BLUE_L')
def attempt(arm):
    if arm in OLDARMS:return BASE/'local/blue-lifelong-b100x100'/('attempt-cell0-checkpoint-r3' if arm==OLDARMS[0] else 'attempt-checkpoint-r2')
    return BASE/('local/fixed10k-native-baselines/attempt-v1' if arm.startswith('BASE_') else 'local/blue-lifelong-b100x100-l567/attempt-v1')
def cell(arm):return (1 if arm=='BASE_ALPHAEDIT' else 2) if arm.startswith('BASE_') else (OLDARMS.index(arm) if arm in OLDARMS else NEWARMS.index(arm))
def root(arm):return attempt(arm)/'output'/f'main-cell-{cell(arm)}'
def sample():
    return read(BASE/'local/datasets/counterfact-fixed-10k-v1/source-sample.lock.json')
def p99(rows):
    from collections import defaultdict
    result={}
    for field in ['new_nll','true_nll','margin']:
        result[field+'_prompt_p99']=float(np.quantile([r[field] for r in rows],.99))
        g=defaultdict(list)
        for r in rows:g[r['case_id']].append(r[field])
        result[field+'_request_p99']=float(np.quantile([np.mean(v) for v in g.values()],.99))
    return result
def reductions(ev,ids,arm,batch,scope):
    rr=reduce_eval(ev,ids,arm,batch,scope)
    for r in rr:r.update(p99(ev['metrics'][r['metric']]['rows']))
    return rr
def checked(rootpath,relative,manifest):
    p=rootpath/relative;m=manifest[relative]
    assert not p.is_symlink() and p.resolve()==p
    h=sha(p);assert (p.stat().st_size,h)==(m['bytes'],m['sha256']),str(p)
    return dict(path=str(p),bytes=m['bytes'],sha256=h,level='NEW_FULL_FILE_SHA')
def family_table(summary,method):
    rows=[dict(arm='PRE_EDIT W0 (공통 1회)',status='REUSED_SEALED_PUBLICATION',job='42673 (SH2)',RS='791/10000 (7.910%)',PS='1997/20000 (9.985%)',NS='89212/100000 (89.212%)')]
    names=[a for a in ARMS if family(a)==method]
    names.sort(key=lambda a:0 if a.startswith('BASE') else 1 if 'ORIGINAL' in a else int(a[-6]))
    for a in names:
        r=next(x for x in summary if x['arm']==a);x=dict(arm=label(a),status=r['status'],job=r['job'])
        for tag in MULT:
            x[tag]=f"{int(r[tag+'_numerator'])}/{int(r[tag+'_denominator'])} ({100*float(r[tag+'_rate']):.3f}%)" if r.get(tag+'_numerator','')!='' else 'NA'
        rows.append(x)
    return table(rows,['arm','status','job','RS','PS','NS'])
