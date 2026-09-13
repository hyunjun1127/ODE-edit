"""Classify original refs without changing any original branch/worktree."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
def git(*a):return subprocess.check_output(['git',*a],text=True).strip()
def main():
    j=json.loads((ROOT/'inventory-before.json').read_text());rows=j['branches']
    for name in ('p1r54-pdz-ablation-v1','p1r54-pdz-time-sweep-analysis-v1'):
        r=copy.deepcopy(next(x for x in rows if x['branch']=='codex/'+name))
        r.update(branch='origin/codex/'+name,ref='refs/remotes/origin/codex/'+name,worktree='',worktree_tracked_dirty='NO_LOCAL_WORKTREE')
        assert git('rev-parse',r['ref'])==r['head'];rows.append(r)
    taskmap={
        'baseline-mechanism':'ODEEDIT-S06-BASELINE-MECHANISM-FIRST-E01-SH1-V1',
        'e01-':'ODEEDIT-S06-BASELINE-MECHANISM-FIRST-E01-SH1-V1',
        'multilayer-joint-edit':'ODEEDIT-S06-MULTILAYER-JOINT-EDIT-A-SH1-V1',
        'p1r54-pdz':'ODEEDIT-S05-P1R54-PDZ-PRC-TIME-SWEEP-T2-T3-T5-DIRECT-V1',
        'p1r55':'ODEEDIT-S05-P1R55-REQUEST-LOCAL-RMS-PDZ-FLOORED-RATE-V1',
        'piru-seq-10xb100-hon':'ODEEDIT-S05-P1R52-PIR-U-SEQUENTIAL-10XB100-STRUCTURALH-ON-V1-B4-DIAG-C2',
        'bgode-fbp-f0':'BGODE-FBP-F0-R1', 'bgode-r1-s1-analysis':'BGODE-R1-S1-MATCHED-ANALYSIS',
        'single-layer-cumulative':'ODEEDIT-S06-SINGLE-LAYER-CUMULATIVE-RISK-ABC-SH1-V1',
        'l4-two-memory':'ODEEDIT-S06-L4-TWO-MEMORY-CONFLICT-ROUTING-SH1-V2'}
    exclusions={'functional.py','linear_solve.py','elastic_qp.py'}
    btests={'test_b_protocol.py','test_elastic.py','test_functional.py','test_pcg.py'}
    residual=[]
    for r in rows:
        r['task_id']=next((v for k,v in taskmap.items() if k in r['branch']),'HISTORICAL_TASK_ID_NOT_EXTRACTED; source/ref evidence retained')
        if r['branch'].startswith('codex/publish-p1r52-piru-'):
            r['owner']='SH1';r['owner_evidence']='head-server1-gh author + server1 report/SH1 source diff; patch/ancestry in main'
            r['status']='ALREADY_IN_MAIN'
        reason='ancestor proof' if r['ancestry'] else 'patch-equivalent cherry-picks' if r['patch_equivalent'] else 'changed file blobs/modes all equal' if r['all_changed_file_bytes_equal'] else ''
        if 'bgode-fbp-f0-foundation' in r['branch']:
            r['status']='NOT_READY';reason='GH F0_R1_HOLD_F0_R2_REQUIRED: full-W0 rollback/accounting/device-order/receipt defects unresolved in4891906a; no runtime repair authorized by this Git task'
        elif 'piru-seq-10xb100-hon' in r['branch']:
            r['status']='CONFLICT_REQUIRES_DECISION';reason='3 runtime/router files conflict with newer main; original worktree additionally has3 dirty files. Old observer changes not blindly restored; choose later scoped reconciliation or retain archive.'
        elif r['status']=='PENDING_REVIEW':
            r['status']='INTEGRATED';reason='verified SH1 scope integrated; original source/report blobs preserved except documented additive dispatch composition or existing repaired E01 successor'
            if 'multilayer-joint-edit-a-v1' in r['branch']:
                r['status']='PARTIALLY_INTEGRATED';reason='all SH1-owned scope integrated; SH2 shared kernels/track_b/tests/server2 audits excluded and handed to SH2, no duplicate merge'
        r['integration_reason']=reason
        rem=[]
        for m in r['members']:
            p=m['path']
            other=p.startswith('audits/servers/server2/') or '/track_b/' in p or (p.startswith('project/run_scripts/multilayer_joint_compensation/') and Path(p).name in exclusions|btests)
            if r['status'] in ('NOT_READY','CONFLICT_REQUIRES_DECISION') or (r['status']=='PARTIALLY_INTEGRATED' and other):
                rem.append(p)
                residual.append({'branch':r['branch'],'head':r['head'],'status':r['status'],'path':p,'reason':reason,'decision_owner':'SH2' if other else 'GH/USER for future scoped work'})
        r['remaining_files']=rem
        r['comparison_after_head']=git('rev-parse','HEAD')
    keys=['branch','head','tree','owner','task_id','original_base_reflog','comparison_merge_base','status','integration_reason','worktree','worktree_tracked_dirty','changed_files','remaining_files']
    with (ROOT/'branch-inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader()
        for r in rows:
            q={k:r.get(k,'') for k in keys};q['changed_files']=';'.join(m['path'] for m in r['members']);q['remaining_files']=';'.join(r['remaining_files']);w.writerow(q)
    with (ROOT/'remaining-files.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['branch','head','status','path','reason','decision_owner'],lineterminator='\n');w.writeheader();w.writerows(residual)
    counts={s:sum(r['status']==s for r in rows) for s in sorted(set(r['status'] for r in rows))}
    payload={'base_main':j['base_main'],'source_publication_head':git('rev-parse','HEAD'),'branch_ref_count':len(rows),
             'unique_heads':len(set(r['head'] for r in rows)),'classification_counts':counts,'branches':rows,
             'root_dirty_preserved':True,'new_experiment_actions':0,'all_branches_integrated':not residual}
    (ROOT/'inventory-after.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='branches'},ensure_ascii=False))
if __name__=='__main__':main()
