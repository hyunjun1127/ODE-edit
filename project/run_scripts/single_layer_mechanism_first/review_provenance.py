"""CPU-only create-once identity rebind for the completed exact B1 attempt."""
import argparse
import hashlib
from pathlib import Path
from .review_completed_b1 import read, member
from .review_b1 import dump


def run(attempt, repo, destination):
    attempt,repo,dest=map(Path,(attempt,repo,destination))
    lock=read(attempt/'execution.lock.json'); rows=[]
    assert member(attempt/'execution.lock.json')['sha256']=='6a14ebaf32549cc9479f2d112ba1954b06ef00380fffc1091cd70090f9098f61'
    assert lock['execution']['commit']=='5f79085629b10b2bb8bdee88d017e18a46bb4c74'
    changed={'PROTOCOL.md','servers/connection-inventory.md'}
    for r in lock['full_read']:
        now=member(repo/r['repo_path']);same=now['sha256']==r['sha256']
        if not same and r['repo_path'] not in changed:raise ValueError('UNEXPECTED_AUTHORITY_CHANGE:'+r['repo_path'])
        rows.append(dict(repo_path=r['repo_path'],prior_sha256=r['sha256'],current=now,
            reading='PRIOR_FULL_READ_EXACT_REUSE' if same else 'CHANGED_DOCUMENT_OR_FULL_DIFF_DIRECT_READ'))
    closure=[]
    namespaces=('single_layer_mechanism_first/','single_layer_edit_preserving_correction/',
                'en_execution_reuse/','low_cost_write_donor_pilot/')
    for r in lock['execution']['members']:
        if not any('/project/run_scripts/'+n in r['path'] for n in namespaces):continue
        now=member(r['path'])
        if (now['sha256'],now['bytes'])!=(r['sha256'],r['bytes']):raise ValueError('SOURCE_SEAL_CHANGED')
        closure.append(now)
    archive=member(lock['execution']['archive']['path'])
    assert archive['sha256']==lock['execution']['archive']['sha256']
    prior=[]
    for item in lock['prior_large_asset_binding']:
        old=item['prior'];p=Path(old['path']);s=p.stat()
        observed=[s.st_dev,s.st_ino,s.st_mtime_ns]
        expected=old.get('stat',item.get('current_stat'))
        prior.append(dict(path=str(p),bytes=s.st_size,prior_sha256=old['sha256'],
                          stable_stat=observed==expected,verification='PRIOR_FULLSHA_CURRENT_STAT_NOT_REHASH'))
    dump(dest,dict(lock=member(attempt/'execution.lock.json'),execution={k:v for k,v in lock['execution'].items() if k!='members'},
        archive_full_rehash=archive,authority=rows,source_full_rehash=closure,large_asset_reuse=prior,
        new_envelope=member(repo/'messages/head/2026-09-20-sh4-slmf-b1-completed-review.md'),
        latest_policy=member(repo/'plans/global/2026-09-19-default-no-experiment-checkpoints.md'),
        original_plan_full_read_receipt='/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/receipts/T0-attempt-v1-input-plan.json',
        analysis_base='0150da2f02c830baa070944e1f504852f2fe6b01',server='server4',
        session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',new_GPU=0,other_scheduler_queries=0))
    return dict(authority=len(rows),source=len(closure),large_asset_stat=len(prior),
                large_asset_all_stable=all(r['stable_stat'] for r in prior))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--repo',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();print(run(a.attempt,a.repo,a.destination))
