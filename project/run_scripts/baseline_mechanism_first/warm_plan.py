"""Seal the already-local exact L4 B010 entry and bounded B011..B020 window."""
import argparse
import csv
import json
from pathlib import Path
from .assets import ABC
from .contracts import Cell,ContractBoundary,digest,member,save
from .panels import panel_manifest
from .case_population import source_digest


def make(worktree,root,cold_lock):
    from scripts.fixed_counterfact import load_prefix
    wt,root=Path(worktree).absolute(),Path(root).absolute()
    original=json.loads(Path(cold_lock).read_text());plan=json.loads((root/'imports/server4-allowlist-v1.json').read_text())
    mapping=list(csv.DictReader((wt/'transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/migration-map.csv').open()))
    entry=next(r for r in mapping if r['arm']=='AlphaEdit_L4_ONLY' and int(r['batch'])==10)
    expected=next(r for r in mapping if r['arm']=='AlphaEdit_L4_ONLY' and int(r['batch'])==20)
    cp=member(ABC/'imports/entries/B010/W-method-state.pt',expected=entry['sha256'])
    assert cp['bytes']==int(entry['bytes'])
    records=load_prefix(original['dataset_root'],10000)
    companions={};members={}
    for name in ('entry.json','contexts.json','native-targets.pt','native-observation.json','current.json'):
        r=next(r for r in plan['members'] if r['source_path'].endswith('blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/B011/'+name))
        p=r['local_reuse'] or r['destination'];row=member(p,expected=r['sha256']);members[name]=row;companions[name]=row['path']
    entry_json=json.loads(Path(companions['entry.json']).read_text())
    assert entry_json['request_ids']==[r['case_id'] for r in records[1000:1100]]
    assert entry_json['request_hashes']==[source_digest(r['requested_rewrite']) for r in records[1000:1100]]
    original.update(cell=Cell(4,1000).plan(),native_batches=list(range(11,21)),entry_checkpoint=cp,
        entry_mapping=entry,comparison_checkpoint=None,comparison_expected_catalog=expected,
        comparison_status='PENDING_S2_SELECTIVE_RECEIPT_NOT_REQUIRED_TO_REUSE_VERIFIED_S1_ENTRY',
        companions=companions,companion_members=members,
        records=[dict(case_id=r['case_id']) for r in records[1000:1100]],
        prior_gate=member(root/'attempts/cold-l4-native-r1/output/INITIAL_VALID.json',
                         expected='eec8f26e8c76932a2a845989cc28e3ce4b5bc76a3f3063da9c6b71ab4cbe18f9'),
        panel=panel_manifest(records,1000),
        first_wave=dict(native_batches=10,first_batch_shared_with_E1=True,next_submission_after_user_recall=True),
        checkpoint_reference_status='EXACT_LOCAL_ENTRY_PRESENT_COMPARISON_NOT_YET_RECEIVED',
        diagnostics=dict(repeated_FD=0,prior_code_and_L4_mapping_gate_reused=True,
            general='NOT_YET_MEASURED',full_query_exposure='PENDING_OBSERVATION_ONLY_SUPPLEMENT',
            source_equivalence='NOT_ASSUMED',priority_backward=False),
        cold_reference_input=member(cold_lock))
    original['historical_json_identity']='BLUE ensure_ascii=True; local manifest canonicalization unchanged'
    return save(root/'locks/warm-l4-n1000-r1/input.lock.json',original)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--root',required=True);p.add_argument('--cold-lock',required=True)
    a=p.parse_args();print(json.dumps(make(a.worktree,a.root,a.cold_lock)))
