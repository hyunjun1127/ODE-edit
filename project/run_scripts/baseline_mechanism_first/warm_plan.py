"""Seal an exact local L4 entry and its bounded ten-batch native window."""
import argparse
import csv
import json
from pathlib import Path
from .assets import ABC
from .contracts import Cell,ContractBoundary,digest,member,save
from .panels import panel_manifest
from .case_population import source_digest


def make(worktree,root,cold_lock,entry_n=1000,instrument=False):
    from scripts.fixed_counterfact import load_prefix
    wt,root=Path(worktree).absolute(),Path(root).absolute()
    original=json.loads(Path(cold_lock).read_text());plan=json.loads((root/'imports/server4-allowlist-v1.json').read_text())
    mapping=list(csv.DictReader((wt/'transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/migration-map.csv').open()))
    cell=Cell(4,entry_n);assert entry_n in (1000,5000,9000)
    entry=next(r for r in mapping if r['arm']=='AlphaEdit_L4_ONLY' and int(r['batch'])==entry_n//100)
    expected=next(r for r in mapping if r['arm']=='AlphaEdit_L4_ONLY' and int(r['batch'])==cell.comparison_batch)
    cp=member(ABC/'imports/entries'/f'B{entry_n//100:03d}'/'W-method-state.pt',expected=entry['sha256'])
    assert cp['bytes']==int(entry['bytes'])
    records=load_prefix(original['dataset_root'],10000)
    companions={};members={}
    for name in ('entry.json','contexts.json','native-targets.pt','native-observation.json','current.json'):
        r=next(r for r in plan['members'] if r['source_path'].endswith(f'blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/B{cell.next_batch:03d}/'+name))
        p=r['local_reuse'] or r['destination'];row=member(p,expected=r['sha256']);members[name]=row;companions[name]=row['path']
    entry_json=json.loads(Path(companions['entry.json']).read_text())
    assert entry_json['request_ids']==[r['case_id'] for r in records[entry_n:entry_n+100]]
    assert entry_json['request_hashes']==[source_digest(r['requested_rewrite']) for r in records[entry_n:entry_n+100]]
    original.update(cell=cell.plan(),native_batches=cell.plan()['native_batches'],entry_checkpoint=cp,
        entry_mapping=entry,comparison_checkpoint=None,comparison_expected_catalog=expected,
        comparison_status='PENDING_S2_SELECTIVE_RECEIPT_NOT_REQUIRED_TO_REUSE_VERIFIED_S1_ENTRY',
        companions=companions,companion_members=members,
        records=[dict(case_id=r['case_id']) for r in records[entry_n:entry_n+100]],
        prior_gate=member(root/'attempts/cold-l4-native-r1/output/INITIAL_VALID.json',
                         expected='eec8f26e8c76932a2a845989cc28e3ce4b5bc76a3f3063da9c6b71ab4cbe18f9'),
        panel=panel_manifest(records,entry_n),
        first_wave=dict(native_batches=10,first_batch_shared_with_E1=True,next_submission_after_user_recall=True),
        checkpoint_reference_status='EXACT_LOCAL_ENTRY_PRESENT_COMPARISON_NOT_YET_RECEIVED',
        diagnostics=dict(**original['diagnostics'],repeated_FD=0,prior_code_and_L4_mapping_gate_reused=True,
            general='NOT_YET_MEASURED',full_query_exposure='PENDING_OBSERVATION_ONLY_SUPPLEMENT',
            source_equivalence='NOT_ASSUMED',priority_backward=False),
        cold_reference_input=member(cold_lock))
    original['historical_json_identity']='BLUE ensure_ascii=True; local manifest canonicalization unchanged'
    ref=root/'imports/server2-selected-v1/AlphaEdit_L4_ONLY'/f'B{cell.comparison_batch:03d}'/'W-method-state.pt'
    if ref.exists():
        original['comparison_checkpoint']=member(ref,expected=expected['sha256'])
        original['comparison_status']='EXACT_REFERENCE_RECEIVED_CPU_SHA_VERIFIED_GPU_EQUIVALENCE_NOT_CLAIMED'
    if instrument:
        from .observation_panels import signed_inventory
        from .panels import historical_ordinals
        general=root/'locks/general-panel-r1/general-manifest.json'
        original['general_manifest']=member(general)
        g=json.loads(general.read_text())
        inventory=signed_inventory(records,list(range(entry_n,entry_n+100)),historical_ordinals(entry_n),g['rows'])
        original['observations']=dict(layer=4,entry_n=entry_n,signed_enabled=True,signed_panel_identity=digest(inventory),
            signed_panel=inventory,query_scope='ALL_CURRENT_HISTORICAL_RPN_AND_GENERAL',full_geometry=True)
        old=json.loads(Path(original['original_execution_lock']['path']).read_text())
        c=next(r for r in old['stats_members'] if 'model.layers.4.mlp.down_proj_' in r['path'])
        path='/mnt/raid5/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/'+Path(c['path']).name
        original['covariance']=member(path,expected=c['sha256'])
        original['diagnostics'].update(priority_backward=True,initial_probe_new_direction=True,
            general='SEALED_EXISTING_WIKIPEDIA_128',full_query_exposure='SEALED_FULL_PAIRS_AFTER_NATIVE',
            spectrum='EXACT_CPU_FULL_DIMENSION_SVD_DIAGNOSTIC_ONLY_AFTER_INITIAL_GATE')
    seal=f'warm-l4-n{entry_n}-'+('instrument-r1' if instrument else 'r1')
    return save(root/'locks'/seal/'input.lock.json',original)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--root',required=True);p.add_argument('--cold-lock',required=True)
    p.add_argument('--entry-n',type=int,default=1000);p.add_argument('--instrument',action='store_true')
    a=p.parse_args();print(json.dumps(make(a.worktree,a.root,a.cold_lock,a.entry_n,a.instrument)))
