"""Read the first durable S boundary only; no model/GPU/forward/replay."""
import argparse
import json
from pathlib import Path
import torch
from .common import ROOT,member,write,digest,tensor_sha
from .sequential_state import state_identity,require_next,require_finalizer

def main():
    torch.set_num_threads(8)
    p=argparse.ArgumentParser();p.add_argument('--observation',type=Path,required=True);a=p.parse_args()
    observation=json.loads(a.observation.read_text())
    available=[arm for arm,d in observation['arms'].items() if 'S_INITIAL_VALID.json' in d]
    if not available:raise ValueError('INITIAL_NOT_OBSERVED')
    arm='EN-F' if 'EN-F' in available else available[0]
    root=ROOT/'S/attempt-v1/arms'/arm/'attempt-v1';b=root/'B001'
    marker=json.loads((root/'S_INITIAL_VALID.json').read_text())
    for key in ('parent_commit','next_entry'):
        if member(marker[key]['path'])!=marker[key]:raise ValueError('INITIAL_MEMBER_'+key)
    commit=json.loads(Path(marker['parent_commit']['path']).read_text())
    next_entry=json.loads(Path(marker['next_entry']['path']).read_text())
    cp=commit['checkpoint']
    if member(cp['path'])!=cp:raise ValueError('CHECKPOINT_MEMBER')
    payload=torch.load(cp['path'],weights_only=True,map_location='cpu',mmap=True)
    for key,shape in [('weight',(4096,14336)),('M4',(1,14336,14336))]:
        t=payload[key]
        if tuple(t.shape)!=shape or t.dtype!=torch.float32 or not torch.isfinite(t).all():
            raise ValueError('CHECKPOINT_TENSOR_'+key)
    if payload['next_batch']!=2 or payload['arm']!=arm or len(payload['received_ledger'])!=100:
        raise ValueError('CHECKPOINT_SEQUENTIAL_SCHEMA')
    current=state_identity(payload['weight'],payload['M4'],payload['RNG'],payload['received_ledger'],payload['contexts'])
    require_next(commit['state'],current);require_next(current,next_entry['identity'])
    entry=json.loads((b/'ENTRY.json').read_text())
    require_finalizer(commit['history'],entry['identity']['M'],current['W'],current['M'])
    if [r['case_id'] for r in payload['received_ledger']]!=payload['sample_order'][:100]:
        raise ValueError('RECEIVED_LEDGER_FIRST100')
    if next_entry['case_ids']!=payload['sample_order'][100:200]:raise ValueError('NEXT_BATCH_ORDER')
    observed=json.loads((b/'observers/selected-current.json').read_text())
    if observed['compatibility']['endpoint_weight_sha256']!=current['W']:
        raise ValueError('SELECTED_OBSERVER_ENDPOINT')
    for metric,n in [('RS',100),('PS',200),('NS',1000)]:
        if observed['metrics'][metric]['denominator']!=n:raise ValueError('INITIAL_DENOMINATOR_'+metric)
    if not observed['entry_selected_weight_restored_exact'] or not observed['RNG_unchanged_and_restored']:
        raise ValueError('OBSERVER_NONMUTATION_RUNTIME_RECEIPT')
    selection=json.loads((b/'selection-ledger.json').read_text())
    receipt=write(ROOT/'S/receipts/initial-boundary-cpu.json',dict(status='STORED_BOUNDARY_CPU_CONSISTENT',
        arm=arm,observation=member(a.observation),initial=member(root/'S_INITIAL_VALID.json'),checkpoint=cp,
        state=current,history_appends=1,next_batch=2,received=100,selected_stop=selection['stop_reason'],
        selected_native_fallback=selection['native_fallback'],selected_correction_norm=selection['actual_delta_norm'],
        selected_canonical_denominators={'RS':100,'PS':200,'NS':1000},
        observed_nonmutation='RUNTIME_GUARD_RECEIPT_NOT_GPU_REPLAY',source=payload['source'],
        CPU_weights_only_reload=True,GPU_continuation='NOT_TESTED',FD='NOT_RUN',
        T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',efficacy_PASS=False))
    print(json.dumps(receipt))

if __name__=='__main__':main()
