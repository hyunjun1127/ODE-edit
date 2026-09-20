"""C02/F00 analysis without diagnostic prerequisites; no model-state artifact."""
import argparse
import gc
import time
import traceback
from pathlib import Path
import torch
from .common import *
from . import model_runtime as rt
from .validation import elementwise_gate
from .geometry import activation_bookkeeping
from .model_gate import tensor_artifact


def run(gate,output):
    gate=Path(gate);historical_gate=read(gate)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    model=None;w0=None;stage='LOAD'
    try:
        b=rt.bind_sources();rows,probe=rt.stream_and_probe();mapping=source_map()
        model,tok,evaltok,w0,runtime=rt.load();guard=rt.pointer_versions(model)
        write_json(out/'runtime.json',runtime)
        native={};capture_checks={}
        for batch,entry in ((1,0),(2,1),(6,5),(11,10),(21,20),(51,50),(91,90)):
            stage=f'B{batch:03d}'
            contexts=read(mapped(f'{S4CELL}/B{batch:03d}/contexts.json',mapping))
            selected=rows[(batch-1)*100:batch*100];rt.set_weight(model,w0)
            old=ATTEMPT/'results/model-gate-r1'/f'B{batch:03d}-keys.pt'
            if batch in (1,2) and old.exists():
                reused=read(ATTEMPT/'receipts/gate-technical-r1-reuse.json')
                member=next(m for m in reused['members'] if m['name']==old.name)
                assert old.stat().st_size==member['bytes'] and sha256(old)==member['sha256']
                c=torch.load(old,weights_only=True,map_location='cpu')
                assert c['ids']==[r['case_id'] for r in selected]
                assert c['contexts_sha256']==digest(contexts)
                c['reused_capture']=dict(path=str(old),sha256=sha256(old),new_capture_seconds=0)
            else:
                c=rt.capture(model,tok,selected,contexts,b,early=True)
            if entry:
                state=rt.checkpoint(entry);we=state['weights'][WEIGHT]
                rt.set_weight(model,we)
                physical=rt.capture(model,tok,selected,contexts,b,early=True)
                c.update(h_entry=physical['h0'],entry_weight_sha256=tensor_sha(we),entry_packing_sha256=physical['packing_sha256'])
                del state,we,physical
            else:c.update(h_entry=c['h0'],entry_weight_sha256=tensor_sha(w0))
            c['entry_batch']=entry
            c['h0W0']=c['h0']
            assert all(torch.isfinite(c[k]).all() for k in ('K','bare_K','h0','h_entry'))
            targets=torch.load(mapped(f'{S4CELL}/B{batch:03d}/native-targets.pt',mapping),weights_only=True,map_location='cpu')
            assert all(x['layer']==4 for x in targets['identities'])
            assert [x['case_id'] for x in targets['identities']]==c['ids']
            assert [tensor_sha(x) for x in targets['values']]==[x['sha256'] for x in targets['identities']]
            native[stage]=c
            print('KEY_BANK_CAPTURE',stage,c['seconds'],flush=True)
        rt.set_weight(model,w0);stage='GEOMETRY512'
        contexts=read(mapped(S4CELL+'/B001/contexts.json',mapping))
        geometry=rt.capture(model,tok,probe,contexts,b,early=True)
        streamgroups={(r['requested_rewrite']['subject'],r['requested_rewrite']['relation_id']) for r in rows}
        overlap=sum((r['requested_rewrite']['subject'],r['requested_rewrite']['relation_id']) in streamgroups for r in probe)
        allpass=True  # Computation completed; no numerical certification asserted.
        bank=dict(native=native,geometry=geometry,w0_tensor_sha256=tensor_sha(w0),
            probe_policy='sha256(20260920|canonical_case_id), numeric case-id tiebreak; full dataset exclude stream10k IDs',
            stream_probe_case_overlap=0,subject_relation_overlap=overlap,
            gate_sha256=sha256(gate),capture_checks=capture_checks,save_checkpoints=False,
            status='PASS',readiness='READY_WITHOUT_DIAGNOSTIC_GATES',**EXECUTION_POLICY)
        bank_receipt=tensor_artifact(out/'key-bank.pt',bank)
        write_json(out/'C02.json',dict(status='PASS' if allpass else 'FAILED',native_requests=700,geometry_requests=512,
            capture_checks=capture_checks,bank=bank_receipt,subject_relation_overlap=overlap,case_id_overlap=0,**EXECUTION_POLICY))
        # F00 is algebra on existing actual W and verified fixed geometry keys.
        stage='F00';keys=geometry['K'];table=[];previous=1
        for end in CONTRACT['inputs']['checkpoint_batches'][1:]:
            a=rt.checkpoint(previous);z=rt.checkpoint(end)
            e=a['weights'][WEIGHT].double()-w0.double()
            d=z['weights'][WEIGHT].double()-a['weights'][WEIGHT].double()
            values=activation_bookkeeping(e,d,keys)
            for i,case in enumerate(geometry['ids']):
                table.append(dict(interval_start=previous,interval_end=end,case_id=case,key_type='native_group_mean_geometry_probe',
                    **{k:v[i] for k,v in values.items() if isinstance(v,list)},maximum_closure_error=values['maximum_closure_error']))
            del a,z,e,d;previous=end
        write_csv(out/'fixed_probe_activation.csv',table)
        write_json(out/'F00.json',dict(status='PASS' if allpass else 'BLOCKED',rows=len(table),
            key_bank_sha256=bank_receipt['sha256'],table_sha256=sha256(out/'fixed_probe_activation.csv'),
            precision='float64 actual weight subtraction/contraction',algebraic_only=True,model_locality_claim=False,**EXECUTION_POLICY))
        rt.set_weight(model,w0);assert tensor_sha(model.get_parameter(WEIGHT))==tensor_sha(w0)
        assert rt.pointer_versions(model)==guard
        write_json(out/'terminal.json',dict(status='PASS' if allpass else 'FAILED',C02=allpass,F00=allpass,
            bank=bank_receipt,elapsed_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            W0_restore=True,nonselected_guard=True,checkpoint_saved=False,z_optimization=0,history_append=0,**EXECUTION_POLICY))
    except BaseException as ex:
        restored=None
        if model is not None and w0 is not None:
            rt.set_weight(model,w0);restored=tensor_sha(model.get_parameter(WEIGHT))==tensor_sha(w0)
        write_json(out/'failure.json',dict(stage=stage,error=repr(ex),traceback=traceback.format_exc(),
            status='FAILED_TECHNICAL',elapsed_seconds=time.perf_counter()-start,W0_restore=restored))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gate',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.gate,a.output)
