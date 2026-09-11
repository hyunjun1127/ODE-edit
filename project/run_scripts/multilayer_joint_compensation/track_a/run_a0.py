"""Middle A0 execution from the shared READY bundle, with initial-valid receipt.

The process naturally finishes its sealed A0 endpoint. Agent monitoring pauses
after the actual first-update gate and peer input verification, independently
of the process's optimization/evaluation schedule.
"""
import argparse
import dataclasses
import importlib
import json
import os
from pathlib import Path
import subprocess
import traceback
import torch
from ..contracts import *
from ..observations import JointView,pack,current_callback,teacher
from ..evaluation import panel,materialized,measure,generation,observe_panel
from ..history import finalize_history_once
from ..banks import current_rows,protection_rows
from ..common_reference.prepare import progress
from .native_geometry import NativeWriterMetric
from .planner import plan_joint_targets
from .initial_checks import run_initial_checks
from .delivered import capture as capture_delivered

def main(args):
    out=Path(args.output).absolute();out.mkdir(parents=True,exist_ok=False)
    ledger=Ledger();phase='INPUT';view=None
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if head!=args.source_head:raise RuntimeError('EXECUTION_SOURCE_MISMATCH')
    save(out/'run.lock.json',dict(source_head=head,args=vars(args),job=os.environ.get('SLURM_JOB_ID'),
      science=dataclasses.asdict(Science()),arm='A0',entry='Middle',scientific_promotion=False,
      actual_initial_gate='shared actual derivative/transaction and first joint optimizer update observed',
      observer_policy='first valid receipt does not stop this sealed process; agent pauses'))
    try:
        if os.environ.get('CUMRISK_ASSET_ROOT')!=str(ABC):raise RuntimeError('ASSET_ROOT_BINDING')
        from project.run_scripts.single_layer_cumulative_risk import binding
        from scripts.fixed_counterfact import load_prefix
        root=Path(args.common).absolute();ready=json.loads((root/'READY.json').read_text())
        if ready['status']!='COMMON_GPU_PREPARED_READY_FOR_SH2_VERIFY' or ready['entry']!='Middle':
            raise RuntimeError('COMMON_NOT_READY_OR_WRONG_ENTRY')
        for m in ready['members']:
            if member(m['path'])!=m:raise RuntimeError('COMMON_MEMBER_HASH_MISMATCH')
        if digest(ready['members'])!=ready['members_root']:raise RuntimeError('COMMON_MEMBERS_ROOT')
        save(out/'common-binding.json',dict(ready=member(root/'READY.json'),members_root=ready['members_root'],
          common_source_head=ready['source_head'],same_input_cross_server_verified='EXTERNAL_SH2_ACK_REQUIRED'))
        records=load_prefix(DATA,10000)
        entry=torch.load(root/'entry.pt',map_location='cpu',weights_only=True,mmap=True)
        rows=torch.load(root/'prediction-rows.pt',map_location='cpu',weights_only=True)
        saved={r:torch.load(root/f'We-teacher-{r}.pt',map_location='cpu',weights_only=True) for r in rows}
        geometries=[]
        for l in (4,8):
            with ledger.time(f'geometry_reuse_L{l}'):
                state=torch.load(root/f'native-geometry-L{l}.pt',map_location='cpu',weights_only=True,mmap=True)
                geometries.append(NativeWriterMetric.from_state(state))
            del state
        save(out/'model-tokenizer-source.json',dict(revision=MODEL.name,
          members=[hf_member(MODEL/name) for name in ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json')],
          timing='before A0 model/tokenizer load'))
        torch.manual_seed(20260911)
        phase='MODEL';model,tok,evaltok=binding.load_model(ledger);params=dict(model.named_parameters())
        names=tuple(entry['names'])
        for name,w0 in zip(names,entry['W0']):
            if not torch.equal(params[name].detach().cpu(),w0):raise RuntimeError('MODEL_W0_IDENTITY')
        with torch.no_grad():
            for name,w in zip(names,entry['We']):params[name].copy_(w.to('cuda'))
        view=JointView(model,names,ledger)
        if dict(zip(names,view.entry_sha))!=entry['entry_identity']['weights']:
            raise RuntimeError('ACTUAL_ENTRY_IDENTITY')
        inventory=entry['raw_effective_inventory']
        repacked={'Current':current_rows(records,inventory['current_effective'],entry['contexts'],tok)}
        for role,ids in inventory['bank'].items():repacked[role]=protection_rows(records,ids,role,tok)
        if digest(repacked)!=digest(rows):raise RuntimeError('ACTUAL_TOKENIZER_PACKING_IDENTITY')
        save(out/'packing-identity.json',dict(status='PASS',all_panels_sha=digest(rows),
          native_padding_side=tok.padding_side,evaluator_padding_side=evaltok.padding_side))
        def count_forward(module,positional,kw):
            ids=kw.get('input_ids',positional[0] if positional else None)
            ledger.add('actual_model_forward_invocations')
            if ids is not None:ledger.add('actual_forward_padded_tokens',ids.numel())
            if kw.get('attention_mask') is not None:ledger.add('actual_forward_nonpadding_tokens',int(kw['attention_mask'].sum()))
        model.register_forward_pre_hook(count_forward,with_kwargs=True)
        native=binding.kernel();hp=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams.from_json(ABC/'imports/config.json')
        current=[records[i] for i in entry['raw_effective_inventory']['current_effective']]
        delivered_entry=capture_delivered(model,tok,native,hp,entry['contexts'],current,ledger)
        tensor_save(out/'entry-module-readout.pt',delivered_entry)
        phase='ACTUAL_INITIAL_CHECKS'
        first=rows['Current'][0]['ordinal'];probe=[dict(r) for r in rows['Current'] if r['ordinal']==first]
        for r in probe:r['context_weight']=1/len(probe)
        batches=list(pack(view,probe,tok.pad_token_id,2))
        probe_panel=panel(view,probe,saved['Current'],tok.pad_token_id,'current',2)
        with ledger.time('initial_actual_derivative_transaction'):
            checks=run_initial_checks(view,batches,probe_panel,projectors=tuple(g.projector for g in geometries))
        save(out/'gpu-initial-checks.json',checks)
        progress(out,'ACTUAL_DERIVATIVE_TRANSACTION_VALID',ledger,checks=member(out/'gpu-initial-checks.json'))
        del probe_panel,batches
        phase='A0_JOINT_TARGETS';callback=current_callback(view,rows['Current'],tok.pad_token_id,2);calls=0
        def observed(deltas):
            nonlocal calls
            value,gradient,info=callback(deltas);calls+=1
            identities=[tensor_sha(x) for x in deltas]
            save(out/'current-objectives'/f'{calls-1:03d}.json',dict(completed_optimizer_updates=calls-1,
              objective=value,delta_sha=identities,info=info,compute=ledger.receipt()))
            if calls==2:
                tensor_save(out/'initial-joint-update.pt',dict(deltas=tuple(x.detach().cpu() for x in deltas),
                  entry_identity=entry['entry_identity'],completed_optimizer_updates=1))
                save(out/'INITIAL_VALID.json',dict(status='ACTUAL_A0_INITIAL_VALID',completed_optimizer_updates=1,
                  finite_objective=value,finite_gradient=True,full_logical_B=100,logical_contexts=len(rows['Current']),
                  physical_microbatch=2,joint_layers=[4,8],actual_full_sequence_weight_forward=True,
                  delta_sha=identities,current_observation=member(out/'current-objectives/001.json'),
                  derivative_transaction_receipt=member(out/'gpu-initial-checks.json'),
                  entry_identity=entry['entry_identity'],common_ready=member(root/'READY.json'),
                  source_head=head,scientific_endpoint_complete=False,scientific_promotion=False))
                progress(out,'ACTUAL_A0_INITIAL_VALID',ledger,receipt=member(out/'INITIAL_VALID.json'))
            return value,gradient,info
        with ledger.time('A0_joint_target_optimization'):
            result=plan_joint_targets(tuple(g.writer_factor.to('cuda') for g in geometries),
              tuple(g.image_metric.to('cuda') for g in geometries),tuple(w.shape[0] for w in view.entry),observed)
        tensor_save(out/'A0-plan.pt',dict(rhs=tuple(r.cpu() for r in result.rhs),
          deltas=tuple(d.cpu() for d in result.deltas),entry_identity=entry['entry_identity']))
        plan_receipt={k:v for k,v in vars(result).items() if k not in ('rhs','deltas','trajectory')}
        save(out/'A0-trajectory.json',dict(trajectory=result.trajectory,receipt=plan_receipt,
          entry_identity=entry['entry_identity'],common_ready_sha=sha(root/'READY.json')))
        phase='ENDPOINT';weights=view.weights_for_delta(result.deltas)
        if not all(torch.isfinite(w).all() for w in weights):raise FloatingPointError('A0_ENDPOINT_NONFINITE')
        tensor_save(out/'endpoint.pt',dict(weights=tuple(w.detach().cpu() for w in weights),names=names,
          entry_identity=entry['entry_identity'],arm='A0'))
        endpoint_identity=dict(weights=dict(zip(names,[tensor_sha(w) for w in weights])),arm='A0',source_head=head)
        observations={}
        for role,rr in rows.items():
            pp=panel(view,rr,saved[role],tok.pad_token_id,
                'current' if role=='Current' else ('past' if role.startswith('Past') else 'base'),2)
            observations[role]=observe_panel(pp,weights);del pp
        save(out/'functional-endpoint.json',observations)
        phase='TERMINAL_MATERIALIZATION'
        with materialized(model,names,weights,ledger,purpose='terminal'):
            ledger.add('authoritative_endpoint_materializations')
            actual=tuple(params[n].detach().clone() for n in names)
            if not all(torch.equal(a,b) for a,b in zip(actual,weights)):raise RuntimeError('ACTUAL_ENDPOINT_APPLY')
            delivered_end=capture_delivered(model,tok,native,hp,entry['contexts'],current,ledger)
            tensor_save(out/'delivered-responses.pt',dict(requested_rhs=tuple(r.cpu() for r in result.rhs),
              native_map_response=tuple(d.cpu()@g.keys.float() for d,g in zip(result.deltas,geometries)),
              module_coordinate_actual={l:delivered_end[l]['output']-delivered_entry[l]['output'] for l in (4,8)},
              endpoint_readout=delivered_end,block_z_not_subtracted=True))
            histories,history_receipt=finalize_history_once(model,tok,native,hp,entry['contexts'],current,
              {4:entry['M4'],8:entry['M8']},ledger,out/'history-finalization',
              endpoint_identity=endpoint_identity,source_identity=entry['source_identity'])
            tensor_save(out/'final-history.pt',histories)
            phase='EVALUATION'
            measure(model,evaltok,records,entry['panel'],ledger,out/'endpoint-metrics.json')
            generation(model,evaltok,records,entry['panel']['generation'],ledger,out/'generation.json')
        view.assert_live(bytes_check=True)
        phase='SIGNED_ATTRIBUTION'
        for label,keep in (('We',(False,False)),('We_plus_D4',(True,False)),('We_plus_D8',(False,True))):
            ww=tuple(w+(d if use else torch.zeros_like(d)) for w,d,use in zip(view.entry,result.deltas,keep))
            with materialized(model,names,ww,ledger,purpose='attribution_observation'):
                measure(model,evaltok,records,entry['panel'],ledger,out/f'attribution-{label}.json',current_only=True)
        view.assert_live(bytes_check=True)
        members=[member(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.pt','.json')]
        save(out/'terminal.json',dict(status='TERMINAL_VALID',arm='A0',entry='Middle',source_head=head,
          members=members,members_root=digest(members),endpoint_identity=endpoint_identity,
          history=history_receipt,compute=ledger.receipt(),peak_GPU_bytes=torch.cuda.max_memory_allocated(),
          actual_initial_gate_pass=True,full_campaign_completed=False,scientific_promotion=False))
        progress(out,'A0_TERMINAL_VALID',ledger,terminal=member(out/'terminal.json'))
    except BaseException as e:
        restored=None
        if view is not None:
            try:view.assert_live(bytes_check=True);restored=True
            except BaseException:restored=False
        save(out/'failure-boundary.json',dict(stage=phase,exception=type(e).__name__,message=str(e),
          traceback=traceback.format_exc(),restore=restored,compute=ledger.receipt(),
          source_head=head,scientific_promotion=False,original_exception_receipt=getattr(e,'receipt',None)))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--common',required=True);p.add_argument('--output',required=True)
    p.add_argument('--source-head',required=True);main(p.parse_args())
