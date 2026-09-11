"""Single sealed B endpoint process; agent monitoring ends at initial validity.

All shared preparation is consumed, never regenerated here. The program keeps
running after INITIAL_VALID.json; that marker is for the agent pause policy.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import torch

from ..contracts import MODEL,DATA,WEIGHTS,Science,Ledger,sha,digest,save,tensor_save,tensor_sha
from ..observations import JointView,pack,teacher
from ..evaluation import panel,materialized,measure
from ..history import finalize_history_once
from ..track_a.native_geometry import NativeWriterMetric
from .protocol import BProblem,run,ARMS
from .native_adapter import operators
from .receive_common import LANDING


class SelectedView:
    """B support8 (or named auxiliary4) inside immutable full WN4/WN8 state."""
    def __init__(self,full,wn,support):
        self.full=full;self.wn=tuple(w.detach().to(full.entry[0].device) for w in wn)
        self.indices=tuple((4,8).index(layer) for layer in support)
        self.entry=tuple(self.wn[i] for i in self.indices);self.ledger=full.ledger
    def full_weights(self,weights):
        if len(weights)!=len(self.indices):raise ValueError('SELECTED_VIEW_SUPPORT')
        out=list(self.wn)
        for i,w in zip(self.indices,weights):out[i]=w
        return tuple(out)
    def logits(self,weights,*args):return self.full.logits(self.full_weights(weights),*args)


def verify_bundle(path):
    path=Path(path).absolute();path.relative_to(LANDING)
    receiver=json.loads((path/'receiver-ready.json').read_text())
    if receiver['status']!='COMMON_DESTINATION_FULL_SHA_VERIFIED':raise ValueError('COMMON_RECEIVER_NOT_READY')
    for member in receiver['members']:
        p=path/member['destination_relative']
        if p.is_symlink() or p.stat().st_size!=member['bytes'] or sha(p)!=member['sha256']:
            raise ValueError('COMMON_MEMBER_CHANGED:'+str(p))
    source_ready=path/'payload/READY.json'
    if sha(source_ready)!=receiver['source_ready_sha256']:raise ValueError('COMMON_READY_CHANGED')
    return path/'payload',receiver


def runtime(args):
    out=Path(args.output).absolute();out.mkdir(parents=True,exist_ok=False)
    ledger=Ledger();phase='INPUT';view=None
    started=time.monotonic();head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if head!=args.source_head:raise RuntimeError('EXECUTION_SOURCE_HEAD_MISMATCH')
    try:
        execution=json.loads(Path(args.execution_lock).read_text())
        if execution['source_head']!=head:raise RuntimeError('EXECUTION_LOCK_SOURCE')
        for m in execution['source_members']:
            if sha(m['path'])!=m['sha256']:raise RuntimeError('EXECUTION_MEMBER_CHANGED:'+m['path'])
        if sha(execution['input_lock']['path'])!=execution['input_lock']['sha256']:raise RuntimeError('INPUT_LOCK_CHANGED')
        inputs=json.loads(Path(execution['input_lock']['path']).read_text())
        if inputs['common']!=str(Path(args.common).absolute()):raise RuntimeError('INPUT_LOCK_COMMON')
        for m in inputs['assets']+inputs['native_source']:
            if Path(m['path']).stat().st_size!=m['bytes'] or sha(m['path'])!=m['sha256']:
                raise RuntimeError('INPUT_ASSET_CHANGED:'+m['path'])
        import transformers
        if transformers.__version__!='4.44.2':raise RuntimeError('PINNED_TRANSFORMERS_4_44_2_REQUIRED')
        source,receiver=verify_bundle(args.common)
        if receiver['entry']!=args.entry:raise RuntimeError('COMMON_ENTRY_MISMATCH')
        from scripts.fixed_counterfact import load_prefix
        from project.run_scripts.single_layer_cumulative_risk import binding
        records=load_prefix(DATA,10000)
        torch.set_num_threads(8);torch.manual_seed(20260911)
        entry=torch.load(source/'entry.pt',weights_only=True,map_location='cpu',mmap=True)
        if entry['schema']!='common-entry-v1' or tuple(entry['names'])!=(WEIGHTS[4],WEIGHTS[8]):raise RuntimeError('COMMON_ENTRY_SCHEMA')
        for key in ('We','W0','WN'):
            if any(w.dtype!=torch.float32 or tuple(w.shape)!=(4096,14336) or not torch.isfinite(w).all() for w in entry[key]):raise RuntimeError('COMMON_WEIGHT_SCHEMA')
        if not torch.equal(entry['WN'][1],entry['We'][1]):raise RuntimeError('N4_WN8_MISMATCH')
        for name in ('M4','M8'):
            m=entry[name]
            if m.dtype!=torch.float32 or tuple(m.shape)!=(14336,14336) or not torch.isfinite(m).all():
                raise RuntimeError('COMMON_HISTORY_SCHEMA:'+name)
        packed=torch.load(source/'prediction-rows.pt',weights_only=True,map_location='cpu')
        calibration=json.loads((source/'calibration.json').read_text())
        support,steps,frozen=ARMS[args.arm]
        geo=[NativeWriterMetric.from_state(torch.load(source/f'native-geometry-L{l}.pt',weights_only=True,map_location='cpu',mmap=True)) for l in support]
        geometry_ops=operators(geo)
        lock=dict(source_head=head,source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),
            entry=args.entry,arm=args.arm,execution_lock_sha=sha(args.execution_lock),common_receiver_sha=sha(Path(args.common)/'receiver-ready.json'),
            common_source_head=receiver['source_head'],common_ready_sha=receiver['source_ready_sha256'],
            torch=torch.__version__,transformers=transformers.__version__,python=platform.python_version(),
            model_revision=MODEL.name,model_dtype='float32',TF32=False,physical_microbatch=args.physical_microbatch,
            project_cap=2,mem_mib=60416,GPU=1,gpu_hour_cap=None,job=os.environ.get('SLURM_JOB_ID'),
            monitoring_policy='ODEEDIT-INITIAL-GATE-ONLY-USER-RECALL-20260911',
            after_initial_valid_agent='MONITORING_PAUSED_AWAITING_USER',program_continues=True,
            native_geometry=[g.receipt for g in geo],native_geometry_reuse=[g.reuse_receipt for g in geo],
            independent_bank_generation=0,WN_regeneration=0,M8_regeneration=0,scientific_promotion=False)
        save(out/'run.lock.json',lock)
        phase='MODEL';model,tok,evaltok=binding.load_model(ledger)
        names=tuple(entry['names']);params=dict(model.named_parameters())
        for name,weight in zip(names,entry['W0']):
            if not torch.equal(params[name].detach().cpu(),weight):raise RuntimeError('W0_MODEL_EXACT_BYTES_MISMATCH')
        with materialized(model,names,entry['We'],ledger,purpose='B_fixture_We'):
            full=JointView(model,names,ledger);view=SelectedView(full,entry['WN'],support)
            fixed_wn_sha={name:tensor_sha(w) for name,w in zip(names,view.wn)}
            phase='CURRENT_REFERENCE'
            wn_teacher=teacher(view,view.entry,packed['Current'],tok.pad_token_id,args.physical_microbatch)
            tensor_save(out/'WN-current-teacher.pt',wn_teacher)
            teacher_map={role:torch.load(source/f'We-teacher-{role}.pt',weights_only=True,map_location='cpu') for role in ('Base','Past')}
            current=panel(view,packed['Current'],wn_teacher,tok.pad_token_id,'current',args.physical_microbatch)
            base=panel(view,packed['Base'],teacher_map['Base'],tok.pad_token_id,'base',args.physical_microbatch)
            past=panel(view,packed['Past'],teacher_map['Past'],tok.pad_token_id,'past',args.physical_microbatch)
            weights=tuple(w.detach().cpu().clone() for w in view.entry)
            # Reference is fixed WN; same forward creates scalar+context receipt.
            reference=current.observe(weights)
            ref_nll=reference['mean_nll']
            if abs(reference['value'])>2e-6:raise RuntimeError('WN_CURRENT_REFERENCE_SELF_PARITY')
            def validate_state(ws):
                full.assert_live()
                if any(w.shape!=r.shape or w.dtype!=torch.float32 or not torch.isfinite(w).all() for w,r in zip(ws,weights)):
                    raise RuntimeError('B_STATE_SCHEMA')
                for i,name in enumerate(names):
                    if i not in view.indices and tensor_sha(view.wn[i])!=fixed_wn_sha[name]:raise RuntimeError('B_FIXED_WN4_CHANGED')
                return dict(selected_fp32_shape_valid=True,nonselected_unchanged=True,
                            applied_fixed_WN=fixed_wn_sha,live_We_guard='pointer/version',compute_z_count=0,history_append_count=0)
            problem=BProblem(weights,support,base,past,current,geometry_ops['native_metric'],geometry_ops['native_metric_inverse'],
                geometry_ops['project'],tuple(calibration['raw_native_risk'][r] for r in ('Base','Past')),
                ref_nll,validate_state,dict(entry=args.entry,common_ready_sha=receiver['source_ready_sha256'],WN=fixed_wn_sha))
            phase='ACTUAL_GATE'
            # Gate implementation is shared separately; never label preparation
            # or teacher capture alone as initial-valid.
            from .runtime_gate import initial_gate
            evidence=initial_gate(problem,view,packed,wn_teacher,tok,model,ledger,out)
            save(out/'INITIAL_VALID.json',dict(status='MINIMUM_ACTUAL_INITIAL_VALID',evidence=evidence,
                source_head=head,common_ready_sha=receiver['source_ready_sha256'],job=os.environ.get('SLURM_JOB_ID'),
                full_B100_endpoint_complete=False,agent_after_this='MONITORING_PAUSED_AWAITING_USER',
                program_continues=True,compute=ledger.receipt(),scientific_promotion=False))
            print('MINIMUM_ACTUAL_INITIAL_VALID',flush=True)
            phase='B_CORRECTION'
            def on_node(row,ws,actual_delta):
                save(out/'nodes'/f'node-{row["node"]:02d}.json',row)
                tensor_save(out/'nodes'/f'node-{row["node"]:02d}.pt',dict(weights=ws,actual_delta=actual_delta,support=support,WN=fixed_wn_sha))
                print(json.dumps(dict(stage='B_NODE_COMPLETE',node=row['node'],arm=args.arm,solver=row['solver']['status'],wall=row['wall_seconds'])),flush=True)
            endpoint,rows=run(problem,args.arm,on_node=on_node)
            full.assert_live(bytes_check=True)
            phase='TERMINAL_MATERIALIZATION'
            full_endpoint=view.full_weights(tuple(w.to(full.entry[0].device) for w in endpoint))
            endpoint_sha={name:tensor_sha(w) for name,w in zip(names,full_endpoint)}
            if support==(8,) and endpoint_sha[names[0]]!=fixed_wn_sha[names[0]]:raise RuntimeError('B_L4_BYTE_MISMATCH')
            # Avoid retaining teacher tensors during endpoint evaluation.
            del base,past,current,problem,teacher_map,wn_teacher
            with materialized(model,names,full_endpoint,ledger,purpose='B_terminal'):
                if any(tensor_sha(params[n])!=endpoint_sha[n] for n in names):raise RuntimeError('ENDPOINT_PHYSICAL_BYTES')
                evalrows=measure(model,evaltok,records,entry['panel'],ledger,out/'endpoint-full.json')
                native=binding.kernel();hp=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams.from_json(binding.ROOT/'imports/config.json')
                effective=[records[i] for i in entry['raw_effective_inventory']['current_effective']]
                final_hist,history_receipt=finalize_history_once(model,tok,native,hp,entry['contexts'],effective,
                    {4:entry['M4'],8:entry['M8']},ledger,out/'history-finalization',endpoint_identity=endpoint_sha,
                    source_identity=dict(common_ready_sha=receiver['source_ready_sha256'],source_head=head))
                tensor_save(out/'endpoint.pt',dict(schema='B-endpoint-v1',names=names,weights=tuple(w.cpu() for w in full_endpoint),
                    histories=final_hist,contexts=entry['contexts'],WN=fixed_wn_sha,entry_identity=entry['entry_identity'],
                    effective_current=entry['raw_effective_inventory']['current_effective'],history_receipt=history_receipt))
            full.assert_live(bytes_check=True)
        phase='TERMINAL'
        save(out/'terminal.json',dict(status='B_ENDPOINT_FINITE_RECORDED',arm=args.arm,entry=args.entry,source_head=head,
            endpoint_sha=endpoint_sha,solver_statuses=[row['solver']['status'] for row in rows],full_pairs=len(evalrows),
            W0_selected_restored=all(torch.equal(params[n].detach().cpu(),w) for n,w in zip(names,entry['W0'])),
            compute=ledger.receipt(),total_wall_seconds=time.monotonic()-started,
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),
            pending_observation_scope=['additional independent audit/attribution aggregation on USER recall'],
            full_experiment_complete=False,scientific_promotion=False))
    except BaseException as error:
        save(out/'failure.json',dict(stage=phase,type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            source_head=head,compute=ledger.receipt(),scientific_promotion=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--common',required=True);p.add_argument('--entry',choices=('Early','Middle','Late'),required=True)
    p.add_argument('--arm',choices=tuple(ARMS),required=True);p.add_argument('--source-head',required=True)
    p.add_argument('--execution-lock',required=True)
    p.add_argument('--output',required=True);p.add_argument('--physical-microbatch',type=int,choices=(1,2,4),default=2)
    runtime(p.parse_args())
