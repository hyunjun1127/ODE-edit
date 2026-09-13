"""One entry's immutable shared assets; no independent SH2 fixture generation.

Full P* is built before loading the model. History and teacher preparation are
measured GPU work, not scientific endpoints. The READY seal is only published
after all members are complete and hashed. Failed namespaces remain immutable.
"""
import argparse
import dataclasses
import importlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import torch
from ..contracts import ABC,DATA,MODEL,P_RAW,ENTRIES,WEIGHTS,Science,member,hf_member,sha,digest,save,tensor_save,tensor_sha,Ledger
from ..observations import JointView,teacher
from ..evaluation import panel
from ..history import capture_native_keys,reconstruct_history
from ..banks import current_rows,protection_rows
from ..track_a.native_geometry import NativeWriterMetric
from project.run_scripts.l4_two_memory_conflict_routing.geometry import Projector

OLD_P4=Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-l4-two-memory-routing-v2/local/l4-two-memory-conflict-routing/20260911-v2/attempt-r1/Middle-B100/projector.pt')
OLD_P4_SHA='ced724ea45b27650b86226fe3795217b933cd47c891a55171a56b3fa908eedc2'
P_RAW_SHA='6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec'

def progress(out,phase,ledger,**extra):
    row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),stage=phase,
      compute=ledger.receipt(),**extra)
    save(out/'progress'/f'{time.time_ns()}.json',row)
    print(json.dumps(row,allow_nan=False),flush=True)

def projector_state(p):
    return dict(vectors=p.vectors.cpu(),complement=p.complement,dimension=p.dimension,
      spectrum=p.spectrum.cpu(),receipt=p.receipt)

def projector_from_state(s):
    p=Projector(s['vectors'],s['complement'],s['dimension'],s['spectrum']);p.receipt=s['receipt'];return p

def prepare(args):
    out=Path(args.output).absolute();out.mkdir(parents=True,exist_ok=False)
    ledger=Ledger();phase='INPUT';model=None;view=None
    execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if execution_head!=args.source_head:raise RuntimeError('EXECUTION_HEAD_MISMATCH')
    save(out/'run.lock.json',dict(args=vars(args),source_head=execution_head,
      source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),
      science=dataclasses.asdict(Science()),job=os.environ.get('SLURM_JOB_ID'),
      mode='COMMON_ASSET_PREPARATION_ONLY',GPU=1,mem_mib=182272,cap=2,
      model_forward='FULL_FP32_EAGER',model_revision=MODEL.name,history_forward_microbatch=8,
      teacher_microbatch=2,scientific_endpoint_actions=0,scientific_promotion=False))
    try:
        torch.set_num_threads(8);torch.manual_seed(20260911)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if os.environ.get('CUMRISK_ASSET_ROOT')!=str(ABC):raise RuntimeError('LEGACY_ASSET_BINDING')
        from scripts.fixed_counterfact import load_prefix
        from project.run_scripts.single_layer_cumulative_risk import binding
        records=load_prefix(DATA,10000)
        cfg=json.loads(Path(args.cpu_input).read_text())
        if cfg['entry']!=args.entry:raise RuntimeError('CPU_ENTRY_MISMATCH')
        for m in cfg['members']:
            if member(m['path'])!=m:raise RuntimeError('CPU_MEMBER_MISMATCH')
        if hf_member(MODEL/'config.json')!=cfg['model_config']:raise RuntimeError('MODEL_CONFIG_MISMATCH')
        if sha(P_RAW)!=P_RAW_SHA:raise RuntimeError('RAW_PROJECTOR_MISMATCH')
        save(out/'input.lock.json',dict(cpu_input=member(args.cpu_input),binding=cfg,
          raw_projector=member(P_RAW),source_native=[member(ABC/'imports/blue-source/AlphaEdit'/n)
            for n in ('AlphaEdit_main.py','compute_ks.py','compute_z.py')]))
        phase='PROJECTORS';projections={}
        if sha(OLD_P4)!=OLD_P4_SHA:raise RuntimeError('EXACT_P4_REFERENCE_MISMATCH')
        projections[4]=projector_from_state(torch.load(OLD_P4,map_location='cpu',weights_only=True))
        ledger.add('exact_P4_eigen_reuse')
        save(out/'P4-reuse.json',dict(member=member(OLD_P4),raw_projector_sha=P_RAW_SHA,
          original_index=0,full_space=projections[4].receipt))
        if args.p8_reference:
            reference=Path(args.p8_reference);sealed=json.loads(reference.with_suffix('.json').read_text())
            if sealed['raw_projector_sha']!=P_RAW_SHA or sealed['sha256']!=sha(reference):
                raise RuntimeError('EXACT_P8_REFERENCE_MISMATCH')
            projections[8]=projector_from_state(torch.load(reference,map_location='cpu',weights_only=True))
            ledger.add('exact_P8_eigen_reuse')
        else:
            with ledger.time('P8_full_eigendecomposition'):
                raw=torch.load(P_RAW,map_location='cpu',weights_only=True,mmap=True)[4].to('cuda')
                projections[8]=Projector.from_raw(raw)
                projections[8].to('cpu');projections[8].spectrum=projections[8].spectrum.cpu()
                del raw;torch.cuda.empty_cache()
        for layer,p in projections.items():
            path=out/f'P{layer}.pt';tensor_save(path,projector_state(p))
            save(path.with_suffix('.json'),dict(sha256=sha(path),raw_projector_sha=P_RAW_SHA,
              raw_index=layer-4,layer=layer,receipt=p.receipt))
        progress(out,'FULL_PROJECTORS_READY',ledger,geometry={str(l):p.receipt for l,p in projections.items()})
        phase='MODEL';model,tok,evaltok=binding.load_model(ledger)
        def forward_count(module,positional,kw):
            ids=kw.get('input_ids',positional[0] if positional else None)
            ledger.add('actual_model_forward_invocations')
            if ids is not None:ledger.add('actual_forward_padded_tokens',ids.numel())
            if kw.get('attention_mask') is not None:
                ledger.add('actual_forward_nonpadding_tokens',int(kw['attention_mask'].sum()))
        model.register_forward_pre_hook(forward_count,with_kwargs=True)
        params=dict(model.named_parameters());names=(WEIGHTS[4],WEIGHTS[8])
        _,offset,attempt=ENTRIES[args.entry]
        prepared=torch.load(ABC/'A'/args.entry/attempt/'prepared.pt',map_location='cpu',weights_only=True,mmap=True)
        cp,targets,current=binding.load_entry(args.entry,records)
        if not torch.equal(params[names[0]].detach().cpu(),prepared['W0']):raise RuntimeError('MODEL_W0_MISMATCH')
        if not torch.equal(cp['weights'][names[0]],prepared['We']):raise RuntimeError('CHECKPOINT_We_MISMATCH')
        if not torch.equal(cp['cache_c'],prepared['M']):raise RuntimeError('CHECKPOINT_M4_MISMATCH')
        if digest(cp['metadata']['contexts'])!=digest(prepared['contexts']):raise RuntimeError('CONTEXT_MISMATCH')
        if not cfg['native_reuse_eligible']:raise RuntimeError('NATIVE_REPREPARATION_REQUIRED_EFFECTIVE_INVENTORY')
        w0=tuple(params[n].detach().cpu().clone() for n in names)
        with torch.no_grad():params[names[0]].copy_(prepared['We'].to('cuda'))
        view=JointView(model,names,ledger)
        entry_identity=dict(model_revision=MODEL.name,weights=dict(zip(names,view.entry_sha)),
          M4=tensor_sha(cp['cache_c']),contexts=digest(prepared['contexts']),cpu_input_sha=sha(args.cpu_input))
        native=binding.kernel();hp=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams.from_json(ABC/'imports/config.json')
        if not hp.blue or hp.L2!=1 or hp.layers!=[4]:raise RuntimeError('NATIVE_HPARAM_MISMATCH')
        native_source={n:sha(ABC/'imports/blue-source/AlphaEdit'/n) for n in ('compute_ks.py','compute_z.py','AlphaEdit_main.py')}
        source_identity=dict(native=native_source,model_revision=MODEL.name)
        inv=cfg['inventory'];effective=[records[i] for i in inv['current_effective']]
        phase='ENTRY_KEYS';keys={}
        for layer in (4,8):
            keys[layer]=capture_native_keys(model,tok,native,hp,prepared['contexts'],effective,layer,ledger,input_width=14336)
            tensor_save(out/f'K{layer}.pt',keys[layer])
        parity=dict(entry=entry_identity,K4_stored_max_abs=float((keys[4]-prepared['K']).abs().max()),
          K4_stored_relative=float((keys[4]-prepared['K']).norm()/prepared['K'].norm()),
          request_order=[r['case_id'] for r in effective],native_endpoint_reused=True,
          physical_microbatch_difference='recorded; original compute_ks operator unchanged',
          WN4_sha=tensor_sha(prepared['WN']),M4_preserved=True)
        save(out/'native-response-parity.json',parity)
        progress(out,'ENTRY_NATIVE_KEYS_VALID',ledger,shapes={str(l):list(k.shape) for l,k in keys.items()},parity=parity)
        phase='HISTORY_L8'
        history_out=Path(args.history_resume).absolute() if args.history_resume else out/'history-L8'
        if args.history_resume and not (history_out/'complete.json').is_file():
            raise RuntimeError('INCOMPLETE_OLD_HISTORY_IMMUTABLE_USE_NEW_PREFIX_NAMESPACE')
        m8,history_receipt=reconstruct_history(model,tok,native,hp,prepared['contexts'],records,offset,ledger,history_out,
          entry_identity=entry_identity,source_identity=source_identity,input_width=14336,
          verify_entry=lambda:view.assert_live(),progress=lambda row:progress(out,'HISTORY_PROGRESS',ledger,history=row))
        histories={4:cp['cache_c'][0].clone(),8:m8}
        tensor_save(out/'entry.pt',dict(schema='common-entry-v1',names=names,W0=w0,
          We=tuple(w.cpu() for w in view.entry),WN=(prepared['WN'].clone(),w0[1].clone()),
          M4=histories[4],M8=histories[8],N4_M_after=prepared['MN'].clone(),
          contexts=prepared['contexts'],native_targets=targets,entry_identity=entry_identity,
          raw_effective_inventory=inv,panel=cfg['panel'],source_identity=source_identity))
        phase='NATIVE_GEOMETRY'
        for layer in (4,8):
            with ledger.time(f'full_native_geometry_L{layer}'):
                geo=NativeWriterMetric(keys[layer],histories[layer],projections[layer],storage_device='cpu')
            tensor_save(out/f'native-geometry-L{layer}.pt',geo.to_state())
            save(out/f'native-geometry-L{layer}.json',geo.receipt)
            progress(out,'NATIVE_GEOMETRY_READY',ledger,layer=layer,receipt=geo.receipt)
            del geo
        phase='PACKING_TEACHER'
        packed={'Current':current_rows(records,inv['current_effective'],prepared['contexts'],tok)}
        for role,indices in inv['bank'].items():packed[role]=protection_rows(records,indices,role,tok)
        save(out/'bank-manifest.json',dict(inventory=inv,entry=entry_identity,
          packing_identity={k:digest(v) for k,v in packed.items()},context_counts={k:len(v) for k,v in packed.items()}))
        tensor_save(out/'prediction-rows.pt',packed)
        teachers={}
        for role,rows in packed.items():
            teachers[role]=teacher(view,view.entry,rows,tok.pad_token_id,2)
            tensor_save(out/f'We-teacher-{role}.pt',teachers[role])
            progress(out,'TEACHER_READY',ledger,role=role,contexts=len(rows))
        phase='NATIVE_RISKS';wn=(prepared['WN'].to('cuda'),view.entry[1])
        risks={}
        for role in ('Base','Past'):
            pp=panel(view,packed[role],teachers[role],tok.pad_token_id,role.lower(),2)
            risks[role]=pp.value(wn);del pp
        if not all(torch.isfinite(torch.tensor(v)) for v in risks.values()):raise FloatingPointError('NATIVE_RISK_NONFINITE')
        calibration=dict(raw_native_risk=risks,sigma={k:max(v,1e-3) for k,v in risks.items()},
          past_tau=.1,reference='same We native WN; not OS',Base_teacher='We',W0_controller_access=0)
        save(out/'calibration.json',calibration)
        view.assert_live(bytes_check=True)
        phase='SEAL';members=[member(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.pt')]
        save(out/'READY.json',dict(status='COMMON_GPU_PREPARED_READY_FOR_SH2_VERIFY',entry=args.entry,
          source_head=execution_head,members=members,members_root=digest(members),
          external_history=None if not args.history_resume else dict(path=str(history_out),receipt=history_receipt),
          entry_identity=entry_identity,calibration=calibration,compute=ledger.receipt(),
          actual_model_derivative_gate='PENDING_A0_AND_SHARED_KERNEL_RUNTIME',
          peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),
          peak_reserved_GPU_bytes=torch.cuda.max_memory_reserved(),
          cross_server_verified=False,scientific_endpoint_actions=0,scientific_promotion=False))
        progress(out,'COMMON_ASSETS_READY',ledger,ready=member(out/'READY.json'))
    except BaseException as exc:
        restore=None
        if view is not None:
            try:view.assert_live(bytes_check=True);restore=True
            except BaseException:restore=False
        save(out/'failure-boundary.json',dict(stage=phase,exception_type=type(exc).__name__,message=str(exc),
          traceback=traceback.format_exc(),entry_restore=restore,compute=ledger.receipt(),
          source_head=execution_head,scientific_endpoint_actions=0,scientific_promotion=False))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',choices=ENTRIES,required=True)
    p.add_argument('--cpu-input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--source-head',required=True);p.add_argument('--p8-reference');p.add_argument('--history-resume')
    prepare(p.parse_args())
