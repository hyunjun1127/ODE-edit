"""One independent A-OS endpoint from an immutable completed A0.

The agent pauses at the first full finite quadratic action; this process keeps
the sealed two-RHS/max20 solve, endpoint evaluation and terminal finalization.
No A0 reoptimization and no BF/follow-up cascade occur in this process.
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
from ..contracts import (ABC, MODEL, Science, Ledger, member, sha, digest, save,
                         tensor_save, tensor_sha)
from ..common_reference.session import open_session, verify_publication
from ..common_reference.prepare import progress
from ..evaluation import panel, materialized, measure, generation, observe_panel
from ..observations import teacher, pack
from ..history import finalize_history_once
from ..track_b.native_adapter import operators
from .protocol import AProblem, run_os


def storage_probe(view, rows, tokenizer, host_weights, host_direction):
    """Same tensors, same full model: host AD transfer vs GPU-resident AD.

    Validation only, no FD/refinement/new scientific tolerance. The dtype
    roundoff bound 64 eps32 is fixed independent of model outcomes. Prior A0
    FD/JVP and virtual/materialized gates remain independently SHA-bound.
    """
    first = rows[0]['ordinal']
    rr = [dict(r) for r in rows if r['ordinal'] == first][:2]
    b = next(pack(view, rr, tokenizer.pad_token_id, 2))
    gpu_weights = tuple(w.to(view.entry[0].device) for w in host_weights)
    gpu_direction = tuple(d.to(view.entry[0].device) for d in host_direction)
    fn = lambda *w: b.logits(w)
    ph, jh = torch.func.jvp(fn, host_weights, host_direction)
    pg, jg = torch.func.jvp(fn, gpu_weights, gpu_direction)
    receipts = {}
    for name, x, y in [('primal', ph, pg), ('jvp', jh, jg)]:
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise FloatingPointError('HOST_DEVICE_NONFINITE')
        error = float(torch.linalg.vector_norm((x-y).double()))
        scale = float(torch.linalg.vector_norm(x.double())+torch.linalg.vector_norm(y.double()))
        bound = 64*torch.finfo(torch.float32).eps*scale
        receipts[name] = dict(l2_error=error, roundoff_bound=bound,
                              byte_equal=torch.equal(x,y), passed=error<=bound)
        if error>bound: raise RuntimeError('HOST_DEVICE_AD_STORAGE_MISMATCH')
    view.assert_live(bytes_check=True)
    return dict(status='PASS', **receipts, physical_contexts=len(rr),
                direction='saved actual A0 batch delta', full_sequence=True,
                production_fd=0, teacher_or_sample_selection=0,
                live_pointer_version_bytes_unchanged=True,
                rule='64 eps_FP32 * (norm(host output)+norm(device output))')


def main(args):
    out = Path(args.output).absolute(); out.mkdir(parents=True,exist_ok=False)
    ledger=Ledger(); session=None; phase='SOURCE_INPUT';stage_counter=0
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    try:
        if head != args.source_head: raise RuntimeError('EXECUTION_SOURCE_MISMATCH')
        save(out/'run.lock.json', dict(source_head=head, args=vars(args),
            job=os.environ.get('SLURM_JOB_ID'), science=dataclasses.asdict(Science()),
            arm='A-OS', h=1., inner_steps=1, logical_current=100,
            physical_microbatch=2, PCG_vector_storage='CPU_FP32',
            native_factors_contractions='CPU_FP64', model='FULL_FP32',
            initialization='stored A0 endpoint; no optimizer/native recomputation',
            initial_gate='first full projected quadratic action finite at actual A0 anchor',
            total_campaign_complete=False,scientific_promotion=False))
        a0=Path(args.a0).absolute()
        terminal=verify_publication(a0/'terminal.json',status='TERMINAL_VALID',
                                   expected_sha=args.a0_terminal_sha)
        if (terminal['arm'],terminal['entry'])!=('A0','Middle'):
            raise RuntimeError('A0_PARENT_ARM_ENTRY')
        session=open_session(args.common,out,ledger,common_sha=args.common_sha)
        view=session.view;entry=session.entry;rows=session.rows;tok=session.tokenizer
        endpoint=torch.load(a0/'endpoint.pt',map_location='cpu',weights_only=True)
        if endpoint['entry_identity']!=entry['entry_identity'] or tuple(endpoint['names'])!=view.names:
            raise RuntimeError('A0_PARENT_ENTRY_IDENTITY')
        if json.loads((a0/'common-binding.json').read_text())['ready']['sha256']!=args.common_sha:
            raise RuntimeError('A0_PARENT_COMMON_IDENTITY')
        plan=json.loads((a0/'A0-trajectory.json').read_text())['receipt']
        we=tuple(w.detach().cpu() for w in view.entry)
        wa=tuple(w.detach() for w in endpoint['weights'])
        d0=tuple((w.double()-e.double()).float() for w,e in zip(wa,we))
        if dict(zip(view.names,map(tensor_sha,wa)))!=terminal['endpoint_identity']['weights']:
            raise RuntimeError('A0_ENDPOINT_TENSOR_IDENTITY')
        save(out/'parent-binding.json',dict(terminal=member(a0/'terminal.json'),
            a0_execution_source=terminal['source_head'], a0_endpoint=member(a0/'endpoint.pt'),
            a0_plan=member(a0/'A0-trajectory.json'),
            initial_derivative_transaction=member(a0/'gpu-initial-checks.json'),
            actual_anchor_weights=[tensor_sha(w) for w in wa],
            common_ready_sha=args.common_sha, reoptimization_count=0,
            delta_definition='FP32 storage of FP64 stored WA minus We',
            delta_roundtrip_exact=all(torch.equal(e+d,w) for e,d,w in zip(we,d0,wa))))
        phase='HOST_AD_STORAGE'
        with ledger.time('host_device_storage_identity'):
            probe=storage_probe(view, rows['Current'], tok, wa, d0)
        save(out/'host-device-identity.json',probe)
        phase='CURRENT_REFERENCE_CAPTURE'
        saved_current=teacher(view,wa,rows['Current'],tok.pad_token_id,2)
        tensor_save(out/'WA-Current-teacher.pt',saved_current)
        saved=dict(session.saved_teachers);saved['Current']=saved_current
        panels={role:panel(view,rows[role],saved[role],tok.pad_token_id,
            {'Base':'base','Past':'past','Current':'current'}[role],2)
            for role in ('Base','Past','Current')}
        calibration=json.loads((session.root/'calibration.json').read_text())
        ops=operators(session.geometries)
        def raw_native_metric(values):
            return tuple(g.s_action(v,normalized=False,projected=True).to(v)
                         for g,v in zip(session.geometries,values))
        def validate_state(weights):
            view.assert_live()
            ok=len(weights)==2 and all(w.shape==e.shape and w.dtype==torch.float32
                and bool(torch.isfinite(w).all()) for w,e in zip(weights,we))
            if not ok: raise RuntimeError('AOS_SELECTED_STATE')
            return dict(selected_fp32_shape_valid=True, nonselected_unchanged=True,
                        history_append_count=0,compute_z_count=0,
                        selected_state_storage='CPU; differentiable full-sequence transfer')
        def on_stage(stage,receipt):
            nonlocal stage_counter
            stage_counter+=1
            save(out/'stages'/f'{stage_counter:03d}-{stage}.json',
                 dict(stage=stage,receipt=receipt,compute=ledger.receipt()))
            if stage=='FIRST_FULL_OPERATOR_MATVEC_COMPLETE':
                save(out/'INITIAL_VALID.json',dict(status='ACTUAL_AOS_OPERATOR_INITIAL_VALID',
                    source_head=head,common_ready_sha=args.common_sha,
                    parent_terminal_sha=args.a0_terminal_sha,
                    reused_A0_derivative_transaction=member(a0/'gpu-initial-checks.json'),
                    host_storage_identity=member(out/'host-device-identity.json'),
                    exact_WA_teacher=member(out/'WA-Current-teacher.pt'),
                    initial_operator_receipt=receipt, full_logical_current=100,
                    full_protection_banks={'Base':128,'Past':128},
                    current_equality_count=1, full_cross_GGN=True,
                    pcg_complete=False,endpoint_complete=False,
                    scientific_promotion=False,agent_policy='pause; sealed process continues'))
            progress(out,stage,ledger,stage_receipt=receipt)
        problem=AProblem(we=we,d0=d0,wa=wa,support=(4,8),
            base=panels['Base'],past=panels['Past'],current=panels['Current'],
            raw_native_metric=raw_native_metric, **ops,
            native_risks=tuple(calibration['raw_native_risk'][r] for r in ('Base','Past')),
            q_balanced=tuple(plan['q_balanced']),sigma_e=plan['sigma_e'],
            validate_state=validate_state,fixture_identity=dict(common=args.common_sha,
                entry=entry['entry_identity'],parent=args.a0_terminal_sha))
        phase='AOS_QUADRATIC'
        with ledger.time('AOS_protection_solve'):
            weights,node=run_os(problem,on_stage=on_stage)
        save(out/'trajectory.json',node)
        tensor_save(out/'endpoint.pt',dict(weights=tuple(w.cpu() for w in weights),
            names=view.names,entry_identity=entry['entry_identity'],arm='A-OS'))
        endpoint_identity=dict(weights=dict(zip(view.names,map(tensor_sha,weights))),
            arm='A-OS',source_head=head)
        phase='ENDPOINT_OBSERVATIONS'
        observations={}
        for role, rr in rows.items():
            pp=panels.get(role)
            if pp is None:
                pp=panel(view,rr,saved[role],tok.pad_token_id,
                         'past' if role.startswith('Past') else 'base',2)
            observations[role]=observe_panel(pp,weights)
        save(out/'functional-endpoint.json',dict(observations=observations,
            Current_reference='fixed WA',Base_Past_reference='fixed We'))
        from project.run_scripts.single_layer_cumulative_risk import binding
        native=binding.kernel()
        hp=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams.from_json(ABC/'imports/config.json')
        current=[session.records[i] for i in entry['raw_effective_inventory']['current_effective']]
        phase='TERMINAL_MATERIALIZATION'
        batch=next(pack(view,rows['Current'][:2],tok.pad_token_id,2))
        with torch.no_grad(): virtual_logits=batch.logits(weights).detach()
        with materialized(session.model,view.names,weights,ledger,purpose='terminal'):
            ledger.add('authoritative_endpoint_materializations')
            if tuple(tensor_sha(view.parameters[n]) for n in view.names)!=tuple(map(tensor_sha,weights)):
                raise RuntimeError('AOS_TERMINAL_APPLY')
            # Direct actual model forward, not a functional call over stale live state.
            with torch.no_grad():
                actual_logits=session.model(input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,use_cache=False).logits[batch.positions]
            error=float(torch.linalg.vector_norm((actual_logits-virtual_logits).double()))
            bound=64*torch.finfo(torch.float32).eps*float(
                torch.linalg.vector_norm(actual_logits.double())+torch.linalg.vector_norm(virtual_logits.double()))
            if not torch.isfinite(actual_logits).all() or error>bound:
                raise RuntimeError('AOS_TERMINAL_FORWARD_PARITY')
            save(out/'terminal-forward-parity.json',dict(passed=True,
                byte_equal=torch.equal(actual_logits,virtual_logits),l2_error=error,
                dtype_roundoff_bound=bound,observed_contexts=2))
            del actual_logits,virtual_logits,batch
            histories,history=finalize_history_once(session.model,tok,native,hp,entry['contexts'],current,
                {4:entry['M4'],8:entry['M8']},ledger,out/'history-finalization',
                endpoint_identity=endpoint_identity,source_identity=entry['source_identity'])
            tensor_save(out/'final-history.pt',histories)
            phase='EVALUATION'
            measure(session.model,session.evaluator_tokenizer,session.records,entry['panel'],
                    ledger,out/'endpoint-metrics.json')
            generation(session.model,session.evaluator_tokenizer,session.records,entry['panel']['generation'],
                       ledger,out/'generation.json')
        view.assert_live(bytes_check=True)
        phase='ATTRIBUTION'
        deltas=tuple(w-e for w,e in zip(weights,we))
        save(out/'attribution-We-reuse.json',dict(member=member(a0/'attribution-We.json'),
            entry_identity=entry['entry_identity'],common=args.common_sha,
            reason='exact same We/panel; no duplicate evaluation'))
        for label, keep in [('We_plus_D4',(True,False)),('We_plus_D8',(False,True))]:
            ww=tuple(e+(d if yes else torch.zeros_like(d)) for e,d,yes in zip(we,deltas,keep))
            with materialized(session.model,view.names,ww,ledger,purpose='attribution_observation'):
                measure(session.model,session.evaluator_tokenizer,session.records,entry['panel'],
                        ledger,out/f'attribution-{label}.json',current_only=True)
        view.assert_live(bytes_check=True)
        members=[member(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.pt','.json')]
        save(out/'terminal.json',dict(status='TERMINAL_VALID',numerical_status=node['solver']['status'],
            arm='A-OS',entry='Middle',source_head=head,members=members,members_root=digest(members),
            endpoint_identity=endpoint_identity,history=history,compute=ledger.receipt(),
            peak_GPU_bytes=torch.cuda.max_memory_allocated(),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
            full_campaign_completed=False,scientific_promotion=False))
    except BaseException as error:
        restored=None
        if session is not None:
            try: session.view.assert_live(bytes_check=True);restored=True
            except BaseException: restored=False
        save(out/'failure-boundary.json',dict(stage=phase,exception=type(error).__name__,
            message=str(error),original_exception_receipt=getattr(error,'receipt',None),
            traceback=traceback.format_exc(),restore=restored,compute=ledger.receipt(),
            source_head=head,scientific_promotion=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('common','common-sha','a0','a0-terminal-sha','output','source-head'):
        p.add_argument('--'+name,required=True)
    main(p.parse_args())
