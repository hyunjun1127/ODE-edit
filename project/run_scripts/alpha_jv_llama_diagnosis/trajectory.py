"""Configurable orchestration; native builds, raw JVP, NNLS and overlay reused.

This is not a second writer/solver. All model-dependent kernels are the pinned
native-v3.1 public primitives. D and S share this loop without monkeypatching.
"""
from dataclasses import asdict
import time
import torch
from project.run_scripts.native_response_ode_v31.algebra import nnls_response, identities
from project.run_scripts.native_response_ode_v31.provenance import save
from project.run_scripts.ordered_response_barrier_ode import runtime as old
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import GroupedFP32Overlay,tensor_sha256,tensor_set_sha256
from project.run_scripts.ordered_response_barrier_ode.terminal_jvp import TerminalResponseObserver
from .contracts import LAYERS, BindingBoundary, primary_residual
from .fixture import observe_prefix, observation_callback_guard
from .observations import controller_matrices, initial_history_gram, layer_actions, actual_delta_rows, same_state_shadows


def solve_current(target, terminal, responses, q, normalization, config):
    if normalization.normalization_id!=config.normalization_id:
        raise BindingBoundary('NORMALIZATION_LABEL_MISMATCH')
    e=normalization.weight(primary_residual(target,terminal))
    psi=torch.stack([normalization.weight(r)/q[i].sqrt() for i,r in enumerate(responses)],dim=1) if responses else torch.empty((e.numel(),0),dtype=torch.float64)
    metric=torch.eye(len(responses),dtype=torch.float64)
    solution=nnls_response(e,psi,metric,config.lambda_response)
    fact=identities(e,psi,metric,solution.coefficients,config.lambda_response)
    bound=1e-10*max(1.,fact['gain'],fact['response_sq'],fact['qN'])
    if abs(fact['dissipation_residual'])>bound or fact['speed_excess']>bound:
        raise BindingBoundary('NNLS_DISSIPATION_BOUNDARY')
    return e,psi,metric,solution,fact


def run_joint(family,dictionary,normalization,config,*,output,arm='JV_NATIVE',prefixes=None,
              raw_sink=None,endpoint_sink=None,observe_history_shadow=False):
    """Return only after exact entry restore. Endpoint evaluation happens inside finalize.

    raw_sink receives detached copies for local raw evidence only, with no return
    value consumed. endpoint_sink receives the actual materialized shadow, not W0.
    Neither hook supplies any parameter of the solve or subsequent path.
    """
    prefixes=dict(prefixes or {})
    if any(not isinstance(n,int) or not 0<n<config.N for n in prefixes):
        raise BindingBoundary('PREFIX_SCHEDULE')
    family.reset_entry();target=family.fixed_z.values
    if normalization.entry_sha!=family.w0_sha256 or dictionary.family is not family:
        raise BindingBoundary('CURRENT_FIXTURE_BINDING')
    reference=(dictionary.qref,dictionary.qfref)
    entry=family.terminal();v0=float(normalization.weight(primary_residual(target,entry)).square().sum()/2)
    names={l:f'{family.hparams.rewrite_module_tmp.format(l)}.weight' for l in LAYERS}
    overlay=GroupedFP32Overlay(family.model,names)
    observer=TerminalResponseObserver(model=family.model,overlay=overlay,capture_terminal_graph=family.terminal_graph)
    nodes=[];endpoints={};energy=0.;length=0.;previous=family.w0;started=time.perf_counter()
    build_before=dictionary.build_count;solve_before=dictionary.solve_count
    stage='ENTRY';shadow=None
    try:
        with overlay,torch.no_grad():
            for node in range(config.N):
                stage=f'NODE_{node}';old._sync();node_started=time.perf_counter()
                terminal=family.terminal();version=overlay.state_version
                native_before=dictionary.wall;jvp_before=observer.ledger.wall_seconds
                builds=dictionary.build(terminal,version)
                active,q,gf=dictionary.whiten(builds);layers=[b.layer for b in active]
                if (dictionary.qref,dictionary.qfref)!=reference:
                    raise BindingBoundary('ENTRY_QREF_REFRESHED')
                responses=[]
                for b in active:
                    response=observer.observe(b,expected_state_version=version)
                    scale=max(1.,float(terminal.abs().max()))
                    if not torch.allclose(response.terminal,terminal,rtol=64*torch.finfo(torch.float32).eps,
                                         atol=64*torch.finfo(torch.float32).eps*scale):
                        raise BindingBoundary('JVP_CURRENT_PRIMAL_BOUNDARY')
                    responses.append(response.response)
                controller_started=time.perf_counter()
                e,psi,metric,solution,fact=solve_current(target,terminal,responses,q,normalization,config)
                c=solution.coefficients
                controller_seconds=time.perf_counter()-controller_started
                shadow_start=time.perf_counter()
                gram=initial_history_gram(dictionary,active,q) if observe_history_shadow and node==0 else None
                shadows=same_state_shadows(e,psi,metric,gf,q,layers,c,config,gram)
                actions=layer_actions(dictionary,active,q,c)
                shadow_seconds=time.perf_counter()-shadow_start
                # All-row attribution is deliberately absent from write dispatch.
                if raw_sink is not None:
                    with observation_callback_guard(family,overlay):
                        raw_sink(node,dict(terminal=terminal.clone(),target=target.clone(),
                            raw_responses=torch.stack(responses) if responses else torch.empty((0,*terminal.shape),dtype=torch.float32),
                            q_layers=q.clone(),layers=list(layers),e=e.clone(),psi=psi.clone(),
                            increments_before=tuple(d.with_coefficient(d.coefficient) for d in overlay.deltas),
                            config=config.receipt()))
                token=overlay.seal_sweep_entry(node)
                for i,b in enumerate(active):
                    overlay.append(b.overlay_delta(float(config.h*c[i]/q[i].sqrt())),sweep_token=token)
                overlay.close_sweep(token)
                exit_terminal=family.terminal();exit_e=normalization.weight(primary_residual(target,exit_terminal))
                shadow=overlay.materialize_shadow(device='cpu')
                physical=actual_delta_rows(shadow,previous,family.w0,names);previous=shadow
                normalized_delta=normalization.weight(exit_terminal-terminal)
                raw_prediction=sum((responses[i].double()*float(c[i]/q[i].sqrt()) for i in range(len(c))),torch.zeros_like(terminal,dtype=torch.float64))
                v_after=float(exit_e.square().sum()/2);energy+=config.h*fact['qN'];length+=config.h*fact['qN']**.5
                old._sync()
                record=dict(node=node,arm=arm,state_version=version,exit_state_version=overlay.state_version,
                    **config.receipt(),t=(node+1)*config.h,V_before=fact['V'],V_after=v_after,V0=v0,
                    V_ratio=v_after/v0 if v0 else None,E=energy,native_path_length_normalized=length,
                    actual_barrier_increment=fact['V']-v_after-config.lambda_response*config.h*fact['qN'],
                    active_layers=layers,c=c.tolist(),raw_physical_coefficients=(c/q.sqrt()).tolist(),
                    actual_step_coefficients=(config.h*c/q.sqrt()).tolist(),q_layers=q.tolist(),qN_ref=dictionary.qref,
                    normalization=normalization.receipt(),source_entry_sha256=family.w0_sha256,
                    fixed_z_sha256=family.fixed_z.identity_sha256,terminal_sha256=tensor_sha256(exit_terminal),
                    model_error_normalized=float((normalized_delta-config.h*(psi@c)).norm()),
                    model_error_raw_activation=float((exit_terminal.double()-terminal.double()-config.h*raw_prediction).norm()),
                    finite_step_dissipation_defect=v_after-fact['V']+config.h*(fact['response_sq']+config.lambda_response*fact['qN'])-.5*config.h**2*fact['response_sq'],
                    layer_actions=actions,actual_physical=physical,shadows=shadows,
                    KKT_stationarity=solution.stationarity,KKT_complementarity=solution.complementarity,
                    build_ids=[b.build_identity for b in builds],solve_backward_errors=[b.solve_backward_error for b in builds],
                    inner_weight_mutation_count=0,inner_history_append_count=0,controller_heldout_access_count=0,
                    node_wall_seconds=time.perf_counter()-node_started,
                    compute_seconds=dict(native_dictionary=dictionary.wall-native_before,
                        main_jvp=observer.ledger.wall_seconds-jvp_before,primary_NNLS=controller_seconds,
                        shadow_metric_NNLS=shadow_seconds),**controller_matrices(e,psi,metric,c),**fact)
                save(output/f'node-{node:02}.json',record);nodes.append(record)
                if node+1 in prefixes:
                    label=prefixes[node+1]
                    def prefix_sink(ep,weights):
                        if endpoint_sink is not None:
                            endpoint_sink(label,dict(ep,**config.endpoint_clock(node+1),candidate_id=label),weights,exit_terminal.clone())
                    ep=observe_prefix(family,overlay,shadow,arm=label,on_endpoint=prefix_sink)
                    ep.update(config.endpoint_clock(node+1),candidate_id=label)
                    save(output/f'endpoint-{label}.json',ep);endpoints[label]=ep
            stage='TERMINAL'
            endpoint_activation=family.terminal()
            with overlay.suspend(authoritative=True):
                endpoint=family.finalize(arm=arm,terminal_state_version=overlay.state_version,
                    shadow_weights=shadow,deltas=overlay.deltas,derived_observation_only=False)
            overlay.assert_w0_unchanged(full_bytes=True)
            if family.last_terminal is None:raise BindingBoundary('MATERIALIZED_TERMINAL_MISSING')
            scale=max(1.,float(endpoint_activation.abs().max()))
            if not torch.allclose(family.last_terminal,endpoint_activation,rtol=64*torch.finfo(torch.float32).eps,
                atol=64*torch.finfo(torch.float32).eps*scale):raise BindingBoundary('VIRTUAL_MATERIALIZED_TERMINAL_PARITY')
            endpoint.update(config.endpoint_clock(config.N),candidate_id=arm)
            if endpoint_sink is not None:
                with observation_callback_guard(family,overlay,tensors=tuple(shadow.values())):
                    endpoint_sink(arm,dict(endpoint),shadow,endpoint_activation.clone())
            save(output/f'endpoint-{arm}.json',endpoint);endpoints[arm]=endpoint
            return dict(status='TERMINAL_VALID',arm=arm,endpoints=endpoints,nodes=nodes,
                jvp_ledger=asdict(observer.ledger),main_jvp_count=observer.ledger.jvp_call_count,
                dictionary_build_count=dictionary.build_count-build_before,solve_count=dictionary.solve_count-solve_before,
                materialization_discrepancy_normalized=float(normalization.weight(family.last_terminal-endpoint_activation).norm()),
                total_write_and_endpoint_seconds=time.perf_counter()-started,overlay=overlay.receipt(),
                **config.receipt(),**dictionary.net_action(overlay.deltas))
    except BaseException as exc:
        restore_error=None
        try:family.reset_entry()
        except BaseException as err:restore_error=repr(err)
        save(output/'failure-boundary.json',dict(stage=stage,error=repr(exc),completed_nodes=len(nodes),
            entry_restore=tensor_set_sha256(family.parameters)==family.w0_sha256,restore_error=restore_error,
            science_change_count=0,tolerance_change_count=0,retry_count=0,**config.receipt()))
        raise
    finally:family.reset_entry()
