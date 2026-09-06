"""Sequential JV/L8 Euler adapter reusing pinned dictionaries, NNLS and overlay."""
from dataclasses import asdict
import time
import torch
from project.run_scripts.native_response_ode_v31.algebra import nnls_response, identities
from project.run_scripts.native_response_ode_v31.provenance import save
from project.run_scripts.ordered_response_barrier_ode import runtime as old
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import GroupedFP32Overlay,tensor_sha256
from project.run_scripts.ordered_response_barrier_ode.terminal_jvp import TerminalResponseObserver
from .contracts import H,N,LAMBDA,LAYERS,CHECKPOINTS
from .telemetry import restricted_l8,controller_matrices,initial_history_gram,same_state_shadows,layer_actions,actual_delta_rows


def run_joint(family,arm,dictionary,normalization,*,batch_index,output):
    if arm not in ('JV_NATIVE','L8_ONLY_NATIVE'):raise ValueError('EXPLICIT_ARM_DISPATCH')
    family.reset_entry();target=family.fixed_z.values;entry=family.terminal()
    v0=float(normalization.weight(target-entry).square().sum()/2)
    names={l:f'{family.hparams.rewrite_module_tmp.format(l)}.weight' for l in LAYERS}
    overlay=GroupedFP32Overlay(family.model,names)
    observer=TerminalResponseObserver(model=family.model,overlay=overlay,capture_terminal_graph=family.terminal_graph)
    nodes=[];energy=0.;previous=family.w0;started=time.perf_counter()
    try:
        with overlay,torch.no_grad():
            for node in range(N):
                old._sync();node_started=time.perf_counter()
                terminal=family.terminal();e=normalization.weight(target-terminal)
                version=overlay.state_version
                build_time_before=dictionary.wall;jvp_time_before=observer.ledger.wall_seconds
                builds=dictionary.build(terminal,version)
                # qref was captured from all five ENTRY directions even for L8.
                if arm=='L8_ONLY_NATIVE':builds=[b for b in builds if b.layer==8]
                active,q,gf=dictionary.whiten(builds);layers=[b.layer for b in active]
                responses=[]
                for b in active:
                    response=observer.observe(b,expected_state_version=version)
                    if not torch.equal(response.terminal,terminal):
                        scale=max(1.,float(terminal.abs().max()))
                        if not torch.allclose(response.terminal,terminal,rtol=64*torch.finfo(torch.float32).eps,
                            atol=64*torch.finfo(torch.float32).eps*scale):raise RuntimeError('JVP_CURRENT_PRIMAL_BOUNDARY')
                    responses.append(response.response)
                psi=torch.stack([normalization.weight(r)/q[i].sqrt() for i,r in enumerate(responses)],dim=1) if responses else torch.empty((e.numel(),0),dtype=torch.float64)
                metric=torch.eye(len(active),dtype=torch.float64)
                controller_start=time.perf_counter()
                solution=nnls_response(e,psi,metric,LAMBDA)
                c=solution.coefficients if arm=='JV_NATIVE' else restricted_l8(e,psi,layers)
                fact=identities(e,psi,metric,c,LAMBDA)
                bound=1e-10*max(1.,fact['gain'],fact['response_sq'],fact['qN'])
                if abs(fact['dissipation_residual'])>bound or fact['speed_excess']>bound:raise RuntimeError('NNLS_DISSIPATION_BOUNDARY')
                controller_seconds=time.perf_counter()-controller_start;shadow_start=time.perf_counter()
                shadows=None
                if arm=='JV_NATIVE':
                    gram=initial_history_gram(dictionary,active,q) if batch_index in CHECKPOINTS and node==0 else None
                    shadows=same_state_shadows(e,psi,metric,gf,q,layers,c,gram)
                layers_action=layer_actions(dictionary,active,q,c)
                shadow_seconds=time.perf_counter()-shadow_start
                token=overlay.seal_sweep_entry(node)
                for i,b in enumerate(active):overlay.append(b.overlay_delta(float(H*c[i]/q[i].sqrt())),sweep_token=token)
                overlay.close_sweep(token)
                post_start=time.perf_counter()
                exit_terminal=family.terminal();exit_e=normalization.weight(target-exit_terminal)
                old._sync();post_seconds=time.perf_counter()-post_start;physical_start=time.perf_counter()
                shadow=overlay.materialize_shadow(device='cpu')
                physical=actual_delta_rows(shadow,previous,family.w0,names);previous=shadow
                physical_seconds=time.perf_counter()-physical_start
                normalized_delta=normalization.weight(exit_terminal-terminal)
                raw_prediction=sum((responses[i].double()*float(c[i]/q[i].sqrt()) for i in range(len(c))),torch.zeros_like(terminal,dtype=torch.float64))
                v_after=float(exit_e.square().sum()/2);energy+=H*fact['qN']
                old._sync()
                record=dict(batch_index=batch_index,node=node,arm=arm,state_version=version,
                    exit_state_version=overlay.state_version,t=(node+1)*H,h=H,lambda_value=LAMBDA,
                    V_before=fact['V'],V_after=v_after,V0=v0,V_ratio=v_after/v0 if v0 else None,E=energy,
                    active_layers=layers,c=c.tolist(),raw_physical_coefficients=(c/q.sqrt()).tolist(),q_layers=q.tolist(),
                    qN_ref=dictionary.qref,normalization_scales=normalization.scales.tolist(),
                    normalization_entry_sha256=normalization.entry_sha,source_entry_sha256=family.w0_sha256,
                    fixed_z_sha256=family.fixed_z.identity_sha256,terminal_sha256=tensor_sha256(exit_terminal),
                    model_error_normalized=float((normalized_delta-H*(psi@c)).norm()),
                    model_error_raw_activation=float((exit_terminal.double()-terminal.double()-H*raw_prediction).norm()),
                    finite_step_dissipation_defect=v_after-fact['V']+H*(fact['response_sq']+LAMBDA*fact['qN'])-.5*H*H*fact['response_sq'],
                    layer_actions=layers_action,actual_physical=physical,shadows=shadows,
                    KKT_stationarity=solution.stationarity,KKT_complementarity=solution.complementarity,
                    build_ids=[b.build_identity for b in builds],solve_backward_errors=[b.solve_backward_error for b in builds],
                    inner_weight_mutation_count=0,inner_history_append_count=0,controller_heldout_access_count=0,
                    normalized_model_error_space='SOURCE_EXACT_N0_FROZEN_PER_BATCH',
                    node_wall_seconds=time.perf_counter()-node_started,
                    compute_seconds=dict(native_dictionary=dictionary.wall-build_time_before,
                        main_jvp=observer.ledger.wall_seconds-jvp_time_before,primary_NNLS=controller_seconds,
                        shadow_metric_NNLS_and_layer_observers=shadow_seconds,post_step_forward=post_seconds,
                        snapshot_materialization_and_actual_norm=physical_seconds),
                    **controller_matrices(e,psi,metric,c),**fact)
                save(output/f'node-{node:02}.json',record);nodes.append(record)
            endpoint_activation=family.terminal()
            with overlay.suspend(authoritative=True):
                endpoint=family.finalize(arm=arm,terminal_state_version=overlay.state_version,
                    shadow_weights=shadow,deltas=overlay.deltas,derived_observation_only=False)
            overlay.assert_w0_unchanged(full_bytes=True)
            if family.last_terminal is None:raise RuntimeError('MATERIALIZED_TERMINAL_MISSING')
            discrepancy=float(normalization.weight(family.last_terminal-endpoint_activation).norm())
            # The source tolerance belongs to primal parity, never semantic success.
            scale=max(1.,float(endpoint_activation.abs().max()))
            if not torch.allclose(family.last_terminal,endpoint_activation,rtol=64*torch.finfo(torch.float32).eps,
                atol=64*torch.finfo(torch.float32).eps*scale):raise RuntimeError('VIRTUAL_MATERIALIZED_TERMINAL_PARITY')
            return dict(status='TERMINAL_VALID',arm=arm,endpoint=endpoint,nodes=nodes,
                materialization_discrepancy_normalized=discrepancy,jvp_ledger=asdict(observer.ledger),
                main_jvp_count=observer.ledger.jvp_call_count,dictionary_build_count=dictionary.build_count,
                solve_count=dictionary.solve_count,build_wall_seconds=dictionary.wall,
                total_write_and_endpoint_seconds=time.perf_counter()-started,overlay=overlay.receipt(),
                **dictionary.net_action(overlay.deltas))
    finally:family.reset_entry()
