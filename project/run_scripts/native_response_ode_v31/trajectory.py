"""Fixed-grid joint Euler. All native builds/JVPs see one version per node."""
from dataclasses import asdict
import time
import torch
from project.run_scripts.ordered_response_barrier_ode import runtime as old
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import GroupedFP32Overlay, tensor_sha256
from project.run_scripts.ordered_response_barrier_ode.terminal_jvp import TerminalResponseObserver
from .algebra import nnls_response, identities, ray_solution, turning, FrozenNormalization
from .native_binding import NativeDictionary
from .provenance import save


def _snapshot(arm, e, psi, metric, q, gf, lam):
    joint = nnls_response(e,psi,metric,lam)
    ray, raw = ray_solution(e,psi,metric,q,lam)
    diag,_ = ray_solution(e,psi,metric,q,lam,diagonal=True)
    frob = nnls_response(e,psi,gf,lam)
    values = {'JV_NATIVE':joint.coefficients,'ORB_RAY_N':ray,
              'ORB_SNAPSHOT':raw,'DIAG_RAY_N':diag,'JV_FROB':frob.coefficients}
    rows=[]
    for label,c in values.items():
        rows.append(dict(state_owner=arm, comparator=label,lambda_value=lam,c=c.tolist(),
                         **identities(e,psi,metric,c,lam),
                         qF=float(c@gf@c), **turning(joint.coefficients,c,metric)))
    return joint,ray,rows


def run_joint(family, arm, dictionary, normalization, *, n=4, output, fixture):
    family.reset_entry()
    target = family.fixed_z.values
    entry = family.terminal()
    e0 = normalization.weight(target-entry)
    v0 = float(e0.square().sum()/2)
    names = {l:f'{family.hparams.rewrite_module_tmp.format(l)}.weight' for l in (4,5,6,7,8)}
    overlay = GroupedFP32Overlay(family.model,names)
    observer = TerminalResponseObserver(model=family.model,overlay=overlay,capture_terminal_graph=family.terminal_graph)
    h=2./n; energy=length=0.; nodes=[]; shadows=[]
    forwards_before=old._model_forward_count(family.model)
    old._sync(); start=time.perf_counter()
    try:
        with overlay,torch.no_grad():
            for node in range(n):
                terminal = family.terminal()
                e = normalization.weight(target-terminal)
                version = overlay.state_version
                builds = dictionary.build(terminal,version)
                active,q,gf = dictionary.whiten(builds)
                responses=[];primals=[terminal]
                for b in active:
                    response = observer.observe(b,expected_state_version=version)
                    # Every raw directional observation must have the same primal.
                    if not torch.equal(response.terminal,terminal):
                        scale = max(1.,float(terminal.abs().max()))
                        if not torch.allclose(response.terminal,terminal,rtol=64*torch.finfo(torch.float32).eps,
                                              atol=64*torch.finfo(torch.float32).eps*scale):
                            raise RuntimeError('JVP_CURRENT_PRIMAL_BOUNDARY')
                    responses.append(response.response)
                    primals.append(response.terminal)
                if node==0 and not hasattr(dictionary,'numeric_floor'):
                    # Reuse already required primal captures: no calibration forward.
                    max_error=torch.stack([torch.linalg.vector_norm((a-terminal).double(),dim=0) for a in primals]).max(0).values
                    dictionary.numeric_floor=torch.maximum(8*max_error,
                        64*torch.finfo(torch.float32).eps*torch.linalg.vector_norm(terminal.double(),dim=0))
                psi = torch.stack([normalization.weight(r)/q[i].sqrt() for i,r in enumerate(responses)],dim=1) if responses else torch.empty((e.numel(),0),dtype=torch.float64)
                metric = torch.eye(len(active),dtype=torch.float64)  # disjoint native-whitened blocks, not Frobenius
                all_shadows=[]
                for lam in (.01,.1,1.):
                    joint,ray,rows = _snapshot(arm,e,psi,metric,q,gf,lam)
                    all_shadows.extend(rows)
                    if lam == .1:
                        selected = joint.coefficients if arm=='JV_NATIVE' else ray
                        solution=joint
                fact=identities(e,psi,metric,selected)
                if arm=='JV_NATIVE':
                    bound=1e-10*max(1.,fact['gain'],fact['response_sq'],fact['qN'])
                    if abs(fact['dissipation_residual'])>bound or fact['speed_excess']>bound:
                        raise RuntimeError('NNLS_DISSIPATION_BOUNDARY')
                predicted = sum((responses[i].double()*(selected[i]/q[i].sqrt()) for i in range(len(active))),
                                torch.zeros_like(terminal,dtype=torch.float64))
                norm_diag = normalization_diagnostics(target,terminal,normalization,responses)
                alternate=alternate_normalizations(target,terminal,normalization,responses,
                                                     dictionary.numeric_floor,q,metric)
                token=overlay.seal_sweep_entry(node)
                for i,b in enumerate(active):
                    overlay.append(b.overlay_delta(float(h*selected[i]/q[i].sqrt())),sweep_token=token)
                overlay.close_sweep(token)
                exit_terminal=family.terminal()
                exit_e=normalization.weight(target-exit_terminal)
                energy+=h*fact['qN']; length+=h*fact['qN']**.5
                record=dict(fixture=fixture,arm=arm,N=n,node=node,h=h,t=(node+1)*h,
                    state_version=version,exit_state_version=overlay.state_version,
                    active_layers=[b.layer for b in active],build_ids=[b.build_identity for b in builds],
                    solve_backward_errors=[b.solve_backward_error for b in builds],
                    q_layers=q.tolist(),coefficients=selected.tolist(),coefficient_raw=(selected/q.sqrt()).tolist(),
                    V0=v0,V_exit=float(exit_e.square().sum()/2),E=energy,L_N=length,
                    barrier=v0-float(exit_e.square().sum()/2)-.1*energy,
                    model_error=float(torch.linalg.vector_norm(exit_terminal.double()-terminal.double()-h*predicted)),
                    terminal_sha256=tensor_sha256(exit_terminal),source_entry_sha256=family.w0_sha256,
                    fixed_z_sha256=family.fixed_z.identity_sha256,normalization_entry_sha256=normalization.entry_sha,
                    qN_ref=dictionary.qref,qF_ref=dictionary.qfref,
                    kkt_stationarity=solution.stationarity,kkt_complementarity=solution.complementarity,
                    controller_heldout_access_count=0,dynamic_z_count=0,inner_history_append_count=0,
                    **fact,normalization_diagnostics=norm_diag,normalization_shadows=alternate)
                nodes.append(record);shadows.extend(dict(node=node,fixture=fixture,N=n,**r) for r in all_shadows)
                save(output/f'node-{node:02}.json',record)
                save(output/f'shadow-{node:02}.json',all_shadows)
            shadow=overlay.materialize_shadow(device='cpu')
            endpoint_activation=family.terminal()
            net=dictionary.net_action(overlay.deltas)
            if fixture=='D2':
                # Local raw only: enable physical endpoint Cauchy distances, not hash-only claims.
                torch.save(dict(activation=endpoint_activation,
                    delta={key:shadow[key].cpu()-family.w0[key].cpu() for key in shadow}),output/'endpoint-state.pt')
            old._sync(); edit_seconds=time.perf_counter()-start
            with overlay.suspend(authoritative=True):
                endpoint=family.finalize(arm=arm,terminal_state_version=overlay.state_version,
                    shadow_weights=shadow,deltas=overlay.deltas,derived_observation_only=False)
            overlay.assert_w0_unchanged(full_bytes=True)
            overlay_receipt=overlay.receipt()
        old._sync()
        vt=float(normalization.weight(target-endpoint_activation).square().sum()/2)
        return dict(status='TERMINAL_VALID',arm=arm,fixture=fixture,N=n,endpoint=endpoint,
            V0=v0,VT=vt,V_ratio=vt/v0 if v0 else None,near_stall=(v0-vt)/v0<1e-4 if v0 else None,
            E_T=energy,L_N=length,raw_integrated_work=energy*dictionary.qref,
            raw_path_length=length*dictionary.qref**.5,**net,nodes=nodes,same_state_fields=shadows,
            edit_core_seconds=edit_seconds,case_seconds=time.perf_counter()-start,
            main_jvp_count=observer.ledger.jvp_call_count,diagnostic_jvp_count=0,
            jvp_ledger=asdict(observer.ledger),overlay=overlay_receipt,
            forward_count=old._model_forward_count(family.model)-forwards_before,
            w0_restore=True,method_state_entry_sha=family.method_state_identity())
    finally:
        family.reset_entry()


def normalization_diagnostics(target,terminal,norm,responses):
    """Observation only. Does not return control weights or modify N0."""
    r=(target-terminal).double(); active=norm.active
    rows=[]
    count=int(active.sum())
    for j,response in enumerate(responses):
        gain=torch.zeros(r.shape[1],dtype=torch.float64)
        energy=torch.zeros_like(gain)
        if count:
            denom=count*norm.scales[active].double().square()
            gain[active]=(r[:,active]*response[:,active].double()).sum(0)/denom
            energy[active]=response[:,active].double().square().sum(0)/denom
        total=float(energy.sum())
        pi=energy/total if total>0 else torch.zeros_like(energy)
        rows.append(dict(column=j,gain_by_request=gain.tolist(),response_energy_by_request=energy.tolist(),
            pi=pi.tolist(),Neff=1/float(pi.square().sum()) if total>0 else None,
            top1=float(pi.max()),top5=float(torch.topk(pi,min(5,len(pi))).values.sum()),
            positive_gain=int((gain>0).sum()),negative_gain=int((gain<0).sum()),
            s_min=float(norm.scales.min()),s_median=float(norm.scales.median()),
            absolute_residual=torch.linalg.vector_norm(r,dim=0).tolist()))
    return rows


def alternate_normalizations(target,terminal,norm,responses,nu,q,metric):
    active=norm.active
    if not bool(active.any()) or not responses:
        return []
    s=norm.scales.double()[active]
    r=(target-terminal).double()[:,active]
    variants={'NNUM':torch.maximum(s,nu[active]),'NRMS':torch.full_like(s,float(s.square().mean().sqrt()))}
    rows=[]
    for name,shat in variants.items():
        aw=float((s.square()/shat.square()).sum().rsqrt())
        e=(aw*r/shat).reshape(-1)
        psi=torch.stack([(aw*x.double()[:,active]/shat).reshape(-1)/q[j].sqrt()
                          for j,x in enumerate(responses)],1)
        solution=nnls_response(e,psi,metric)
        rows.append(dict(policy=name,status='SHADOW_ONLY',primary_influence_count=0,aw=aw,
                         shat=shat.tolist(),nu=nu[active].tolist(),coefficients=solution.coefficients.tolist(),
                         **identities(e,psi,metric,solution.coefficients)))
    return rows
