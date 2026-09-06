"""Outcome-independent raw-direction FD and joint overlay GPU fidelity gate."""
import torch
from project.run_scripts.ordered_response_barrier_ode import runtime as old
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import GroupedFP32Overlay, tensor_set_sha256
from project.run_scripts.ordered_response_barrier_ode.terminal_jvp import TerminalResponseObserver
from .native_binding import NativeDictionary
from .algebra import FrozenNormalization, nnls_response
from .provenance import save


def full_logits(f):
    prompts=[str(r['prompt']).format(str(r['subject'])) for r in f.requests]
    batch=f.tokenizer(prompts,padding=True,return_tensors='pt').to(f.device)
    return f.model(**batch,use_cache=False).logits.detach().float().cpu()


def _parity(a,b):
    scale=max(1.,float(a.abs().max()),float(b.abs().max()))
    tol=64*torch.finfo(torch.float32).eps
    return dict(max_abs=float((a-b).abs().max()),relative_l2=float(torch.linalg.vector_norm((a-b).double())/torch.linalg.vector_norm(a.double())) if bool(a.any()) else None,
                atol=tol*scale,rtol=tol,pass_check=bool(torch.allclose(a,b,atol=tol*scale,rtol=tol)))


def run_fidelity(f, output):
    receipt=dict(schema='native-v31.gpu-fidelity.v1',status='RUNNING',layers=[],fd_epsilons=[2**-7,2**-8,2**-9],
                 fixed_z_sha=f.fixed_z.identity_sha256,context_sha=f.fixed_z.target_context_identity_sha256,
                 entry_sha=f.w0_sha256,tokenizer_padding_side=f.tokenizer.padding_side)
    dictionary=NativeDictionary(f)
    names={l:f'{f.hparams.rewrite_module_tmp.format(l)}.weight' for l in (4,5,6,7,8)}
    overlay=GroupedFP32Overlay(f.model,names)
    observer=TerminalResponseObserver(model=f.model,overlay=overlay,capture_terminal_graph=f.terminal_graph)
    try:
        with overlay,torch.no_grad():
            terminal=f.terminal(); builds=dictionary.build(terminal,0)
            halves=dictionary.build(terminal,0,audit_residual_divisor=2)
            scaling={b.layer:bool(torch.equal(b.left/2,c.left) and torch.equal(b.right,c.right))
                     for b,c in zip(builds,halves,strict=True)}
            if not all(scaling.values()):
                raise RuntimeError('NATIVE_RHS_SCALING_BOUNDARY')
            receipt['metric']=dictionary.capture_reference(builds)
            active,q,gf=dictionary.whiten(builds)
            norm=FrozenNormalization.capture(f.fixed_z.values,terminal,f.w0_sha256)
            responses=[]
            for b in active:
                jvp=observer.observe(b,expected_state_version=0)
                function=observer._function(b)
                eps_receipts=[]
                for epsilon in (2**-7,2**-8,2**-9):
                    plus=function(torch.tensor(epsilon,device=f.device)).detach().float().cpu()
                    minus=function(torch.tensor(-epsilon,device=f.device)).detach().float().cpu()
                    fd=(plus.double()-minus.double())/(2*epsilon)
                    reference=jvp.response.double()
                    signal=float(torch.linalg.vector_norm(plus.double()-minus.double()))
                    noise=64*torch.finfo(torch.float32).eps*max(float(torch.linalg.vector_norm(plus.double())),float(torch.linalg.vector_norm(minus.double())))
                    nr=float(torch.linalg.vector_norm(reference));nf=float(torch.linalg.vector_norm(fd))
                    cosine=float((fd*reference).sum())/(nr*nf) if nr>0 and nf>0 else None
                    rel=float(torch.linalg.vector_norm(fd-reference))/nr if nr>0 else None
                    eps_receipts.append(dict(epsilon=epsilon,signal=signal,rounding_envelope=noise,
                        sufficient_signal=signal>noise,cosine=cosine,relative_L2=rel,
                        passed=bool(signal>noise and cosine is not None and cosine>=.99 and rel<=.05)))
                passed=any(eps_receipts[i]['passed'] and eps_receipts[i+1]['passed'] for i in range(2))
                row=dict(layer=b.layer,native_solve_backward_error=b.solve_backward_error,
                         eps=eps_receipts,adjacent_pass=passed,
                         residual_scaling_half_exact=scaling[b.layer],
                         orientation=list(f.parameters[b.weight_name].shape),
                         factor_shape=[b.left.shape[0],b.right.shape[0]])
                if f.family=='AlphaEdit':
                    p=f.module.P[b.layer-4]
                    projected=p.double()@b.right.double()
                    row['stored_projector_operator_action_relative']=float(torch.linalg.norm(projected-b.right.double())/torch.linalg.norm(b.right.double()))
                receipt['layers'].append(row)
                save(output/f'fd-layer{b.layer}.json',row)
                if not passed:
                    receipt['status']='JVP_NUMERICALLY_UNRESOLVED'
                    raise RuntimeError('JVP_NUMERICALLY_UNRESOLVED')
                responses.append(jvp.response)
            e=norm.weight(f.fixed_z.values-terminal)
            psi=torch.stack([norm.weight(r)/q[i].sqrt() for i,r in enumerate(responses)],1) if responses else torch.empty((e.numel(),0),dtype=torch.float64)
            solution=nnls_response(e,psi,torch.eye(len(active),dtype=torch.float64))
            token=overlay.seal_sweep_entry(0)
            for i,b in enumerate(active):
                overlay.append(b.overlay_delta(float(.5*solution.coefficients[i]/q[i].sqrt())),sweep_token=token)
            overlay.close_sweep(token)
            virtual=f.terminal(); virtual_logits=full_logits(f)
            shadow=overlay.materialize_shadow(device='cpu')
            with overlay.suspend(authoritative=False):
                try:
                    f._apply_shadow(shadow)
                    physical=f.terminal(); physical_logits=full_logits(f)
                finally:
                    old._restore_selected(f.parameters,f.w0)
            receipt['activation_parity']=_parity(virtual,physical)
            receipt['logit_parity']=_parity(virtual_logits,physical_logits)
            receipt['one_nonzero_joint_step']=bool(solution.coefficients.any())
            receipt['finite_zero_step']=not bool(solution.coefficients.any())
            receipt['entry_pointer_bytes_restore']=tensor_set_sha256(f.parameters)==f.w0_sha256
            receipt['overlay']=overlay.receipt()
            if not receipt['activation_parity']['pass_check'] or not receipt['logit_parity']['pass_check']:
                raise RuntimeError('OVERLAY_MATERIALIZED_PARITY_BOUNDARY')
            receipt['status']='PASS' if receipt['one_nonzero_joint_step'] else 'FINITE_NO_ACTION_NONZERO_GATE_NOT_OBSERVED'
        # Stock entrypoint provenance is tested independently of ours efficacy.
        official=f.run_official(fixed_z=f.fixed_z)
        receipt['official']={k:v for k,v in official.items() if k not in ('evaluation','semantic_observation')}
        save(output/'official-endpoint.json',official)
        save(output/'gpu_fidelity_checks.json',receipt)
        return receipt
    except BaseException:
        f.reset_entry()
        receipt['restored_after_boundary']=tensor_set_sha256(f.parameters)==f.w0_sha256
        save(output/'gpu-fidelity-boundary.json',receipt)
        raise
    finally:
        f.reset_entry()
