"""One-state raw capture using the existing native dictionary and serial JVP.

No model loader, fixed-target optimization, write, history append, or selection.
The caller must admit the resource and bind the full immutable fixture first.
"""
from dataclasses import asdict
import torch
from project.run_scripts.native_response_ode_v31.algebra import FrozenNormalization
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import GroupedFP32Overlay, tensor_sha256
from project.run_scripts.ordered_response_barrier_ode.terminal_jvp import TerminalResponseObserver
from .contracts import LAYERS, BindingBoundary
from .diagnosis import capture_repeatability
from .fixture import observation_callback_guard
from .normalization_views import NormalizationView


def capture_entry(family, dictionary):
    """Return raw local tensors separately from a raw-free observation receipt.

    N0 is initialized from capture 0, never a mean of three observations. Native
    RHS retains all B rows. Whitening is downstream of detached raw JVP capture.
    """
    if dictionary.family is not family or family.fixed_z is None:
        raise BindingBoundary('CAPTURE_FIXTURE_BINDING')
    family.reset_entry()
    names={l:f'{family.hparams.rewrite_module_tmp.format(l)}.weight' for l in LAYERS}
    overlay=GroupedFP32Overlay(family.model,names)
    observer=TerminalResponseObserver(model=family.model,overlay=overlay,capture_terminal_graph=family.terminal_graph)
    M=tensor_sha256(family.module.cache_c)
    try:
        with overlay,torch.no_grad(),observation_callback_guard(family,overlay):
            captures=[family.terminal() for _ in range(3)]
            source=FrozenNormalization.capture(family.fixed_z.values,captures[0],family.w0_sha256)
            builds=dictionary.build(captures[0],overlay.state_version)
            reference=dictionary.capture_reference(builds) if dictionary.qref is None else dict(
                qN_ref=dictionary.qref,qF_ref=dictionary.qfref,capture_count=0,status='BOUND_EXISTING_ENTRY_REFERENCE')
            active,q,gf=dictionary.whiten(builds)
            responses=[]
            for build in active:
                value=observer.observe(build,expected_state_version=overlay.state_version)
                tolerance=64*torch.finfo(torch.float32).eps
                if not torch.allclose(value.terminal,captures[0],rtol=tolerance,
                                      atol=tolerance*max(1.,float(captures[0].abs().max()))):
                    raise BindingBoundary('RAW_JVP_PRIMAL_NOT_ENTRY')
                responses.append(value.response.detach().clone())
            raw=dict(target32=family.fixed_z.values,entry_captures32=tuple(captures),
                raw_responses32=torch.stack(responses) if responses else torch.empty((0,*captures[0].shape),dtype=torch.float32),
                q_layers=q,metric=torch.eye(len(active),dtype=torch.float64),
                frobenius_metric=gf,layer_ids=tuple(b.layer for b in active),
                native_builds=tuple(builds),source_normalization=source)
            receipt=dict(status='CAPTURED_OBSERVATION_ONLY',entry_W_sha256=family.w0_sha256,
                entry_M_sha256=M,fixed_z_sha256=family.fixed_z.identity_sha256,
                semantic_inventory_sha256=family.fixed_z.target_context_identity_sha256,
                repeatability=capture_repeatability(family.fixed_z.values,captures,
                    semantic_identities=[family.fixed_z.target_context_identity_sha256]*3),
                normalization=NormalizationView.from_source(source).receipt(),reference=reference,
                raw_response_sha256=tensor_sha256(raw['raw_responses32']),
                raw_response_shape=list(raw['raw_responses32'].shape),raw_response_layout='m,D,B',
                active_layers=list(raw['layer_ids']),native_RHS_request_count=len(family.requests),
                jvp_ledger=asdict(observer.ledger),actual_write_count=0,history_append_count=0,
                controller_influence_count=0,fixed_z_recompute_count=0)
        return raw,receipt
    finally:
        family.reset_entry()
