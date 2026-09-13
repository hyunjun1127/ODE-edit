"""The already sealed single-rewrite signed probe, reusable at a warm entry."""
import time
import torch

from .contracts import ContractBoundary, digest
from .evaluation import diagnostic_margin
from .fixtures import capture_rng, restore_rng, tensor_sha
from .signed_response import AllPositionContraction, finite_difference_audit


def probe(model,tok,weight,module_name,entry_weight,endpoint_weight,record,diagnostics):
    start=time.monotonic();rng=capture_rng()
    saved=weight.detach().clone();pointer=weight.data_ptr();version=weight._version
    flags=[(p,p.requires_grad) for p in model.parameters()]
    entry=entry_weight.to(weight);model.requires_grad_(False)
    delta64=endpoint_weight.double()-entry_weight.double()
    direction=delta64.float().to(weight.device)
    rounding=float((direction.cpu().double()-delta64).norm());del delta64
    try:
        with torch.no_grad():weight.copy_(entry)
        with AllPositionContraction(model.get_submodule(module_name),direction) as capture:
            scalar=diagnostic_margin(model,tok,record)
            contraction=capture.compute(scalar)
        alpha=diagnostics['fd_alpha']
        assert alpha==2**-8 and diagnostics['fd_relative_tolerance']==.05
        with torch.no_grad():
            zero=float(diagnostic_margin(model,tok,record));repeat=float(diagnostic_margin(model,tok,record))
            values=[]
            for sign in (-1,1):
                weight.copy_(entry+(sign*alpha)*direction)
                values.append(float(diagnostic_margin(model,tok,record)))
        noise=abs(repeat-zero)
        absolute=(8*torch.finfo(torch.float32).eps*max(1.,abs(zero))+noise)/alpha
        fd=finite_difference_audit(contraction['event_derivative'],values[0],zero,values[1],
            alpha=alpha,forward_noise=noise,absolute_tolerance=absolute,relative_tolerance=.05)
    finally:
        with torch.no_grad():weight.copy_(saved)
        for p,flag in flags:p.requires_grad_(flag)
        restore_rng(rng)
        if weight.data_ptr()!=pointer or not torch.equal(weight,saved):
            raise ContractBoundary('SIGNED_PROBE_RESTORE')
    return dict(contraction=contraction,finite_difference=fd,
        physical_endpoint_difference_FP32_rounding_norm=rounding,
        direction_source='ACTUAL_ENDPOINT_DIFFERENCE_FP32',record_sha256=digest(record),
        selected_pointer_bytes_restored=True,selected_version_increment=weight._version-version,
        RNG_flags_restored=True,diagnostic_forwards=5,backwards=1,
        native_alpha1_repeated=False,observer_only=True,seconds=time.monotonic()-start)
