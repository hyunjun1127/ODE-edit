"""Post-native CPU geometry diagnostics; none of these tensors enter a writer."""
import time
import numpy as np
import torch

from .contracts import member, save
from .geometry import projected_geometry, covariance_diagnostics
from .runner import tensor_artifact


def load_native_moment(path):
    # Native SecondMoment stores an unnormalised sum and a token count.
    # Torch FP32 division matches its source moment(), not NumPy promotion.
    with np.load(path,allow_pickle=False) as z:
        count=int(z['mom2.count']);assert count>0
        value=torch.from_numpy(z['mom2.mom2'])/count
    assert value.dtype==torch.float32 and torch.isfinite(value).all()
    return value,count


def run(P,M,K,actual_delta,entry_weight,w0_weight,covariance,output):
    """Exact full-dimension SVD summaries; explicitly costed CPU-only stage.

    U is the eigenspace of symmetric(P) above .5, a diagnostic coordinate
    choice only. Neither this symmetrisation nor U is passed to native code.
    Native-system diagnostics retain raw P/M and source-order FP32 operations.
    """
    from pathlib import Path
    root=Path(output);started=time.monotonic()
    binding=member(covariance);C0,count=load_native_moment(covariance)
    P=P.detach().cpu();M=M.detach().cpu();K=K.detach().cpu();delta=actual_delta.detach().cpu()
    assert P.dtype==M.dtype==K.dtype==C0.dtype==torch.float32
    times=[]
    with torch.no_grad():
        t=time.monotonic();sym=(P.double()+P.double().T)*.5
        values,vectors=torch.linalg.eigh(sym);U=vectors[:,values>.5].float()
        times.append(dict(component='diagnostic_P_sym_FP64_eigh',seconds=time.monotonic()-t))
        basis=tensor_artifact(root/'diagnostic-basis.pt',dict(U=U,eigenvalues=values,
            cutoff=.5,source_P_shape=list(P.shape),role='DIAGNOSTIC_ONLY_NEVER_WRITER'))
        del sym,values,vectors
        t=time.monotonic()
        geometry=projected_geometry(P,C0,M,K,basis=U,ridge=1.,compute_spectrum=True,exact_max_dimension=None)
        times.append(dict(component='exact_full_and_reduced_SVD',seconds=time.monotonic()-t));del U
        t=time.monotonic()
        S=(entry_weight.detach().cpu().double()-w0_weight.detach().cpu().double()).float()
        leakage=covariance_diagnostics(S,delta,C0)
        times.append(dict(component='FP32_covariance_contractions',seconds=time.monotonic()-t))
    return save(root/'projected-geometry.json',dict(covariance=binding,source_moment_token_count=count,
        matrix_shape=list(C0.shape),moment_normalisation='NATIVE_TORCH_FP32_SUM_DIV_COUNT',
        basis=basis,geometry=geometry,leakage=leakage,components=times,
        seconds=time.monotonic()-started,rank_truncation_in_writer=0,
        native_inputs_changed=False,source_C0_not_checkpoint_covariance=True,
        diagnostic_dtype='FP64_sym_P_eigh; FP32_native_system_reduced_SVD_and_contractions',scientific_promotion=False))
