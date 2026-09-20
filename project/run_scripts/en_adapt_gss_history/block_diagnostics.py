"""Algebra-only reference/history spectral decomposition using the one SVD."""
import time
import numpy as np
from threadpoolctl import threadpool_limits
from project.run_scripts.en_adaptive_nullspace.geometry import array


def block_spectrum(geometry, reference, history):
    started=time.monotonic()
    gr,gh=np.asarray(array(reference)),np.asarray(array(history))
    if gr.shape!=gh.shape or gr.dtype!=np.float64 or gh.dtype!=np.float64:
        raise ValueError('FP64_BLOCK_GRADIENT_SCHEMA')
    mode_r=np.zeros(geometry.rank);mode_h=np.zeros(geometry.rank);cross=np.zeros(geometry.rank)
    exact_r=exact_h=exact_cross=0.
    with threadpool_limits(limits=8):
        for start in range(0,gr.shape[0],128):
            r=gr[start:start+128]@geometry.basis
            h=gh[start:start+128]@geometry.basis
            mr=r@geometry.vectors;mh=h@geometry.vectors
            mode_r+=np.sum(mr*mr,axis=0);mode_h+=np.sum(mh*mh,axis=0)
            cross+=2*np.sum(mr*mh,axis=0)
            er=(r-mr@geometry.vectors.T)@geometry.basis.T
            eh=(h-mh@geometry.vectors.T)@geometry.basis.T
            exact_r+=float(np.sum(er*er));exact_h+=float(np.sum(eh*eh))
            exact_cross+=2*float(np.sum(er*eh))
    return dict(order='ascending singular values, identical to parent spectrum',
        mode_reference_energy=mode_r[::-1].tolist(),mode_history_energy=mode_h[::-1].tolist(),
        mode_cross_term=cross[::-1].tolist(),exact_reference_energy=exact_r,
        exact_history_energy=exact_h,exact_cross_term=exact_cross,
        reconstructed_exact_energy=exact_r+exact_h+exact_cross,
        extra_model_passes=0,extra_SVDs=0,seconds=time.monotonic()-started)
