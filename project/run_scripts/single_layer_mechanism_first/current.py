"""Current observations reuse immutable inputs without repeated byte audits.

Same head shapes, geometry backend and actual per-candidate tests as EN.
T0 independently compares physical execution; ownership/version guards are not
a substitute for that numerical check or an adversarial mutation guarantee.
"""
import math
import numpy as np
import scipy.linalg as la
import torch
from project.run_scripts.en_execution_reuse.current_observation import CurrentObservationController
from project.run_scripts.en_execution_reuse.endpoint_session import EndpointSession, RuntimePolicy
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import model_guard
from project.run_scripts.single_layer_edit_preserving_correction.common import digest
from project.run_scripts.single_layer_edit_preserving_correction import geometry


class BoundCurrent(CurrentObservationController):
    def _check_inputs(self):
        if self._closed:raise RuntimeError('CURRENT_OBSERVATION_CLOSED')
        self.oracle._guard()
        if self.oracle.head_chunk_positions!=16 or tuple(self.oracle.shape)!=self.session.shape:
            raise RuntimeError('CURRENT_LAYOUT_CHANGED')
        # _input_manifest is computed once by base construction. Cached tensors
        # have private ownership and version/shape/epoch guards on consumption.


class CurrentGuard:
    def __init__(self, rt, WN, oracle, rows, keys, allowed, meta):
        self.rt,self.WN,self.oracle,self.keys,self.allowed=rt,WN,oracle,keys,allowed
        ids=dict(current=digest(meta),teacher=rt.generated_store.receipt['manifest_sha256'])
        epoch=model_guard(rt.model)
        policy=RuntimePolicy(tuple(WN.shape),oracle.device,epoch,ids,rt.lock['execution']['commit'],
            epoch_getter=lambda:model_guard(rt.model),input_identity_getter=lambda:ids,
            source_identity_getter=lambda:rt.lock['execution']['commit'],verify_bytes=False)
        self.session=EndpointSession(rt.identity,dict(current=meta['K_sha256']),policy)
        self.native=self.session.bind_native(WN)
        self.controller=BoundCurrent(WN,oracle,rows,self.session,self.native,
            teacher_identity=ids['teacher'],input_identity=ids['current'])
        self.anchor=self.controller.anchor_rows

    def candidate(self, weight, ideal, trial):
        handle=self.session.bind_candidate(weight,dict(trial=trial))
        passed,reasons=self.controller.guard(handle)
        out=dict(quality_pass=passed,reasons=reasons,rows=self.controller.candidate_rows)
        if passed:
            out['invariant']=self.controller.invariant(handle,ideal,self.keys,self.allowed)
        out['pass']=bool(passed and out['invariant']['pass'])
        return out

    def close(self):
        self.controller.close();self.session.close()


def fixed_panel(items, identity, count=4):
    """Prospective hash-only panel, never selected using error or performance."""
    import hashlib
    order=sorted(range(len(items)),key=lambda i:(hashlib.sha256(
        ('SL-MECHANISM-T0|20260919|'+str(identity(items[i]))).encode()).hexdigest(),i))
    return sorted(order[:count])
