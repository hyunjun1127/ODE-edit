"""Fresh own-entry native fitting and selected-endpoint history, RAM only.

The production S2 Git hook is reused without changing its source. Its presence
is source evidence, not evidence of use in an earlier S2 allocation. New T0 is
responsible for documenting batch arithmetic differences; this module does not
inherit another task's waiver or numerical pass.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import torch
from project.run_scripts.single_layer_mechanism_first import z_hook
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import WEIGHT, model_guard

HOOK_SHA256 = '722a0c35d91e3733b1962f9d07c016436a609b3b5c0cdd391b6ac80ed8ecca8e'
NATIVE_SHA256 = '79da927aad5ab817fd008c5958768adcd00556989a8efbaa2c4bdc80d8fc842e'
Z_CHUNK_SIZE = 16


def requests_from_records(records):
    return [dict(deepcopy(r['requested_rewrite']), case_id=r['case_id']) for r in records]


def _nonselected_guard(model):
    tensors, modules = model_guard(model)
    return tuple(row for row in tensors if row[1] != WEIGHT), modules


class NativeRunner:
    """Caller installs its own W and supplies its own M before every fit.

    fit() leaves the physical model at WN, but does not append M. finalize()
    appends the source-exact native key outer product at the caller-installed
    SELECTED endpoint and returns an owned RAM history. No tensor is saved.
    """
    def __init__(self, model, tok, hp, native_module, contexts, projector, *,
                 batch_size=Z_CHUNK_SIZE, expected_native_sha256=NATIVE_SHA256):
        if hashlib.sha256(Path(z_hook.__file__).read_bytes()).hexdigest() != HOOK_SHA256:
            raise ValueError('S2_GIT_Z_HOOK_SOURCE_MISMATCH')
        if hp.layers != [4] or hp.v_loss_layer != 31:
            raise ValueError('SINGLE_L4_FINAL_LAYER_NATIVE_REQUIRED')
        if projector.shape != (1, 14336, 14336) or projector.dtype != torch.float32:
            raise ValueError('PHYSICAL_L4_SINGLETON_PROJECTOR_REQUIRED')
        self.model, self.tok, self.hp = model, tok, hp
        self.projector = projector
        self.weight = dict(model.named_parameters())[WEIGHT]
        self.batch_size = batch_size
        self.fitter = z_hook.HookedNativeSingletonFitter(native_module,
            expected_source_sha256=expected_native_sha256, contexts=contexts,
            config=z_hook.ZHookConfig(batch_size=batch_size, record_vectors=False))
        self.guard = _nonselected_guard(model)
        self.fit_batches = self.fit_requests = self.history_appends = 0

    def _check(self):
        if _nonselected_guard(self.model) != self.guard:
            raise RuntimeError('NONSELECTED_PARAMETER_OR_HOOK_MUTATION')
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('FROZEN_MODEL_REQUIRED')

    def fit(self, requests, history, *, capture=False):
        self._check()
        if not requests or len(requests) > 100:
            raise ValueError('BOUNDED_NATIVE_BATCH_REQUIRED')
        if len({r['case_id'] for r in requests}) != len(requests):
            raise ValueError('DUPLICATE_CASE_WITHIN_NATIVE_BATCH')
        entry = self.weight.detach().cpu().clone()
        result = self.fitter.fit(self.model, self.tok, self.hp, history,
            self.projector, requests, layer=4, capture=capture)
        self._check()
        self.fit_batches += 1
        self.fit_requests += len(requests)
        result['actual_delta'] = result['weight'].double() - entry.double()
        result['receipt'].update(save_checkpoints=False, disk_tensor_writes=0,
            z_hook_source_sha256=HOOK_SHA256, z_chunk_size=self.batch_size,
            earlier_S2_actual_hook_usage='NOT_VERIFIED',
            new_path_numeric_equivalence='T0_EVIDENCE_REQUIRED',
            fresh_own_entry=True, history_commit='DEFERRED_TO_SELECTED_ENDPOINT',
            case_ids=[r['case_id'] for r in requests])
        # Hook vectors are observations, not resumable state. Retain only when
        # the caller explicitly requests the bounded technical capture.
        if capture:
            result['target_observations'] = self.fitter.target_observations
        self.fitter.target_observations = []
        return result

    def finalize(self, requests, entry_history):
        self._check()
        history = entry_history.detach().cpu().clone()
        receipts = self.fitter.finalize(self.model, self.tok, requests,
            [(4, self.hp, history, self.projector)])
        self._check()
        self.history_appends += 1
        return dict(history=history, receipt=dict(native=receipts,
            selected_endpoint_only=True, save_checkpoints=False,
            disk_tensor_writes=0, case_ids=[r['case_id'] for r in requests]))
