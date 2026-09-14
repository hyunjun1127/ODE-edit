"""External absolute-Z binding for the unchanged native singleton writer.

Only the private compiled function namespace substitutes ``compute_z``.  Native
request normalization, fresh keys/readout, residual, direct solve, stored-weight
addition and the separately invoked native finalizer retain their source AST.
These helpers do not implement a target optimizer or choose a policy by score.
"""
import ast
from copy import deepcopy
import hashlib
import json
import time

from .fitting import FitBoundary, NativeSingletonFitter, tensor_sha


POLICIES = {
    'N4': ((24,), (1.,)),
    'REFIT4': ((24, 24), (.75, 1.)),
    'FROZEN2': ((24, 0), (.75, 1.)),
    'I2': ((12, 12), (.75, 1.)),
    'FROZEN4': ((24, 0, 0, 0), (.75, .75, .75, 1.)),
    'I4': ((6, 6, 6, 6), (.75, .75, .75, 1.)),
}


def request_sha(request):
    """Full normalized request identity, not case_id or row index alone."""
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def native_request_normalizer(split_tree):
    """Compile the already guarded native deepcopy/normalization statements."""
    finalizer = split_tree.body[1]
    fn = ast.parse('def normalize(requests):\n    return requests\n').body[0]
    fn.body = deepcopy(finalizer.body[:2]) + fn.body
    tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = {'deepcopy': deepcopy}
    exec(compile(tree, '<native-request-normalization>', 'exec'), ns)
    return ns['normalize']


class ExternalTargetSingletonFitter(NativeSingletonFitter):
    """Use ``bind_targets(requests, ordered_1d_vectors)`` before fit_targets.

    The binding records native-normalized request hashes and copies targets to
    immutable-by-convention CPU snapshots.  Both hashes and tensor bytes are
    rechecked before consumption.  No state survives a fit call in this adapter.
    Inherited ``fit`` remains the native reference; inherited ``finalize`` is
    called ONCE by the batch orchestrator after all subwrites/materializations.
    """
    def __init__(self, module, *, expected_source_sha256, contexts):
        super().__init__(module, expected_source_sha256=expected_source_sha256,
                         contexts=contexts)
        self.normalize_requests = native_request_normalizer(self.tree)

    def bind_targets(self, requests, vectors):
        normalized = self.normalize_requests(requests)
        if not normalized or len(vectors) != len(normalized):
            raise FitBoundary('EXTERNAL_TARGET_COUNT')
        bound = []
        for request, vector in zip(normalized, vectors):
            if (not self.torch.is_tensor(vector) or vector.ndim != 1 or
                    vector.dtype != self.torch.float32 or
                    not self.torch.isfinite(vector).all().item()):
                raise FitBoundary('EXTERNAL_TARGET_FP32_VECTOR')
            value = vector.detach().cpu().clone()
            bound.append(dict(request_sha256=request_sha(request), z=value,
                              z_sha256=tensor_sha(value)))
        return dict(schema='native-normalized-absolute-targets-v1', members=bound)

    def fit_targets(self, model, tok, hp, history, projector, requests, targets,
                    *, layer=4, capture=True):
        if layer != 4:
            raise FitBoundary('WRITE_REFRESH_PHYSICAL_L4_ONLY')
        name, weight = self._check(model, tok, hp, history, projector, layer)
        normalized = self.normalize_requests(requests)
        if not isinstance(targets, dict) or targets.get('schema') != 'native-normalized-absolute-targets-v1':
            raise FitBoundary('EXTERNAL_TARGET_BINDING_REQUIRED')
        members = targets.get('members', [])
        if not normalized or len(members) != len(normalized):
            raise FitBoundary('EXTERNAL_TARGET_COUNT')
        # Validate the entire B100 target barrier BEFORE the first writer call.
        for request, member in zip(normalized, members):
            z = member.get('z')
            if member.get('request_sha256') != request_sha(request):
                raise FitBoundary('EXTERNAL_TARGET_REQUEST_ORDER')
            if (not self.torch.is_tensor(z) or z.dtype != self.torch.float32 or
                    tuple(z.shape) != (weight.shape[0],) or
                    not self.torch.isfinite(z).all().item() or
                    member.get('z_sha256') != tensor_sha(z)):
                raise FitBoundary('EXTERNAL_TARGET_BYTES_OR_SCHEMA')

        counts, tensors = {'compute_z': 0, 'target_supply': 0}, {}
        h_sha, p_sha = tensor_sha(history), tensor_sha(projector)
        before = weight.detach().cpu().clone()
        ptr = weight.data_ptr()
        fit, _ = self._functions(counts, tensors)
        ns = fit.__globals__
        if ns is vars(self.module):
            raise FitBoundary('EXTERNAL_TARGET_NAMESPACE_NOT_ISOLATED')
        original_compute_z = self.module.compute_z
        mode_before = model.training
        grad_before = tuple(p.requires_grad for p in model.parameters())

        def supplied_z(actual_model, actual_tok, request, actual_hp, actual_layer, contexts):
            i = counts['target_supply']
            if (actual_model is not model or actual_tok is not tok or actual_hp is not hp or
                    actual_layer != layer or contexts != self.contexts or i >= len(members) or
                    request_sha(request) != members[i]['request_sha256']):
                raise FitBoundary('EXTERNAL_TARGET_NATIVE_ASSOCIATION')
            begin = time.monotonic()
            # This conversion performs no target optimization or numerical cast.
            result = members[i]['z'].to(device=weight.device).detach().clone()
            tensors.setdefault('compute_z', []).append(result.detach().cpu().clone())
            counts['target_supply'] += 1
            counts['target_supply_seconds'] = counts.get('target_supply_seconds', 0.) + time.monotonic() - begin
            return result

        # The native fit AST is untouched. Only its disposable globals differ.
        ns['compute_z'] = supplied_z
        begin = time.monotonic()
        returned, cache = fit(model, tok, requests, hp, cache_template=None,
                              cache_c=history, P=projector)
        if returned is not model or cache is not history or weight.data_ptr() != ptr:
            raise FitBoundary('EXTERNAL_FIT_RETURN_OR_POINTER')
        if tensor_sha(history) != h_sha or tensor_sha(projector) != p_sha:
            raise FitBoundary('EXTERNAL_FIT_HISTORY_OR_PROJECTOR_MUTATION')
        if self.module.compute_z is not original_compute_z:
            raise FitBoundary('EXTERNAL_FIT_MODULE_GLOBAL_MUTATION')
        if model.training != mode_before or tuple(p.requires_grad for p in model.parameters()) != grad_before:
            raise FitBoundary('EXTERNAL_FIT_MODE_OR_GRAD_MUTATION')
        if (counts['compute_z'] != 0 or counts['target_supply'] != len(requests) or
                counts.get('compute_ks') != 1 or counts.get('solve') != 1 or
                counts.get('get_module_input_output_at_words') != 1):
            raise FitBoundary('EXTERNAL_FIT_CALL_COUNTS')
        if not self.torch.isfinite(weight).all().item():
            raise FitBoundary('NONFINITE_EXTERNAL_FIT')
        endpoint = weight.detach().cpu().clone()
        # These diagnostic operations happen AFTER the native stored-weight write.
        current_y = tensors['get_module_input_output_at_words'][0].T.contiguous()
        absolute_z = self.torch.stack([m['z'].detach().cpu() for m in members], dim=1)
        if current_y.shape != absolute_z.shape:
            raise FitBoundary('EXTERNAL_TARGET_CANONICAL_Y_SHAPE')
        residual = absolute_z - current_y
        return dict(weight=endpoint, captures=tensors if capture else None,
                    current_y=current_y, residual=residual, absolute_z=absolute_z,
                    receipt=dict(**counts, layer=layer, weight_name=name, history_append=0,
                                 seconds=time.monotonic()-begin,
                                 entry_weight_sha256=tensor_sha(before),
                                 endpoint_weight_sha256=tensor_sha(endpoint),
                                 history_sha256=h_sha, projector_sha256=p_sha,
                                 actual_delta_norm=float((endpoint.double()-before.double()).norm()),
                                 source=self.source_evidence,
                                 target_mode='externally-bound-own-absolute-Z',
                                 request_sha256=[m['request_sha256'] for m in members],
                                 target_sha256=[m['z_sha256'] for m in members],
                                 current_y_sha256=tensor_sha(current_y),
                                 residual_sha256=tensor_sha(residual),
                                 native_writer_ast_changes=0,
                                 module_global_patch_count=0,
                                 cache_reuse=False))
