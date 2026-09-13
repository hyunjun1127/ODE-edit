"""Source-exact BLUE singleton fitting, separated from endpoint history commit.

No editor globals are patched. The pinned apply function is compiled with only
its final history loop removed; that very loop is compiled into a finalizer.
The caller owns branch W/M/RNG transactions and immutable import-closure locks.
"""
import ast
from copy import deepcopy
import hashlib
import inspect
import time


class FitBoundary(RuntimeError):
    pass


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def split_native_source(source):
    """Only rename function / remove final append loop; refuse unknown layout."""
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'apply_AlphaEdit_to_model']
    if len(functions) != 1:
        raise FitBoundary('SOURCE_FUNCTION_LAYOUT')
    original = functions[0]
    # Source ends with the endpoint history for-loop, print, return.
    history = original.body[-3]
    expected = ast.parse('''for i, layer in enumerate(hparams.layers):
    layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
    cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T
''').body[0]
    if ast.dump(history, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise FitBoundary('SOURCE_FINAL_HISTORY_LOOP_MISMATCH')
    if not isinstance(original.body[-1], ast.Return):
        raise FitBoundary('SOURCE_RETURN_LAYOUT')
    fit = deepcopy(original)
    fit.name = '_pilot_fit_without_history'
    del fit.body[-3]
    # Reuse native deepcopy/leading-space normalization exactly, not a new
    # tokenizer or request-format implementation. Ignore native sample prints.
    if not (isinstance(original.body[1], ast.Assign)
            and isinstance(original.body[2], ast.For)):
        raise FitBoundary('SOURCE_REQUEST_NORMALIZATION_LAYOUT')
    context = [n for n in original.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'context_templates' for t in n.targets)]
    if len(context) != 1:
        raise FitBoundary('SOURCE_CONTEXT_LAYOUT')
    final = deepcopy(original)
    final.name = '_pilot_endpoint_history'
    final.body = deepcopy(original.body[1:3]) + deepcopy(context) + [deepcopy(history), deepcopy(original.body[-1])]
    for fn in (fit, final):
        fn.decorator_list = []
    transformed = ast.fix_missing_locations(ast.Module(body=[fit, final], type_ignores=[]))
    evidence = dict(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_function_ast_sha256=hashlib.sha256(ast.dump(original).encode()).hexdigest(),
        transformed_ast_sha256=hashlib.sha256(ast.dump(transformed).encode()).hexdigest(),
        history_loop_ast_sha256=hashlib.sha256(ast.dump(history).encode()).hexdigest(),
        change='fit: remove final history loop only; finalize: reuse request normalization, context lookup, exact history loop',
        optimizer_or_solve_equation_changes=0, module_global_patch_count=0)
    return transformed, evidence


class NativeSingletonFitter:
    """One instance per isolated process, supplied a locked BLUE module.

    `expected_source_sha256` hashes inspect.getsource(module), UTF-8. `contexts`
    must already be the exact capsule contexts; generation is forbidden here.
    Physical projector selection is explicit and always creates singleton P[0].
    Captured z/K/R are local-only tensors, never publish them.
    """
    def __init__(self, module, *, expected_source_sha256, contexts):
        self.module = module
        source = inspect.getsource(module)
        self.tree, self.source_evidence = split_native_source(source)
        if self.source_evidence['source_sha256'] != expected_source_sha256:
            raise FitBoundary('PINNED_EDITOR_SOURCE_SHA')
        if contexts is None or module.CONTEXT_TEMPLATES_CACHE != contexts:
            raise FitBoundary('EXACT_CONTEXT_CACHE_REQUIRED')
        self.contexts = deepcopy(contexts)
        self.torch = module.torch

    def _check(self, model, tok, hp, history, projector, layer):
        t = self.torch
        if hp.layers != [layer] or not hp.blue or hp.L2 != 1 or layer not in (4, 8):
            raise FitBoundary('PILOT_SINGLETON_HPARAMS')
        if hp.v_num_grad_steps != 25 or tok.padding_side != 'right':
            raise FitBoundary('NATIVE_OPTIMIZER_OR_PADDING_POLICY')
        if self.module.CONTEXT_TEMPLATES_CACHE != self.contexts:
            raise FitBoundary('CONTEXT_CACHE_CHANGED')
        name = hp.rewrite_module_tmp.format(layer) + '.weight'
        weights = dict(model.named_parameters())
        w = weights[name]
        if (any(x.dtype != t.float32 for x in weights.values()) or
                history.dtype != t.float32 or projector.dtype != t.float32 or
                history.shape != (1, w.shape[1], w.shape[1]) or projector.shape != history.shape):
            raise FitBoundary('FP32_SINGLETON_SCHEMA')
        if any(not t.isfinite(x).all().item() for x in (w, history, projector)):
            raise FitBoundary('NONFINITE_ENTRY')
        return name, w

    def _functions(self, counts, capture):
        ns = dict(vars(self.module))
        def wrap(name):
            original = ns[name]
            def called(*args, **kwargs):
                begin = time.monotonic()
                result = original(*args, **kwargs)
                counts[name] = counts.get(name, 0) + 1
                counts[name + '_seconds'] = counts.get(name + '_seconds', 0.) + time.monotonic() - begin
                if capture is not None:
                    out = result[1] if name == 'get_module_input_output_at_words' else result
                    capture.setdefault(name, []).append(out.detach().cpu().clone())
                return result
            ns[name] = called
        for name in ('compute_z', 'compute_ks', 'get_module_input_output_at_words'):
            wrap(name)
        original_torch = self.torch
        original_solve = original_torch.linalg.solve
        def solve(*args, **kwargs):
            begin = time.monotonic()
            result = original_solve(*args, **kwargs)
            counts['solve'] = counts.get('solve', 0) + 1
            counts['solve_seconds'] = counts.get('solve_seconds', 0.) + time.monotonic() - begin
            return result
        class Proxy:
            def __init__(self, base, **overrides):
                self.base, self.overrides = base, overrides
            def __getattr__(self, name):
                return self.overrides[name] if name in self.overrides else getattr(self.base, name)
        # Only the copied function global sees this proxy. torch itself and the
        # compute_z module retain their original objects and numerical paths.
        ns['torch'] = Proxy(original_torch, linalg=Proxy(original_torch.linalg, solve=solve))
        exec(compile(self.tree, self.module.__file__ + ':pilot-history-split', 'exec'), ns)
        return ns['_pilot_fit_without_history'], ns['_pilot_endpoint_history']

    def fit(self, model, tok, hp, history, projector, requests, *, layer, capture=False):
        name, weight = self._check(model, tok, hp, history, projector, layer)
        counts, tensors = {}, {} if capture else None
        h_sha, p_sha = tensor_sha(history), tensor_sha(projector)
        before = weight.detach().cpu().clone()
        ptr = weight.data_ptr()
        fit, _ = self._functions(counts, tensors)
        begin = time.monotonic()
        returned, cache = fit(model, tok, requests, hp, cache_template=None, cache_c=history, P=projector)
        if returned is not model or cache is not history or weight.data_ptr() != ptr:
            raise FitBoundary('FIT_RETURN_OR_POINTER')
        if tensor_sha(history) != h_sha or tensor_sha(projector) != p_sha:
            raise FitBoundary('FIT_HISTORY_OR_PROJECTOR_MUTATION')
        if (counts.get('compute_z', 0) != len(requests) or counts.get('compute_ks', 0) != 1
                or counts.get('solve', 0) != 1 or counts.get('get_module_input_output_at_words', 0) != 1):
            raise FitBoundary('FIT_CALL_COUNTS')
        if not self.torch.isfinite(weight).all().item():
            raise FitBoundary('NONFINITE_FIT')
        endpoint = weight.detach().cpu().clone()
        return dict(weight=endpoint, captures=tensors, receipt=dict(
            **counts, layer=layer, weight_name=name, history_append=0, seconds=time.monotonic()-begin,
            entry_weight_sha256=tensor_sha(before), endpoint_weight_sha256=tensor_sha(endpoint),
            history_sha256=h_sha, projector_sha256=p_sha,
            actual_delta_norm=float((endpoint.double()-before.double()).norm()),
            source=self.source_evidence, target_mode='same-host-fresh-current-state'))

    def finalize(self, model, tok, requests, bindings):
        """bindings=[(physical_layer, singleton_hp, singleton_M, singleton_P)].

        Invoke exactly once at the final actual W for the unique selected layers.
        A refit of L4 still supplies ONE L4 binding. Caller owns commit token so
        intentional suffix batches may append again at their own endpoints.
        """
        layers = [b[0] for b in bindings]
        if not layers or len(layers) != len(set(layers)):
            raise FitBoundary('DUPLICATE_FINAL_HISTORY_LAYER')
        receipts = []
        for layer, hp, history, projector in bindings:
            name, weight = self._check(model, tok, hp, history, projector, layer)
            w_sha, h_sha, p_sha = tensor_sha(weight), tensor_sha(history), tensor_sha(projector)
            counts = {}
            _, finalize = self._functions(counts, None)
            returned, cache = finalize(model, tok, requests, hp, cache_template=None, cache_c=history, P=projector)
            if (returned is not model or cache is not history or tensor_sha(weight) != w_sha
                    or tensor_sha(projector) != p_sha):
                raise FitBoundary('FINALIZE_MODEL_MUTATION')
            if counts.get('compute_ks') != 1 or counts.get('compute_z', 0) or counts.get('solve', 0):
                raise FitBoundary('FINALIZE_CALL_COUNTS')
            if not self.torch.isfinite(history).all().item():
                raise FitBoundary('NONFINITE_HISTORY')
            receipts.append(dict(layer=layer, history_append=1, **counts,
                                 before_sha256=h_sha, after_sha256=tensor_sha(history), weight_sha256=w_sha))
        return receipts


def select_projector(full_stack, physical_layer, *, source_layers=(4, 5, 6, 7, 8)):
    if len(source_layers) != full_stack.shape[0] or physical_layer not in source_layers:
        raise FitBoundary('PROJECTOR_PHYSICAL_MAPPING')
    index = source_layers.index(physical_layer)
    selected = full_stack[index:index+1].detach().clone()
    return selected, dict(physical_layer=physical_layer, source_index=index, local_index=0,
                          source_tensor_sha256=tensor_sha(full_stack[index]),
                          selected_tensor_sha256=tensor_sha(selected[0]))


def materialize_alpha(weight, entry, native, alpha):
    """Exact endpoint copies for alpha0/1, otherwise FP32 D4/materialization."""
    import torch
    if any(x.dtype != torch.float32 for x in (weight, entry, native)):
        raise FitBoundary('ALPHA_FP32_REQUIRED')
    if entry.shape != native.shape or weight.shape != entry.shape or alpha not in (0., .75, .875, 1.):
        raise FitBoundary('ALPHA_SCHEMA_OR_UNREGISTERED_VALUE')
    with torch.no_grad():
        if alpha == 0.:
            result = entry
        elif alpha == 1.:
            result = native
        else:
            result = entry + alpha * (native - entry)
        weight.copy_(result.to(weight.device))
    if not torch.isfinite(weight).all().item():
        raise FitBoundary('NONFINITE_ALPHA')
    return dict(alpha=alpha, weight_sha256=tensor_sha(weight), dtype='float32',
                materialization='endpoint-copy' if alpha in (0., 1.) else 'FP32(entry + alpha * FP32(native-entry))',
                arithmetic_device=str(entry.device))
