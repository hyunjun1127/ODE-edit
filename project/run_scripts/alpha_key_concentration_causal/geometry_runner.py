"""E1/E2 controller using the task-owned runtime; no CLI or job submission.

Only keys, means, scalar spectra and provenance are persistent. Weight/history
state stays in the runtime. The original model/observer/solve is never replaced.
All calibration selection is fixed before observations; N/P never enter here.
"""
from __future__ import annotations

import csv
import itertools
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .common import digest, file_sha, save, tensor_file, tensor_sha
from .geometry import GeometryError, compare_keys, context_geometry, geometry_summary


STATES = (0, 10, 50, 70, 80, 90, 100)
LAYERS = (4, 5, 6, 7, 8)
COHORTS = ('early', 'middle', 'onset', 'late')
HYBRID_STATES = (80, 100)


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype not in (np.dtype('float32'), np.dtype('float64')) or not np.isfinite(result).all():
        raise GeometryError('capture/P must be finite FP32/FP64')
    return result


def _cpu_tensor(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    return torch.from_numpy(np.ascontiguousarray(value))


def _pick(mapping, layer):
    return mapping[layer] if layer in mapping else mapping[str(layer)]


def _projector(runtime, layer):
    source = runtime.P
    if isinstance(source, Mapping):
        return _numpy(_pick(source, layer))
    return _numpy(source[layer - 4])


def project_keys(keys: np.ndarray, projector: np.ndarray, block_rows: int = 128) -> np.ndarray:
    """Stored FP32 P applied to FP32 keys, with bounded sample-row scratch.

    This is a diagnostic Pk operation, not a writer replacement. Diagnostic
    stats subsequently use FP64. No feature-square copy/orthogonalization occurs.
    """
    keys, projector = _numpy(keys), _numpy(projector)
    if keys.ndim not in (2, 3) or projector.ndim != 2 or projector.shape != (keys.shape[-1], keys.shape[-1]):
        raise GeometryError('project_keys dimension mismatch')
    if block_rows < 1:
        raise GeometryError('project_keys block_rows must be positive')
    if keys.dtype != np.float32 or projector.dtype != np.float32:
        raise GeometryError('capture and stored P must be FP32; implicit precision conversion forbidden')
    rows = keys.reshape(-1, keys.shape[-1])
    result = np.empty_like(rows)
    for first in range(0, len(rows), block_rows):
        result[first:first + block_rows] = rows[first:first + block_rows] @ projector.T
    if not np.isfinite(result).all():
        raise GeometryError('nonfinite projected keys')
    return result.reshape(keys.shape)


def validate_panels(panels: Mapping[str, Any], byid: Mapping[Any, Any]) -> dict[str, Any]:
    """Fail closed on a changed fixed cohort/calibration/assessment membership."""
    cohorts = panels['cohorts']
    if set(cohorts) != set(COHORTS):
        raise GeometryError('expected exact four predeclared cohorts')
    all_ids, cal_ids = [], []
    for name in COHORTS:
        panel = cohorts[name]
        whole = list(panel['case_ids'])
        cal = list(panel['calibration_case_ids'])
        assess = list(panel['assessment_case_ids'])
        if (len(whole), len(cal), len(assess)) != (1000, 128, 872):
            raise GeometryError(f'{name}: wrong cohort/calibration/assessment count')
        if len(set(whole)) != 1000 or len(set(cal)) != 128 or len(set(assess)) != 872:
            raise GeometryError(f'{name}: duplicate IDs')
        if set(cal) & set(assess) or set(cal) | set(assess) != set(whole):
            raise GeometryError(f'{name}: split partition mismatch')
        for case_id in whole:
            record = byid[case_id] if case_id in byid else byid[str(case_id)]
            if 'case_id' in record and record['case_id'] != case_id:
                raise GeometryError('byid record case identity mismatch')
        all_ids.extend(whole)
        cal_ids.extend(cal)
    if len(set(all_ids)) != 4000 or len(set(cal_ids)) != 512:
        raise GeometryError('cross-cohort overlap')
    return {'cohorts': cohorts, 'calibration_ids': cal_ids,
            'panel_sha256': digest(panels), 'whole_order_sha256': digest(all_ids),
            'calibration_order_sha256': digest(cal_ids)}


def _records(ids, byid):
    return [byid[i] if i in byid else byid[str(i)] for i in ids]


def _gate_up(features, layer, point):
    aliases = {'gate': ('gate', 'gate_preactivation'), 'up': ('up', 'up_projection')}
    keys = [f'L{layer}/{name}' for name in aliases[point] if f'L{layer}/{name}' in features]
    if not keys:
        raise GeometryError(f'L{layer}: missing {point} feature')
    if len(keys) == 2 and not np.array_equal(_numpy(features[keys[0]]), _numpy(features[keys[1]])):
        raise GeometryError(f'L{layer}: ambiguous {point} aliases differ')
    return features[keys[0]]


def _validate_capture(capture, ids, *, require_features=False):
    for layer in LAYERS:
        keys = _numpy(_pick(capture['keys'], layer))
        means = _numpy(_pick(capture['means'], layer))
        if keys.ndim != 3 or keys.shape[:2] != (len(ids), 6) or means.shape != (len(ids), keys.shape[-1]):
            raise GeometryError(f'L{layer}: capture shape/count mismatch')
        if keys.dtype != np.float32 or means.dtype != np.float32:
            raise GeometryError('runtime keys/means must retain captured FP32 bytes')
    if require_features:
        for layer in LAYERS:
            for point in ('gate', 'up'):
                key = f'L{layer}/{point}'
                feature = _numpy(_gate_up(capture['features'], layer, point))
                if feature.shape != _numpy(_pick(capture['keys'], layer)).shape:
                    raise GeometryError(f'{key}: feature orientation/count mismatch')
    if 'token_receipt' not in capture:
        raise GeometryError('capture token receipt missing')


def _flatten_summary(summary, *, phase, state, condition, panel, layer, space, context):
    return {
        'phase': phase, 'state': state, 'condition': condition, 'panel': panel,
        'layer': layer, 'space': space, 'context': context, 'n': summary['n'],
        'd': summary['d'], 'raw_pr': summary['raw']['participation_ratio'],
        'centered_pr': summary['centered']['participation_ratio'],
        'unit_raw_pr': summary['unit_raw']['participation_ratio'],
        'unit_centered_pr': summary['unit_centered']['participation_ratio'],
        'top1_eigen_share': summary['raw']['top1_energy_share'],
        'top5_eigen_share': summary['raw']['top5_energy_share'],
        'top20_eigen_share': summary['raw']['top20_energy_share'],
        'mean_energy_fraction': summary['mean_energy_fraction'],
        'norm_energy_ess': summary['norm_energy_ess'],
        'top1pct_sample_energy_share': summary['top1pct_sample_energy_share'],
        'top5pct_sample_energy_share': summary['top5pct_sample_energy_share'],
        'zero_norm_count': summary['zero_norm_count'],
    }


def _save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['status'])
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


class _PerRequestSink:
    """Streaming local-only norms/provenance; parquet availability is explicit."""
    def __init__(self, root):
        self.root, self.count, self.closed = Path(root), 0, False
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema = None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            self.pa = pa
            self.schema = pa.schema([
                ('phase', pa.string()), ('state', pa.int32()), ('condition', pa.string()),
                ('panel', pa.string()), ('layer', pa.int8()), ('case_id', pa.int64()),
                ('request_index', pa.int32()), ('space', pa.string()), ('context', pa.string()),
                ('key_norm', pa.float64()), ('key_energy', pa.float64()), ('nonzero', pa.bool_()),
                ('key_dimension', pa.int32()), ('input_dtype', pa.string()), ('reduction_dtype', pa.string()),
                ('context_weight', pa.float64()), ('source_tensor_sha256', pa.string()),
                ('source_tensor_path', pa.string()), ('source_context_tensor_retained', pa.bool_()),
                ('token_receipt_sha256', pa.string()), ('P_tensor_sha256', pa.string()), ('P_asset_index', pa.int8()),
            ])
            self.target = self.root / 'per_request_key_geometry.parquet'
            self.partial = self.target.with_suffix('.parquet.atomic-partial')
            self.file = self.partial.open('xb')
            self.writer = pq.ParquetWriter(self.file, self.schema, compression='zstd')
            self.status, self.version = 'AVAILABLE', pa.__version__
        except ImportError as exc:
            self.target = self.root / 'per_request_key_geometry.csv'
            self.partial = self.target.with_suffix('.csv.atomic-partial')
            self.file = self.partial.open('x', newline='', encoding='utf-8')
            self.writer = None
            self.status, self.version = 'PARQUET_NOT_AVAILABLE_CSV_FALLBACK', str(exc)
        save(self.root / 'per-request-format.json', {'status': self.status, 'pyarrow': self.version,
             'path': str(self.target), 'no_prompts': True, 'norms_are_not_persistent_model_state': True})

    def write(self, rows):
        if not rows:
            return
        if self.closed:
            raise GeometryError('per-request sink already closed')
        if self.schema is not None:
            self.writer.write_table(self.pa.Table.from_pylist(rows, schema=self.schema))
        else:
            if self.writer is None:
                self.writer = csv.DictWriter(self.file, fieldnames=list(rows[0]))
                self.writer.writeheader()
            self.writer.writerows(rows)
        self.count += len(rows)

    def close(self, *, complete):
        if self.closed:
            return None
        self.closed = True
        if self.schema is not None:
            self.writer.close()
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        if complete:
            os.link(self.partial, self.target)
            self.partial.unlink()
            path = self.target
        else:
            path = self.partial
        return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha(path),
                'rows': self.count, 'complete': complete, 'format_status': self.status}


class _Output:
    def __init__(self, runtime, root):
        self.rt, self.root = runtime, Path(root)
        self.rows, self.artifacts, self.cells, self.costs = [], [], [], []
        self.e1_refs = {}
        self.per_request = _PerRequestSink(self.root)
        self.P_hashes = {}

    def close(self, complete=True):
        receipt = self.per_request.close(complete=complete)
        if receipt is not None:
            self.artifacts.append(receipt)
        return receipt

    def capture(self, ids, byid, *, features, label):
        started = time.monotonic()
        result = self.rt.capture(_records(ids, byid), contexts=True, features=features, full=False)
        _validate_capture(result, ids, require_features=features)
        self.costs.append({'kind': 'capture_inclusive', 'cell': label, 'seconds': time.monotonic() - started,
                           'runtime_seconds': result.get('seconds'), 'n': len(ids), 'features': features,
                           'nested_runtime_seconds_do_not_add': True})
        return result

    def analyze(self, capture, ids, *, phase, state, condition, panel, save_context, features=False):
        started = time.monotonic()
        target = self.root / phase / f'W{state:03d}' / condition / panel
        provenance = {'phase': phase, 'state': state, 'condition': condition, 'panel': panel,
                      'case_ids': list(ids), 'case_order_sha256': digest(list(ids)),
                      'token_receipt': capture['token_receipt'], 'contexts_retained': save_context,
                      'writer_means_retained': True, 'feature_tensors_persisted': False,
                      'feature_status': 'MEASURED' if features else 'NOT_REQUESTED',
                      'captured_feature_names': sorted(capture.get('features', {})),
                      'P_mapping': {str(layer): layer - 4 for layer in LAYERS}}
        self.artifacts.append(save(target / 'capture.json', provenance))
        for layer in LAYERS:
            keys = _numpy(_pick(capture['keys'], layer))
            means = _numpy(_pick(capture['means'], layer))
            blob = {'writer_mean': _cpu_tensor(means), 'case_ids': list(ids)}
            if save_context:
                blob['context_keys'] = _cpu_tensor(keys)
            artifact = tensor_file(target / f'L{layer}-keys.pt', blob)
            self.artifacts.append(artifact)
            layer_meta = dict(phase=phase, state=state, condition=condition, panel=panel, layer=layer)
            self._stats(keys, means, target, layer_meta, ids, artifact, capture['token_receipt'], save_context)
            if phase == 'E1':
                self._e1_compare(keys, means, target, layer_meta, ids, artifact)
        if features:
            self._features(capture['features'], target, phase, state, condition, panel)
        self.cells.append(dict(phase=phase, state=state, condition=condition, panel=panel, n=len(ids),
                               status='COMPLETED', capture=provenance['case_order_sha256']))
        self.costs.append({'kind': 'cpu_geometry_inclusive', 'phase': phase, 'state': state,
                           'condition': condition, 'panel': panel, 'seconds': time.monotonic() - started})

    def _stats(self, keys, means, target, meta, ids, artifact, token_receipt, context_retained):
        for space in ('raw', 'projected'):
            if space == 'raw':
                ck, cm = keys, means
            else:
                p = _projector(self.rt, meta['layer'])
                ck, cm = project_keys(keys, p), project_keys(means, p)
            summary = context_geometry(ck, layer=meta['layer'], space=space, writer_mean=cm, metadata=meta)
            summary['projection_reduction'] = 'CPU_FP32_Pk_before_FP64_geometry' if space == 'projected' else 'NONE'
            self.artifacts.append(save(target / f'L{meta["layer"]}-{space}-geometry.json', summary))
            if space == 'projected' and meta['layer'] not in self.P_hashes:
                self.P_hashes[meta['layer']] = tensor_sha(_cpu_tensor(p))
            for context, item in enumerate(summary['contexts']):
                self.rows.append(_flatten_summary(item, **meta, space=space, context=context))
            self.rows.append(_flatten_summary(summary['writer_mean'], **meta, space=space, context='writer_mean'))
            per_request_rows = []
            for context, item in [(str(j), x) for j, x in enumerate(summary['contexts'])] + [('writer_mean', summary['writer_mean'])]:
                for index, (case_id, energy) in enumerate(zip(ids, item['norm_energy_per_sample'], strict=True)):
                    per_request_rows.append({**meta, 'case_id': int(case_id), 'request_index': index,
                        'space': space, 'context': context, 'key_norm': float(np.sqrt(energy)),
                        'key_energy': float(energy), 'nonzero': bool(energy > 0),
                        'key_dimension': int(item['d']), 'input_dtype': item['input_dtype'],
                        'reduction_dtype': item['reduction_dtype'],
                        'context_weight': None if context == 'writer_mean' else .5 if context == '0' else .1,
                        'source_tensor_sha256': artifact['sha256'],
                        'source_tensor_path': artifact['path'],
                        'source_context_tensor_retained': bool(context_retained or context == 'writer_mean'),
                        'token_receipt_sha256': digest(token_receipt),
                        'P_tensor_sha256': self.P_hashes.get(meta['layer']) if space == 'projected' else None,
                        'P_asset_index': meta['layer'] - 4 if space == 'projected' else None})
            self.per_request.write(per_request_rows)
            del ck, cm, summary

    def _e1_compare(self, keys, means, target, meta, ids, artifact):
        import torch
        reference_key = (meta['panel'], meta['layer'])
        if meta['state'] == 0:
            self.e1_refs[reference_key] = artifact['path']
            return
        reference = torch.load(self.e1_refs[reference_key], map_location='cpu', weights_only=True, mmap=True)
        if reference['case_ids'] != list(ids):
            raise GeometryError('W0 paired comparison IDs mismatch')
        baseline_keys, baseline_means = _numpy(reference['context_keys']), _numpy(reference['writer_mean'])
        results = {}
        for space in ('raw', 'projected'):
            if space == 'raw':
                ak, am, bk, bm = baseline_keys, baseline_means, keys, means
            else:
                p = _projector(self.rt, meta['layer'])
                ak, am = project_keys(baseline_keys, p), project_keys(baseline_means, p)
                bk, bm = project_keys(keys, p), project_keys(means, p)
            results[space] = {'contexts': [compare_keys(ak[:, j], bk[:, j], request_ids_a=ids, request_ids_b=ids) for j in range(6)],
                              'writer_mean': compare_keys(am, bm, request_ids_a=ids, request_ids_b=ids)}
        self.artifacts.append(save(target / f'L{meta["layer"]}-paired-W0.json', results))

    def _features(self, features, target, phase, state, condition, panel):
        for name, value in features.items():
            if '/' not in name:
                raise GeometryError(f'unknown feature naming: {name}')
            layer_name, point = name.split('/', 1)
            feature = _numpy(value)
            if feature.ndim != 3 or feature.shape[1] != 6 or len(feature) > 1000:
                raise GeometryError(f'feature shape mismatch: {name}')
            # CPU group mean has a distinct label, never native-writer parity.
            summary = context_geometry(feature, layer=int(layer_name[1:]), space='internal_feature',
                                       metadata={'phase': phase, 'state': state, 'condition': condition,
                                                 'panel': panel, 'feature': point})
            self.artifacts.append(save(target / f'feature-{layer_name}-{point.replace("/", "_")}.json', summary))


def _restore_lower(runtime, mask):
    import torch
    with torch.no_grad():
        for layer, use_checkpoint in zip((4, 5, 6), mask):
            if not use_checkpoint:
                runtime.weights[layer].copy_(runtime.base_weights[layer])


def _upper_control(runtime, baseline, ids, byid, output, state):
    import torch
    runtime.set_state(state)
    with torch.no_grad():
        for layer in (7, 8):
            runtime.weights[layer].copy_(runtime.base_weights[layer])
    capture = output.capture(ids, byid, features=False, label=f'E2-W{state}-upper-control')
    comparisons = {}
    for layer in (6, 7):
        a, b = _numpy(_pick(baseline['keys'], layer)), _numpy(_pick(capture['keys'], layer))
        comparisons[str(layer)] = {'bitwise_equal': bool(np.array_equal(a, b)),
                                   'max_abs': float(np.max(np.abs(a.astype(np.float64) - b))),
                                   'baseline_sha256': tensor_sha(_cpu_tensor(a)),
                                   'control_sha256': tensor_sha(_cpu_tensor(b))}
    output.artifacts.append(save(output.root / 'E2' / f'W{state:03d}' / 'upper-control-parity.json', comparisons))
    if not all(c['bitwise_equal'] for c in comparisons.values()):
        raise GeometryError('upper-layer negative control changed K6/K7 in identical capture kernel')
    return capture


def _cross_gate_up(base_features, current_features, output, state):
    """W0/current gate x up crossing: attribution, not a weight-only model."""
    from scipy.special import expit
    for layer in LAYERS:
        for gate_state, up_state in itertools.product((0, state), repeat=2):
            gates = base_features if gate_state == 0 else current_features
            ups = base_features if up_state == 0 else current_features
            gate, up = _numpy(_gate_up(gates, layer, 'gate')), _numpy(_gate_up(ups, layer, 'up'))
            if gate.shape != up.shape:
                raise GeometryError('gate/up cross shape mismatch')
            # FP32 diagnostic combination. This must not be interpreted as
            # native torch SiLU bitwise parity; the actual captured key is kept.
            mixed = np.empty_like(gate)
            for first in range(0, len(gate), 64):
                part = gate[first:first + 64]
                mixed[first:first + 64] = (part * expit(part)) * up[first:first + 64]
            summary = context_geometry(mixed, layer=layer, space='cross_gate_up', metadata={
                'gate_state': gate_state, 'up_state': up_state, 'state_pair': [0, state],
                'diagnostic_only': True, 'CPU_FP32_silu_not_native_bitwise': True,
                'not_weight_realizable_claim': True})
            output.artifacts.append(save(output.root / 'E2' / f'W{state:03d}' / 'gate-up-cross' /
                                         f'L{layer}-g{gate_state}-u{up_state}.json', summary))
            del mixed


def run_geometry(rt, panels, byid, out):
    """Execute exactly E1(28) and E2(18) families in one admitted runtime.

    All eight masks get calibration512 and assessment3488 (four fixed872
    cohorts). Assessment context tensors are *summarized*, not retained; all
    writer means and calibration context keys are retained. No state checkpoint.
    runtime.set_state must independently restore approved W/M/RNG before each
    cell. A final W0 restore also runs on exceptions. This does not resume cells
    based on existing files: a new attempt/output directory is required.
    """
    checked = validate_panels(panels, byid)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    output = _Output(rt, root)
    started = time.monotonic()
    save(root / 'plan.json', {
        'schema': 'alpha-geometry-run-v1', 'E1_families': 28, 'E2_families': 18,
        'states': list(STATES), 'hybrid_states': list(HYBRID_STATES),
        'hybrid_masks': [''.join(map(str, x)) for x in itertools.product((0, 1), repeat=3)],
        'hybrid_layer_order': [4, 5, 6], 'mask0': 'W0', 'mask1': 'checkpoint',
        'panel_sha256': checked['panel_sha256'], 'N_P_inputs': 'NOT_USED',
        'E1_context_and_means_saved': True, 'E2_calibration_context_and_means_saved': True,
        'E2_assessment_context_saved': False, 'E2_assessment_means_saved': True,
        'feature_tensors_saved': False, 'new_W_M_checkpoints': False,
        'assessment_contrasts': 'all8masks+upperControl; no outcome-selected contrast',
        'exact_geometry_split': 'per1000E1/per512calibration/per872assessment; no3488Gram',
    })
    restoration = 'NOT_OBSERVED'
    try:
        for state in STATES:
            rt.set_state(state)
            for name in COHORTS:
                ids = checked['cohorts'][name]['case_ids']
                capture = output.capture(ids, byid, features=False, label=f'E1-W{state}-{name}')
                output.analyze(capture, ids, phase='E1', state=state, condition='native', panel=name, save_context=True)
                del capture
        cal_ids = checked['calibration_ids']
        rt.set_state(0)
        base = output.capture(cal_ids, byid, features=True, label='E2-W0-calibration-feature-anchor')
        # Persist statistics but only preserve necessary gate/up RAM references.
        output._features(base['features'], root / 'E2' / 'W000' / 'calibration-anchor', 'E2', 0, 'native', 'calibration512')
        base_features = {f'L{layer}/{point}': _gate_up(base['features'], layer, point)
                         for layer in LAYERS for point in ('gate', 'up')}
        del base
        for state in HYBRID_STATES:
            native_calibration = None
            for mask in itertools.product((0, 1), repeat=3):
                rt.set_state(state)
                _restore_lower(rt, mask)
                condition = ''.join(map(str, mask))
                capture = output.capture(cal_ids, byid, features=True, label=f'E2-W{state}-{condition}-calibration')
                output.analyze(capture, cal_ids, phase='E2', state=state, condition=condition,
                               panel='calibration512', save_context=True, features=True)
                if mask == (1, 1, 1):
                    _cross_gate_up(base_features, capture['features'], output, state)
                    native_calibration = {'keys': capture['keys'], 'means': capture['means']}
                del capture
                for cohort in COHORTS:
                    ids = checked['cohorts'][cohort]['assessment_case_ids']
                    capture = output.capture(ids, byid, features=False, label=f'E2-W{state}-{condition}-{cohort}')
                    output.analyze(capture, ids, phase='E2', state=state, condition=condition,
                                   panel=f'assessment-{cohort}', save_context=False)
                    del capture
            control = _upper_control(rt, native_calibration, cal_ids, byid, output, state)
            output.analyze(control, cal_ids, phase='E2', state=state, condition='upper-control', panel='calibration512', save_context=True)
            del control, native_calibration
            for cohort in COHORTS:
                ids = checked['cohorts'][cohort]['assessment_case_ids']
                capture = output.capture(ids, byid, features=False, label=f'E2-W{state}-upper-control-{cohort}')
                output.analyze(capture, ids, phase='E2', state=state, condition='upper-control',
                               panel=f'assessment-{cohort}', save_context=False)
                del capture
        _save_csv(root / 'state_cohort_geometry.csv', output.rows)
        save(root / 'cost-breakdown.json', {'events': output.costs, 'nested_timer_policy': 'capture runtime_seconds nested; do not add'})
        output.close(complete=True)
        save(root / 'artifact-index.json', output.artifacts)
        save(root / 'cell-receipts.json', output.cells)
    except BaseException as exc:
        output.close(complete=False)
        save(root / 'failure.json', {'status': 'TECHNICAL_FAILED', 'type': type(exc).__name__, 'message': str(exc),
                                     'completed_panels': output.cells, 'seconds': time.monotonic() - started})
        raise
    finally:
        try:
            rt.set_state(0)
            restoration = 'W0_APPROVED_INPUT_RESTORED'
        except BaseException as exc:
            save(root / 'restore-failure.json', {'status': 'TECHNICAL_FAILED', 'type': type(exc).__name__, 'message': str(exc)})
            raise
    terminal = {'status': 'COMPLETED', 'E1_families': 28, 'E2_families': 18,
                'completed_panel_observations': len(output.cells), 'seconds': time.monotonic() - started,
                'final_restore': restoration, 'new_checkpoint_saved': False, 'no_N_P_selection': True,
                'NOT_APPLICABLE': ['N-based_basis_selection', 'method_claim', 'native_solver_replacement']}
    save(root / 'terminal.json', terminal)
    return terminal
