"""Read-only failed-attempt inventory and tiny frozen-controller serialization repro.

No model import/load, scheduler calls, CUDA, or production edits. Writes only
new analysis receipts. Partial JSON is never repaired into a completion receipt.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.attempt / 'source'))
    import numpy as np
    from project.run_scripts.en_adaptive_nullspace.geometry import build_geometry
    from project.run_scripts.en_adaptive_nullspace.controller import run_controller
    inventory = []
    paths = list((a.attempt / 'output').rglob('*'))
    paths += list((a.attempt / 'logs').glob('*'))
    paths += [a.attempt / x for x in ('execution.lock.json', 'submission.json', 'held-inspection.json')]
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        row = dict(path=str(path), relative_path=str(path.relative_to(a.attempt)),
                   bytes=path.stat().st_size, sha256=sha(path))
        if path.suffix == '.json':
            try:
                json.loads(path.read_text())
                row['json'] = 'VALID'
            except json.JSONDecodeError as e:
                row['json'] = 'INVALID_PARTIAL'
                row['parse_error'] = str(e)
        inventory.append(row)
    geo = build_geometry(np.eye(2), [.5, .5], [0, 1], np.eye(2))
    wn = np.array([[.1, .2]], dtype=np.float32)
    g = np.array([[1., 0.]])
    def objective(w):
        x = float(w[0, 0] - wn[0, 0])
        return {'J': 1. + x + 2*x*x, 'L_R': 1. + x + 2*x*x, 'L_H': 0.}
    result = run_controller(wn, g, np.array([[-1., 0.]]), {'J': 1.}, objective,
                            geo, native_norm=2., native_action=100., input_identity='CPU_REPRO_ONLY')
    receipt = {k: v for k, v in result.items() if k != 'weight'}
    try:
        json.dumps(receipt, allow_nan=False)
        raise AssertionError('Expected unmodified source serialization failure')
    except TypeError as e:
        error = repr(e)
    numpy_scalars = []
    def walk(v, path='$'):
        if isinstance(v, dict):
            for k, x in v.items(): walk(x, path+'.'+k)
        elif isinstance(v, (list, tuple)):
            for k, x in enumerate(v): walk(x, path+f'[{k}]')
        elif isinstance(v, np.generic):
            numpy_scalars.append(dict(path=path, type=type(v).__module__+'.'+type(v).__name__, value=v.item()))
    walk(receipt)
    # Test a serialization-only proposal, NOT a production repair.
    def scalar_only(v):
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    encoded = json.dumps(receipt, allow_nan=False, default=scalar_only)
    json.loads(encoded)
    negatives = {}
    for name, value in [('nan', float('nan')), ('inf', float('inf')), ('tensor_array', np.zeros((2,2)))]:
        try:
            json.dumps({'value': value}, allow_nan=False, default=scalar_only)
            negatives[name] = 'WRONGLY_ACCEPTED'
        except (TypeError, ValueError):
            negatives[name] = 'REJECTED'
    assert all(x == 'REJECTED' for x in negatives.values())
    raw = (a.attempt/'output/B1/EN_EXACT-controller.json').read_text()
    # Only already-complete leading values; never infer the truncated ledger's
    # trailing accepted/status/evaluations/selected_trial fields.
    leading, _ = json.JSONDecoder().raw_decode(raw[raw.index('"objective":')+len('"objective":'):].lstrip())
    trial_pos = raw.index('"trial":') + len('"trial":')
    first_trial, _ = json.JSONDecoder().raw_decode(raw[trial_pos:].lstrip())
    trial_obj_pos = raw.index('"objective":', trial_pos) + len('"objective":')
    trial_obj, _ = json.JSONDecoder().raw_decode(raw[trial_obj_pos:].lstrip())
    out = dict(numpy=np.__version__, frozen_source=str(a.attempt/'source'),
               error=error, numpy_scalars=numpy_scalars,
               synthetic_controller_status=result['status'],
               cpu_fixture_only=True, model_loaded=False, gpu_used=False,
               serialization_only_proposal_roundtrip='PASS', negative_cases=negatives,
               partial_actual=dict(valid_receipt=False, leading_return_objective_J=leading['J'],
                   leading_return_objective_seconds=leading.get('seconds'),
                   first_trial=first_trial, first_trial_objective_J=trial_obj['J'],
                   first_trial_seconds=trial_obj.get('seconds'),
                   selected_trial='NOT_RECORDED', status='NOT_RECORDED',
                   warning='Complete leading JSON values only; not a scientific seal or commit.'))
    a.destination.mkdir(parents=True, exist_ok=True)
    for name, value in [('raw-inventory.json', inventory), ('cpu-reproduction.json', out)]:
        with (a.destination/name).open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
