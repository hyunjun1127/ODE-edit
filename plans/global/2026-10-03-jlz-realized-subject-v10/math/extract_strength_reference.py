"""Historical v9 diagnostic only; not a v10 calibration target or run prerequisite."""
import hashlib
import json
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[4]
    snapshot = root / 'local/jlz-v9-deep-review/20261003-r1/snapshot/attempt-r2'
    arms = {}
    for arm in ['A', 'B']:
        p = snapshot / f'main-{arm}/batch-01/fit/candidate-25.json'
        raw = p.read_bytes()
        j = json.loads(raw)
        layers = j['layer']
        means = {l: statistics.mean(x['realized_rho']) for l, x in layers.items()}
        counts = {l: len(x['realized_rho']) for l, x in layers.items()}
        assert len(set(counts.values())) == 1
        all_values = [v for x in layers.values() for v in x['realized_rho']]
        arms[arm] = dict(
            source=str(p.relative_to(root)),
            sha256=hashlib.sha256(raw).hexdigest(),
            candidate=j['candidate'],
            completed_updates_before=j['completed_updates_before'],
            B=next(iter(counts.values())),
            eligible_layers=list(layers),
            mean_by_layer=means,
            mean=statistics.mean(all_values),
        )
    result = dict(
        schema='jlz-v10-v9-B1-strength-reference-v1',
        role='historical_diagnostic_only_not_calibration_target',
        calibration_withdrawn=True,
        definition='mean over layer/request of ||Y_l[:,r]||/a_l,r; ideal FP64 Y=R P.T K_actual',
        context_norm_mean=False,
        source_commit='b8c4c96fef79d3ad6ba37b1f6f4055bd58343e66',
        arms=arms,
    )
    out = Path(__file__).with_name('v9-b1-strength-reference.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
