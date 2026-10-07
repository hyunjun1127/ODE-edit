"""Read-only compact snapshot. No model imports, raw text export, or job actions."""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


METRICS = {'denominator', 'numerator', 'rate', 'true_nll_mean', 'new_nll_mean',
           'desired_token_count', 'desired_token_correct', 'token_micro', 'prompt_macro',
           'strict_numerator', 'strict_denominator', 'new_strict_numerator'}


def describe(values):
    values = sorted(float(v) for v in values)
    assert values and all(math.isfinite(v) for v in values)
    return dict(count=len(values), min=values[0], median=statistics.median(values),
                mean=statistics.mean(values), max=values[-1])


def collect(cells):
    output = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), cells=[])
    for cell in cells:
        row = dict(cell, summaries=[], batches=[], prices=[], fits=[], generation=[], files=[], issues=[])
        root = Path(cell['root'])

        def read(path):
            raw = path.read_bytes()
            value = json.loads(raw)
            row['files'].append(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            return value

        lock = root.parent/'execution.lock.json'
        if lock.exists():
            d = read(lock)
            row['execution'] = {k: d[k] for k in ['source', 'source_commit', 'source_sha', 'config_sha256',
                'source_tree', 'task', 'task_id'] if k in d and isinstance(d[k], (str, int, bool))}
        row['root_exists'] = root.exists()
        terminal = root/'terminal.json'
        if terminal.exists():
            d = read(terminal)
            row['terminal'] = {k: d[k] for k in ['status', 'commits', 'job', 'source', 'arm', 'task',
                'checkpoint_saved', 'seconds', 'peak_RSS_KiB', 'peak_VRAM_bytes'] if k in d}
        error = root/'first-error.json'
        if error.exists():
            d = read(error)
            row['error_type'] = d.get('type')
            # Error text/tracebacks can contain arbitrary inputs: retain only the artifact hash.
        paths = sorted(root.glob('batch-*/pre/summary.json')) + sorted(root.glob('batch-*/post/summary.json'))
        if (root/'W0/summary.json').exists(): paths.append(root/'W0/summary.json')
        for path in paths:
            d = read(path)
            batch = 0 if path.parent.name == 'W0' else int(path.parent.parent.name.split('-')[1])
            phase = 'W0' if batch == 0 else path.parent.name
            expected_requests = 2000 if batch == 0 else batch*100 if phase == 'post' and batch in [5, 10, 15, 20] else 100
            assert d['requests'] == expected_requests, (str(path), d['requests'], expected_requests)
            if 'row_count' in d: assert d['row_count'] == expected_requests*13, str(path)
            assert d['no_mutation'] is True, str(path)
            for group in ['summary', 'current']:
                if group not in d: continue
                metrics = d[group]
                n = expected_requests if group == 'summary' else 100
                # W0 producers may expose the full cohort as current; preserve the actual denominator.
                if batch == 0 and group == 'current': n = metrics['R']['denominator']
                compact = {}
                for kind, factor in [('R', 1), ('P', 2), ('N', 10)]:
                    m = metrics[kind]
                    assert m['denominator'] == n*factor and 0 <= m['numerator'] <= m['denominator'], str(path)
                    assert abs(m['rate']-m['numerator']/m['denominator']) < 1e-12, str(path)
                    assert all(not isinstance(v, float) or math.isfinite(v) for v in m.values()), str(path)
                    compact[kind] = {k: v for k, v in m.items() if k in METRICS}
                scope = 'W0_first2000' if batch == 0 else 'all_seen' if group == 'summary' and phase == 'post' and batch in [5, 10, 15, 20] else 'current'
                if group == 'current' and scope == 'W0_first2000': continue
                if group == 'current' and expected_requests == 100: continue
                row['summaries'].append(dict(batch=batch, phase=phase, scope=scope, requests=n,
                    endpoint=d['endpoint'], metrics=compact, source_path=str(path), source_sha256=row['files'][-1]['sha256']))
        for folder in sorted(root.glob('batch-*')):
            batch = int(folder.name.split('-')[1])
            commit = folder/'commit.json'
            if commit.exists():
                d = read(commit)
                row['batches'].append({k: d[k] for k in ['batch', 'seen_requests', 'source', 'config', 'seconds',
                    'observer_no_mutation', 'post_scope', 'history_appends', 'native_counts', 'fit_count',
                    'accepted_candidate', 'checkpoint_saved'] if k in d})
            price = folder/'entry-price.json'
            if price.exists():
                d = read(price)['payload']; layers=d['layers']; prices=d['effective_pi']
                assert len(layers)==len(prices)
                cheapest=collections.Counter()
                for column in zip(*prices):
                    for layer, v in zip(layers, column):
                        if v == min(column): cheapest[str(layer)] += 1
                row['prices'].append(dict(batch=batch, layers=layers,
                    effective_pi={str(l):describe(v) for l,v in zip(layers,prices)},
                    cheapest_layer_count_ties_included=dict(cheapest),
                    beta_base=describe(d['beta_base']), beta_max=describe(d['beta_max']),
                    price_extra_model_calls=d.get('price_extra_model_calls'), price_extra_solves=d.get('price_extra_solves')))
            fit = folder/'fit.json'
            if fit.exists():
                d=read(fit); controller=d.get('controller',{})
                row['fits'].append(dict(batch=batch,status=d.get('status'),seconds=d.get('seconds'),
                    terminal_states=dict(collections.Counter(d.get('terminal_states',[]))),
                    expansion_counts=dict(collections.Counter(map(str,controller.get('expansion',[])))),
                    request_updates=d.get('request_updates'),candidates=d.get('candidates')))
        gen=root/'generation'
        if gen.exists():
            # Endpoint summaries have complete observation guards. Per-case partial output is counted only.
            row['partial_generation_observation_files']=len(list((gen/'observations').glob('*.json')))
            for path in sorted(gen.rglob('*.json')):
                if 'observations' in path.relative_to(gen).parts: continue
                d=read(path)
                if 'summary' in d:
                    assert d.get('RNG_restored') is True and d.get('observer_no_mutation') is True
                    row['generation'].append(dict(source_path=str(path),source_sha256=row['files'][-1]['sha256'],
                        identity_sha256=d.get('identity_sha256'),endpoint=d.get('identity',{}).get('endpoint'),
                        cohort=d.get('identity',{}).get('cohort'),summary=d['summary']))
        output['cells'].append(row)
    output['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    return output
