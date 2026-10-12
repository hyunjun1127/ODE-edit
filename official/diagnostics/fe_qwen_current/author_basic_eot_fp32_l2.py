"""FP32 / author numeric settings / current-entry z / basic EOT, with the author's own ridge `algs.L2` from a
pre-registered grid, on the held-out CF selection slice or the original main streams.

Reuses `author_basic_eot_fp32` unchanged (run, observer, writer observation, checkpoint, collector) and patches only
its stream table, settings audit, tracking and identity for the duration of one call:

* `algs.L2` must equal the binding's L2, which must be in L2_GRID.  Every other author numeric/mode field is audited
  by the original FP32 audit (the L2 field is compared as 0 there, i.e. exactly the author value except L2).
* split `heldout`: CF only; the stream's case_id order must equal fixed10k[4000:6000] of the pinned fixed10k file and
  be disjoint from main [0:2000] and the earlier alternative [2000:4000].  The stream file identity is recorded.
* split `main`: the original CF / zsRE main streams pinned by the FP32 module.

Selection (`select`): among held-out W20 reports, arms meeting W20 Eff>=95, Gen>=85, Loc>=70 are ranked by the
unrounded W20 Score; ties go to the smaller L2.  No arm meeting the gate means no selection.  The grid was fixed
on 2026-10-13 (janghj-7d) before any FP32+EOT held-out result existed.
"""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import author_basic_eot_fp32 as base
from official.experiments.prepare import digest, write_new
from official.runners.server1.common import read, verify, member, require

SCHEMA = 'author-basic-eot-fp32-l2-v1'
PROFILE = 'FE_AUTHOR_HPARAMS_CURRENT_Z_BASIC_EOT_FP32_L2'
L2_GRID = (10.0, 100.0)
GATE = dict(Efficacy=95.0, Generalization=85.0, Specificity=70.0)
FIXED10K = (16679956, '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1')
HELDOUT_REFERENCE = 'heldout4000-6000'


def _l2(b):
    value = float(b['L2'])
    require(value in L2_GRID, 'L2_NOT_PREREGISTERED')
    return value


def _heldout_rows_check(b):
    """Case-id order of the held-out stream == fixed10k[4000:6000]; disjoint from [0:4000]."""
    src = b['heldout_source']
    require((src['bytes'], src['sha256']) == FIXED10K, 'FIXED10K_SOURCE')
    fixed = read(verify(src))
    ids = lambda rows: [r['case_id'] for r in rows]
    stream = read(verify(b['main_stream']))
    require(ids(stream) == ids(fixed[4000:6000]), 'HELDOUT_ORDER_FIXED10K_4000_6000')
    require(not set(ids(stream)) & set(ids(fixed[:4000])), 'HELDOUT_DISJOINT_FROM_0_4000')
    return dict(order_sha256=digest(ids(stream)), main_0_2000_sha256=digest(ids(fixed[:2000])),
                alt_2000_4000_sha256=digest(ids(fixed[2000:4000])), stream=b['main_stream'])


def _patches(b):
    l2 = _l2(b)
    split = b['split']
    require(split in ('heldout', 'main'), 'SPLIT')
    streams, refs = dict(base.STREAMS), dict(base.REFERENCES)
    heldout = None
    if split == 'heldout':
        require(b['dataset'] == 'cf', 'HELDOUT_IS_CF_ONLY')
        heldout = _heldout_rows_check(b)
        streams['cf'] = (b['main_stream']['bytes'], b['main_stream']['sha256'])
        refs['cf'] = HELDOUT_REFERENCE
    original_audit, original_tracking, original_identity = base.audit_settings, base.tracking, base.identity

    def audit(cfg, profiles):
        actual = OmegaConf.to_container(cfg, resolve=True)
        require(float(actual['algs']['L2']) == l2, 'NATIVE_L2_EQUALS_BINDING')
        check = OmegaConf.create(actual)
        check.algs.L2 = 0
        original_audit(check, profiles)
        return actual

    def tracking(bb, env=None):
        values = original_tracking(bb, env)
        values.update(task_id='fe-qwen-author-basic-eot-fp32-l2-20261013',
                      arm=f'AUTHOR-{bb["dataset"]}-{split}-FP32-L2-{l2:g}-clamp1-basicEOT',
                      baseline='FE-author-hparams-current-z-basicEOT-FP32-L2', evaluation_profile=PROFILE + '-' + split)
        return values

    def identity(bb, resolved):
        value = original_identity(bb, resolved)
        value.update(stage=PROFILE, L2=l2, split=split, l2_grid=list(L2_GRID), variant=member(__file__),
                     heldout=heldout)
        return value

    return [patch.object(base, 'SCHEMA', SCHEMA), patch.object(base, 'STREAMS', streams),
            patch.object(base, 'REFERENCES', refs), patch.object(base, 'audit_settings', audit),
            patch.object(base, 'tracking', tracking), patch.object(base, 'identity', identity),
            patch.object(base, 'PROFILE', PROFILE)]


def _with(b, fn, *args):
    with ExitStack() as stack:
        for p in _patches(b):
            stack.enter_context(p)
        return fn(*args)


def validate(bindings):
    b = read(bindings)
    return _with(b, base.validate, b)


def run(bindings, output):
    return _with(read(bindings), base.run, bindings, output)


def collect(result, output):
    proof = read(result)
    b = read(verify(proof['bindings']))
    return _with(b, base.collect, result, output)


def select(arm_outputs, output):
    """Each arm output folder holds tracking-finish.json; a W20 arm also holds collect-report.json (this module's
    collect).  QUALITY_EARLY_STOP is a gate failure; any other non-W20 status blocks the selection."""
    rows = []
    for folder in map(Path, arm_outputs):
        runtime = read(folder / 'runtime.json')
        ident = runtime['identity']
        require(ident.get('stage') == PROFILE and ident.get('split') == 'heldout', 'HELDOUT_ARM')
        l2 = float(ident['L2'])
        require(l2 in L2_GRID, 'L2_NOT_PREREGISTERED')
        finish = read(folder / 'tracking-finish.json')
        if finish['status'] == base.STATUS:
            rep = read(folder / 'collect-report.json')
            require(rep['status'] == 'AUTHOR_BASIC_EOT_FP32_W20_CPU_AUDITED', 'HELDOUT_AUDITED_W20')
            require(read(verify(rep['result']))['identity'] == ident, 'REPORT_RUNTIME_IDENTITY')
            w = rep['W20']
            rows.append(dict(L2=l2, folder=str(folder), status='W20', report=member(folder / 'collect-report.json'),
                             W20=w, gate=all(float(w[k]) >= v for k, v in GATE.items())))
        elif finish['status'] == 'QUALITY_EARLY_STOP':
            rows.append(dict(L2=l2, folder=str(folder), status='QUALITY_EARLY_STOP', W20=None, gate=False,
                             finish=member(folder / 'tracking-finish.json')))
        else:
            require(False, 'TECHNICAL_FAILURE_BLOCKS_SELECTION')
    require(sorted(r['L2'] for r in rows) == sorted(L2_GRID), 'ALL_PREREGISTERED_ARMS')
    eligible = [r for r in rows if r['gate']]
    chosen = max(eligible, key=lambda r: (float(r['W20']['Score']), -r['L2']))['L2'] if eligible else None
    out = dict(status='SELECTED' if chosen is not None else 'NO_SELECTION', selected_L2=chosen, gate=GATE,
               rule='gate then max unrounded W20 Score, tie smaller L2', arms=rows, variant=member(__file__))
    write_new(Path(output), out)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('validate'); r.add_argument('bindings')
    r = sub.add_parser('run'); r.add_argument('bindings'); r.add_argument('output')
    r = sub.add_parser('collect'); r.add_argument('result'); r.add_argument('output')
    r = sub.add_parser('select'); r.add_argument('output'); r.add_argument('arm_outputs', nargs='+')
    a = p.parse_args()
    if a.command == 'validate':
        cfg, resolved, rows = validate(a.bindings)
        print(json.dumps(dict(status='VALID', rows=len(rows), L2=resolved['algs']['L2'])))
    elif a.command == 'run':
        print(run(a.bindings, a.output))
    elif a.command == 'collect':
        print(collect(a.result, a.output))
    else:
        print(json.dumps(select(a.arm_outputs, a.output)))
