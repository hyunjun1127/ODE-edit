"""Deterministic aggregate-only figures; no model/evaluator imports or calls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .analysis import member, require, _write_json


def render(aggregates: str | Path, output: str | Path) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'figure.dpi': 120, 'savefig.dpi': 120, 'axes.grid': True,
        'grid.alpha': .22, 'axes.spines.top': False, 'axes.spines.right': False})
    aggregates, output = Path(aggregates).absolute(), Path(output).absolute()
    require(not output.exists(), 'plot output must be create-once')
    sources = [member(aggregates / name) for name in ('metrics.csv', 'batch.csv')]
    with (aggregates / 'metrics.csv').open() as stream:
        metrics = list(csv.DictReader(stream))
    with (aggregates / 'batch.csv').open() as stream:
        batches = list(csv.DictReader(stream))
    require(len(batches) == 10 and [int(r['batch_index']) for r in batches] == list(range(1, 11)),
            'figures require verified complete ten batches')
    output.mkdir(parents=True, mode=0o700)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), constrained_layout=True)
    for column, tag in enumerate(('RS', 'PS', 'NS')):
        current = [r for r in metrics if r['metric'] == tag and r['scope'] == 'current']
        require(len(current) == 10, 'plot current inventory')
        x = [int(r['batch_index']) * 100 for r in current]
        axes[0, column].plot(x, [float(r['percent']) for r in current], marker='o', label='Current B100')
        axes[0, column].set(title=tag + ': online current cohort', xlabel='Processed request index',
                           ylabel='Preference success (%)', ylim=(0, 100))
        sl = [r for r in metrics if r['metric'] == tag and r['scope'] == 'seen-full' and r['batch_index'] == '10']
        n4 = [r for r in metrics if r['metric'] == tag and r['scope'] == 'historical-n4']
        require(len(sl) == 1 and len(n4) <= 1, 'plot terminal inventory')
        entries = n4 + sl
        labels = (['N4 (historical)'] if n4 else []) + ['SL-ZFlow']
        bars = axes[1, column].bar(labels, [float(r['percent']) for r in entries], color=['#737373', '#2877ae'][-len(entries):])
        for bar, row in zip(bars, entries):
            axes[1, column].annotate(row['numerator'] + '/' + row['denominator'],
                (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0, 4),
                textcoords='offset points', ha='center', fontsize=8)
        axes[1, column].set(title=tag + ': final W10, same 1000 requests',
                           ylabel='Preference success (%)', ylim=(0, 108))
    fig.suptitle('SL-ZFlow: online cohort scores are not final retention\nN4 reused; host/backend parity is not claimed', fontsize=11)
    fig.savefig(output / 'quality.png', metadata={'Software': 'SL_ZFLOW_CODE_PLOT_V1'})
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    xs = list(range(1, 11))
    axes[0].bar(xs, [int(r['accepted']) for r in batches], label='Accepted')
    axes[0].bar(xs, [int(r['rejected']) for r in batches], bottom=[int(r['accepted']) for r in batches], label='Rejected')
    axes[0].set(title='Oracle candidates (initial call excluded)', ylabel='Calls', xlabel='Batch')
    axes[0].legend()
    axes[1].plot(xs, [float(r['actual_delta_cost_fp64']) for r in batches], marker='o')
    axes[1].set(title='Stored FP32 delta: actual native cost', ylabel='C', xlabel='Batch')
    times = [r['flow_seconds'] for r in batches]
    if all(t not in ('', 'None') for t in times):
        axes[2].plot(xs, [float(t) for t in times], marker='o')
    else:
        axes[2].text(.5, .5, 'Interrupted timing: NOT_RECORDED', ha='center', transform=axes[2].transAxes)
    axes[2].set(title='Edit flow only (not total/evaluation)', ylabel='Seconds', xlabel='Batch')
    fig.savefig(output / 'work.png', metadata={'Software': 'SL_ZFLOW_CODE_PLOT_V1'})
    plt.close(fig)
    require(sources == [member(aggregates / name) for name in ('metrics.csv', 'batch.csv')], 'plot inputs changed')
    receipt = dict(format='SL_ZFLOW_CODE_PLOT_V1', inputs=sources, code=member(Path(__file__).resolve()),
        outputs=[member(output / name) for name in ('quality.png', 'work.png')],
        matplotlib=matplotlib.__version__, model_evaluator_GPU_calls=0,
        command='python -m project.run_scripts.single_layer_zflow.plots --aggregates ' + str(aggregates) + ' --output ' + str(output))
    _write_json(output / 'plot-receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregates', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.aggregates, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
