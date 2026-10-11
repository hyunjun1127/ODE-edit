"""Plot native FE controls from saved logs and common-evaluation receipts only."""
import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_log(path):
    if not path.exists():
        return dict(path=str(path), prefix_bytes=0, prefix_sha256=None, completed_batches={})
    # Capture one immutable prefix even if a monitored job is still writing.
    data = path.read_bytes()
    data = data[:data.rfind(b'\n') + 1]
    batches = {}
    batch = layer = None
    for line in data.decode(errors='replace').splitlines():
        match = re.match(r'NATIVE_FE_BATCH_START (\d+) ', line)
        if match:
            batch = int(match[1])
            layer = None
            batches[batch] = dict(complete=False, layers={})
        elif line.startswith('NATIVE_FE_BATCH_COMPLETE '):
            batches[int(line.split()[-1])]['complete'] = True
        elif batch is not None:
            match = re.match(r'LAYER (\d+)', line)
            if match:
                layer = match[1]
                batches[batch]['layers'][layer] = {}
            match = re.match(r'(z error|orig norm|upd norm) tensor\(([^,\)]+)', line)
            if match and layer is not None:
                key = {'z error': 'residual', 'orig norm': 'weight', 'upd norm': 'update'}[match[1]]
                batches[batch]['layers'][layer][key] = float(match[2])
    return dict(path=str(path), prefix_bytes=len(data),
                prefix_sha256=hashlib.sha256(data).hexdigest(),
                completed_batches={k: v for k, v in batches.items() if v['complete']})


def read_scores(folder):
    result = []
    complete_path = folder / 'COMPLETE.json'
    complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
    for path in sorted(folder.glob('B*-receipt.json')):
        batch = int(path.name[1:3])
        if batch == 0:
            continue  # W0 is evaluated on 2K, whereas edited milestones use all seen.
        data = path.read_bytes()
        receipt = json.loads(data)
        if complete is not None:
            assert receipt == complete['results'][str(batch)], path
        summary = receipt['summary']
        assert summary['requests'] == batch * 100
        result.append(dict(edits=batch * 100, summary=summary, path=str(path),
                           sha256=hashlib.sha256(data).hexdigest()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output prefix; no extension')
    args = parser.parse_args()
    specs = {
        'GPT-J': ('3', (
            ('Current-model z / original C0', 'native-r2/gptj-63386.out', 'native-eval-r1/results/gptj'),
            ('Current-model z / repaired C0', 'native-causal-r1/gptj_repaired-63389.out', 'native-causal-r1/evaluations/gptj_repaired'),
            ('W0 z on arrival / repaired C0', 'native-causal-r2/gptj_online_w0_repaired-63425.out', 'native-causal-r2/evaluations/gptj_online_w0_repaired'),
        )),
        'Qwen2.5-7B': ('5', (
            ('Current-model z', 'native-r2/qwen25-63385.out', 'native-eval-r1/results/qwen25'),
            ('W0 z, current requests only', 'native-causal-r1/qwen_online_w0-63391.out', 'native-causal-r1/evaluations/qwen_online_w0'),
        )),
    }
    result = dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  matplotlib_version=matplotlib.__version__,
                  scope='One identical ordered CF 2K stream; author main.py and hparams; persistent history',
                  score_scope='Common CF evaluator; all seen requests at each checkpoint; W0 excluded',
                  norm_precision='Displayed native log precision, not recomputed tensor norms',
                  runs={})
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex='col')
    for col, (model, (layer, arms)) in enumerate(specs.items()):
        colors = ('#bd3b36', '#78629a', '#2368a2') if len(arms) == 3 else ('#bd3b36', '#2368a2')
        for color, (label, log_name, eval_name) in zip(colors, arms):
            log = read_log(args.root / log_name)
            scores = read_scores(args.root / eval_name)
            result['runs'][model + ': ' + label] = dict(log=log, common_metrics=scores)
            batches = log['completed_batches']
            x = [b * 100 for b in sorted(batches)]
            y = [batches[b]['layers'][layer]['update'] for b in sorted(batches)]
            axes[0, col].plot(x, y, color=color, marker='.', linewidth=1.6, label=label)
            axes[1, col].plot([s['edits'] for s in scores], [s['summary']['Score'] for s in scores],
                             color=color, marker='o', linewidth=1.6)
        axes[0, col].set_title(model)
        axes[0, col].set_yscale('log')
        axes[0, col].set_ylabel(f'L{layer} update norm (log scale)')
        handles, labels = axes[0, col].get_legend_handles_labels()
        # All measured scores exceed 45%; use the empty lower score region
        # so the legend does not hide the large original-C0 update spikes.
        axes[1, col].legend(handles, labels, loc='lower left', fontsize=8)
        axes[1, col].set_ylabel('CF Score (%)')
        axes[1, col].set_ylim(0, 100)
        axes[1, col].set_xlabel('Number of sequential edits')
        for ax in axes[:, col]:
            ax.set_xlim(0, 2050)
            ax.set_xticks([0, 500, 1000, 1500, 2000])
            ax.grid(alpha=.18)
    complete = all(len(row['log']['completed_batches']) == 20 and len(row['common_metrics']) == 4
                   for row in result['runs'].values())
    result['all_runs_and_evaluations_complete'] = complete
    fig.suptitle('Native FE: numerical stability and retained edit performance'
                 + ('' if complete else ' (in progress)'), fontsize=14)
    fig.text(.5, .015, 'Same ordered CF 2K stream; batch size 100. Score uses all seen requests at each checkpoint.\n'
             'Single-stream comparison; W0 reference is a separately labeled protocol for both models.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        fig.savefig(args.output.with_suffix(suffix), dpi=180)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'runs': {name: {
        'completed_batches': len(row['log']['completed_batches']),
        'evaluated_checkpoints': len(row['common_metrics'])} for name, row in result['runs'].items()}}))


if __name__ == '__main__':
    main()
