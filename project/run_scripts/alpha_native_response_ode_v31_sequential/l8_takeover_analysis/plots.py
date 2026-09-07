"""Deterministic CPU-only plotting of sealed scalar review tables.

No runtime/raw paths are accepted. Missing values remain missing: in particular,
cumulative PS/NS are drawn only at the three actually evaluated checkpoints.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np

MODELS = ('llama3-8b-inst', 'qwen2.5-7b-inst')
ARMS = ('O_NATIVE', 'JV_NATIVE', 'L8_ONLY_NATIVE')
MODEL_NAMES = ('Llama', 'Qwen')
ARM_NAMES = ('Official O', 'JV', 'L8-only')
METRICS = ('RS', 'PS', 'NS')
ARM_COLORS = ('#657b83', '#0072b2', '#d55e00')
LAYER_COLORS = ('#0072b2', '#009e73', '#e69f00', '#cc79a7', '#d55e00')
DPI = 150
SEED = 20260907
MISSING_POLICY = 'NOT_RECORDED remains missing; no interpolation, replacement, or imputation.'
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
         'axes.labelsize': 9, 'figure.titlesize': 15, 'axes.spines.top': False,
         'axes.spines.right': False, 'axes.grid': False, 'savefig.facecolor': 'white',
         'figure.facecolor': 'white', 'svg.hashsalt': 'alpha-jv-l8-review-v1'}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def number(row, key):
    value = row.get(key, '')
    if value is None or value == '' or str(value).startswith('NOT_'):
        return np.nan
    result = float(value)
    if not np.isfinite(result):
        raise ValueError('NONFINITE_SEALED_SCALAR: ' + key)
    return result


def select(rows, alias, arm):
    return [r for r in rows if r['alias'] == alias and r['arm'] == arm]


def ordered_unique(rows, alias, arm, keys=('batch',)):
    rr = select(rows, alias, arm)
    ids = [tuple(int(r[k]) for k in keys) for r in rr]
    if len(ids) != len(set(ids)):
        raise ValueError('DUPLICATE_PLOT_KEY')
    return sorted(rr, key=lambda r: tuple(int(r[k]) for k in keys))


def axes_grid(title, cols=3, figsize=(13.2, 7.2)):
    fig, axs = plt.subplots(2, cols, figsize=figsize, squeeze=False, layout='constrained')
    fig.suptitle(title)
    return fig, axs


def legend_arms(fig):
    handles = [plt.Line2D([], [], color=c, marker='o', label=n)
               for c, n in zip(ARM_COLORS, ARM_NAMES)]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False)


def final_performance(rows):
    fig, axs = axes_grid('Final W10: all 1,000 edited requests')
    for i, alias in enumerate(MODELS):
        for j, key in enumerate(METRICS):
            ax = axs[i, j]
            for a, arm in enumerate(ARMS):
                rr = select(rows, alias, arm)
                if len(rr) != 1:
                    raise ValueError('FINAL_SIX_ARM_CARDINALITY')
                r = rr[0]; n, d = int(r[key+'_num']), int(r[key+'_den'])
                ax.bar(a, 100*n/d, color=ARM_COLORS[a], width=.65)
                ax.text(a, 100*n/d+2, f'{n:,}/{d:,}', ha='center', va='bottom', fontsize=8)
            ax.set(title=f'{MODEL_NAMES[i]} · {key}', ylabel='Canonical success (%)', ylim=(0, 112))
            ax.set_xticks(range(3), ARM_NAMES)
    return fig


def performance_trajectory(rows, cumulative):
    title = ('Cumulative performance at frozen W1 / W5 / W10: all seen requests'
             if cumulative else 'Current B100 immediately after each batch write')
    fig, axs = axes_grid(title)
    for i, alias in enumerate(MODELS):
        for j, key in enumerate(METRICS):
            ax = axs[i, j]
            for a, arm in enumerate(ARMS):
                rr = ordered_unique(rows, alias, arm)
                x = np.asarray([int(r['batch']) for r in rr])
                y = np.asarray([100*number(r, key+'_num')/number(r, key+'_den') for r in rr])
                # Cumulative missing checkpoints are not connected/interpolated.
                ax.plot(x, y, color=ARM_COLORS[a], marker='o', linestyle='none' if cumulative else '-',
                        label=ARM_NAMES[a], markersize=5 if cumulative else 3)
            ax.set(title=f'{MODEL_NAMES[i]} · {key}', xlabel='Completed sequential batch',
                   ylabel='Canonical success (%)', ylim=(-2, 102), xlim=(.65, 10.35))
            ax.set_xticks([1, 5, 10] if cumulative else range(1, 11))
    legend_arms(fig)
    return fig


def retention_heatmaps(rows):
    fig, axs = axes_grid('Rewrite retention: cohort B1–B10 under each subsequent W')
    image = None
    for i, alias in enumerate(MODELS):
        for a, arm in enumerate(ARMS):
            rr = ordered_unique(rows, alias, arm, ('batch', 'cohort'))
            matrix = np.full((10, 10), np.nan)
            for r in rr:
                b, c = int(r['batch']), int(r['cohort'])
                if not 1 <= c <= b <= 10:
                    raise ValueError('RETENTION_COHORT_BOUNDARY')
                matrix[c-1, b-1] = 100*number(r, 'current_success')/number(r, 'canonical_denominator')
            if np.count_nonzero(np.isfinite(matrix)) != len(rr):
                raise ValueError('RETENTION_ROW_COUNT')
            cmap = matplotlib.colormaps['viridis'].copy(); cmap.set_bad('#eeeeee')
            image = axs[i, a].imshow(np.ma.masked_invalid(matrix), origin='upper',
                                     vmin=0, vmax=100, cmap=cmap, interpolation='none')
            axs[i, a].set(title=f'{MODEL_NAMES[i]} · {ARM_NAMES[a]}',
                         xlabel='Evaluation W after batch', ylabel='Original edit cohort')
            axs[i, a].set_xticks(range(10), range(1, 11)); axs[i, a].set_yticks(range(10), range(1, 11))
    fig.colorbar(image, ax=axs.ravel().tolist(), label='Rewrite success (%; 100 requests/cohort)', shrink=.8)
    return fig


def weight_magnitude(rows):
    fig, axs = axes_grid('Layer-wise Update Magnitude')
    for i, alias in enumerate(MODELS):
        for a, arm in enumerate(ARMS):
            rr = ordered_unique(rows, alias, arm, ('batch', 'layer'))
            ax = axs[i, a]
            for layer, color in zip(range(4, 9), LAYER_COLORS):
                part = [r for r in rr if int(r['layer']) == layer]
                ax.plot([int(r['batch']) for r in part], [number(r, 'batch_net_norm') for r in part],
                        color=color, marker='o', markersize=3, label=f'L{layer}')
            ax.set(title=f'{MODEL_NAMES[i]} · {ARM_NAMES[a]}', xlabel='Sequential batch',
                   ylabel=r'$\|W_{b,l}-W_{b-1,l}\|_F$', xlim=(.65, 10.35))
            ax.set_xticks(range(1, 11)); ax.set_ylim(bottom=0)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False)
    return fig


def mechanism_curves(rows, title, columns):
    fig, axs = axes_grid(title, len(columns), (4.4*len(columns), 7.2))
    for i, alias in enumerate(MODELS):
        for j, (key, label) in enumerate(columns):
            ax = axs[i, j]
            for a, arm in enumerate(ARMS):
                rr = ordered_unique(rows, alias, arm)
                x = [int(r['batch']) for r in rr]; y = [number(r, key) for r in rr]
                if np.isfinite(y).any():
                    ax.plot(x, y, color=ARM_COLORS[a], marker='o', markersize=3)
                else:
                    ax.text(.02, .97-.055*a, ARM_NAMES[a]+': NOT_RECORDED',
                            transform=ax.transAxes, va='top', color=ARM_COLORS[a], fontsize=7)
            ax.set(title=f'{MODEL_NAMES[i]}\n{label}', xlabel='Sequential batch',
                   ylabel=label, xlim=(.65, 10.35))
            ax.set_xticks(range(1, 11))
    legend_arms(fig)
    return fig


def cost_plot(rows):
    fig, axs = axes_grid('Recorded compute: cross-server descriptive comparison', 2, (10, 7.2))
    for i, alias in enumerate(MODELS):
        for j, (key, label, factor) in enumerate([
                ('process_seconds', 'Process wall time (hours)', 1/3600),
                ('peak_gpu_bytes', 'Recorded peak GPU memory (GiB)', 1/2**30)]):
            ax = axs[i, j]
            values = []
            for arm in ARMS:
                rr = select(rows, alias, arm)
                if len(rr) != 1:
                    raise ValueError('COST_SIX_ARM_CARDINALITY')
                values.append(number(rr[0], key)*factor)
            ax.bar(range(3), values, color=ARM_COLORS, width=.65)
            ax.set(title=MODEL_NAMES[i], ylabel=label, ylim=(0, max(values)*1.1))
            ax.set_xticks(range(3), ARM_NAMES)
    return fig


def encode(builder, rows, rendering=None):
    with plt.rc_context(STYLE):
        np.random.seed(SEED)
        fig = builder(rows)
        if rendering is not None:
            rendering.update(figure_size_inches=[float(x) for x in fig.get_size_inches()],
                axes=[dict(title=ax.get_title(), xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(),
                           xlim=[float(x) for x in ax.get_xlim()], ylim=[float(x) for x in ax.get_ylim()],
                           xscale=ax.get_xscale(), yscale=ax.get_yscale()) for ax in fig.axes])
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=DPI,
                    metadata={'Software': 'ODE-edit deterministic l8_takeover_analysis.plots/v1'})
        plt.close(fig)
        return buffer.getvalue()


def build(performance, mechanism, output):
    performance, mechanism, output = map(Path, (performance, mechanism, output))
    jobs = [
        ('final_six_arm_performance', performance/'main_six_arm_table.csv', final_performance,
         'Final W10 fixed endpoint: six arms; RS 1,000 rewrite, PS 2,000 paraphrase, NS 10,000 neighborhood prompts per arm. Strict NLL-pair inequalities; ties fail.'),
        ('current_batch_performance', performance/'current_batch_metrics.csv', lambda r: performance_trajectory(r, False),
         '60 current-batch endpoints; each point only the just-written B100: RS 100, PS 200, NS 1,000 prompts. Not cumulative performance.'),
        ('cumulative_seen_prefix_performance', performance/'seen_prefix_metrics.csv', lambda r: performance_trajectory(r, True),
         '18 frozen-W seen-prefix endpoints, B1/B5/B10 only: requests=100/500/1,000; RS=n, PS=2n, NS=10n. Markers only, no interpolation at unrecorded checkpoints.'),
        ('six_arm_rewrite_retention', performance/'retention_cohort_metrics.csv', retention_heatmaps,
         '330 observed cohort×evaluation-batch cells; 100 rewrite requests in every cohort. Grey cells are not-yet-edited cohorts, not failed observations.'),
        ('layer_wise_update_magnitude', mechanism/'batch_layer_actions.csv', weight_magnitude,
         '300 physical batch×layer observations: ten sequential B100 writes per arm, five layers. Absolute Frobenius magnitude of actual materialized batch delta, not squared magnitude or per-request samples. No ideal/equal-share reference.'),
        ('signed_progress_and_native_work', mechanism/'batch_mechanism.csv',
         lambda r: mechanism_curves(r, 'Recorded signed predicted progress and native work', [
             ('signed_predicted_progress', 'Signed predicted progress'), ('raw_native_work', 'Raw native work')]),
         '60 batch rows; only recorded dynamic-arm values are drawn. Signed predicted progress is controller-predicted, not an actual activation-realization ratio. O dynamics are NOT_RECORDED; work and progress have distinct units.'),
        ('v_and_barrier_diagnostics', mechanism/'batch_mechanism.csv',
         lambda r: mechanism_curves(r, 'V/V0 and barrier diagnostic: recorded dynamic arms', [
             ('final_V_ratio', 'Final V / entry V'), ('barrier_end_minus_entry', r'$\Delta(V+\lambda E)$'),
             ('barrier_positive_increment_count', r'Positive $\Delta(V+\lambda E)$ count')]),
         'Ten batch summaries per dynamic arm, four configured nodes per batch; Official O lacks these dynamics. Delta(V+lambda E)>0 means reserve b=V0-V-lambda E decreases; it is not a positive reserve change. Positive increments are telemetry, not an exclusion rule. No causal or barrier-benefit conclusion follows from these curves.'),
        ('n0_normalization', mechanism/'normalization.csv',
         lambda r: mechanism_curves(r, 'Frozen batch-entry N0 normalization diagnostics', [
             ('qN_ref', 'Entry qN reference'), ('minimum_active_scale', 'Minimum active scale'),
             ('maximum_active_scale', 'Maximum active scale')]),
         '60 batch-entry normalization rows. Min/max are across the active requests of that batch; qN reference unavailable for Official O remains NOT_RECORDED. Batch-specific N0 values are not refreshed within its trajectory.'),
        ('compute_comparison', mechanism/'cost_by_arm.csv', cost_plot,
         'Six arm-level recorded process-time/peak-memory values. O/JV executed on Server2, L8 on Server4; hardware/concurrency are uncontrolled. Wall-time differences are descriptive, not algorithm-only overhead.'),
    ]
    output.mkdir(parents=True, exist_ok=False)
    command = ('/data/janghj/EasyEdit/.venv/bin/python -m '
               'project.run_scripts.alpha_native_response_ode_v31_sequential.l8_takeover_analysis.plots '
               f'--performance {performance} --mechanism {mechanism} --output <NEW_OUTPUT_DIR>')
    manifest = dict(schema='l8-takeover-deterministic-plots/v1', seed=SEED, dpi=DPI,
                    backend='Agg', style=STYLE, model_order=MODELS, arm_order=ARMS,
                    matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                    source=dict(path=str(Path(__file__).resolve()), sha256=sha_bytes(Path(__file__).read_bytes())),
                    command=command, missing_policy=MISSING_POLICY, figures=[])
    for name, source, builder, caption in jobs:
        content = source.read_bytes(); rows = read_csv(source)
        rendering = {}
        first = encode(builder, rows, rendering); second = encode(builder, rows)
        if first != second:
            raise ValueError('PLOT_BYTE_STABILITY_FAILURE: '+name)
        destination = output/(name+'.png')
        with destination.open('xb') as f:
            f.write(first)
        manifest['figures'].append(dict(name=name, output=destination.name,
            output_sha256=sha_bytes(first), output_bytes=len(first), caption=caption,
            input_path=str(source), input_sha256=sha_bytes(content), input_rows=len(rows),
            byte_stable_rerender=True, command=command, rendering=rendering, missing_policy=MISSING_POLICY))
    manifest['figure_count'] = len(jobs)
    payload = json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)+'\n'
    with (output/'plot-manifest.json').open('x') as f:
        f.write(payload)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--performance', type=Path, required=True)
    parser.add_argument('--mechanism', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.performance, args.mechanism, args.output)
    print(json.dumps({'figure_count': result['figure_count'], 'byte_stable': True}))
