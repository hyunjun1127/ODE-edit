"""Small synthetic checks for plotting semantics; no model or raw input."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from . import plots as p


def fixture():
    final, current, seen, cohorts, layers, batches, normalization, costs = ([] for _ in range(8))
    for alias in p.MODELS:
        for a, arm in enumerate(p.ARMS):
            base = dict(alias=alias, arm=arm)
            row = dict(base)
            for metric, factor in zip(p.METRICS, (1, 2, 10)):
                row.update({metric+'_num': (700+a*30)*factor, metric+'_den': 1000*factor})
            final.append(row)
            for batch in range(1, 11):
                row = dict(base, batch=batch)
                for metric, factor in zip(p.METRICS, (1, 2, 10)):
                    row.update({metric+'_num': (70+a*3)*factor, metric+'_den': 100*factor})
                current.append(row)
                if batch in (1, 5, 10):
                    seen.append({k: (v*batch if k.endswith('_num') or k.endswith('_den') else v)
                                 for k, v in row.items()})
                for cohort in range(1, batch+1):
                    cohorts.append(dict(base, batch=batch, cohort=cohort,
                                        canonical_denominator=100, current_success=80))
                for layer in range(4, 9):
                    layers.append(dict(base, batch=batch, layer=layer, batch_net_norm=layer+batch*.1))
                measured = 0.5 if a else 'NOT_RECORDED'
                batches.append(dict(base, batch=batch, signed_predicted_progress=measured,
                    raw_native_work=measured, final_V_ratio=measured, barrier_end_minus_entry=measured,
                    barrier_positive_increment_count=measured))
                normalization.append(dict(base, batch=batch, qN_ref=measured,
                                          minimum_active_scale=2., maximum_active_scale=4.))
            costs.append(dict(base, process_seconds=3600, peak_gpu_bytes=2**30))
    return final, current, seen, cohorts, layers, batches, normalization, costs


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)


class PlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_panel_order_and_title_no_weight_reference(self):
        data = fixture()
        fig = p.weight_magnitude(data[4])
        self.assertEqual(fig._suptitle.get_text(), 'Layer-wise Update Magnitude')
        self.assertEqual([a.get_title() for a in fig.axes],
                         [m+' · '+a for m in p.MODEL_NAMES for a in p.ARM_NAMES])
        for ax in fig.axes:
            self.assertEqual(len(ax.lines), 5)
            self.assertEqual([line.get_label() for line in ax.lines], ['L4', 'L5', 'L6', 'L7', 'L8'])
            self.assertTrue(all(len(line.get_xdata()) == 10 for line in ax.lines))
        text = ' '.join(t.get_text().lower() for t in fig.findobj(match=matplotlib_text))
        self.assertNotIn('bars:', text)
        self.assertNotIn('ideal', text)

    def test_missing_cumulative_is_not_interpolated(self):
        fig = p.performance_trajectory(fixture()[2], True)
        for ax in fig.axes:
            self.assertEqual(len(ax.lines), 3)
            for line in ax.lines:
                self.assertEqual(list(line.get_xdata()), [1, 5, 10])
                self.assertEqual(line.get_linestyle(), 'None')

    def test_retention_row_count_and_mask(self):
        cohorts = fixture()[3]
        self.assertEqual(len(cohorts), 330)
        fig = p.retention_heatmaps(cohorts)
        for ax in fig.axes[:6]:
            arr = ax.images[0].get_array()
            self.assertEqual(np.count_nonzero(~arr.mask), 55)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            p.retention_heatmaps(cohorts+[cohorts[0]])

    def test_primary_success_not_teacher_accuracy(self):
        rows = fixture()[0]
        for row in rows:
            row['rewrite_acc'] = 0.
        fig = p.final_performance(rows)
        self.assertEqual([x.get_height() for x in fig.axes[0].patches], [70., 73., 76.])

    def test_create_and_byte_stability(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); perf = root/'performance'; mech = root/'mechanism'
            perf.mkdir(); mech.mkdir()
            for directory, filename, rows in [
                (perf, 'main_six_arm_table.csv', data[0]), (perf, 'current_batch_metrics.csv', data[1]),
                (perf, 'seen_prefix_metrics.csv', data[2]), (perf, 'retention_cohort_metrics.csv', data[3]),
                (mech, 'batch_layer_actions.csv', data[4]), (mech, 'batch_mechanism.csv', data[5]),
                (mech, 'normalization.csv', data[6]), (mech, 'cost_by_arm.csv', data[7])]:
                write_csv(directory/filename, rows)
            output = root/'out'
            manifest = p.build(perf, mech, output)
            self.assertEqual(manifest['figure_count'], 9)
            self.assertEqual(len(list(output.glob('*.png'))), 9)
            self.assertEqual(json.loads((output/'plot-manifest.json').read_text())['seed'], p.SEED)
            for fig in manifest['figures']:
                self.assertTrue(fig['byte_stable_rerender'])
                self.assertEqual(p.sha_bytes((output/fig['output']).read_bytes()), fig['output_sha256'])
            with self.assertRaises(FileExistsError):
                p.build(perf, mech, output)


def matplotlib_text(value):
    from matplotlib.text import Text
    return isinstance(value, Text)


if __name__ == '__main__':
    unittest.main()
