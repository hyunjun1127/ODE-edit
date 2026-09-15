"""Synthetic aggregate plotting: deterministic bytes, labels and source guard."""
import csv
from pathlib import Path
import tempfile
import unittest

from project.run_scripts.single_layer_zflow.plots import render


def csv_file(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)


class PlotTests(unittest.TestCase):
    def test_same_aggregate_produces_exact_png_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'aggregates'; source.mkdir()
            rows = []
            for tag, den in [('RS', 100), ('PS', 200), ('NS', 1000)]:
                for index in range(1, 11):
                    rows.append(dict(metric=tag, scope='current', batch_index=index,
                                     numerator=den//2, denominator=den, percent=50.))
                for scope in ('seen-full', 'historical-n4'):
                    rows.append(dict(metric=tag, scope=scope, batch_index=10,
                                     numerator=5*den, denominator=10*den, percent=50.))
            csv_file(source / 'metrics.csv', rows)
            csv_file(source / 'batch.csv', [dict(batch_index=i, accepted=1, rejected=2,
                actual_delta_cost_fp64=.125, flow_seconds=1.5) for i in range(1, 11)])
            first = render(source, root / 'first'); second = render(source, root / 'second')
            self.assertEqual([r['sha256'] for r in first['outputs']], [r['sha256'] for r in second['outputs']])
            self.assertEqual(first['inputs'], second['inputs'])
            self.assertEqual(first['model_evaluator_GPU_calls'], 0)
            with self.assertRaisesRegex(ValueError, 'create-once'):
                render(source, root / 'first')


if __name__ == '__main__':
    unittest.main()
