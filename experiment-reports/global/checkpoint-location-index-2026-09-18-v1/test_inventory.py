"""Inventory bookkeeping tests only; no model or remote access."""
import csv
import hashlib
import json
from pathlib import Path
import re
import unittest

from build_index import HERE, INDEX_ROLES, family, role
from checkpoint_inventory import candidate


def row(path, keys=''):
    return {'path': path, 'schema_keys': keys, 'category': 'UNCLASSIFIED_TENSOR'}


class InventoryTests(unittest.TestCase):
    def test_weight(self):
        self.assertEqual(role(row('/x/B010/W-method-state.pt', 'weights|rng')), 'WEIGHT_STATE_INDICATORS')

    def test_rsync_partial(self):
        self.assertTrue(candidate('.state-01000.pt.2xsQeF'))
        self.assertEqual(role(row('/x/.state-01000.pt.2xsQeF', 'weights')), 'PARTIAL_OR_TEMP')

    def test_model_index_not_weight(self):
        self.assertEqual(role(row('/x/model.safetensors.index.json')), 'SIDECAR_METADATA')

    def test_fixture_not_checkpoint(self):
        self.assertEqual(role(row('/x/cpu-fixtures/state.pt', 'weights')), 'TEST_FIXTURE')

    def test_delta_not_complete_weight(self):
        self.assertEqual(role(row('/x/actual-increments.pt', 'weights|M')), 'DELTA_OR_RECONSTRUCTION_COMPONENT')

    def test_native_capsule_separate(self):
        self.assertEqual(role(row('/x/own-N4.pt', 'W|M')), 'PREPARED_OR_NATIVE_CAPSULE')

    def test_archive_family(self):
        p = '/data/janghj/ODE-edit/local/checkpoint-archives/v1/payload/local/blue-lifelong/B010/W.pt'
        self.assertEqual(family(p), 'blue-lifelong')

    def test_nested_local_family(self):
        self.assertEqual(family('/data/janghj/ODE-edit/local/local-z/technical/local/own-N4.pt'), 'local-z')

    def test_target_snapshot(self):
        self.assertEqual(role(row('/x/snapshots/Zpm/h-0.2.pt')), 'TARGET_KEY_OR_Z_ARTIFACT')

    def test_totals(self):
        with (HERE/'checkpoint-index.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len({(r['server'], r['path']) for r in rows}))
        self.assertTrue(all(r['role'] in INDEX_ROLES for r in rows))
        with (HERE/'server-summary.csv').open() as f:
            summaries = list(csv.DictReader(f))
        self.assertEqual(sum(int(r['indexed_paths']) for r in summaries), len(rows))

    def test_migration_counts(self):
        with (HERE/'migration-status.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(sum(r['migration'] == '2026-09-11' for r in rows), 183)
        self.assertEqual(sum(r['migration'] == '2026-09-18' for r in rows), 34)
        self.assertTrue(all('NOT_CURRENT_REHASH' in r['verification_scope'] for r in rows))

    def test_report_tables_and_links(self):
        text = (HERE/'checkpoint-location-report-ko.md').read_text()
        width = None
        for line in text.splitlines():
            if line.startswith('|'):
                columns = len(line.split('|'))
                if width is None:
                    width = columns
                self.assertEqual(columns, width)
            else:
                width = None
        for ref in re.findall(r'\]\(([^)]+)\)', text):
            self.assertTrue((HERE/ref).is_file(), ref)

    def test_manifest(self):
        manifest = json.loads((HERE/'report-manifest.json').read_text())
        for r in manifest['members']:
            b = (HERE/r['path']).read_bytes()
            self.assertEqual(len(b), r['bytes'])
            self.assertEqual(hashlib.sha256(b).hexdigest(), r['sha256'])


if __name__ == '__main__':
    unittest.main()
