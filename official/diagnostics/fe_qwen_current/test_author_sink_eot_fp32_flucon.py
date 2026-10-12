"""CPU only.  Scoped hooks, FP32 restore, tracking parent, final binding; and on rent, every preparation step on the
actual S (FP32 CF sink_region) W20 run before the source-on-main check: the independent re-audit, stream/reference/
assets binding and the FP32 checkpoint payload validation (no model, no GPU, no generation)."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from official.diagnostics.fe_qwen_current import flucon as base
from official.diagnostics.fe_qwen_current import author_sink_eot_fp32_flucon as v
from official.experiments.prepare import digest
from official.runners.server1.common import read, verify, member

RUN = Path('/workspace/checkpoints/fe-qwen-working-20261012/author-sink-eot-fp32-cf-r1')
COLLECTOR = Path('/workspace/envs/fe-qwen-current-py312-20261012/bin/python')
REFERENCE = Path('/workspace/data/fe-qwen-working-20261012/generation-reference/reference-binding.json')


class Hooks(unittest.TestCase):
    def test_scoped_substitution(self):
        before = {n: getattr(base, n) for n in ('TASK', 'audit_final', 'validate_payload', 'restore', 'load_reference', 'tracking_values')}
        with v.bindings():
            self.assertIs(base.audit_final, v.audit_final);self.assertIs(base.restore, v.restore)
            self.assertIs(base.validate_payload, v.validate_payload);self.assertEqual(base.EDITING_PRECISION, 'FP32_UNCHANGED')
        self.assertEqual({n: getattr(base, n) for n in before}, before);self.assertFalse(hasattr(base, 'EDITING_PRECISION'))

    def test_restore_is_exact_fp32(self):
        seen = {}
        with patch.object(base, 'payload_weights', lambda o: ({'w': torch.zeros(1)}, 'i' * 64)), \
             patch.object(v, 'bind_weights', lambda m, w, o, d: seen.update(dtype=d) or {'w': 's'}):
            out = v.restore(None, dict(checkpoint='cp', identity='id'))
        self.assertIs(seen['dtype'], torch.float32);self.assertEqual(out['cast'], 'NONE_FP32_CHECKPOINT_EXACT')

    def test_tracking_group_and_parent(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'tracking').mkdir();(Path(d) / 'tracking/receipt.json').write_text(json.dumps(
                dict(run_id='7dd69bca29af45b0', job_identity=dict(job_id='130333'))))
            c = dict(original=dict(terminal=dict(path=str(Path(d) / 'benchmark-result.json')),
                                   identity=dict(job_id='130333', eot_policy='sink_region')))
            with patch.object(v, '_tracking', lambda c, s: dict(server='rent', task_id='x', arm='y', attempt='a', source_sha='0' * 40)), \
                 patch.object(base, 'validate_tracking', lambda x: x):
                t = v.tracking_values(c, '0' * 40)
        self.assertEqual((t['task_id'], t['parent_run_id']), ('fe-qwen-author-sink-eot-fp32-20261013', '7dd69bca29af45b0'))
        self.assertEqual(t['arm'], 'AUTHOR-cf-FP32-L2-0-clamp1-sink_regionEOT-W20-flucon')

    def test_final_binding_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'final.json';p.write_text(json.dumps(dict(schema='other')))
            with self.assertRaises(Exception):
                v.audit_final(member(p))


@unittest.skipUnless(RUN.is_dir() and COLLECTOR.is_file() and REFERENCE.is_file(), 'rent S run, collector env, reference')
class RentPreparation(unittest.TestCase):
    def test_actual_S_preparation_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            final = Path(d) / 'final.json'
            final.write_text(json.dumps(dict(schema=v.FINAL_SCHEMA, collect_report=member(RUN / 'collect-report.json'),
                                             collector_python=str(COLLECTOR))))
            with v.bindings():
                audited = base.audit_final(member(final))
                records = read(verify(audited['config']['stream']))
                self.assertEqual((len(records), digest([r['case_id'] for r in records])), (2000, base.ORDER))
                refs = base.load_reference(str(REFERENCE));self.assertEqual(refs.sha, base.REFERENCE)
                assets = read(verify(audited['config']['model_assets']))
                for m in assets['model']['members']:verify(m)
                original = dict(checkpoint=audited['row']['checkpoint'], identity=audited['identity'], pointer=audited['pointer'],
                                commit=audited['commit'], terminal=audited['terminal'], config=audited['row']['config'],
                                hparams=base.HP, job_id=str(audited['identity']['job_id']))
                weights, integrity = base.payload_weights(original)
                self.assertEqual({w.dtype for w in weights.values()}, {torch.float32});self.assertEqual(len(integrity), 64)
                bad = {k: w.to(torch.bfloat16) for k, w in weights.items()}
                with self.assertRaises(Exception):
                    v.validate_payload(dict(torch.load(verify(original['checkpoint']), map_location='cpu', mmap=True,
                                                       weights_only=False), weights=bad), original)


if __name__ == '__main__':
    unittest.main()
