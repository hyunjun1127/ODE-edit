import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import author_basic_eot_fp32 as base
from official.diagnostics.fe_qwen_current import author_basic_eot_fp32_l2 as v
from official.runners.server1.common import member


class L2VariantTest(unittest.TestCase):
    def bindings(self, l2=10.0, split='main', dataset='cf'):
        return dict(L2=l2, split=split, dataset=dataset, main_stream=dict(bytes=1, sha256='0' * 64))

    def test_grid_and_split(self):
        with self.assertRaises(Exception):
            v._patches(self.bindings(l2=1.0))
        with self.assertRaises(Exception):
            v._patches(self.bindings(split='other'))
        with self.assertRaises(Exception):
            v._patches(self.bindings(split='heldout', dataset='zsre'))

    def test_audit_passes_l2_zero_to_original_and_requires_binding_l2(self):
        seen = []
        cfg = OmegaConf.create(dict(algs=dict(L2=100.0, add_old_keys=True), llms=dict(clamp_norm_factor=1), model_dtype='float32'))
        with patch.object(base, 'audit_settings', lambda c, p: seen.append(OmegaConf.to_container(c))):
            patches = v._patches(self.bindings(l2=100.0))
            with patches[3]:
                resolved = base.audit_settings(cfg, None)
        self.assertEqual(seen[0]['algs']['L2'], 0)
        self.assertEqual(seen[0]['llms'], {'clamp_norm_factor': 1})
        self.assertEqual(resolved['algs']['L2'], 100.0)
        with patch.object(base, 'audit_settings', lambda c, p: None):
            patches = v._patches(self.bindings(l2=10.0))
            with patches[3], self.assertRaises(Exception):
                base.audit_settings(cfg, None)

    def test_tracking_and_identity_labels(self):
        with patch.object(base, 'tracking', lambda b, env=None: dict(task_id='x', arm='y')), \
             patch.object(base, 'identity', lambda b, r: dict(stage='old', L2=0)):
            patches = v._patches(self.bindings(l2=10.0))
            with patches[4], patches[5]:
                t = base.tracking(self.bindings(l2=10.0))
                i = base.identity(self.bindings(l2=10.0), {})
        self.assertEqual(t['arm'], 'AUTHOR-cf-main-FP32-L2-10-clamp1-basicEOT')
        self.assertEqual(i['L2'], 10.0)
        self.assertEqual(i['stage'], v.PROFILE)
        self.assertEqual(i['split'], 'main')

    def arm(self, root, l2, status, w20=None):
        d = Path(root) / f'L2-{l2:g}'
        d.mkdir()
        ident = dict(stage=v.PROFILE, split='heldout', L2=l2)
        (d / 'runtime.json').write_text(json.dumps(dict(identity=ident)))
        (d / 'tracking-finish.json').write_text(json.dumps(dict(status=status)))
        if w20 is not None:
            res = d / 'benchmark-result.json'
            res.write_text(json.dumps(dict(identity=ident)))
            (d / 'collect-report.json').write_text(json.dumps(dict(status='AUTHOR_BASIC_EOT_FP32_W20_CPU_AUDITED',
                                                                   result=member(res), W20=w20)))
        return str(d)

    def test_select_gate_score_and_stop(self):
        with tempfile.TemporaryDirectory() as root:
            a = self.arm(root, 10.0, base.STATUS, dict(Efficacy=99, Generalization=95, Specificity=72, Score=88.0))
            b = self.arm(root, 100.0, base.STATUS, dict(Efficacy=99, Generalization=94, Specificity=75, Score=88.9))
            out = v.select([a, b], Path(root) / 'sel.json')
            self.assertEqual(out['selected_L2'], 100.0)
        with tempfile.TemporaryDirectory() as root:
            a = self.arm(root, 10.0, 'QUALITY_EARLY_STOP')
            b = self.arm(root, 100.0, base.STATUS, dict(Efficacy=99, Generalization=94, Specificity=69.9, Score=88.9))
            out = v.select([a, b], Path(root) / 'sel.json')
            self.assertEqual(out['status'], 'NO_SELECTION')
        with tempfile.TemporaryDirectory() as root:
            a = self.arm(root, 10.0, 'TECHNICAL_FAILURE')
            b = self.arm(root, 100.0, base.STATUS, dict(Efficacy=99, Generalization=94, Specificity=75, Score=88.9))
            with self.assertRaises(Exception):
                v.select([a, b], Path(root) / 'sel.json')


if __name__ == '__main__':
    unittest.main()
