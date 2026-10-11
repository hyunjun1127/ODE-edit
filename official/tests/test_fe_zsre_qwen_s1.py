"""Server1 Qwen FE zsRE (native harness): 63385 template derivation and serial lane width (CPU only)."""
import json
from pathlib import Path
import unittest

from official.experiments.prepare import digest
from official.runners.server1 import fe_native_zsre_qwen as qwen
from official.runners.server1 import submit as control
from official.runners.server4 import fe_zsre_lane as lane

T = '/fixture/native-r2'
TEMPLATE_OVERRIDES = ['llms=qwen2.5-7b', 'algs=memit', 'bs=100', 'num_edits=2000', 'seed=0', 'data=counterfact_stream_2000',
                      f'data_dir={T}/data', f'cache_dir={T}/runs/qwen25/cache', f'results_dir={T}/runs/qwen25/results',
                      f'zs_cache_dir={T}/runs/qwen25/unused-z-cache', 'save_name=native-sequential-qwen25',
                      'load_name=native-sequential-qwen25', f'+diagnostic_checkpoint_dir={T}/runs/qwen25/checkpoints',
                      '+diagnostic_reserve_bytes=68719476736', f'hydra.run.dir={T}/runs/qwen25/hydra-edit', 'hydra.job.chdir=False']


def template():
    c = dict(model='qwen25', author_commit=lane.AUTHOR_COMMIT, ode_patch_source=qwen.PATCH_SOURCE,
             entrypoint='python main.py', root=T + '/author', runtime=dict(torch='2.9.1+cu128'), python='/venv/bin/python',
             source_members=[dict(path=T + '/author/main.py', sha256='a', bytes=1)], model_assets=dict(path='/assets'),
             covariances=[dict(layer=4, original=dict(path='/c0'), link=T + '/runs/qwen25/cache/stats/x', repair=False)],
             stream=dict(path=T + '/data/cf.json'), original_stream=dict(path='/cf-stream.json'),
             overrides=list(TEMPLATE_OVERRIDES), model_dtype='bfloat16', current_batch_targets_only=True,
             history='native add_old_keys=True', covariance_repair=False, context_override=False, writer_override=False,
             requests=2000, batch_size=100, gpu_authorization='OLD', output=T + '/runs/qwen25', tracking='native stdout')
    c['identity_sha256'] = digest(c)
    return c


class QwenLaneTests(unittest.TestCase):
    def test_overrides_move_only_the_dataset_name_and_run_paths(self):
        mine = qwen.zsre_overrides(TEMPLATE_OVERRIDES, T, '/new/out')
        self.assertEqual([x.split('=', 1)[0] for x in mine], [x.split('=', 1)[0] for x in TEMPLATE_OVERRIDES])
        changed = {a.split('=', 1)[0] for a, b in zip(mine, TEMPLATE_OVERRIDES) if a != b}
        self.assertEqual(changed, {'data', *qwen.PATH_KEYS})
        self.assertIn('data=' + lane.DATA_NAME, mine)
        self.assertIn('cache_dir=/new/out/runs/qwen25/cache', mine)
        self.assertNotIn('+diagnostic_z_reference=batch_entry', mine)
        self.assertNotIn('+diagnostic_history_batches=[20]', ' '.join(mine))
        with self.assertRaises(AssertionError):
            qwen.zsre_overrides(TEMPLATE_OVERRIDES + [f'+other={T}/x'], T, '/new/out')

    def test_spec_changes_only_dataset_paths_and_provenance(self):
        t = template()
        spec, changed = qwen.zsre_spec(t, root='/new/author', output='/new/out/runs/qwen25', dataset='zsre',
                                       overrides=qwen.zsre_overrides(t['overrides'], T, '/new/out'), task_id=qwen.TASK)
        self.assertEqual(spec['identity_sha256'], digest({k: v for k, v in spec.items() if k != 'identity_sha256'}))
        self.assertEqual(set(changed), {'root', 'output', 'dataset', 'overrides', 'task_id', 'identity_sha256'})
        for key in ('model_dtype', 'ode_patch_source', 'covariance_repair', 'runtime', 'python', 'history', 'model_assets'):
            self.assertEqual(spec[key], t[key])
        for bad in (dict(model_dtype='float32'), dict(covariance_repair=True), dict(z_reference='initial_model_on_arrival')):
            with self.assertRaises(AssertionError):
                qwen.zsre_spec(t, **bad)

    def test_serial_chain_keeps_devbox_width_two(self):
        flucon = [dict(key='63537', gpus=1, parents=[]), dict(key='63538', gpus=1, parents=['63537']),
                  dict(key='63539', gpus=0, parents=['63537', '63538'])]
        self.assertEqual(control.graph_width(qwen.chain_rows(flucon)), 2)
        extra = flucon + [dict(key='63600', gpus=1, parents=[])]
        self.assertEqual(control.graph_width(qwen.chain_rows(extra)), 3)

    @unittest.skipUnless((qwen.TEMPLATE/'configs/qwen25.json').is_file() and (qwen.TEMPLATE_EVAL/'qwen25.sh').is_file(),
                         'devbox 63385/63387 templates')
    def test_launchers_reproduce_63385_and_63387_bytes(self):
        c = json.loads((qwen.TEMPLATE/'configs/qwen25.json').read_text())
        root = qwen.TEMPLATE
        actual = c['overrides'] + ['+diagnostic_identity_sha256=' + c['identity_sha256']]
        evaluation = [x for x in actual if not x.startswith('hydra.run.dir=')] + [
            'test_only=True', f'hydra.run.dir={root}/runs/qwen25/hydra-eval']
        self.assertEqual(qwen.edit_script(c['python'], root/'hf-cache', root/'author', actual, evaluation),
                         (root/'qwen25.sh').read_text())
        self.assertEqual(qwen.evaluation_script(c['python'], qwen.TEMPLATE_EVAL/'source', root/'configs/qwen25.json',
                                                qwen.TEMPLATE_EVAL/'results/qwen25', qwen.TEMPLATE_EVAL/'execution-lock.json'),
                         (qwen.TEMPLATE_EVAL/'qwen25.sh').read_text())
        mine = qwen.zsre_overrides(c['overrides'], root, qwen.LOCAL/'qwen-zsre-native-r1')
        self.assertEqual(len(mine), len(c['overrides']))
        self.assertEqual(c['identity_sha256'], qwen.TEMPLATE_IDENTITY)


if __name__ == '__main__':
    unittest.main()
