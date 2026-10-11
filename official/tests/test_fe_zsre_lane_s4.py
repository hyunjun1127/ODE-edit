"""Server4 lane-B FE zsRE preparation: config derivation, native zsRE rows, lane width (CPU only)."""
import copy
import json
import os
from pathlib import Path
import unittest

from official.baselines.fe_author_profile import INSTRUCTION, METHOD, PROFILE, TASK, profile
from official.experiments.prepare import digest, file_sha
from official.runners import fe_author_history
from official.runners.server1 import submit as control
from official.runners.server4 import fe_zsre_lane as lane
from official.tracking.schema import config as tracking_config


def template_63357():
    c = dict(schema='official-fe-author-history-v1', instruction_id=INSTRUCTION, task_id=TASK, method=METHOD,
             model='llama3', server='server4', dataset='zsre', batch_size=100, batches=20, requests=2000, edit_seed=0,
             qualification='NOT_RUN_USER_DISABLED', milestones=[5, 10, 15, 20], hparams=profile('llama3')['hparams'],
             author_profile_sha256='0'*64, storage_min_free_bytes=8 << 30, arm='zsre-MEMIT_FE_HISTORY_FE_AUTHOR_HPARAMS',
             output='/data/old/runs/zsre', tracking_env_file='/data/old/tracking.env', assets={'path': '/old/assets'},
             stream_member={'path': '/old/stream'}, stream_sha256=lane.ZSRE_RECORDS_DIGEST, runtime={'python': 'x'},
             checkpoint_archive=dict(host='codex-server1', root='/archive'), checkpoint_policy='REMOTE',
             dispatch_instruction='OLD', zsre_query_parity={'path': '/old/parity'},
             history=dict(solve='lambda_C*C0+H_entry+K@K.T'))
    c['config_sha256'] = digest(c)
    return c


def official_record(case_id, target='Paris', neighbors=1):
    return dict(case_id=case_id, occurrence_index=case_id + 1, paraphrase_prompts=['Where is X located?'],
                neighborhood_prompts=[dict(prompt='nq question: who is it?', target='Someone')] * neighbors,
                requested_rewrite=dict(prompt='What city is {} in?', subject='X', target_new=dict(str=target),
                                       target_true=dict(str='<|endoftext|>')),
                attribute_prompts=[], generation_prompts=[])


def author_record(case_id):
    return dict(id=case_id, subject='X', prompt='What city is {} in?', target_new='Paris', target_true='',
                paraphrase_prompts=['Where is X located?'], neighborhood_prompts=['nq question: who is it?'],
                neighborhood_prompts_answers=['Someone'])


class LaneTests(unittest.TestCase):
    def test_llama_config_keeps_63357_science_and_drops_the_remote_archive(self):
        t = template_63357()
        c = lane.llama_config(t, assets={'path': '/new/assets'}, stream={'path': '/new/stream'},
                              runtime_info={'python': 'x'}, output=lane.LOCAL/'llama-zsre-local-r1',
                              tracking_env=lane.LOCAL/'llama-zsre-local-r1/tracking.env', parity={'path': '/new/parity'},
                              template_member={'path': '/old/config', 'sha256': t['config_sha256']},
                              profile_sha256=file_sha(PROFILE))
        self.assertNotIn('checkpoint_archive', c)
        self.assertEqual(c['config_sha256'], digest({k: v for k, v in c.items() if k != 'config_sha256'}))
        changed = {k for k in set(c) | set(t) if c.get(k, lane.MISSING) != t.get(k, lane.MISSING)}
        self.assertLessEqual(changed, lane.LLAMA_OPERATIONAL)
        for key in ('hparams', 'stream_sha256', 'milestones', 'batch_size', 'requests', 'edit_seed', 'history', 'arm'):
            self.assertEqual(c[key], t[key])
        self.assertEqual((c['covariance_policy'], c['rerun_of']['job']), ('strict', '63357'))
        self.assertEqual(c['output'], str(lane.LOCAL/'llama-zsre-local-r1/runs/zsre'))
        fe_author_history.validate_config(c)
        values = tracking_config(fe_author_history.tracking_values(c, {'source_commit': 'a'*40}))
        self.assertEqual((values['server'], values['dataset'], values['model']), ('server4', 'zsre', 'llama3'))
        local = copy.deepcopy(t)
        local.pop('checkpoint_archive')
        with self.assertRaises(KeyError):
            lane.llama_config(local, assets={}, stream={}, runtime_info={}, output=lane.LOCAL, tracking_env='x',
                              parity={}, template_member={}, profile_sha256='x')

    def test_native_zsre_rows_preserve_order_and_equal_author_fields(self):
        records = [official_record(i) for i in range(3)]
        rows = lane.native_zsre_rows(records)
        self.assertEqual([r['case_id'] for r in rows], [0, 1, 2])
        self.assertEqual(rows[0]['target_new'], 'Paris')
        self.assertEqual(rows[0]['neighborhood_prompts'], ['nq question: who is it?'])
        self.assertEqual(rows[0]['neighborhood_prompts_answers'], ['Someone'])
        check = lane.compare_author_zsre(rows, [author_record(i) for i in range(5)])
        self.assertEqual((check['requests'], check['target_true_differs']), (3, 3))
        self.assertFalse(any(check['mismatches'].values()))
        author = [author_record(i) for i in range(3)]
        author[1]['target_new'] = 'Lyon'
        with self.assertRaises(AssertionError):
            lane.compare_author_zsre(rows, author)
        author = [author_record(i) for i in (0, 2, 1)]
        with self.assertRaises(AssertionError):
            lane.compare_author_zsre(rows, author)
        for bad in (official_record(0, target=' Paris'), official_record(0, neighbors=2)):
            with self.assertRaises(AssertionError):
                lane.native_zsre_rows([bad])

    def test_native_overrides_match_the_cf_harness_except_dataset_paths_and_storage(self):
        run = lane.LOCAL/'gptj-zsre-native-r1/runs'/lane.ARM
        overrides = lane.native_overrides(run, lane.LOCAL/'gptj-zsre-native-r1/data')
        self.assertEqual([x for x in overrides if x in lane.FIXED_OVERRIDES], list(lane.FIXED_OVERRIDES))
        self.assertIn('data=' + lane.DATA_NAME, overrides)
        self.assertIn('zsre_mend_eval', lane.DATA_NAME)
        self.assertIn('+diagnostic_history_batches=[20]', overrides)
        self.assertIn(f'+diagnostic_reserve_bytes={8 << 30}', overrides)
        self.assertEqual(len({x.split('=', 1)[0] for x in overrides}), len(overrides))
        need = lane.native_disk_need()
        weights, history = 6*4096*16384*2, 6*16384*16384*4
        self.assertEqual(need, 5*(weights + (16 << 20)) + history + weights + (8 << 30))
        self.assertLess(need, lane.native_disk_need(history_batches=[0, 5, 10, 15, 20]))

    def test_scripts_run_preflight_then_native_main_then_author_metrics(self):
        text = lane.native_script('/venv/bin/python', '/a', '/hf', '/tmp-own', '/run', ['bs=100'], ['test_only=True'], 7)
        lines = text.splitlines()
        self.assertIn('export TMPDIR=/tmp-own', lines)
        self.assertEqual(lines[-3].split()[:3], ['/venv/bin/python', '-B', '-c'])
        self.assertIn('need=7', lines[-3])
        self.assertEqual(lines[-2], '/venv/bin/python -B -u main.py bs=100')
        self.assertEqual(lines[-1], '/venv/bin/python -B -u main.py test_only=True')
        text = lane.evaluation_script('/venv/bin/python', '/src', '/tmp-own', '/c.json', '/out', '/lock.json')
        self.assertTrue(text.splitlines()[-1].startswith('exec /venv/bin/python -B -u -m official.diagnostics.fe_native.evaluate'))

    def test_serial_lane_adds_one_gpu_to_the_server4_frontier(self):
        lane_a = [dict(key='63529', gpus=1, parents=[])]
        self.assertEqual(control.graph_width(lane.lane_rows(lane_a)), 2)
        self.assertEqual(control.graph_width(lane.lane_rows([])), 1)
        two = lane_a + [dict(key='63530', gpus=1, parents=[])]
        self.assertEqual(control.graph_width(lane.lane_rows(two)), 3)
        chained = lane_a + [dict(key='63530', gpus=1, parents=['63529'])]
        self.assertEqual(control.graph_width(lane.lane_rows(chained)), 2)

    def test_repair_and_runtime_comparisons(self):
        repair = dict(policy='project_roundoff', repaired=True, corrected_eigenvalues=6, shape=[4, 4],
                      original_eigenvalues=[-2e-6, 1e-6], eigenvalue_floor=8e-6, matrix_frobenius=67.2,
                      correction_frobenius=1.8e-5)
        devbox = dict(source=dict(sha256='s'), count=5, repair=repair, derived=dict(sha256='a'))
        server4 = dict(copy.deepcopy(devbox), derived=dict(sha256='b'))
        server4['repair']['original_eigenvalues'] = [-2e-6 + 1e-15, 1e-6]
        check = lane.compare_repair(server4, devbox)
        self.assertFalse(check['bitwise_equal'])
        for field, value in (('count', 6), ('source', dict(sha256='t'))):
            bad = dict(copy.deepcopy(server4), **{field: value})
            with self.assertRaises(AssertionError):
                lane.compare_repair(bad, devbox)
        bad = copy.deepcopy(server4)
        bad['repair']['corrected_eigenvalues'] = 5
        with self.assertRaises(AssertionError):
            lane.compare_repair(bad, devbox)
        rows = {'torch': dict(version='2.9.1+cu128', origin='external', files_sha256=None),
                'transformers': dict(version='4.51.3', origin='venv', files_sha256='x')}
        versions = dict(torch='2.9.1+cu128', transformers='4.51.3', hydra='1.3.2', numpy='1.26.4')
        devbox_rt = dict(python='3.12.0', versions=versions, distributions=dict(rows, pip=dict(version='23.2.1')))
        server4_rt = dict(python='3.12.3', versions=versions, distributions=rows)
        self.assertEqual(lane.compare_runtime(server4_rt, devbox_rt)['differences'], [])
        changed = copy.deepcopy(server4_rt)
        changed['distributions']['transformers']['files_sha256'] = 'y'
        self.assertEqual(lane.compare_runtime(changed, devbox_rt)['differences'], ['transformers'])
        changed['versions'] = dict(versions, numpy='2.2.6')
        with self.assertRaises(AssertionError):
            lane.compare_runtime(changed, devbox_rt)

    @unittest.skipUnless(os.environ.get('FE_ZSRE_OFFICIAL_STREAM') and os.environ.get('FE_NATIVE_AUTHOR_ROOT'),
                         'optional official zsRE stream and pinned author checkout')
    def test_actual_official_first2000_equal_author_zsre_rows(self):
        records = json.loads(Path(os.environ['FE_ZSRE_OFFICIAL_STREAM']).read_text())
        self.assertEqual((len(records), digest(records)), (2000, lane.ZSRE_RECORDS_DIGEST))
        author = json.loads((Path(os.environ['FE_NATIVE_AUTHOR_ROOT'])/'data/zsre_mend_eval_19086.json').read_text())
        check = lane.compare_author_zsre(lane.native_zsre_rows(records), author)
        self.assertEqual(check['requests'], 2000)


if __name__ == '__main__':
    unittest.main()
