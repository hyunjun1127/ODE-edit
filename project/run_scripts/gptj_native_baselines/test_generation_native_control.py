"""Narrow CPU controls for new source/telemetry/resource binding, not GPU PASS."""
import ast
import copy
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

from project.run_scripts.experiment_tracking import schema
from project.run_scripts.experiment_generation_eval.metrics import PUBLIC_REASONS
from . import generation_native_common as common
from . import generation_native_submit as submit
from . import generation_native_tracking as tracking
from .generation_native_run import arm_configuration


def progress():
    result = {key:0 for key in schema.GENERATION_PROGRESS_METRICS}
    result.update({'phase':'W20_generation', 'generation_progress/total_cases':2000,
        'generation_progress/total_prompts':20000, 'generation_progress/step':1})
    return result


def summary():
    return dict(planned_count=2000, fluency_count=2000, consistency_count=1999,
        ngram_entropy=2.5, reference_score=.25, generation_prompt_count=20000,
        generated_token_count=1800000, missing_reason_counts={key:0 for key in PUBLIC_REASONS})


class NativeControlTests(unittest.TestCase):
    def test_actual_config_bound_no_legacy_generation_gates_science_projection(self):
        config = common.read(common.PREPARATION/'config.json')
        before = copy.deepcopy(config)
        self.assertTrue(common.ready(config))
        for arm in common.ARMS:
            projected = arm_configuration(config, arm)
            self.assertEqual(projected['native'], config['arm_configs'][arm]['native'])
            self.assertEqual(projected['cold_W'], config['arm_configs'][arm]['cold_W'])
            self.assertEqual(projected['task_id'], common.TASK)
        self.assertEqual(config, before)
        self.assertFalse({'repair', 'qualification_plan', 'W0_cache'} & config['generation'].keys())
        self.assertEqual(config['generation']['profile'], common.PROFILE)
        self.assertEqual(config['generation']['generation_schedule'], 'W20_ONLY_FIRST2000')

    def test_count_plan_no_double_generation_noCP(self):
        counts = common.counts()
        self.assertEqual(counts['edit_applications'], 12000)
        self.assertEqual(counts['planned_generation_case_observations'], 12000)
        self.assertEqual(counts['generation_endpoint_calls_per_arm'], 1)
        for key in ('cold_W0_generation_case_observations', 'intermediate_generation_case_observations',
                'qualification_prompt_evaluations', 'additional_generation_per_metric', 'checkpoint_saves'):
            self.assertEqual(counts[key], 0)

    def _graph(self, frontier, cap):
        jobs, graph = {}, {}
        for index, role in enumerate((*common.ARMS, 'collector')):
            graph[role] = submit.dependencies(role, frontier, jobs, cap)
            if role != 'collector': jobs[role] = str(80000+index)
        return jobs, graph

    def test_cap2_protected_memit_and_independent_lane(self):
        jobs, graph = self._graph(['61428'], 2)
        self.assertEqual(graph['BASE_MEMIT'], ['61428'])
        self.assertEqual(graph['BASE_ALPHAEDIT'], [])
        self.assertEqual(graph['CAKE'], [jobs['BASE_MEMIT']])
        self.assertEqual(graph['ALPHAEDIT_BLUE'], [jobs['BASE_ALPHAEDIT']])
        self.assertEqual(graph['PRUNE'], [jobs['CAKE']])
        self.assertEqual(graph['RECT'], [jobs['ALPHAEDIT_BLUE']])
        self.assertEqual(graph['collector'], list(jobs.values()))

    def test_cap2_additional_existing_frontier_both_heads_wait(self):
        for frontier in (['61428','80050'], ['80050'], []):
            _, graph = self._graph(frontier, 2)
            self.assertEqual(graph['BASE_MEMIT'], frontier)
            self.assertEqual(graph['BASE_ALPHAEDIT'], frontier)

    def test_stricter_cap1_single_serial_lane(self):
        jobs, graph = self._graph(['61428','80050'], 1)
        self.assertEqual(graph['BASE_MEMIT'], ['61428','80050'])
        for previous, role in zip(common.ARMS, common.ARMS[1:]):
            self.assertEqual(graph[role], [jobs[previous]])

    def test_scheduler_identity_errors_no_guessed_ids(self):
        for frontier in (['61428','61428'], ['UNKNOWN'], [0]):
            with self.assertRaises((RuntimeError, AttributeError)):
                submit.dependencies('BASE_MEMIT', frontier, {}, 2)
        with self.assertRaises(RuntimeError):
            submit.dependencies('collector', [], {}, 2)
        with self.assertRaises(RuntimeError):
            submit.dependencies('BASE_MEMIT', [], {}, 3)

    def test_launcher_explicit_new_entry_cpu_threads_noCP_no_secret(self):
        config = common.read(common.PREPARATION/'config.json')
        for role in (*common.ARMS,'collector'):
            text = submit.launcher(Path('/fixed/attempt'), role, config, 'CPU_SOURCE_FIXTURE')
            module = 'generation_native_collect' if role=='collector' else 'generation_native_run'
            self.assertIn(module, text)
            for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                self.assertIn('export '+name+'=6', text)
            self.assertNotIn('WANDB_API_KEY', text)
            self.assertNotIn('qualification', text)
            self.assertIn('HF_HUB_OFFLINE=1', text)
            self.assertEqual('CUDA_VISIBLE_DEVICES=' in text, role=='collector')

    def test_real_job_config_native_profile_shared_whitelist(self):
        config = common.read(common.PREPARATION/'config.json')
        lock = dict(source_commit='a'*40, config_sha256='b'*64)
        with patch.dict('os.environ', {'SLURM_JOB_ID':'80000', 'SLURM_ARRAY_JOB_ID':'79999',
                'SLURM_ARRAY_TASK_ID':'0', 'SLURM_STEP_ID':'-1'}, clear=True):
            values = schema.bind_job_identity(tracking.tracking_config(config,lock,'BASE_MEMIT','80000'))
        self.assertEqual(values['job_id'], '80000')
        self.assertEqual(values['job_display_id'], '79999_0')
        self.assertEqual(values['array_task_id'], '0')
        self.assertEqual(values['step_id'], '-1')
        self.assertEqual(values['generation_profile'], common.PROFILE)
        self.assertEqual(values['generation_schedule'], 'W20_ONLY_FIRST2000')
        self.assertEqual(values['generation_repair_instruction'], common.REPAIR_INSTRUCTION)
        self.assertNotIn('qualification', values)

    def test_progress_exact_shared_scalar_api_private_metadata_rejected(self):
        value = progress()
        self.assertEqual(tracking.generation_progress_values(value), value)
        for key, extra in (('route', common.ROUTE), ('job_id', '80000'), ('prompt', 'PRIVATE')):
            with self.assertRaises(RuntimeError):
                tracking.generation_progress_values(dict(value, **{key:extra}))
        invalid = dict(value, phase='W0_generation')
        with self.assertRaises(RuntimeError): tracking.generation_progress_values(invalid)
        invalid = dict(value, **{'generation_progress/completed_cases':1})
        with self.assertRaises(RuntimeError): tracking.generation_progress_values(invalid)

    def test_final_generation_keys_same_scores_single_final_not_midpoint(self):
        result = tracking.generation_values('all_seen/post', summary(), 2000, 1900, 2000)
        self.assertEqual(result['all_seen/post/fluency/ngram_entropy'], 2.5)
        self.assertEqual(result['all_seen/post/consistency/reference_score'], .25)
        self.assertEqual(result['all_seen/post/generation/planned_count'], 2000)
        self.assertEqual(result['edits'], result['post_state_edits'])
        for prefix, edits, post in (('current/post',2000,2000), ('all_seen/post',500,500),
                ('W0_first2000',0,0)):
            with self.assertRaises(RuntimeError):
                tracking.generation_values(prefix, summary(), edits, 0, post)

    def test_submit_contains_no_cancellation_or_science_retry(self):
        source = inspect.getsource(submit)
        literals = {node.value for node in ast.walk(ast.parse(source))
            if isinstance(node,ast.Constant) and isinstance(node.value,str)}
        self.assertFalse({'scancel','requeue','update'} & literals)
        self.assertIn('PROTECTED_MEMIT_COUNTED_IN_FRONTIER', literals)
        self.assertIn('NATIVE_FREE_SECOND_LANE_EXACT_PROOF', literals)
        self.assertIn('scontrol', literals)
        self.assertIn('release', literals)


if __name__ == '__main__': unittest.main()
