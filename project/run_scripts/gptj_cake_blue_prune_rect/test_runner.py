"""CPU metadata/fake runner tests; no model, tensor allocation, GPU or jobs."""
import copy
import shlex
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import collect, common, logic, metrics, native_cake_blue, native_prune_rect, run, submit


class FakeWeight:
    """Physical layout metadata only, never a reconstruction-equivalent tensor."""
    def __init__(self, label, shape=(4096, 16384)):
        self.label, self.shape = label, shape
        self.dtype, self.device, self._version = 'torch.float32', 'cpu', 0

    def data_ptr(self):
        return id(self)


class FakeTransformer:
    def __init__(self):
        self.h = [SimpleNamespace(mlp=SimpleNamespace(fc_out=SimpleNamespace(
            weight=FakeWeight('W' + str(index))))) for index in range(9)]
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(last_hidden_state='already-ln-f-hidden')


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(model_type='gptj')
        self.transformer = FakeTransformer()

    def parameters(self):
        return (block.mlp.fc_out.weight for block in self.transformer.h)

    def named_parameters(self):
        return ((f'transformer.h.{index}.mlp.fc_out.weight', block.mlp.fc_out.weight)
                for index, block in enumerate(self.transformer.h))

    def named_modules(self):
        return [('model', self)]


class FakeReader:
    def __init__(self, values):
        self.values, self.files = values, {}

    def json(self, path):
        return copy.deepcopy(self.values[str(path)])

    def bound(self, row):
        return copy.deepcopy(self.values[row['path']])


def fake_batches(records):
    for index in range(20):
        yield index + 1, records[index * 100:(index + 1) * 100], records[:(index + 1) * 100]


@contextmanager
def fake_chain(commits=1, terminal=None, invalid_second=False):
    """Fabricated metadata tests reducer prefix handling, not scientific rows."""
    with tempfile.TemporaryDirectory(prefix='gptj-fourarm-review-') as folder:
        attempt = Path(folder)
        arm = 'ALPHAEDIT_BLUE'
        out = attempt / arm
        out.mkdir()
        layers = ('3', '8')
        initial = dict(W={layer: 'cold-' + layer for layer in layers},
                       H={layer: 'zero-H-' + layer for layer in layers})
        config = dict(arm_layers={arm: [3, 8]}, cold_W=initial['W'])
        lock = dict(source_commit='fake-frozen-source')
        records = [dict(case_id=index) for index in range(2000)]
        values = {str(out / 'runtime.json'): dict(source=lock['source_commit'], arm=arm,
            initial_history_zero=True, initial_state=initial)}
        (out / 'runtime.json').touch()
        previous = initial
        summary = {kind: dict(denominator=count) for kind, count in dict(R=100, P=200, N=1000).items()}
        measured = logic.expected_counts(arm)
        for number in range(1, commits + 1):
            batch = out / f'batch-{number:02d}'
            batch.mkdir()
            (batch / 'commit.json').touch()
            after = dict(W={layer: f'W{number}-{layer}' for layer in layers},
                         H={layer: f'H{number}-{layer}' for layer in layers})
            receipt = dict(task=common.TASK, arm=arm, batch=number,
                source=lock['source_commit'], config=common.digest(config),
                case_ids=[row['case_id'] for row in records[(number - 1) * 100:number * 100]],
                before=previous, after=after, native_counts=measured,
                native=dict(delta=measured), observer_no_mutation=True,
                checkpoint_saved=False, exact_resume='NOT_AVAILABLE',
                pre=summary, post=summary, post_current=summary, seconds=1.0)
            if invalid_second and number == 2:
                receipt['before'] = initial
            values[str(batch / 'commit.json')] = receipt
            previous = after
        if terminal is not None:
            (out / 'terminal.json').touch()
            values[str(out / 'terminal.json')] = terminal
        reader = FakeReader(values)

        def endpoint(_reader, path, _identities, ids, name, _state, _records):
            if name == 'W0':
                return None
            return dict(rows=[dict(case_id=index) for index in ids], summary=summary,
                        seconds=.1, reference_only=False)

        with ExitStack() as stack:
            stack.enter_context(patch.object(collect, 'batches', side_effect=fake_batches))
            stack.enter_context(patch.object(collect, 'endpoint', side_effect=endpoint))
            stack.enter_context(patch.object(collect, 'reduce_rows', return_value=summary))
            stack.enter_context(patch.object(collect, 'compare_summary'))
            stack.enter_context(patch.object(collect, '_metric_rows', return_value=[]))
            stack.enter_context(patch.object(collect, '_paired_rows', return_value=[]))
            yield reader, attempt, config, lock, arm, records


class ProfileAndLaneTests(unittest.TestCase):
    def test_four_arm_exact_counts(self):
        expected = {
            'CAKE': (100, 6, 6, 6, 6),
            'ALPHAEDIT_BLUE': (200, 2, 2, 2, 2),
            'PRUNE': (100, 6, 0, 6, 0),
            'RECT': (100, 6, 0, 6, 0),
        }
        for arm, counts in expected.items():
            with self.subTest(arm=arm):
                self.assertEqual(logic.expected_counts(arm), dict(zip(collect.COUNT_KEYS, counts)))

    def test_full_history_counts_are_120_40_0_0(self):
        self.assertEqual({arm: 20 * logic.expected_counts(arm)['history_appends'] for arm in common.ARMS},
                         dict(CAKE=120, ALPHAEDIT_BLUE=40, PRUNE=0, RECT=0))

    def test_profile_unknown_arm_is_rejected(self):
        for function in (logic.expected_counts, logic.writer_identity):
            with self.assertRaisesRegex(ValueError, 'UNKNOWN_ARM'):
                function('MEMIT_BLUE')

    def test_two_lanes_preserve_existing_frontier(self):
        frontier, jobs = ['10001', '10002'], dict(CAKE='20001', ALPHAEDIT_BLUE='20002')
        self.assertEqual(logic.lane_dependencies('CAKE', frontier, jobs, cap=2), frontier)
        self.assertEqual(logic.lane_dependencies('ALPHAEDIT_BLUE', frontier, jobs, cap=2), frontier)
        self.assertEqual(logic.lane_dependencies('PRUNE', frontier, jobs, cap=2), ['20001'])
        self.assertEqual(logic.lane_dependencies('RECT', frontier, jobs, cap=2), ['20002'])
        result = logic.lane_dependencies('CAKE', frontier, jobs, cap=2)
        result.append('does-not-mutate-frontier')
        self.assertEqual(frontier, ['10001', '10002'])

    def test_one_lane_serializes_all_four(self):
        jobs = dict(zip(common.ARMS, ('20001', '20002', '20003', '20004')))
        self.assertEqual([logic.lane_dependencies(arm, ['10001'], jobs, cap=1) for arm in common.ARMS],
                         [['10001'], ['20001'], ['20002'], ['20003']])

    def test_lane_cap_and_unknown_arm_are_rejected(self):
        for cap in (0, 3, 4):
            with self.assertRaisesRegex(ValueError, 'TASK_CAP'):
                logic.lane_dependencies('CAKE', [], {}, cap=cap)
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_ARM'):
            logic.lane_dependencies('MEMIT_BLUE', [], {}, cap=2)

    def test_runner_keeps_native_nested_target_schema(self):
        records = [dict(case_id=index, requested_rewrite=dict(prompt='{} says', subject='S',
            target_new={'str': ' leading target'}, target_true={'str': 'true'})) for index in range(100)]
        original = copy.deepcopy(records)
        request_sets = [module.normalize_requests(records)
                        for module in (run, native_cake_blue, native_prune_rect)]
        self.assertEqual(records, original)
        self.assertEqual(request_sets[0], request_sets[1])
        self.assertEqual(request_sets[0], request_sets[2])
        self.assertEqual(request_sets[0][0]['target_new'], {'str': ' leading target'})


class ObservationLayoutTests(unittest.TestCase):
    def test_BLUE_selects_physical_three_eight_only(self):
        model = FakeModel()
        view = metrics.ObservationView(model, (3, 8))
        self.assertEqual(view.sites, (3, 8))
        self.assertEqual(set(view.weights), {3, 8})
        self.assertIs(view.weights[8], model.transformer.h[8].mlp.fc_out.weight)
        self.assertEqual(len(run.nonselected(view)), 7)

    def test_BLUE_history_has_separate_two_slots(self):
        view = metrics.ObservationView(FakeModel(), (3, 8))
        history = {layer: FakeWeight('H' + str(layer), (16384, 16384)) for layer in (3, 8)}
        with patch.object(metrics, 'tensor_sha', side_effect=lambda weight: weight.label):
            self.assertEqual(metrics.state(view, history),
                             dict(W={'3': 'W3', '8': 'W8'}, H={'3': 'H3', '8': 'H8'}))
            self.assertEqual(metrics.state(view, {})['H'], {})
            with self.assertRaisesRegex(RuntimeError, 'ALPHA_HISTORY_LAYERS'):
                metrics.state(view, {3: history[3]})

    def test_six_site_view_keeps_native_fc_out_layout(self):
        view = metrics.ObservationView(FakeModel())
        self.assertEqual(view.sites, common.LAYERS)
        self.assertEqual(set(view.weights), set(common.LAYERS))
        with self.assertRaisesRegex(RuntimeError, 'GPTJ_ARM_PHYSICAL_SITES'):
            metrics.ObservationView(FakeModel(), (3, 4))

    def test_GPT2_layout_is_not_accepted(self):
        model = FakeModel()
        model.transformer.h[3].mlp.fc_out.weight.shape = (16384, 4096)
        with self.assertRaisesRegex(RuntimeError, 'OBSERVER_GPTJ_LINEAR_LAYOUT'):
            metrics.ObservationView(model, (3, 8))

    def test_observer_uses_native_final_ln_once(self):
        model = FakeModel()
        view = metrics.ObservationView(model, (3, 8))
        self.assertEqual(view.observer_hidden(input_ids='metadata-only'), 'already-ln-f-hidden')
        self.assertEqual(model.transformer.calls,
                         [dict(input_ids='metadata-only', use_cache=False)])


class CollectorPrefixAndPrivacyTests(unittest.TestCase):
    def test_partial_committed_prefix_without_terminal_is_kept(self):
        with fake_chain() as (reader, attempt, config, lock, arm, records):
            result = collect.review(reader, attempt, config, lock, arm, {}, records)
        self.assertEqual(result['scientific_status'], 'PARTIAL_OR_FAILED')
        self.assertEqual((result['commits'], result['requests'], result['state_links']), (1, 100, 0))
        self.assertEqual(result['measured_counts'], logic.expected_counts('ALPHAEDIT_BLUE'))

    def test_failed_terminal_does_not_erase_committed_counts_or_leak_error(self):
        terminal = dict(status='FAILED', error_type='RuntimeError',
                        error='PRIVATE_RAW_SENTINEL', state={'prompt': 'PRIVATE_RAW_SENTINEL'},
                        native_counts=logic.expected_counts('ALPHAEDIT_BLUE'), completed_batches=1)
        with fake_chain(terminal=terminal) as (reader, attempt, config, lock, arm, records):
            result = collect.review(reader, attempt, config, lock, arm, {}, records)
        self.assertEqual(result['commits'], 1)
        self.assertEqual(result['scientific_status'], 'PARTIAL_OR_FAILED')
        self.assertNotIn('PRIVATE_RAW_SENTINEL', str(result))

    def test_invalid_next_commit_preserves_prior_progress(self):
        progress = {}
        with fake_chain(commits=2, invalid_second=True) as (reader, attempt, config, lock, arm, records):
            with self.assertRaisesRegex(RuntimeError, 'OWN_STATE_LINK'):
                collect.review(reader, attempt, config, lock, arm, {}, records, progress=progress)
        self.assertEqual((progress['commits'], progress['requests']), (1, 100))
        self.assertEqual(len(progress['counts']), 1)
        self.assertEqual(progress['counts'][0]['native_z'], 200)

    def test_terminal_counts_cannot_undercount_committed_native_calls(self):
        terminal = dict(status='FAILED', native_counts={key: 0 for key in collect.COUNT_KEYS})
        with fake_chain(terminal=terminal) as (reader, attempt, config, lock, arm, records):
            with self.assertRaisesRegex(RuntimeError, 'TOTAL_COUNTER_LOWER_BOUND'):
                collect.review(reader, attempt, config, lock, arm, {}, records)

    def test_compact_terminal_discards_private_payloads(self):
        original = dict(status='FAILED', stage='B2_NATIVE_APPLY', error_type='RuntimeError',
            source='frozen-source', native_counts=logic.expected_counts('CAKE'),
            completed_batches=1, rollback_verified=True,
            error='PRIVATE_RAW_SENTINEL', traceback='PRIVATE_RAW_SENTINEL',
            state={'W': 'PRIVATE_RAW_SENTINEL'}, extra={'prompt': 'PRIVATE_RAW_SENTINEL'})
        compact = collect.compact_terminal(original)
        self.assertEqual(compact['status'], 'FAILED')
        self.assertEqual(compact['error_type'], 'RuntimeError')
        self.assertEqual(compact['native_counts'], logic.expected_counts('CAKE'))
        self.assertNotIn('PRIVATE_RAW_SENTINEL', str(compact))

    def test_collector_reducer_exception_does_not_publish_raw_errors(self):
        with tempfile.TemporaryDirectory(prefix='gptj-fourarm-collector-') as folder:
            attempt = Path(folder)
            config = dict(task_id=common.TASK, instruction_id=common.NONCE,
                observer_identity={'path': 'observer.mock'}, assets=[{'path': 'stream.mock'}], stream='stream.mock')
            lock = dict(config_sha256='config-seal', source_commit='frozen-source')
            reader = FakeReader({str(attempt / 'config.json'): config,
                str(attempt / 'execution.lock.json'): lock,
                'observer.mock': {}, 'stream.mock': [dict(case_id=index) for index in range(2000)]})
            with ExitStack() as stack:
                stack.enter_context(patch.object(collect, 'Reader', return_value=reader))
                stack.enter_context(patch.object(collect, 'sha', return_value='config-seal'))
                stack.enter_context(patch.object(collect, 'batches', side_effect=fake_batches))
                stack.enter_context(patch.object(collect, 'review', side_effect=RuntimeError('PRIVATE_RAW_SENTINEL')))
                stack.enter_context(patch.object(collect, 'accounting', return_value={'status': 'NOT_RECORDED'}))
                result = collect.collect(attempt)
            review = common.read(attempt / 'collector' / 'review.json')
            self.assertFalse(result['scientific_complete'])
            self.assertEqual(len(review['reviews']), 4)
            self.assertNotIn('PRIVATE_RAW_SENTINEL', str(review))


SUBMIT_CONFIG = dict(runtime={'python': common.PYTHON},
    resources=dict(cpu=6, gpu=1, host_mib=59392, wall='2-00:00:00',
                   collector_cpu=6, collector_host_mib=24576, collector_wall='04:00:00'),
    noCP=True, z_disk_cache=False, exact_resume='NOT_AVAILABLE')


def queue_detail(job, *, node='(null)', requested_node='server2', qos='lab_gpu_s2',
                 source='/mnt/raid5/janghj/ODE-edit/local/prior-task/main.sh', allocated=0):
    return (f'JobId={job} UserId=fixture-owner(123) JobState=PENDING '
            f'Command={source} WorkDir=/tmp/prior-working-dir NodeList={node} '
            f'ReqNodeList={requested_node} QOS={qos} '
            f'ReqTRES=cpu=6,mem=59392M,node=1,gres/gpu=1 '
            f'AllocTRES=cpu=6,mem=59392M,node=1,gres/gpu={allocated}')


@contextmanager
def held_metadata(role='ALPHAEDIT_BLUE', observed_dependency='afterany:60656:60657'):
    """Exact fake scheduler response; never submits/releases an actual job."""
    with tempfile.TemporaryDirectory(prefix='gptj-fourarm-held-') as folder:
        attempt = Path(folder)
        (attempt / 'source').mkdir()
        script = submit.launcher(attempt, role, SUBMIT_CONFIG, 'frozen-source')
        (attempt / (role + '.sh')).write_text(script)
        collector = role == 'collector'
        mem = 24576 if collector else 59392
        wall = '04:00:00' if collector else '2-00:00:00'
        argv = ['sbatch', '--parsable', '--hold', '--partition=gpu', '--qos=lab_gpu_s2',
            '--nodelist=server2', '--nodes=1', '--ntasks=1', '--cpus-per-task=6',
            f'--mem={mem}M', '--time=' + wall, '--export=NONE', '--no-requeue',
            '--job-name=' + common.TASK + '-' + role, '--chdir=' + str(attempt / 'source'),
            '--dependency=afterany:60656:60657']
        if not collector:
            argv.append('--gres=gpu:1')
        argv.append(str(attempt / (role + '.sh')))
        detail = (f'JobId=20001 JobName={common.TASK}-{role} JobState=PENDING Reason=JobHeldUser '
            f'UserId=fixture-owner(123) Requeue=0 ReqNodeList=server2 Partition=gpu QOS=lab_gpu_s2 '
            f'TimeLimit={wall} Command={attempt / (role + ".sh")} WorkDir={attempt / "source"} '
            f'CPUs/Task=6 ReqTRES=cpu=6,mem={mem}M,node=1' + ('' if collector else ',gres/gpu=1') +
            f' Dependency={observed_dependency} SubmitLine={shlex.join(argv)} WorkDir={attempt / "source"}')
        calls = []

        def command(argv):
            calls.append(argv)
            if argv == ['scontrol', 'show', 'job', '20001', '--oneliner']:
                return detail
            if argv == ['scontrol', 'write', 'batch_script', '20001', '-']:
                return script
            raise AssertionError('Read-only held inspection only')

        with patch.object(submit, 'command', side_effect=command), \
                patch.object(submit.getpass, 'getuser', return_value='fixture-owner'):
            yield attempt, argv, detail, calls


class SubmissionMetadataTests(unittest.TestCase):
    def test_field_parser_matches_exact_key(self):
        detail = 'JobId=20001 JobName=task NodeList=(null) ReqNodeList=server2 CPUs/Task=6'
        self.assertEqual(submit.field(detail, 'NodeList'), '(null)')
        self.assertEqual(submit.field(detail, 'ReqNodeList'), 'server2')
        self.assertEqual(submit.field(detail, 'CPUs/Task'), '6')
        self.assertIsNone(submit.field(detail, 'AbsentField'))

    def test_gpu_count_generic_typed_and_empty(self):
        for value, expected in ((None, 0), ('', 0), ('cpu=6,mem=58G', 0),
                ('cpu=6,gres/gpu=1', 1), ('gres/gpu:a6000=2', 2),
                ('gres/gpu=1,gres/gpu:a6000=1', 1)):
            with self.subTest(value=value):
                self.assertEqual(submit.gpu_count(value), expected)

    def test_project_job_uses_source_not_job_name(self):
        self.assertTrue(submit.project_job(queue_detail('60656')))
        self.assertTrue(submit.project_job(queue_detail('60657',
            source='/mnt/raid5/janghj/.codex/worktrees/odeedit-fixture/project/run_scripts/main.sh')))
        self.assertFalse(submit.project_job('JobName=ODE-edit-gptj Command=/tmp/other-project/run.sh WorkDir=/tmp'))

    def test_whole_owner_queue_includes_empty_pending_nodelist_and_preserves_prior(self):
        queued = [('60656', 'prior-running', 'RUNNING'), ('60657', 'prior-pending', 'PENDING'),
                  ('60658', 'prior-completing', 'COMPLETING'), ('60659', 'other-node', 'PENDING'),
                  ('60660', 'other-source', 'PENDING')]
        raw = '\n'.join(f'{job}|{name}|{status}|gpu:1||Dependency' for job, name, status in queued)
        details = {'60656': queue_detail('60656', node='server2', allocated=1),
            '60657': queue_detail('60657', node='', requested_node='server2'),
            '60658': queue_detail('60658', node='server2', allocated=1),
            '60659': queue_detail('60659', requested_node='server1'),
            '60660': queue_detail('60660', source='/tmp/other-project/run.sh')}
        calls = []

        def command(argv):
            calls.append(argv)
            if argv[0] == 'squeue':
                self.assertEqual(argv, ['squeue', '-h', '-r', '-u', 'fixture-owner',
                                       '-o', '%i|%j|%T|%b|%N|%R'])
                return raw
            self.assertEqual(argv[:3], ['scontrol', 'show', 'job'])
            return details[argv[3]]

        with patch.object(submit, 'command', side_effect=command), \
                patch.object(submit.getpass, 'getuser', return_value='fixture-owner'):
            inventory = submit.inventory()
        self.assertEqual([row['job'] for row in inventory['project']], ['60656', '60657', '60658'])
        self.assertEqual(inventory['project'][1]['state'], 'PENDING')
        self.assertIsNone(inventory['project'][1]['node'])
        self.assertEqual(inventory['project'][1]['requested_node'], 'server2')
        self.assertEqual(inventory['project'][1]['allocated_GPUs'], 0)
        self.assertEqual({row['job']: row['reason'] for row in inventory['excluded']},
                         {'60659': 'EXPLICIT_OTHER_NODE', '60660': 'OTHER_PROJECT_SOURCE'})
        self.assertEqual(len(calls), 6)
        self.assertFalse(any('release' in call or 'cancel' in call or 'hold' in call for call in calls))

    def test_pending_qos_scope_and_ambiguous_project_node_fail_closed(self):
        raw = '60657|prior-pending|PENDING|gpu:1||(Priority)'
        for qos, should_accept in (('lab_gpu_s2', True), ('unknown_qos', False)):
            def command(argv):
                return raw if argv[0] == 'squeue' else queue_detail('60657', requested_node='(null)', qos=qos)
            with self.subTest(qos=qos), patch.object(submit, 'command', side_effect=command), \
                    patch.object(submit.getpass, 'getuser', return_value='fixture-owner'):
                if should_accept:
                    self.assertEqual(len(submit.inventory()['project']), 1)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'UNRESOLVED_PROJECT_NODE_SCOPE'):
                        submit.inventory()

    def test_inventory_excludes_only_own_registered_ids(self):
        raw = f'60656|prior|RUNNING|gpu:1|server2|None\n20001|{common.TASK}-CAKE|PENDING|gpu:1||JobHeldUser'
        calls = []
        def command(argv):
            calls.append(argv)
            return raw if argv[0] == 'squeue' else queue_detail('60656', node='server2', allocated=1)
        with patch.object(submit, 'command', side_effect=command), \
                patch.object(submit.getpass, 'getuser', return_value='fixture-owner'):
            inventory = submit.inventory(exclude=('20001',))
        self.assertEqual([row['job'] for row in inventory['project']], ['60656'])
        self.assertEqual(len(calls), 2)

    def test_launchers_are_offline_source_pinned_and_collector_GPU_zero(self):
        attempt = Path('/tmp/fake-task/attempt-r1')
        for role in (*common.ARMS, 'collector'):
            script = submit.launcher(attempt, role, SUBMIT_CONFIG, 'frozen-source')
            self.assertIn('set -euo pipefail', script)
            self.assertIn('export ' + common.SOURCE_ENV + '=frozen-source', script)
            self.assertIn('export OMP_NUM_THREADS=6', script)
            self.assertIn('export MKL_NUM_THREADS=6', script)
            self.assertIn('export HF_HUB_OFFLINE=1', script)
            self.assertIn('export TRANSFORMERS_OFFLINE=1', script)
            self.assertNotIn('--checkpoint', script)
            self.assertNotIn('WANDB_API_KEY', script)
            if role == 'collector':
                self.assertIn("export CUDA_VISIBLE_DEVICES=''", script)
                self.assertNotIn('--arm', script)
            else:
                self.assertNotIn('CUDA_VISIBLE_DEVICES', script)
                self.assertIn('--arm ' + role, script)

    def test_held_GPU_job_exact_resources_script_argv_and_afterany(self):
        with held_metadata() as (attempt, argv, _detail, calls):
            inspection = submit.inspect_held('20001', 'ALPHAEDIT_BLUE', attempt, argv,
                                             ['60656', '60657'], SUBMIT_CONFIG)
        self.assertEqual(inspection['job'], '20001')
        self.assertIn('--cpus-per-task=6', inspection['argv'])
        self.assertIn('--mem=59392M', inspection['argv'])
        self.assertIn('--export=NONE', inspection['argv'])
        self.assertIn('--no-requeue', inspection['argv'])
        self.assertEqual(len(calls), 2)

    def test_held_collector_requests_no_GPU(self):
        with held_metadata(role='collector') as (attempt, argv, _detail, calls):
            inspection = submit.inspect_held('20001', 'collector', attempt, argv,
                                             ['60656', '60657'], SUBMIT_CONFIG)
        self.assertIn('--mem=24576M', inspection['argv'])
        self.assertFalse(any(arg.startswith('--gres') for arg in inspection['argv']))
        self.assertEqual(len(calls), 2)

    def test_held_inspection_rejects_afterok_performance_dependency(self):
        with held_metadata(observed_dependency='afterok:60656:60657') as (attempt, argv, _detail, calls):
            with self.assertRaisesRegex(RuntimeError, 'AFTERANY_NOT_PERFORMANCE'):
                submit.inspect_held('20001', 'ALPHAEDIT_BLUE', attempt, argv,
                                    ['60656', '60657'], SUBMIT_CONFIG)
        self.assertEqual(len(calls), 1)

    def test_held_actual_resources_are_not_masked_by_submitline(self):
        mutations = (
            ('ReqTRES=cpu=6,mem=59392M', 'ReqTRES=cpu=6,mem=60416M', 'HELD_CPU_MEMORY'),
            ('ReqTRES=cpu=6,mem=59392M', 'ReqTRES=cpu=4,mem=59392M', 'HELD_CPU_MEMORY'),
            ('node=1,gres/gpu=1', 'node=1,gres/gpu=2', 'HELD_GPU'),
            ('CPUs/Task=6', 'CPUs/Task=8', 'HELD_FIELD:CPUs/Task'),
            ('Requeue=0', 'Requeue=1', 'HELD_FIELD:Requeue'),
            ('UserId=fixture-owner(123)', 'UserId=other-owner(123)', 'HELD_OWNER'),
            ('--export=NONE', '--export=ALL', 'HELD_FULL_ARGV'),
        )
        for old, new, label in mutations:
            with self.subTest(label=label), held_metadata() as (attempt, argv, detail, _calls):
                changed = detail.replace(old, new, 1)
                script = (attempt / 'ALPHAEDIT_BLUE.sh').read_text()
                with patch.object(submit, 'command', side_effect=lambda command:
                                  changed if command[1] == 'show' else script):
                    with self.assertRaisesRegex(RuntimeError, label):
                        submit.inspect_held('20001', 'ALPHAEDIT_BLUE', attempt, argv,
                                            ['60656', '60657'], SUBMIT_CONFIG)

    def test_held_script_bytes_cannot_change_after_registration(self):
        with held_metadata() as (attempt, argv, detail, _calls):
            with patch.object(submit, 'command', side_effect=lambda command:
                              detail if command[1] == 'show' else '#!/bin/bash\nexit 0\n'):
                with self.assertRaisesRegex(RuntimeError, 'HELD_SCRIPT_BYTES'):
                    submit.inspect_held('20001', 'ALPHAEDIT_BLUE', attempt, argv,
                                        ['60656', '60657'], SUBMIT_CONFIG)

    def test_submission_rejects_wrong_resources_or_checkpoint_before_any_command(self):
        for change in ({'cpu': 7}, {'host_mib': 60416}, {'gpu': 2}, {'noCP': False}, {'z_disk_cache': True}):
            config = copy.deepcopy(SUBMIT_CONFIG)
            for key, value in change.items():
                if key in config['resources']:
                    config['resources'][key] = value
                else:
                    config[key] = value
            with self.subTest(change=change), tempfile.TemporaryDirectory(prefix='gptj-fourarm-reject-') as folder:
                with patch.object(submit, 'LOCAL', Path(folder)), \
                        patch.object(submit, 'authority'), \
                        patch.object(submit, 'sha', return_value=submit.ENVELOPE_SHA), \
                        patch.object(submit, 'read', return_value=config), \
                        patch.object(submit, 'command', side_effect=AssertionError('No scheduler calls')) as command:
                    with self.assertRaisesRegex(RuntimeError, 'RESOURCE_NOCP_PROFILE'):
                        submit.submit()
                    command.assert_not_called()

    def test_imported_project_source_closure_is_archive_covered(self):
        required = ('gptj_cake_blue_prune_rect', 'gptj_native_baselines', 'jlz_realized_writer_sequential',
                    'jlz_realization', 'jlz_price_gptj', 'jlz_interference_l1', 'experiment_tracking')
        self.assertTrue(all('project/run_scripts/' + directory in submit.SOURCES for directory in required))
        for name, module in list(sys.modules.items()):
            file = getattr(module, '__file__', None)
            if not name.startswith(('project.', 'scripts.')) or not file:
                continue
            path = Path(file).resolve()
            if not path.is_relative_to(common.ROOT):
                continue
            relative = str(path.relative_to(common.ROOT))
            self.assertTrue(any(relative == source or relative.startswith(source + '/')
                                for source in submit.SOURCES), (name, relative))


class FitPayloadTests(unittest.TestCase):
    def test_already_computed_return_scalars_only(self):
        value=dict(batch=2,request_index=4,native_z=104,fit_global_candidate=300,
            fit_updates=196,prompt='NEVER_UPLOAD')
        trace=dict(request_index=4,loss=.03,nll_loss=.02,kl_loss=.01,target='NEVER_UPLOAD')
        payload=logic.fit_payload(value,trace)
        self.assertEqual(payload,{'batch':2,'candidate':4,'fit/global_candidate':300,
            'optimizer/calls':196,'fit/loss':.03,'fit/nll':.02,'fit/kl':.01})
        from project.run_scripts.experiment_tracking.schema import metrics as validate
        validate(payload)

    def test_no_stale_previous_target_loss_at_new_forward(self):
        value=dict(batch=2,request_index=5,native_z=104,fit_global_candidate=301,fit_updates=196)
        payload=logic.fit_payload(value,dict(request_index=4,loss=.03,nll_loss=.02,kl_loss=.01))
        self.assertEqual(set(payload),{'batch','candidate','fit/global_candidate','optimizer/calls'})


if __name__ == '__main__':
    unittest.main()
