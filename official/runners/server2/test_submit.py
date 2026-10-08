"""Controller CPU fixtures only: no scheduler/network/model/GPU calls."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from official.experiments.prepare import digest, write_new
from official.runners.server2 import submit as controller


def manifest():
    return dict(registration_stage='qualification', code_commit='a'*40, official_tree_sha256='b'*40,
        runtime={'python':'/fixture/python'}, resources=dict(gpu=1, cpu=6, host_mib=59392,
            wall='2-00:00:00', collector_cpu=6, collector_host_mib=24576,
            collector_wall='04:00:00', partition='gpu', qos='lab_gpu_s2', node='server2', reserve_bytes=1),
        base_manifest_sha256='c'*64)


def held_detail(job, role, attempt, argv, deps, value, *, owner='fixture-user'):
    r = value['resources']
    cpu, memory, wall = (r['collector_cpu'], r['collector_host_mib'], r['collector_wall']) if role == 'collector' \
        else (r['cpu'], r['host_mib'], r['wall'])
    import shlex
    return ' '.join([f'JobId={job}', f'JobName={controller.TASK}-{value["registration_stage"]}-{role}',
        'JobState=PENDING', 'Reason=JobHeldUser', 'Requeue=0', 'ReqNodeList=server2',
        'Partition=gpu', 'QOS=lab_gpu_s2', 'TimeLimit='+wall, f'CPUs/Task={cpu}',
        f'Command={attempt}/{role}.sh', f'UserId={owner}(123)',
        f'ReqTRES=cpu={cpu},mem={memory}M,node=1'+('' if role == 'collector' else ',gres/gpu=1'),
        'Dependency='+('afterany:'+':'.join(deps) if deps else '(null)'),
        'SubmitLine='+shlex.join(argv), 'WorkDir='+str(attempt/'source')])


def frozen_attempt(scope):
    """Tiny CPU-only archive reproducing the exact one-held-FT failure shape."""
    attempt = scope/'registration-qualification-r1'
    attempt.mkdir()
    base = scope/'base.json'
    write_new(base, dict(manifest(), fixture=True))
    value = dict(manifest(), model='gptj', instruction_id=controller.INSTRUCTION,
        owner={'server':'server2', 'session':controller.SESSION}, registration_roles=list(controller.METHODS),
        base_manifest_sha256=controller.file_sha(base))
    source = attempt/'source'
    git_rows, source_members = [], []
    value['source_members'] = {}
    for name in ('runners/server2/run.py', 'experiments/prepare.py', 'tracking/__init__.py'):
        path = source/'official'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ('# CPU archive fixture '+name+'\n').encode()
        path.write_bytes(data)
        value['source_members'][name] = hashlib.sha256(data).hexdigest()
        source_members.append(controller.member(path))
        checksum = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        git_rows.append('100644 blob '+checksum+'\t'+name)
    write_new(attempt/'manifest.json', value)
    archive = attempt/'source.tar'
    with tarfile.open(archive, 'w') as stream:
        for row in source_members:
            stream.add(row['path'], arcname=str(Path(row['path']).relative_to(source)), recursive=False)
    launchers = []
    for role in (*controller.METHODS, 'collector'):
        path = attempt/(role+'.sh')
        path.write_text(controller.launcher(attempt, role, value))
        launchers.append(controller.member(path))
    lock = dict(code_commit=value['code_commit'], official_tree_sha256=value['official_tree_sha256'],
        resources=value['resources'], stage='qualification', source={'code_commit':value['code_commit'],
        'official_tree_sha256':value['official_tree_sha256']}, base_manifest=controller.member(base),
        manifest=controller.member(attempt/'manifest.json'), archive=controller.member(archive),
        source_members=source_members, launchers=launchers)
    write_new(attempt/'execution.lock.json', lock)
    deps = ['61538', '61539']
    argv = controller.sbatch_argv(attempt, 'FT', value, deps)
    write_new(attempt/'command-FT.json', dict(role='FT', argv=argv))
    write_new(attempt/'sbatch-FT.json', dict(returncode=0, stdout='61619', stderr='', argv=argv))
    write_new(attempt/'submitted-FT.json', dict(job='61619', argv=argv, dependencies=deps))
    write_new(attempt/'submission-failure.json', dict(status='REGISTRATION_OR_RELEASE_BLOCKED',
        error='HELD_DEPENDENCY', jobs={'FT':'61619'}, lock=controller.member(attempt/'execution.lock.json')))
    return dict(attempt=attempt, base=base, manifest=value, argv=argv, deps=deps,
        git_rows='\n'.join(git_rows))


class ContinuationHarness:
    """Mocks ALL scheduler/admission/Git processes; no production job calls."""
    def __init__(self, fixture, *, fail_role=None, ft_state='PENDING', extra=None, race_edit=None):
        self.fixture = fixture
        self.fail_role, self.ft_state, self.extra, self.race_edit = fail_role, ft_state, extra, race_edit
        self.checked_ft, self.sbatch_roles, self.released, self.inventory_calls = False, [], [], 0
        self.jobs = {'61619':('FT', fixture['argv'], fixture['deps'])}
        self.binding = dict(code_commit='d'*40, official_tree_sha256='e'*40, source_members={},
            exact_official_diff_sha256='f'*64, science_or_archive_hotpatch=False)

    def command(self, argv):
        f = self.fixture
        if argv[:3] == ['git', 'ls-tree', '-r']:
            return f['git_rows']
        if argv[:3] == ['scontrol', 'show', 'job']:
            role, submitted, deps = self.jobs[argv[3]]
            detail = held_detail(argv[3], role, f['attempt'], submitted, deps, f['manifest'])
            # Actual Slurm formatting of a conjunction may split equal kinds.
            rendered = 'afterany:'+':'.join(deps) if deps else '(null)'
            expanded = ','.join('afterany:'+job+'(unfulfilled)' for job in deps) if deps else '(null)'
            detail = detail.replace('Dependency='+rendered, 'Dependency='+expanded)
            if role == 'FT':
                self.checked_ft = True
                detail = detail.replace('JobState=PENDING', 'JobState='+self.ft_state)
            return detail
        if argv[:3] == ['scontrol', 'write', 'batch_script']:
            role = self.jobs[argv[3]][0]
            return (f['attempt']/(role+'.sh')).read_text()
        if argv[:2] == ['scontrol', 'release']:
            self.released.append(argv[2])
            return ''
        if argv[0] == 'squeue':
            return 'CPU_FIXTURE_ONLY|PENDING|Dependency'
        raise AssertionError('Unmocked external command: '+repr(argv))

    def inventory(self, *, exclude):
        assert self.checked_ft, 'Known held job must be checked before exclusion/admission'
        assert '61619' in set(exclude)
        self.inventory_calls += 1
        rows = [dict(job=job, dependency='(null)', command='/fixture/old/'+job+'.sh', workdir='/fixture/old',
            state='RUNNING', allocated_gpus=1) for job in self.fixture['deps']]
        if self.extra:
            rows.append(self.extra)
        if self.inventory_calls == 2 and self.race_edit:
            self.race_edit()
        return dict(project=rows, ambiguous=[], owner='fixture-user')

    def sbatch(self, argv, **kwargs):
        assert argv[0] == 'sbatch', 'All non-sbatch processes must be mocked by command'
        role = Path(argv[-1]).stem
        self.sbatch_roles.append(role)
        if role == self.fail_role:
            return SimpleNamespace(returncode=1, stdout='', stderr='fixed CPU sbatch rejection fixture')
        job = str(62000+len(self.sbatch_roles))
        dep = next((item.partition('=')[2] for item in argv if item.startswith('--dependency=')), '')
        self.jobs[job] = (role, argv, controller.dependency_ids(dep))
        return SimpleNamespace(returncode=0, stdout=job+'\n', stderr='')

    def run(self, scope):
        f = self.fixture
        with patch.object(controller, 'OUTPUT', scope), patch.object(controller, 'command', self.command), \
                patch.object(controller, 'published_control_binding', return_value=self.binding), \
                patch.object(controller, 'tracking_binding', return_value={'CPU_fixture':True}), \
                patch.object(controller, 'inventory', self.inventory), \
                patch.object(controller, 'admission', return_value={'cap':2, 'CPU_fixture':True}), \
                patch.object(controller.getpass, 'getuser', return_value='fixture-user'), \
                patch.object(controller.subprocess, 'run', self.sbatch):
            return controller.continue_held_registration(f['base'], f['attempt'], 'd'*40, 'e'*40)


class OfficialSubmitTests(unittest.TestCase):
    def test_dependency_conjunction_grouping_duplicate_and_index_zero_semantics(self):
        self.assertEqual(controller.typed_dependencies('afterany:61538:61539'),
            controller.typed_dependencies('afterany:61539(unfulfilled),afterany:61538(unfulfilled)'))
        self.assertEqual(controller.typed_dependencies('afterany:71000_0:71001,afterany:71000_0'),
            (('afterany', ('71000_0', '71001')),))
        self.assertNotEqual(controller.typed_dependencies('afterany:61538:61539'),
            controller.typed_dependencies('afterok:61538,afterany:61539'))
        with self.assertRaisesRegex(ValueError, 'SEMANTICS'):
            controller.typed_dependencies('afterany:61538?afterany:61539')

    def test_control_source_binds_exact_import_closure_not_unrelated_live_head(self):
        value = manifest()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            payloads = {}
            for name in controller.CONTROL_IMPORT_PATHS:
                data = ('# Published CPU control fixture '+name+'\n').encode()
                path = repo/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                payloads[name] = data
            changes = 'M\t'+controller.CONTROL_PATHS[0]+'\nM\t'+controller.CONTROL_PATHS[1]
            def run(argv):
                if argv[1:3] == ['remote', 'get-url']:
                    return 'https://github.com/hyunjun1127/ODE-edit.git'
                if argv[1] == 'merge-base' or argv[1:3] == ['status', '--porcelain']:
                    return ''
                if argv[1] == 'diff':
                    return changes
                if argv == ['git', 'rev-parse', value['code_commit']+':official']:
                    return value['official_tree_sha256']
                if argv == ['git', 'rev-parse', 'd'*40+':official']:
                    return 'e'*40
                if argv == ['git', 'rev-parse', 'origin/main']:
                    return 'f'*40
                raise AssertionError('No live HEAD equality needed: '+repr(argv))
            def raw(argv):
                return payloads[argv[2].partition(':')[2]] if argv[1] == 'show' else b'EXACT_DIFF_CPU_FIXTURE\n'
            with patch.object(controller, 'REPO', repo):
                proof = controller.published_control_binding(value, 'd'*40, 'e'*40, run=run, raw=raw)
                self.assertEqual(set(proof['source_members']), set(controller.CONTROL_IMPORT_PATHS))
                self.assertFalse(proof['science_or_archive_hotpatch'])
                (repo/'official/experiments/prepare.py').write_text('# UNBOUND helper\n')
                with self.assertRaisesRegex(ValueError, 'IMPORTED_BYTES'):
                    controller.published_control_binding(value, 'd'*40, 'e'*40, run=run, raw=raw)
                (repo/'official/experiments/prepare.py').write_bytes(payloads['official/experiments/prepare.py'])
                def dirty(argv):
                    return ' M official/experiments/prepare.py' if argv[1:3] == ['status', '--porcelain'] else run(argv)
                with self.assertRaisesRegex(ValueError, 'UNCOMMITTED'):
                    controller.published_control_binding(value, 'd'*40, 'e'*40, run=dirty, raw=raw)
                def scientific(argv):
                    return 'M\tofficial/runners/server2/native.py' if argv[1] == 'diff' else run(argv)
                with self.assertRaisesRegex(ValueError, 'ONLY_SUBMIT_AND_TEST'):
                    controller.published_control_binding(value, 'd'*40, 'e'*40, run=scientific, raw=raw)

    def test_exact_frozen_failure_verification_and_source_launcher_tamper_rejection(self):
        for mutated in (None, 'source', 'launcher', 'archive', 'unknown_submission', 'released'):
            with self.subTest(mutated=mutated), tempfile.TemporaryDirectory() as directory:
                scope = Path(directory)
                f = frozen_attempt(scope)
                if mutated == 'source':
                    (f['attempt']/'source/official/runners/server2/run.py').write_text('# CHANGED\n')
                elif mutated == 'launcher':
                    (f['attempt']/'MEMIT.sh').write_text('# CHANGED\n')
                elif mutated == 'archive':
                    (f['attempt']/'source.tar').write_bytes(b'CHANGED')
                elif mutated == 'unknown_submission':
                    write_new(f['attempt']/'submitted-MEMIT.json', dict(job='99999'))
                elif mutated == 'released':
                    write_new(f['attempt']/'released-FT.json', dict(job='61619'))
                with patch.object(controller, 'OUTPUT', scope):
                    if mutated:
                        with self.assertRaises(ValueError):
                            controller.verify_held_attempt(f['base'], f['attempt'], run=lambda argv:f['git_rows'])
                    else:
                        proof = controller.verify_held_attempt(f['base'], f['attempt'], run=lambda argv:f['git_rows'])
                        self.assertEqual(proof['jobs'], {'FT':'61619'})
                        self.assertEqual(proof['manifest']['code_commit'], 'a'*40)

    def test_control_lock_is_exclusive_even_for_identical_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory)/'control.json'
            controller.control_lock_once(lock, {'fixture':True})
            with self.assertRaisesRegex(ValueError, 'ALREADY_STARTED_NO_RETRY'):
                controller.control_lock_once(lock, {'fixture':True})

    def test_explicit_continuation_reuses_ft_and_preserves_science_failure_and_reverse_release(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = Path(directory)
            f = frozen_attempt(scope)
            before = {path:path.read_bytes() for path in f['attempt'].rglob('*') if path.is_file()}
            mock = ContinuationHarness(f)
            result = mock.run(scope)
            self.assertEqual(mock.sbatch_roles, [*controller.METHODS[1:], 'collector'])
            self.assertEqual(mock.released, [result['jobs'][role] for role in reversed([*controller.METHODS, 'collector'])])
            self.assertEqual(result['jobs']['FT'], '61619')
            self.assertTrue(result['existing_FT_reused_not_resubmitted'])
            self.assertEqual(result['source']['code_commit'], 'a'*40)
            self.assertEqual(result['controller']['code_commit'], 'd'*40)
            self.assertEqual(result['actual_GPU_qualification'], 'NOT_OBSERVED')
            self.assertFalse(result['scientific_complete'])
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data, str(path))
            self.assertEqual(result['dependencies']['MEMIT'], ['61538','61539'])
            self.assertEqual(result['dependencies']['ALPHAEDIT'], ['61619'])
            self.assertEqual(len(result['dependencies']['collector']), 6)
            with self.assertRaisesRegex(ValueError, 'ALREADY_STARTED_OR_RELEASED'):
                mock.run(scope)
            self.assertEqual(len(mock.sbatch_roles), 6)

    def test_continue_blocks_released_or_unknown_sameattempt_new_frontier_before_sbatch(self):
        for kind in ('running_ft', 'unknown_sameattempt', 'new_frontier', 'missing_old_parent'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                scope = Path(directory)
                f = frozen_attempt(scope)
                extra = None
                if kind in ('unknown_sameattempt', 'new_frontier'):
                    extra = dict(job='61699', dependency='(null)', state='PENDING',
                        command=str(f['attempt']/'MEMIT.sh') if kind == 'unknown_sameattempt' else '/fixture/new/job.sh',
                        workdir=str(f['attempt']/'source') if kind == 'unknown_sameattempt' else '/fixture/new')
                mock = ContinuationHarness(f, ft_state='RUNNING' if kind == 'running_ft' else 'PENDING', extra=extra)
                if kind == 'missing_old_parent':
                    original_inventory = mock.inventory
                    mock.inventory = lambda **kwargs:dict(original_inventory(**kwargs), project=[])
                with self.assertRaises(ValueError):
                    mock.run(scope)
                self.assertEqual(mock.sbatch_roles, [])
                self.assertEqual(mock.released, [])

    def test_partial_control_failure_preserves_all_new_ids_and_disallows_second_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = Path(directory)
            f = frozen_attempt(scope)
            old_failure = (f['attempt']/'submission-failure.json').read_bytes()
            mock = ContinuationHarness(f, fail_role='ALPHAEDIT_BLUE')
            with self.assertRaisesRegex(ValueError, 'SBATCH_REJECTED'):
                mock.run(scope)
            failure = controller.read(f['attempt']/'control-repair-failure.json')
            self.assertEqual(set(failure['jobs']), {'FT', 'MEMIT', 'ALPHAEDIT'})
            self.assertEqual(failure['jobs']['FT'], '61619')
            self.assertEqual(mock.released, [])
            self.assertEqual((f['attempt']/'submission-failure.json').read_bytes(), old_failure)
            old_calls = list(mock.sbatch_roles)
            with self.assertRaisesRegex(ValueError, 'ALREADY_STARTED_OR_RELEASED'):
                mock.run(scope)
            self.assertEqual(mock.sbatch_roles, old_calls)

    def test_pre_release_source_launcher_and_untracked_members_rechecked(self):
        for kind in ('source', 'launcher', 'extra_source'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                scope = Path(directory)
                f = frozen_attempt(scope)
                def edit():
                    path = f['attempt']/({'source':'source/official/runners/server2/run.py',
                        'launcher':'FT.sh', 'extra_source':'source/official/unknown.py'}[kind])
                    path.write_text('# CHANGED DURING HELD INSPECTION\n')
                mock = ContinuationHarness(f, race_edit=edit)
                with self.assertRaises(ValueError):
                    mock.run(scope)
                self.assertEqual(mock.released, [])
                self.assertTrue((f['attempt']/'control-repair-failure.json').is_file())

    def test_source_only_published_main_exact_tree_and_dirty_rejection(self):
        def run(argv):
            if argv[1:3] == ['remote', 'get-url']:
                return 'https://github.com/hyunjun1127/ODE-edit.git'
            if argv[1] == 'merge-base':
                return ''
            if argv[1:3] == ['status', '--porcelain']:
                return ''
            return 'b'*40 if argv[-1].endswith(':official') else 'd'*40
        value = controller.sealed_source('a'*40, 'b'*40, run=run)
        self.assertEqual(value['observed_main'], 'd'*40)
        with self.assertRaisesRegex(ValueError, 'TREE_MISMATCH'):
            controller.sealed_source('a'*40, 'f'*40, run=run)
        with self.assertRaisesRegex(ValueError, 'EXACT_SHA_REQUIRED'):
            controller.sealed_source('main', 'b'*40, run=run)
        def dirty(argv):
            return ' M official/runner.py' if argv[1:3] == ['status', '--porcelain'] else run(argv)
        with self.assertRaisesRegex(ValueError, 'SOURCE_UNCOMMITTED'):
            controller.sealed_source('a'*40, 'b'*40, run=dirty)
        def different_imported_tree(argv):
            return 'f'*40 if argv == ['git','rev-parse','HEAD:official'] else run(argv)
        with self.assertRaisesRegex(ValueError, 'IMPORTED_OFFICIAL_TREE_NOT_SELECTED_SOURCE'):
            controller.sealed_source('a'*40, 'b'*40, run=different_imported_tree)

    def test_dependency_parser_preserves_index_zero_and_uses_admitted_gpu_leaves(self):
        self.assertEqual(controller.dependency_ids('afterany:71000_0(unfulfilled):71001,afterok:71002'),
            ['71000_0', '71001', '71002'])
        rows = [{'job':'1', 'dependency':'(null)'}, {'job':'2', 'dependency':'afterany:1(unfulfilled)'},
            {'job':'3', 'dependency':'(null)'}, {'job':'4', 'dependency':'afterany:3(unfulfilled)'}]
        self.assertEqual(controller.frontier(rows), ['2', '4'])
        with self.assertRaisesRegex(ValueError, 'SEMANTICS'):
            controller.dependency_ids('afterany:1?afterany:2')

    def test_two_lanes_and_cap_one_dependencies_preserve_entire_old_frontier(self):
        ordered = list(controller.METHODS)
        for cap in (1, 2):
            jobs = {}
            for index, method in enumerate(ordered):
                expected = ['61428', '61538', '61539'] if index < cap else [jobs[ordered[index-cap]]]
                actual = controller.dependencies(method, ordered, jobs, ['61428', '61538', '61539'], cap)
                self.assertEqual(actual, expected)
                jobs[method] = str(62000+index)
            self.assertEqual(controller.dependencies('collector', ordered, jobs, [], cap), list(jobs.values()))

    def test_sbatch_resource_argv_gpu_and_collector_export_none_requeue_zero(self):
        value = manifest()
        for role, gpu in [('MEMIT', 1), ('collector', 0)]:
            argv = controller.sbatch_argv(Path('/fixture/attempt'), role, value, ['61428', '61538'])
            self.assertIn('--hold', argv)
            self.assertIn('--export=NONE', argv)
            self.assertIn('--no-requeue', argv)
            self.assertIn('--cpus-per-task=6', argv)
            self.assertEqual('--gres=gpu:1' in argv, bool(gpu))
            self.assertIn('--dependency=afterany:61428:61538', argv)
            self.assertIn('--kill-on-invalid-dep=yes', argv)
            self.assertIn('--mem=59392M' if gpu else '--mem=24576M', argv)
            self.assertIn('--time=2-00:00:00' if gpu else '--time=04:00:00', argv)
        illegal = copy.deepcopy(value)
        illegal['resources']['cpu'] = 8
        with self.assertRaisesRegex(ValueError, 'STRICTER'):
            controller.resources(illegal)

    def test_runner_command_science_uses_only_official_native_modes_and_explicit_resume(self):
        value = manifest()
        for stage, mode, dataset in [('qualification', 'qualification', 'cf'), ('cf', 'chain', 'cf'),
                ('zsre', 'chain', 'zsre')]:
            argv = controller.runner_argv('/manifest', 'MEMIT', stage, Path('/attempt'), value)
            self.assertIn('official.runners.server2.run', argv)
            self.assertEqual(argv[argv.index('--mode')+1], mode)
            self.assertEqual(argv[argv.index('--dataset')+1], dataset)
        argv = controller.runner_argv('/manifest', 'ZSRE_SMOKE', 'zsre', Path('/attempt'), value)
        self.assertEqual(argv[argv.index('--mode')+1], 'smoke')
        argv = controller.runner_argv('/manifest', 'W0_CF', 'cf', Path('/attempt'), value)
        self.assertEqual(argv[argv.index('--mode')+1], 'w0')
        self.assertEqual(argv[argv.index('--method')+1], 'MEMIT')
        argv = controller.runner_argv('/manifest', 'W0_ZSRE', 'zsre', Path('/attempt'), value)
        self.assertEqual(argv[argv.index('--mode')+1], 'w0')
        self.assertEqual(argv[argv.index('--dataset')+1], 'zsre')
        script = controller.launcher(Path('/attempt'), 'MEMIT', value, resume='/checkpoint')
        self.assertIn('--resume /checkpoint', script)
        self.assertIn('OFFICIAL_CODE_COMMIT=', script)
        self.assertNotIn('project.run_scripts', script)
        self.assertNotIn('easyeditor', script)
        self.assertNotIn('scancel', script)

    def test_held_owner_fullargv_script_and_resource_inspection(self):
        value = manifest()
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            role, job, deps = 'MEMIT', '62001', ['61428', '61538']
            script = controller.launcher(attempt, role, value)
            (attempt/(role+'.sh')).write_text(script)
            argv = controller.sbatch_argv(attempt, role, value, deps)
            detail = held_detail(job, role, attempt, argv, deps, value)
            def run(args):
                return script if args[1:3] == ['write', 'batch_script'] else detail
            receipt = controller.inspect_held(job, role, attempt, value, argv, deps, run=run, owner='fixture-user')
            self.assertFalse(receipt['actual_GPU_qualification'])
            self.assertEqual(receipt['dependencies'], deps)
            for wrong in (detail.replace('fixture-user(123)', 'other(123)'),
                    detail.replace('cpu=6,', 'cpu=8,'), detail.replace('JobState=PENDING', 'JobState=RUNNING')):
                with self.assertRaisesRegex(ValueError, 'HELD_'):
                    controller.inspect_held(job, role, attempt, value, argv, deps,
                        run=lambda args:script if args[1:3] == ['write', 'batch_script'] else wrong,
                        owner='fixture-user')

    def test_resume_metadata_binds_exact_original_identity_without_tensor_load(self):
        from official.experiments.checkpoint import IDENTITY_FIELDS
        identity = {key:'a'*64 for key in IDENTITY_FIELDS}
        identity.update(code_commit='a'*40, official_tree_sha256='b'*40)
        with tempfile.TemporaryDirectory() as directory:
            scope = Path(directory)
            attempt = scope/'registration-r1'
            root = attempt/'MEMIT/checkpoints'
            root.mkdir(parents=True)
            original = dict(manifest(), instruction_id=controller.INSTRUCTION, model='gptj',
                owner={'server':'server2', 'session':controller.SESSION}, registration_stage='cf',
                registration_roles=list(controller.METHODS))
            write_new(attempt/'manifest.json', original)
            filename = 'batch-02-'+('c'*16)+'.pt'
            (root/filename).write_bytes(b'NOT_A_TENSOR_CPU_FIXTURE')
            pointer = dict(identity_sha256=digest(identity), file=filename, sha256='c'*64, batch=2)
            write_new(root/'latest.json', pointer)
            with patch.object(controller, 'OUTPUT', scope), \
                    patch('official.runners.server2.run.checkpoint_identity', return_value=identity):
                result = controller.resume_binding(root, identity, method='MEMIT', dataset='cf')
                self.assertEqual(result['batch'], 2)
                self.assertFalse(result['controller_tensor_load'])
                with self.assertRaisesRegex(ValueError, 'IDENTITY_MISMATCH'):
                    controller.resume_binding(root, dict(identity, code_commit='d'*40), method='MEMIT', dataset='cf')
                with self.assertRaisesRegex(ValueError, 'OWN_REGISTERED_METHOD_FOLDER'):
                    controller.resume_binding(root, identity, method='FT', dataset='cf')
                with self.assertRaisesRegex(ValueError, 'CHECKPOINT_SCOPE'):
                    controller.resume_binding('/tmp/checkpoints', identity, method='MEMIT', dataset='cf')
                alias = scope/'alias'
                alias.symlink_to(attempt, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'SYMLINK_COMPONENT'):
                    controller.resume_binding(alias/'MEMIT/checkpoints', identity, method='MEMIT', dataset='cf')
                script = attempt/'MEMIT.sh'
                script.write_text('#!/bin/bash\n')
                with self.assertRaisesRegex(ValueError, 'ACTIVE_OR_ADMITTED_WRITER:62001'):
                    controller.resume_writer_guard(result, [{'job':'62001', 'command':str(script), 'state':'PENDING'}])
                unrelated = attempt/'FT.sh'
                unrelated.write_text('#!/bin/bash\n')
                self.assertTrue(controller.resume_writer_guard(result,
                    [{'job':'62002', 'command':str(unrelated), 'state':'RUNNING'}])['single_writer'])

    def test_cf_w0_technical_gate_has_two_native_lanes_and_no_intermediate_generation(self):
        ordered = controller.roles('cf')
        self.assertEqual(ordered, ['W0_CF', *controller.METHODS])
        jobs = {}
        for role in ordered:
            dep = controller.stage_dependencies(role, 'cf', ordered, jobs, ['61538', '61539'], 2)
            if role == 'W0_CF':
                self.assertEqual(dep, ['61538', '61539'])
            else:
                self.assertIn(('afterok', (jobs['W0_CF'],)), controller.typed_dependencies(dep))
                index = list(controller.METHODS).index(role)
                if index >= 2:
                    self.assertIn(('afterany', (jobs[controller.METHODS[index-2]],)),
                        controller.typed_dependencies(dep))
            jobs[role] = str(63000+len(jobs))
        self.assertEqual(controller.stage_dependencies('collector', 'cf', ordered, jobs, [], 2), list(jobs.values()))

    def test_held_dependency_kind_cannot_be_silently_changed(self):
        self.assertEqual(controller.typed_dependencies('afterok:63000(unfulfilled),afterany:63001(unfulfilled)'),
            (('afterany', ('63001',)), ('afterok', ('63000',))))
        self.assertNotEqual(controller.typed_dependencies('afterany:63000:63001'),
            controller.typed_dependencies('afterok:63000,afterany:63001'))

    def test_zsre_smoke_requires_one_fresh_model_w0_then_one_batch(self):
        ordered = controller.roles('zsre', smoke_only=True)
        self.assertEqual(ordered, ['W0_ZSRE', 'ZSRE_SMOKE'])
        self.assertEqual(controller.stage_dependencies('W0_ZSRE', 'zsre', ordered, {}, ['61538'], 2), ['61538'])
        self.assertEqual(controller.stage_dependencies('ZSRE_SMOKE', 'zsre', ordered,
            {'W0_ZSRE':'63000'}, ['61538'], 2), 'afterok:63000')
        self.assertEqual(controller.stage_dependencies('collector', 'zsre', ordered,
            {'W0_ZSRE':'63000', 'ZSRE_SMOKE':'63001'}, [], 2), ['63000', '63001'])

    def test_partial_existing_attempt_cannot_be_silently_resubmitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = dict(stage='qualification', base_manifest_sha256='a'*64,
                status='SUBMISSION_HANDOFF', registration_roles=list(controller.METHODS), jobs={'FT':'62001'})
            write_new(root/'registration-r1/submission.json', receipt)
            self.assertTrue(controller._existing(root, 'a'*64, 'qualification')['duplicate_prevented'])
            self.assertIsNone(controller._existing(root, 'b'*64, 'qualification'))
            with self.assertRaisesRegex(ValueError, 'FULL_DAG_REQUIRED'):
                controller._existing(root, 'a'*64, 'qualification', list(controller.METHODS))

    def test_partial_failed_registration_cannot_be_retried_under_another_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_new(root/'registration-r1/manifest.json', dict(registration_stage='qualification',
                base_manifest_sha256='a'*64, registration_roles=list(controller.METHODS)))
            write_new(root/'registration-r1/submission-failure.json', dict(jobs={'FT':'62001'}))
            with self.assertRaisesRegex(ValueError, 'PARTIAL_REGISTERED_DAG_NO_RETRY'):
                controller._existing(root, 'a'*64, 'qualification', list(controller.METHODS))

    def test_tracking_cpu_source_binding_does_not_claim_online_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'wandb.env'
            config.write_text('WANDB_PROJECT=layer allocation\n')
            value = dict(source_members={'tracking/__init__.py':'a'*64}, tracking=dict(
                namespace='official.tracking', source_sha256='a'*64,
                env_file=str(config), metric_schema='official-baselines-scalar-v1'))
            receipt = controller.tracking_binding(value)
            self.assertEqual(receipt['SDK_auth_remote_status'], 'NOT_CERTIFIED_BY_CPU_BINDING')
            value['tracking']['namespace'] = 'project.run_scripts.experiment_tracking'
            with self.assertRaisesRegex(ValueError, 'API_NOT_READY'):
                controller.tracking_binding(value)

    def test_actual_resume_gate_rejects_cpu_fixtures_context_change_and_wrong_cell(self):
        value = manifest()
        checkpoint_identity = {'identity_fixture':'strict CPU gate fixture, not actual evidence'}
        methods = {method:dict(status='PASS_ACTUAL_QUALIFICATION', method=method, dataset='cf', model='gptj',
            actual_GPU=True, continuous_batches=3, resume_after_batch=2, resumed_batches=[3],
            weights_equal=True, history_equal=True, rng_equal=True, metrics_equal=True, contexts_equal=True,
            checkpoint_identity=checkpoint_identity, code_commit=value['code_commit'],
            official_tree_sha256=value['official_tree_sha256'], manifest_sha256=value['base_manifest_sha256'])
            for method in controller.METHODS}
        receipt = dict(status='PASS_ACTUAL_QUALIFICATION', actual_GPU=True, methods=methods,
            code_commit=value['code_commit'], official_tree_sha256=value['official_tree_sha256'],
            manifest_sha256=value['base_manifest_sha256'],
            CF_original_evaluator_parity='PASS_ACTUAL_ORIGINAL_NATIVE_REFERENCE')
        with tempfile.TemporaryDirectory() as directory, \
                patch('official.runners.server2.run.checkpoint_identity', return_value=checkpoint_identity):
            path = Path(directory)/'qualification.json'
            path.write_text(json.dumps(receipt))
            self.assertEqual(controller.gate(path, manifest=value, kind='qualification')['path'], str(path))
            receipt['CF_original_evaluator_parity']='NOT_ESTABLISHED_BY_OWNER_FORMULA_CONTROL'
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'CF_ORIGINAL_NATIVE_REFERENCE_PARITY'):
                controller.gate(path,manifest=value,kind='qualification')
            receipt['CF_original_evaluator_parity']='PASS_ACTUAL_ORIGINAL_NATIVE_REFERENCE'
            receipt['actual_GPU'] = False
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'ACTUAL_STAGE_RECEIPT'):
                controller.gate(path, manifest=value, kind='qualification')
            receipt['actual_GPU'] = True
            receipt['methods']['MEMIT']['contexts_equal'] = False
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'ACTUAL_RESUME_PROOF'):
                controller.gate(path, manifest=value, kind='qualification')
            receipt['methods']['MEMIT']['contexts_equal'] = True
            receipt['methods']['MEMIT']['checkpoint_identity'] = {'wrong_cell':True}
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'ACTUAL_RESUME_PROOF'):
                controller.gate(path, manifest=value, kind='qualification')

    def test_project_match_uses_registered_repository_roots_not_generic_worktrees(self):
        roots = ('/mnt/raid5/janghj/ODE-edit/', '/mnt/raid5/janghj/.codex/worktrees/this-ode-wt/')
        self.assertTrue(controller._project('WorkDir=/mnt/raid5/janghj/ODE-edit Command=/bin/python', roots))
        self.assertTrue(controller._project('Command=/mnt/raid5/janghj/.codex/worktrees/this-ode-wt/run.sh', roots))
        self.assertFalse(controller._project('Command=/mnt/raid5/janghj/.codex/worktrees/unrelated-repo/run.sh', roots))
        with self.assertRaisesRegex(ValueError, 'CYCLIC_FRONTIER'):
            controller.frontier([{'job':'1', 'dependency':'afterany:2'}, {'job':'2', 'dependency':'afterany:1'}])

    def test_existing_registration_reconciles_exact_scheduler_identity_without_resubmit(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)/'registration-r1'
            attempt.mkdir()
            value = dict(manifest(), model='gptj', instruction_id=controller.INSTRUCTION,
                owner={'server':'server2', 'session':controller.SESSION})
            write_new(attempt/'manifest.json', value)
            write_new(attempt/'execution.lock.json', {'fixture_cpu_only':True})
            argv = controller.sbatch_argv(attempt, 'MEMIT', value, ['61538','61539'])
            write_new(attempt/'command-MEMIT.json', dict(argv=argv))
            detail = held_detail('62001', 'MEMIT', attempt, argv, ['61538','61539'], value)
            receipt = dict(jobs={'MEMIT':'62001'}, manifest=controller.member(attempt/'manifest.json'),
                lock=controller.member(attempt/'execution.lock.json'))
            observed = controller.existing_snapshot(receipt, run=lambda args:detail, owner='fixture-user')
            self.assertTrue(observed['no_new_registration'])
            self.assertEqual(observed['jobs'][0]['state'], 'PENDING')
            self.assertFalse(observed['jobs'][0]['scientific_result_queried'])
            with self.assertRaisesRegex(ValueError, 'CURRENT_JOB_IDENTITY_MISMATCH'):
                controller.existing_snapshot(receipt,
                    run=lambda args:detail.replace('fixture-user(123)', 'other(123)'), owner='fixture-user')

    def test_actual_zsre_smoke_requires_exact_completed_cold_reference_not_fixture_relabel(self):
        value = dict(manifest(), model_revision='r', tokenizer_sha256='t',
            streams={'zsre':{'lock':{'stream_sha256':'s'}}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'READY.json'
            ready = dict(status='READY_COLD_W0_COMPLETE', actual_GPU=True, model='gptj', dataset='zsre',
                code_commit=value['code_commit'], official_tree_sha256=value['official_tree_sha256'],
                model_revision='r', tokenizer_sha256='t', stream_sha256='s')
            path.write_text(json.dumps(ready))
            smoke = dict(status='PASS_ACTUAL_SMOKE', actual_GPU=True, model='gptj', dataset='zsre', batches=1,
                code_commit=value['code_commit'], official_tree_sha256=value['official_tree_sha256'],
                manifest_sha256=value['base_manifest_sha256'], W0member=controller.member(path))
            smoke_path = Path(directory)/'smoke.json'
            smoke_path.write_text(json.dumps(smoke))
            self.assertEqual(controller.gate(smoke_path, manifest=value, kind='smoke')['path'], str(smoke_path))
            ready['dataset'] = 'cf'
            path.write_text(json.dumps(ready))
            smoke['W0member'] = controller.member(path)
            smoke_path.write_text(json.dumps(smoke))
            with self.assertRaisesRegex(ValueError, 'W0_REFERENCE_IDENTITY'):
                controller.gate(smoke_path, manifest=value, kind='smoke')

    def test_official_git_archive_root_directory_is_safe_but_escape_and_symlink_are_not(self):
        root = tarfile.TarInfo('official')
        root.type = tarfile.DIRTYPE
        member = tarfile.TarInfo('official/runners/server2/run.py')
        self.assertTrue(controller.safe_archive_members([root, member]))
        for path in ('project/run.py', '/official/run.py', 'official/../../outside'):
            self.assertFalse(controller.safe_archive_members([root, tarfile.TarInfo(path)]))
        alias = tarfile.TarInfo('official/alias.py')
        alias.type = tarfile.SYMTYPE
        alias.linkname = '/tmp/other'
        self.assertFalse(controller.safe_archive_members([root, alias]))


if __name__ == '__main__':
    unittest.main()
