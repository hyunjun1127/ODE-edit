"""Controller CPU fixtures only: no scheduler/network/model/GPU calls."""
import copy
import json
from pathlib import Path
import tempfile
import tarfile
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


class OfficialSubmitTests(unittest.TestCase):
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
