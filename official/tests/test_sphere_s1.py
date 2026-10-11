"""Server1 resubmission of the cancelled Qwen SPHERE 62873 (CF Flu/Con) and 63303 (zsRE B11-B20): CPU only."""
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from official.experiments.prepare import digest, write_new
from official.runners.server1 import flucon_eval as shared
from official.runners.server1 import sphere_flucon_s1 as flucon
from official.runners.server1 import sphere_resume_s1 as s1
from official.runners.server1 import sphere_s1_submit as submit
from official.runners.server1.common import member
from official.runners.server1.sphere_s1_common import field_diff, host_only, received, transfer_rows
from official.runners.server1.submit import graph_width
from official.runners.server2 import flucon_eval as server2
from official.runners.server4 import sphere_b10_resume as s4

HOST_SUBSTITUTIONS = [("c['server'] == 'server4'", "c['server'] == 'server1'"), ("server='server4'", "server='server1'"),
                      ("'/data/janghj/", "'/mnt/raid5/janghj/")]


def receipt_line(key, source, local, sha=None, size=None, status='OK'):
    found = member(local)
    return '\t'.join([key, 'rke-server2', str(source), sha or found['sha256'], str(size or found['bytes']), str(local),
                      found['sha256'], status, '0']) + '\n'


class ResumeIsServer4ScienceTests(unittest.TestCase):
    def test_validate_prepare_run_are_63303_text_with_only_host_bindings(self):
        changed = 0
        for name in ('validate', 'prepare', 'run'):
            text = inspect.getsource(getattr(s4, name))
            for old, new in HOST_SUBSTITUTIONS:
                text = text.replace(old, new)
            mine = inspect.getsource(getattr(s1, name))
            self.assertEqual(mine, text, name)
            original = inspect.getsource(getattr(s4, name)).splitlines()
            changed += sum(a != b for a, b in zip(original, mine.splitlines()))
        self.assertEqual(changed, 6)  # TRANSFER_SCOPE, snapshot, projector, server, tracking env, tracking label

    def test_every_global_the_loop_calls_is_the_same_object(self):
        code = s4.run.__code__
        names = set(code.co_names) | {n for c in code.co_consts if inspect.iscode(c) for n in c.co_names}
        shared_names = sorted(n for n in names if hasattr(s4, n) and not n.startswith('__')
                              and n not in ('ROOT', 'TASK', 'AUTHORITY', 'CODE', 'validate'))
        self.assertGreater(len(shared_names), 20)
        for name in shared_names:
            self.assertIs(getattr(s1, name), getattr(s4, name), name)
        for name in ('PARENT_SHA', 'PARENT_BYTES', 'PARENT_SOURCE', 'PARENT_CONFIG', 'CONTEXT_SHA'):
            self.assertEqual(getattr(s1, name), getattr(s4, name))
        self.assertIs(s1.validate_parent, s4.validate_parent)
        self.assertIs(s1.save_reserved, s4.save_reserved)
        self.assertEqual(s1.CODE, Path(s1.__file__).resolve().parents[3])

    def test_tracking_labels_are_valid_for_server1(self):
        from official.tracking.schema import config
        value = dict(server='server1', task_id=s1.TASK, arm='qwen25-zsre-sphere-resume-b10', attempt='r1', source_sha='a' * 40,
                     config_sha='b' * 64, model='qwen25', model_family='qwen2', writer='SPHERE', baseline='SPHERE',
                     role='scientific', metric_schema='official-baselines-scalar-v1',
                     instruction_id='USER-OFFICIAL-BASELINES-20261008-R1', dataset='zsre', checkpoint_sha256=s1.PARENT_SHA,
                     source_run_id='job62538-b10')
        self.assertEqual(config(value)['server'], 'server1')


class HostFieldTests(unittest.TestCase):
    def transfer(self):
        member_row = lambda name: dict(path='/data/janghj/snap/' + name, real_path='/data/janghj/blob/' + name, bytes=3,
                                       allocated_bytes=4096, device=1, inode=2, nlink=1, mtime_ns=3, ctime_ns=4,
                                       sha256=name * 4, expected_sha256=name * 4, verification='FRESH_SHA256')
        return dict(authority='OLD', server='server4', start_batch=10, target_batches=list(range(11, 21)),
                    parent_checkpoint=dict(path='/var/tmp/p.pt', bytes=s1.PARENT_BYTES, sha256=s1.PARENT_SHA),
                    evidence=[dict(path='/var/tmp/e0', bytes=1, sha256='e' * 64)], numeric_sources=[dict(relative='x.py', sha256='n' * 64)],
                    model_members=[member_row('a'), member_row('b')], packages=dict(torch='2.9.1'),
                    projector=dict(path='/data/janghj/P.pt', sha256='p' * 64, bytes=7, shape=[5, 2, 2]),
                    storage=dict(path='/var/tmp/storage.json', bytes=1, sha256='s' * 64), tracking_env='/data/w.env',
                    output='/var/tmp/output', model_snapshot='/data/janghj/snap', config_sha256='c' * 64, context_sha256='x' * 64)

    def host_changed(self, old):
        new = copy.deepcopy(old)
        new.update(authority=s1.AUTHORITY, server='server1', output='/mnt/raid5/out', tracking_env='/mnt/raid5/w.env',
                   model_snapshot='/mnt/raid5/janghj/snap', config_sha256='d' * 64)
        new['parent_checkpoint']['path'] = '/mnt/raid5/p.pt'; new['evidence'][0]['path'] = '/mnt/raid5/e0'
        new['projector']['path'] = '/mnt/raid5/janghj/P.pt'; new['storage'] = dict(path='/mnt/raid5/s.json', bytes=2, sha256='t' * 64)
        for m in new['model_members']:
            m.update(path=m['path'].replace('/data/', '/mnt/raid5/'), real_path='/mnt/raid5/blob', device=9, inode=8, mtime_ns=7, ctime_ns=6)
        return new

    def test_63303_host_bindings_only(self):
        old = self.transfer(); new = self.host_changed(old)
        changed = host_only(field_diff(old, new), s1.HOST_FIELDS)
        self.assertIn('model_members[0].inode', changed)
        self.assertIn('parent_checkpoint.path', changed)
        science = [lambda c: c.update(target_batches=list(range(12, 21))), lambda c: c['parent_checkpoint'].update(sha256='0' * 64),
                   lambda c: c['numeric_sources'][0].update(sha256='0' * 64), lambda c: c['packages'].update(torch='2.9.0'),
                   lambda c: c['model_members'][0].update(sha256='0' * 64), lambda c: c['projector'].update(sha256='0' * 64),
                   lambda c: c['evidence'][0].update(sha256='0' * 64), lambda c: c.update(start_batch=9),
                   lambda c: c.update(context_sha256='0' * 64), lambda c: c.update(new_field=1)]
        for change in science:
            bad = copy.deepcopy(new); change(bad)
            with self.assertRaisesRegex(ValueError, 'SCIENTIFIC_FIELD_CHANGED'):
                host_only(field_diff(old, bad), s1.HOST_FIELDS)

    def test_62873_host_bindings_only(self):
        old = dict(registration_authority=server2.AUTHORITY, instruction=shared.INSTRUCTION, key=flucon.KEY, asset_schema='history',
                   original=dict(checkpoint=dict(path='/s2/cp.pt', sha256='a' * 64, bytes=1), identity=dict(config_sha256='i' * 64),
                                 hparams=dict(layers=[4, 5]), job_id='62079', method='SPHERE', ordered_case_ids_sha256='o' * 64),
                   assets=dict(path='/s2/a.json', sha256='b' * 64, bytes=2), stream=dict(path='/s2/cf.json', sha256='c' * 64, bytes=3),
                   reference=dict(path='/s2/ref.json', sha256='d' * 64, bytes=4), reference_identity='r' * 64,
                   evaluator=dict(path='/s2/native_observer.py', sha256='e' * 64, bytes=5),
                   runtime=dict(python='3.12.12', packages=dict(torch='2.9.1')), output='/s2/out', tracking_env_file='/w.env',
                   config_sha256='f' * 64)
        new = copy.deepcopy(old)
        new.update(registration_authority=s1.AUTHORITY, output='/s1/out', config_sha256='0' * 64,
                   assets=dict(path='/s1/a.json', sha256='1' * 64, bytes=9), reference=dict(path='/s1/ref.json', sha256='2' * 64, bytes=8))
        new['original']['checkpoint']['path'] = '/s1/cp.pt'; new['stream']['path'] = '/s1/cf.json'
        new['evaluator']['path'] = '/s1/native_observer.py'; new['runtime']['python'] = '3.12.0'
        self.assertIn('runtime.python', host_only(field_diff(old, new), flucon.HOST_FIELDS))
        science = [lambda c: c['original']['hparams'].update(layers=[4]), lambda c: c.update(reference_identity='0' * 64),
                   lambda c: c['runtime']['packages'].update(torch='2.9.0'), lambda c: c['evaluator'].update(sha256='0' * 64),
                   lambda c: c['stream'].update(sha256='0' * 64), lambda c: c['original']['checkpoint'].update(sha256='0' * 64),
                   lambda c: c.update(instruction='OTHER'), lambda c: c.update(tracking_env_file='/other.env'),
                   lambda c: c['original'].update(ordered_case_ids_sha256='0' * 64), lambda c: c.update(key='other')]
        for change in science:
            bad = copy.deepcopy(new); change(bad)
            with self.assertRaisesRegex(ValueError, 'SCIENTIFIC_FIELD_CHANGED'):
                host_only(field_diff(old, bad), flucon.HOST_FIELDS)

    def test_manifest_host_fields_are_stat_only(self):
        old = dict(model=dict(snapshot='/snap', members=[dict(path='/snap/a', sha256='a' * 64, bytes=1, device=1, inode=2, ctime_ns=3)]))
        new = copy.deepcopy(old); new['model']['members'][0].update(device=5, inode=6, ctime_ns=7)
        host_only(field_diff(old, new), flucon.MANIFEST_HOST_FIELDS)
        for change in (dict(sha256='0' * 64), dict(bytes=2), dict(path='/other/a')):
            bad = copy.deepcopy(new); bad['model']['members'][0].update(change)
            with self.assertRaisesRegex(ValueError, 'SCIENTIFIC_FIELD_CHANGED'):
                host_only(field_diff(old, bad), flucon.MANIFEST_HOST_FIELDS)


class TransferTests(unittest.TestCase):
    def test_receipt_binds_both_sides_and_current_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder) / 'x.json'; local.write_text('{}')
            receipt = Path(folder) / 'receipt.tsv'
            receipt.write_text(receipt_line('k', '/s2/x.json', local))
            row, found = received(transfer_rows(receipt), local, '/s2/x.json')
            self.assertEqual(row['source_sha256'], found['sha256'])
            with self.assertRaisesRegex(ValueError, 'TRANSFER_RECEIPT_MISMATCH'):
                received(transfer_rows(receipt), local, '/s2/other.json')
            receipt.write_text(receipt_line('k', '/s2/x.json', local, sha='0' * 64))
            with self.assertRaisesRegex(ValueError, 'TRANSFER_RECEIPT_MISMATCH'):
                received(transfer_rows(receipt), local, '/s2/x.json')
            receipt.write_text(receipt_line('k', '/s2/x.json', local))
            local.write_text('{"changed": 1}')
            with self.assertRaisesRegex(ValueError, 'TRANSFER_RECEIPT_MISMATCH'):
                received(transfer_rows(receipt), local, '/s2/x.json')

    def test_stage_requires_the_63303_evidence_set_and_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'zsre-b10-resume'; inputs = root / 'inputs'; (inputs / 'parent').mkdir(parents=True)
            lines, files = [], []
            for index, name in enumerate(f'{i:02d}-f{i}.json' for i in range(22)):
                local = inputs / 'parent' / name; local.write_text(json.dumps(index))
                found = member(local)
                files.append(dict(path='/s2/' + name, bytes=found['bytes'], sha256=found['sha256'], local='/old/' + name, name=name))
                lines.append(receipt_line('zsre-sphere-b10-evidence', '/s2/' + name, local))
            log = inputs / '62538-qwen25-zsre-sphere-resume-b9.out'; log.write_text('x' * 9000 + 'RESOURCE_BLOCKED_STORAGE_KEEP_CHECKPOINT\n')
            lines.append(receipt_line('zsre-sphere-b10-log', s1.SERVER2_ROOT + '/logs/qwen25-zsre-sphere-resume-b9-62538.out', log))
            parent = inputs / 'parent-b10.pt'; parent.write_bytes(b'b10')
            lines.append(receipt_line('zsre-sphere-b10-checkpoint',
                                      s1.SERVER2_ROOT + '/runs/qwen25-zsre-sphere/checkpoint/batch-10-6ab0bea43a595f97.pt', parent))
            old = inputs / 'server4-63303/inputs/parent-evidence.json'
            write_new(old, dict(files=files, failure_tail=log.read_text()[-8000:], parent_job='62538', root=s1.SERVER2_ROOT))
            lines.append(receipt_line('server4-63303', s1.SERVER4_ROOT + '/inputs/parent-evidence.json', old))
            receipt = Path(folder) / 'receipt.tsv'; receipt.write_text(''.join(lines))
            found = member(parent)
            with patch.multiple(s1, ROOT=root, PARENT_SHA=found['sha256'], PARENT_BYTES=found['bytes']):
                with self.assertRaisesRegex(ValueError, 'NEW_STAGE_REQUIRED'):
                    s1.stage(Path(folder) / 'elsewhere', receipt)
                rows = s1.stage(root, receipt)
                self.assertEqual([(r['path'], r['sha256'], r['name']) for r in rows], [(f['path'], f['sha256'], f['name']) for f in files])
                staged = json.loads((inputs / 'parent-evidence.json').read_text())
                self.assertEqual(staged['parent_job'], '62538'); self.assertEqual(staged['root'], s1.SERVER2_ROOT)
            # A different evidence set (one SHA changed at the source) is refused.
            (inputs / 'parent-evidence.json').unlink(); (inputs / 'parent' / 'failure-tail.txt').unlink()
            files[3]['sha256'] = '0' * 64; old.unlink(); write_new(old, dict(files=files, failure_tail=log.read_text()[-8000:],
                                                                              parent_job='62538', root=s1.SERVER2_ROOT))
            receipt.write_text(''.join(lines[:-1]) + receipt_line('server4-63303', s1.SERVER4_ROOT + '/inputs/parent-evidence.json', old))
            with patch.multiple(s1, ROOT=root, PARENT_SHA=found['sha256'], PARENT_BYTES=found['bytes']):
                with self.assertRaisesRegex(ValueError, 'EVIDENCE_NOT_63303_SET'):
                    s1.stage(root, receipt)

    def test_reserve_preallocates_two_slots_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'zsre-b10-resume'; root.mkdir()
            with patch.multiple(s1, ROOT=root, PARENT_BYTES=4096, SLOT_MARGIN=4096, RAW_BUDGET=0, MIN_FREE_RUNTIME=0):
                storage = s1.reserve(root)
                self.assertEqual((storage['slots'], storage['slot_capacity'], storage['old_files_deleted']), (2, 8192, 0))
                for i in (0, 1):
                    s = (root / 'output/checkpoint' / f'slot-{i}.pt').stat()
                    self.assertEqual(s.st_size, 8192); self.assertGreaterEqual(s.st_blocks * 512, s.st_size)
                with self.assertRaisesRegex(ValueError, 'NEW_STORAGE_REQUIRED'):
                    s1.reserve(root)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module(); self.mlp.down_proj = torch.nn.Linear(4, 3, bias=False)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList([Block() for _ in range(3)])


class FluconBindingTests(unittest.TestCase):
    def config(self):
        return dict(key=flucon.KEY, model='qwen25', config_sha256='a' * 64,
                    original=dict(job_id='62079', method='SPHERE', checkpoint=dict(sha256='b' * 64), identity=dict(tokenizer_sha256='c' * 64)),
                    evaluator=dict(sha256='d' * 64), stream=dict(sha256='e' * 64), reference_identity='f' * 64)

    def test_labels_on_the_shared_schema(self):
        value = flucon.tracking_values(self.config(), '1' * 40)
        self.assertEqual((value['server'], value['task_id'], value['role']), ('server1', flucon.TASK, 'eval_only'))
        self.assertEqual((value['generation_schedule'], value['generation_eval_seed']), ('W20_ONLY_FIRST2000', 20261007))
        self.assertEqual((value['instruction_id'], value['source_run_id'], value['baseline']), (shared.INSTRUCTION, '62079', 'SPHERE'))
        self.assertEqual((value['arm'], value['attempt']), ('qwen25-sphere-62079', 'r1-62079'))

    def test_bindings_use_62873_restore_and_reset_after_failure(self):
        previous = (shared.restore, shared.tracking_values)
        with self.assertRaisesRegex(ValueError, 'fixture'):
            with flucon.host_bindings():
                self.assertIs(shared.restore, server2.restore)
                self.assertIs(shared.tracking_values, flucon.tracking_values)
                raise ValueError('fixture')
        self.assertEqual((shared.restore, shared.tracking_values), previous)

    def test_62873_restore_copies_the_w20_weights_exactly(self):
        hp = dict(rewrite_module_tmp='model.layers.{}.mlp.down_proj', layers=[1, 2])
        identity = dict(config_sha256='a' * 64, stream_sha256='b' * 64, code_commit='c' * 40, official_tree_sha256='d' * 64,
                        model_revision='e' * 40, tokenizer_sha256='f' * 64, assets_sha256='1' * 64)
        torch.manual_seed(1)
        weights = {f'model.layers.{l}.mlp.down_proj.weight': torch.randn(3, 4) for l in hp['layers']}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); name = 'batch-20-0123456789abcdef.pt'
            torch.save(dict(schema='official-baseline-checkpoint-v1', batch=20, identity=identity, method='SPHERE', weights=weights,
                            cache_c={'1': torch.eye(4), '2': torch.eye(4)}), folder / name)
            checkpoint = member(folder / name)
            write_new(folder / 'latest.json', dict(batch=20, file=name, final_W20=True, identity_sha256=digest(identity),
                                                  sha256=checkpoint['sha256']))
            original = dict(method='SPHERE', identity=identity, checkpoint=checkpoint, pointer=member(folder / 'latest.json'), hparams=hp)
            model = Tiny(); untouched = model.model.layers[0].mlp.down_proj.weight.detach().clone()
            with flucon.host_bindings():
                receipt = shared.restore(model, original)
            self.assertEqual(receipt['restored_parameters'], sorted(weights))
            parameters = dict(model.named_parameters())
            for key, value in weights.items():
                self.assertTrue(torch.equal(parameters[key], value))
            self.assertTrue(torch.equal(model.model.layers[0].mlp.down_proj.weight, untouched))
            original['identity'] = dict(identity, code_commit='0' * 40)
            with self.assertRaises(AssertionError):
                server2.restore(Tiny(), original)

    def test_tracker_env_keys_are_allowlisted(self):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / 'wandb.env'
            env.write_text('# comment\nWANDB_ENTITY=x\nWANDB_PROJECT=\'p q\'\nWANDB_MODE=online\nODEEDIT_WANDB_PYTHON=/p\n')
            self.assertEqual(flucon.wandb_keys(env), ['ODEEDIT_WANDB_PYTHON', 'WANDB_ENTITY', 'WANDB_MODE', 'WANDB_PROJECT'])
            env.write_text('WANDB_ENTITY=x\nWANDB_API_KEY=secret\n')
            with self.assertRaisesRegex(ValueError, 'TRACKER_ENV_KEYS_NOT_ALLOWLISTED'):
                flucon.wandb_keys(env)

    def test_run_scope_and_node(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'c.json'; write_new(path, dict(registration_authority=server2.AUTHORITY, asset_schema='history', key=flucon.KEY))
            with self.assertRaisesRegex(ValueError, 'SPHERE_FLUCON_S1_SCOPE'):
                flucon.run(path, None)
            other = Path(folder) / 'd.json'; write_new(other, dict(registration_authority=flucon.AUTHORITY, asset_schema='history', key=flucon.KEY))
            with patch.dict(os.environ, SLURMD_NODENAME='server2'):
                with self.assertRaisesRegex(ValueError, 'SPHERE_FLUCON_S1_NODE'):
                    flucon.run(other, None)


class SubmitTests(unittest.TestCase):
    def queue(self, extra=()):
        jobs = [dict(key='63537', gpus=1, parents=[]), dict(key='63538', gpus=1, parents=['63537']),
                dict(key='63555', gpus=1, parents=[]), dict(key='63556', gpus=1, parents=['63555']), *extra]
        return dict(jobs=jobs, admitted_DAG_width=graph_width(jobs))

    def test_chain_after_lane_tail_keeps_width_two(self):
        existing = self.queue()
        self.assertEqual(submit.admit(existing, '63538', 2), 2)
        rows = submit.lane_rows(existing['jobs'], '63538')
        self.assertEqual([r['parents'] for r in rows[-2:]], [['63538'], ['planned-flucon']])
        self.assertEqual(submit.admit(existing, '63556', 2), 2)

    def test_non_tail_unknown_or_over_cap_refused(self):
        with self.assertRaisesRegex(ValueError, 'LANE_TAIL_HAS_GPU_DESCENDANT'):
            submit.admit(self.queue(), '63537', 2)
        with self.assertRaisesRegex(ValueError, 'LANE_TAIL_NOT_OWN_DEVBOX_GPU_JOB'):
            submit.admit(self.queue(), '99999', 2)
        with self.assertRaisesRegex(ValueError, 'SERVER1_GPU_CAP'):
            submit.admit(self.queue([dict(key='63600', gpus=1, parents=[])]), '63538', 2)

    def test_launcher_form_reproduces_the_cancelled_scripts(self):
        env = submit.environment('417a12e7db55af93346be60f79138a6923324de1', 'eee5082a6edc68e873dac3dfe9b2f8aeeb64ba93')
        base = '/mnt/raid5/janghj/ODE-edit/local/baseline-refresh-s2-flucon-20261010/'
        text = submit.launcher(submit.FLUCON_ENV, env, base + 'registration-r1/source',
                               [submit.PYTHON, '-B', '-u', '-m', 'official.runners.server2.flucon_eval', 'run', '--config',
                                base + 'preparation-r2/configs/qwen25-sphere-62079.json', '--lock', base + 'registration-r1/execution-lock.json'])
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), flucon.CANCELLED['script'][0])
        root = s1.SERVER4_ROOT
        text = submit.launcher(submit.RESUME_ENV, env, root + '/source', ['/data/janghj/EasyEdit/.venv/bin/python', '-B', '-u', '-m',
                               'official.runners.server4.sphere_b10_resume', 'run', '--root', root])
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), s1.CANCELLED['script_sha256'])

    def test_template_replacement_requires_every_token(self):
        self.assertEqual(submit.replaced('a b c', [('a', 'x'), ('c', 'y')]), 'x b y')
        with self.assertRaisesRegex(ValueError, 'LAUNCHER_TEMPLATE_TOKEN'):
            submit.replaced('a b c', [('z', 'x')])


if __name__ == '__main__':
    unittest.main()
