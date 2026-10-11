"""CPU fixtures for the FE/CAKE saved-W20 Flu/Con restore adapter; no pretrained model or GPU."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from official.experiments.prepare import digest, write_new
from official.runners import fe_cake_flucon as adapter
from official.runners.fe_cake_flucon_prepare import locate, transfer_rows
from official.runners.fe_cake_flucon_submit import lane_rows
from official.runners.server1 import flucon_eval as shared
from official.runners.server1.common import member
from official.runners.server1.native import tensor_sha
from official.runners.server1.submit import graph_width

HP = dict(rewrite_module_tmp='model.layers.{}.mlp.down_proj', layers=[1, 2])
IDS = list(range(7, 2007))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(4, 3)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block() for _ in range(3)])


def edited(dtype):
    torch.manual_seed(1)
    names = adapter.selected_names(HP)
    return {name: (torch.randn(3, 4) * 7).to(dtype) for name in sorted(names)}


class Fixture:
    """Writes one checkpoint of either schema plus its pointer/receipt into a temp folder."""
    def __init__(self, folder, schema, method='CAKE', dtype=torch.float32, mutate=None):
        self.folder, self.weights = Path(folder), edited(dtype)
        identity = dict(config_sha256='a' * 64, stream_sha256='b' * 64, code_commit='c' * 40,
                        official_tree_sha256='d' * 40, model_revision='e' * 40, tokenizer_sha256='f' * 64,
                        assets_sha256='1' * 64)
        if schema == adapter.OFFICIAL:
            cursor = dict(batch=20, edits=2000) if method == 'CAKE' else dict(completed_batch=20)
            contexts = [['{}']] if method == 'CAKE' else dict(successful_calls=20)
            payload = dict(schema=schema, batch=20, identity=identity, method=method, weights=self.weights,
                           cache_c={'1': torch.eye(4), '2': torch.eye(4)}, contexts=contexts, evaluation_cursor=cursor)
        else:
            identity = dict(native_identity_sha256='9' * 64, z_policy='CURRENT_BATCH_ENTRY_NO_FUTURE_TARGETS',
                            model_revision='e' * 40, tokenizer_sha256='f' * 64)
            payload = dict(schema=schema, completed_batches=20, completed_edits=2000, weights=self.weights,
                           history=torch.zeros(2, 4, 4), seen_case_ids=list(IDS), identity_sha256='9' * 64,
                           z_policy=identity['z_policy'],
                           config=dict(model_dtype='bfloat16', bs=100, num_edits=2000, algs=dict(add_old_keys=True),
                                       diagnostic_identity_sha256='9' * 64, llms=dict(HP)))
        if mutate:
            mutate(payload)
        torch.save(payload, self.folder / 'W20.pt')
        checkpoint = member(self.folder / 'W20.pt')
        if schema == adapter.OFFICIAL:
            write_new(self.folder / 'latest.json', dict(batch=20, file='W20.pt', final_W20=True,
                      identity_sha256=digest(identity), sha256=checkpoint['sha256']))
        else:
            write_new(self.folder / 'latest.json', dict(path='/elsewhere/B20.pt', sha256=checkpoint['sha256'],
                      bytes=checkpoint['bytes'], completed_batches=20, completed_edits=2000, identity_sha256='9' * 64))
        self.original = dict(job_id='12345', method=method if schema == adapter.OFFICIAL else 'FE_AUTHOR_NATIVE',
            checkpoint_schema=schema, identity=identity, checkpoint=checkpoint, pointer=member(self.folder / 'latest.json'),
            hparams=dict(HP), ordered_case_ids_sha256=digest(IDS),
            payload_tensor_sha256={k: adapter.raw_sha(v) for k, v in self.weights.items()},
            recorded_selected_weights_sha256=None)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)

    def test_raw_sha_matches_server1_tensor_sha_for_fp32(self):
        value = torch.randn(5, 6)
        self.assertEqual(adapter.raw_sha(value), tensor_sha(value))
        self.assertNotEqual(adapter.raw_sha(value.to(torch.bfloat16)), adapter.raw_sha(value))

    def check_restored(self, model, weights, untouched):
        parameters = dict(model.named_parameters())
        for name, value in weights.items():
            self.assertTrue(torch.equal(parameters[name], value.to(torch.float32)))
            self.assertEqual(parameters[name].dtype, torch.float32)
        for name, value in untouched.items():
            self.assertTrue(torch.equal(parameters[name], value))

    def test_official_cake_copies_fp32_and_keeps_everything_else(self):
        fixture = Fixture(self.tmp.name, adapter.OFFICIAL)
        model = Tiny()
        names = adapter.selected_names(HP)
        untouched = {k: v.detach().clone() for k, v in model.named_parameters() if k not in names}
        receipt = adapter.restore(model, fixture.original)
        self.check_restored(model, fixture.weights, untouched)
        self.assertEqual(receipt['cast'], 'none: FP32 copied exactly')
        self.assertEqual(receipt['restored_selected_weights_fp32_sha256'], fixture.original['payload_tensor_sha256'])
        json.dumps(receipt)

    def test_official_fe_history_must_match_recorded_w20_hashes(self):
        fixture = Fixture(self.tmp.name, adapter.OFFICIAL, method='MEMIT_FE_HISTORY')
        fixture.original['recorded_selected_weights_sha256'] = dict(fixture.original['payload_tensor_sha256'])
        self.assertEqual(adapter.restore(Tiny(), fixture.original)['recorded_W20_hash_match'], 'EXACT')
        name = sorted(fixture.weights)[0]
        fixture.original['recorded_selected_weights_sha256'][name] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'RESTORED_NOT_RECORDED_W20'):
            adapter.restore(Tiny(), fixture.original)

    def test_official_rejects_identity_method_and_cursor_changes(self):
        fixture = Fixture(self.tmp.name, adapter.OFFICIAL)
        fixture.original['method'] = 'MEMIT'
        with self.assertRaisesRegex(ValueError, 'OFFICIAL_CHECKPOINT_IDENTITY'):
            adapter.restore(Tiny(), fixture.original)
        other = Path(self.tmp.name) / 'cursor'; other.mkdir()
        fixture = Fixture(other, adapter.OFFICIAL, mutate=lambda p: p['evaluation_cursor'].update(batch=19))
        with self.assertRaisesRegex(ValueError, 'CAKE_FINAL_CURSOR'):
            adapter.restore(Tiny(), fixture.original)

    def test_payload_hash_mismatch_writes_nothing(self):
        fixture = Fixture(self.tmp.name, adapter.OFFICIAL)
        name = sorted(fixture.weights)[-1]
        fixture.original['payload_tensor_sha256'][name] = '0' * 64
        model = Tiny(); before = {k: v.detach().clone() for k, v in model.named_parameters()}
        with self.assertRaisesRegex(ValueError, 'RESTORE_PAYLOAD_TENSOR_SHA'):
            adapter.restore(model, fixture.original)
        for k, v in model.named_parameters():
            self.assertTrue(torch.equal(v, before[k]))

    def test_changed_checkpoint_file_rejected(self):
        fixture = Fixture(self.tmp.name, adapter.OFFICIAL)
        with (Path(self.tmp.name) / 'W20.pt').open('ab') as handle:
            handle.write(b'0')
        with self.assertRaisesRegex(ValueError, 'IMMUTABLE_INPUT_MEMBER_CHANGED'):
            adapter.restore(Tiny(), fixture.original)

    def test_native_bf16_cast_is_exact_and_keeps_everything_else(self):
        fixture = Fixture(self.tmp.name, adapter.NATIVE, dtype=torch.bfloat16)
        model = Tiny(); names = adapter.selected_names(HP)
        untouched = {k: v.detach().clone() for k, v in model.named_parameters() if k not in names}
        receipt = adapter.restore(model, fixture.original)
        self.check_restored(model, fixture.weights, untouched)
        parameters = dict(model.named_parameters())
        for name, value in fixture.weights.items():
            self.assertTrue(torch.equal(parameters[name].to(torch.bfloat16), value))
            self.assertEqual(receipt['restored_selected_weights_fp32_sha256'][name], tensor_sha(parameters[name]))
        self.assertEqual(receipt['stored_dtype'], 'bfloat16')

    def test_native_rejects_order_dtype_and_config(self):
        cases = [(lambda p: p['seen_case_ids'].reverse(), 'NATIVE_STREAM_ORDER'),
                 (lambda p: p['config'].update(model_dtype='float32'), 'NATIVE_CONFIG'),
                 (lambda p: p['config']['llms'].update(layers=[0, 1]), 'NATIVE_CONFIG'),
                 (lambda p: p.update(completed_batches=15), 'NATIVE_CHECKPOINT_IDENTITY')]
        for index, (mutate, code) in enumerate(cases):
            folder = Path(self.tmp.name) / str(index); folder.mkdir()
            fixture = Fixture(folder, adapter.NATIVE, dtype=torch.bfloat16, mutate=mutate)
            with self.assertRaisesRegex(ValueError, code):
                adapter.restore(Tiny(), fixture.original)
        folder = Path(self.tmp.name) / 'fp32'; folder.mkdir()
        fixture = Fixture(folder, adapter.NATIVE, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, 'RESTORE_SHAPE_DTYPE_FINITE'):
            adapter.restore(Tiny(), fixture.original)

    def test_nonfinite_and_missing_weights_rejected(self):
        fixture = Fixture(self.tmp.name, adapter.OFFICIAL,
                          mutate=lambda p: p['weights'][sorted(p['weights'])[0]].fill_(float('inf')))
        with self.assertRaisesRegex(ValueError, 'RESTORE_SHAPE_DTYPE_FINITE'):
            adapter.restore(Tiny(), fixture.original)
        folder = Path(self.tmp.name) / 'missing'; folder.mkdir()
        fixture = Fixture(folder, adapter.OFFICIAL, mutate=lambda p: p['weights'].pop(sorted(p['weights'])[0]))
        with self.assertRaisesRegex(ValueError, 'RESTORE_SELECTED_WEIGHT_SET'):
            adapter.restore(Tiny(), fixture.original)

    def test_unknown_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CHECKPOINT_SCHEMA_NOT_ALLOWLISTED'):
            adapter.restore(Tiny(), dict(checkpoint_schema='historical-W-method-state'))


class BindingTests(unittest.TestCase):
    def config(self, server):
        return dict(server=server, key='gptj-fe_author_native-63389', model='gptj', config_sha256='a' * 64,
            original=dict(job_id='63389', method='FE_AUTHOR_NATIVE', checkpoint=dict(sha256='b' * 64),
                          identity=dict(tokenizer_sha256='c' * 64)),
            evaluator=dict(sha256='d' * 64), stream=dict(sha256='e' * 64), reference_identity='f' * 64)

    def test_host_labels_on_the_single_shared_schema(self):
        for server in ('server1', 'server2'):
            value = adapter.tracking_values(self.config(server), '1' * 40)
            self.assertEqual((value['server'], value['task_id'], value['role']), (server, adapter.TASK, 'eval_only'))
            self.assertEqual(value['generation_schedule'], 'W20_ONLY_FIRST2000')
            self.assertEqual(value['instruction_id'], shared.INSTRUCTION)
            self.assertEqual((value['source_run_id'], value['baseline']), ('63389', 'FE_AUTHOR_NATIVE'))

    def test_bindings_restored_after_failure(self):
        previous = (shared.restore, shared.tracking_values)
        with self.assertRaisesRegex(ValueError, 'fixture'):
            with adapter.host_bindings():
                self.assertIs(shared.restore, adapter.restore)
                self.assertIs(shared.tracking_values, adapter.tracking_values)
                raise ValueError('fixture')
        self.assertEqual((shared.restore, shared.tracking_values), previous)

    def test_lanes_are_serial_chains_and_count_with_existing_lane(self):
        existing = [dict(key='63526', gpus=1, parents=[]), dict(key='63527', gpus=1, parents=['63526']),
                    dict(key='63528', gpus=1, parents=['63527'])]
        one = lane_rows([['a', 'b']])
        self.assertEqual(one[1]['parents'], ['planned-a'])
        self.assertEqual(graph_width(existing + one), 2)
        self.assertEqual(graph_width(lane_rows([['a', 'b'], ['c', 'd']])), 2)

    def test_copied_input_must_match_transfer_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); local = folder / 'inputs/key/run/latest.json'; local.parent.mkdir(parents=True)
            local.write_text('{}'); found = member(local)
            row = ['key', 'rke-server1', '/source/run/latest.json', found['sha256'], str(found['bytes']), str(local),
                   found['sha256'], 'OK', '0']
            receipt = folder / 'receipt.tsv'; receipt.write_text('\t'.join(row) + '\n')
            spec = dict(files=dict(pointer=('/source', 'run/latest.json')))
            paths, provenance = locate('key', spec, folder / 'inputs', transfer_rows(receipt))
            self.assertEqual(paths['pointer'], local)
            self.assertEqual(provenance['pointer']['source_path'], '/source/run/latest.json')
            local.write_text('{"changed": 1}')
            with self.assertRaisesRegex(ValueError, 'TRANSFER_RECEIPT_POINTER'):
                locate('key', spec, folder / 'inputs', transfer_rows(receipt))


if __name__ == '__main__':
    unittest.main()
