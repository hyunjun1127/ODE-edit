"""New sequential fixtures only; unchanged B1 GPU/math receipts are reused."""
import ast, copy, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .common import *
from .run import BatchTransaction, drive, w0_subset
from .submit import admission_plan, argv
from .collect import validate_commit, harmonic
from project.run_scripts.jlz_realization.writer import rng_snapshot, rng_equal
from project.run_scripts.jlz_realization.observe import active_flags, reduce_rows

class Fake:
    def __init__(self):
        self.weights = {0: torch.zeros(2, 3)}; self.last_virtual = {'stale': 'previous-batch'}; self.capture_virtual = False
    def guard(self): return 'nonselected-unchanged'
    def hook_signature(self): return 'no-hooks'

class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13); torch.set_num_threads(2)

    def test_success_finish_persists_and_failure_restores_prefix(self):
        a = Fake(); H = {0: torch.zeros(3, 3)}; bench = SimpleNamespace(contexts=[['{}']])
        with BatchTransaction(a, H, bench) as tx:
            self.assertEqual(a.last_virtual, {})
            a.weights[0].add_(1); H[0].add_(2); a.last_virtual = {0: 'terminal'}; tx.finish()
        prefix = state(a, H); rng = rng_snapshot()
        self.assertEqual(float(a.weights[0].sum()), 6); self.assertEqual(a.last_virtual, {})
        tx2 = BatchTransaction(a, H, bench)
        with self.assertRaisesRegex(RuntimeError, 'WRITE_FAULT'):
            with tx2:
                a.weights[0].add_(5); H[0].add_(7); a.last_virtual = {1: 'wrong'}
                bench.contexts.append(['mutated']); torch.randn(5)
                raise RuntimeError('WRITE_FAULT')
        self.assertTrue(tx2.rollback_verified and rng_equal(rng)); self.assertEqual(state(a, H), prefix)
        self.assertEqual(bench.contexts, [['{}']]); self.assertEqual(a.last_virtual, {})

    def test_independent_arm_storage(self):
        a, b = Fake(), Fake(); ah = {0: torch.zeros(3, 3)}; bh = {0: torch.zeros(3, 3)}
        before = state(b, bh)
        with BatchTransaction(a, ah, SimpleNamespace(contexts=[])) as tx:
            a.weights[0].add_(1); ah[0].add_(1); tx.finish()
        self.assertEqual(state(b, bh), before)
        self.assertNotEqual(ah[0].data_ptr(), bh[0].data_ptr())

    def test_horizon20_partial_and_milestones_no_future(self):
        rows = list(range(2000)); sequence = list(batches(rows, 100))
        self.assertEqual(len(sequence), 20); self.assertEqual(sequence[-1][0], 20)
        for n, current, seen in sequence:
            self.assertEqual(current, rows[(n-1)*100:n*100]); self.assertEqual(seen, rows[:n*100])
            self.assertEqual(len(selected_for_post(current, seen, n)), n*100 if n in MILESTONES else 100)
        self.assertEqual([len(x[1]) for x in batches(list(range(7)), 3)], [3, 3, 1])

    def test_seen_prefix_active_not_future(self):
        def record(i, target):
            return dict(case_id=i, requested_rewrite=dict(subject='x', relation_id='r', target_new=dict(id=target, str=target)))
        records = [record(1, 'a'), record(2, 'b')]
        self.assertTrue(active_flags(records[:1])[1]); self.assertFalse(active_flags(records)[1])

    def test_cap2_parallel_and_common_external_barrier(self):
        self.assertEqual(admission_plan(dict(jobs=[]), 3), [])
        inventory = dict(jobs=[dict(job='1', gpus=1), dict(job='2', gpus=1)])
        self.assertEqual(admission_plan(inventory, 3), ['1', '2'])
        with self.assertRaises(RuntimeError): admission_plan(inventory, 1)
        r = dict(host_mib=59392, collector_host_mib=24576, wall='1-00:00:00', collector_wall='04:00:00')
        for arm in ARMS:
            args = argv(arm, '', Path('/exact/attempt'), r)
            self.assertIn('--gres=gpu:1', args); self.assertIn('--mem=59392M', args)
            self.assertIn('--cpus-per-task=8', args); self.assertIn('--export=NONE', args)
            self.assertFalse(any(x.startswith('--dependency=') for x in args))
        self.assertNotIn('--gres=gpu:1', argv('collector', 'afterany:1:2', Path('/exact/attempt'), r))

    def test_exact_observer_identity_not_only_count(self):
        row = dict(identity='a', case_id=1, kind='R', prompt_index=0, endpoint='W1', new_token_identity='n', true_token_identity='t',
            new_nll=.5, true_nll=1., margin_true_minus_new=.5, new_token_count=1, true_token_count=1,
            new_token_correct=1, true_token_correct=0, new_strict=True, true_strict=False)
        expected = [{k: row[k] for k in ('identity', 'case_id', 'kind', 'prompt_index', 'new_token_identity', 'true_token_identity')}]
        self.assertEqual(validate_rows([row], expected, [1], 'W1')['R']['numerator'], 1)
        with self.assertRaises(RuntimeError): validate_rows([row | dict(identity='another')], expected, [1], 'W1')
        with self.assertRaises(RuntimeError): validate_rows([row | dict(new_token_identity='changed')], expected, [1], 'W1')

    def test_collector_history_once_weight_state_join(self):
        before = dict(W={'0': 'w0'}, H={'0': 'h0'}); after = dict(W={'0': 'w1'}, H={'0': 'h1'})
        expected = dict(ids=[1, 2], identity='pack')
        entry = dict(source='s', config='c', ids=[1, 2], native_pack='pack', state=before, RNG='rng')
        commit = dict(source='s', config='c', ids=[1, 2], native_pack='pack', before=before, after=after,
            RNG_before='rng', RNG_after='rng', fit_count=1, history_appends=1, observer_no_mutation=True, checkpoint_saved=False)
        writer = dict(history_appends=1, history=[dict(layer=0, append_count=1, columns=2, rewrite_only=True, KL_in_history=False,
            CPU_FP32=True, before='h0', after='h1')], layers={'0': dict(weight_after='w1',
            solver=dict(numerical_projection_verified=True), ideal_effective_parity=dict(pass_=True))})
        self.assertEqual(validate_commit(commit, writer, entry, expected, before, 's', 'c', [0]), after)
        bad = copy.deepcopy(writer); bad['history'][0]['append_count'] = 2
        with self.assertRaises(RuntimeError): validate_commit(commit, bad, entry, expected, before, 's', 'c', [0])
        bad = copy.deepcopy(writer); bad['history'][0]['before'] = 'oldH'
        with self.assertRaises(RuntimeError): validate_commit(commit, bad, entry, expected, before, 's', 'c', [0])

    def test_harmonic_zero_is_measured_zero(self):
        self.assertEqual(harmonic({k: dict(rate=0.) for k in ('R', 'P', 'N')}), 0)
        self.assertAlmostEqual(harmonic({k: dict(rate=.9) for k in ('R', 'P', 'N')}), .9)

    def test_source_no_disk_tensor_no_branch_probe(self):
        folder = ROOT / 'project/run_scripts/jlz_realized_writer_sequential'
        for p in folder.glob('*.py'):
            if p.name.startswith('test_'): continue
            calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n, ast.Call)]
            self.assertNotIn('torch.save', calls); self.assertNotIn('np.save', calls)
        tree = ast.parse((folder / 'run.py').read_text())
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'fit' for n in ast.walk(tree)), 1)
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'apply_sequential' for n in ast.walk(tree)), 1)
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'commit' for n in ast.walk(tree)))

    def test_tiny_two_batches_MD_CD_actual_frozen_helpers(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from project.run_scripts.jlz_realized_writer.capture import Adapter, capture_native_sites, virtual_terminal, cache_identity
        from project.run_scripts.jlz_realized_writer.writer import apply_sequential
        from project.run_scripts.jlz_shared_budget.entry import prepare_entry
        from project.run_scripts.jlz_shared_budget.optimize import fit
        from project.run_scripts.jlz_shared_budget.telemetry import Events
        config = LlamaConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=3,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=31, max_position_embeddings=64, _attn_implementation='eager')
        initial_model = LlamaForCausalLM(config).state_dict()
        def pack(B, ids, salt):
            data = dict(identity=f'CPU:{ids}', n_requests=B, n_rw=2, record_ids=ids, canonical_rows=list(range(0, 3*B, 3)),
                context_group_slices=[(0, 1), (1, 2)], lookup=[1]*(3*B), row_kind=['rewrite', 'rewrite', 'kl']*B,
                row_request=[i//3 for i in range(3*B)], tokens=dict(input_ids=(torch.arange(15*B).reshape(3*B, 5)+salt)%29+1,
                attention_mask=torch.ones(3*B, 5, dtype=torch.long)), targets=torch.full((3*B, 5), -100))
            for i in range(3*B):
                if data['row_kind'][i] == 'rewrite': data['targets'][i, 3:] = torch.tensor([2, 3])
            return data
        ends = {}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stats = {}
            for l in (0, 1):
                path = root / f'stats{l}.npz'; np.savez(path, **{'mom2.mom2': np.eye(16, dtype=np.float32), 'mom2.count': np.array(1)})
                stats[str(l)] = str(path)
            for arm in ARMS:
                model = LlamaForCausalLM(config); model.load_state_dict(initial_model)
                a = Adapter(model, dict(eligible_layers=[0, 1], anchor_layer=1, nll_layer=2, kl_factor=.0625))
                bench = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0), contexts=[['{}'], ['x{}']])
                H = {l: torch.zeros(16, 16) for l in a.sites}; previous = state(a, H); last_anchors = None
                for number, p in enumerate((pack(1, [11], 0), pack(3, [21, 22, 23], 4)), 1):
                    with BatchTransaction(a, H, bench) as tx:
                        self.assertEqual(a.last_virtual, {})
                        entry = prepare_entry(a, bench, p, H, stats, 1)
                        for l in a.sites: self.assertEqual(tensor_sha(entry['entry_weights'][l]), previous['W'][str(l)])
                        initial = capture_native_sites(a, entry, a.sites); cached = cache_identity(entry)
                        a.capture_virtual = True
                        plan, receipt = fit(a, entry, Events(root / f'{arm}-{number}.jsonl', 'CPU:' + arm, number), root / arm / f'fit{number}')
                        a.capture_virtual = False; virtual = virtual_terminal(a, entry, plan)
                        self.assertEqual(state(a, H), previous)
                        self.assertEqual(len(a.last_virtual), 3*p['n_requests'])
                        prior_H = {l: h.clone() for l, h in H.items()}
                        wr = apply_sequential(a, entry, plan, virtual, initial, H, arm, root / arm / f'writer{number}')
                        final = capture_native_sites(a, entry, a.sites)
                        for l in a.sites:
                            k = final['mean'][l]; native = k.T.contiguous().T
                            self.assertTrue(torch.equal(H[l], prior_H[l] + native@native.T))
                            self.assertEqual(wr['history'][a.sites.index(l)]['before'], previous['H'][str(l)])
                        self.assertEqual(cache_identity(entry), cached)
                        self.assertLessEqual(receipt['request_evaluations'], 25*p['n_requests'])
                        if last_anchors is not None: self.assertNotEqual(last_anchors, {l: tensor_sha(v) for l, v in entry['anchors'].items()})
                        last_anchors = {l: tensor_sha(v) for l, v in entry['anchors'].items()}
                        tx.finish()
                    self.assertEqual(a.last_virtual, {}); self.assertFalse(tx.rollback_verified)
                    self.assertNotEqual(state(a, H), previous); previous = state(a, H)
                ends[arm] = previous
        self.assertNotEqual(ends['MD'], ends['CD'])

if __name__ == '__main__':
    unittest.main()
