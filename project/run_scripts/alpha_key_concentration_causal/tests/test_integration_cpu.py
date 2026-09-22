"""Independent bounded integration red fixtures: no GPU/model construction.

These assert contracts across module boundaries, not actual-model gate PASS.
Production modules remain owned by the coordinator/other implementation agents.
"""
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch

from project.run_scripts.alpha_key_concentration_causal import (
    common, component_runner, geometry, geometry_runner, observer, reduce, runner,
    technical, writer_runner,
)


ROOT = Path(__file__).resolve().parents[4]
INPUT_ROOT = Path('/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/inputs/design')


def paired_raw():
    raw = {}
    for category, count in (('rewrite', 1), ('rephrase', 2), ('locality', 10)):
        for side, nll in (('new', 1.), ('true', 2.)):
            raw[f'{category}_target_{side}'] = [
                {'case_id': 1, 'prompt_index': i, 'prompt': f'{category} {i}', 'target': side,
                 'nll': nll, 'target_token_ids': [3], 'token_predictions': [3],
                 'token_correct': [True], 'all_tokens_correct': True,
                 'endpoint_id': 'same-W', 'full_vocab': {'prompt_token_ids_sha256': f'{category}-{i}'}}
                for i in range(count)]
    return raw


class ActionFixtureModel(torch.nn.Module):
    """Tiny CPU-only affine model; never constructs a pretrained model."""
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(9)])
        for layer in self.model.layers:
            layer.mlp = torch.nn.Module()
            layer.mlp.down_proj = torch.nn.Linear(3, 2, bias=False)
            with torch.no_grad():
                layer.mlp.down_proj.weight.copy_(torch.tensor([[1., 2., 3.], [-1., 0., 1.]]))

    @staticmethod
    def keys(ids):
        x = ids.to(torch.float32)
        return torch.stack((x, x + 1, 2 * x - 1), dim=-1)

    def forward(self, input_ids, attention_mask, raise_after=False):
        result = self.model.layers[4].mlp.down_proj(self.keys(input_ids))
        if raise_after:
            raise RuntimeError('CPU_FIXTURE_AFTER_HOOK')
        return result


def sham_fixture():
    value = {'factors': {layer: {field: torch.tensor([[1., 2.], [3., 4.]])
                               for field in ('K', 'R', 'delta')} for layer in (4, 5, 6, 7, 8)}}
    observed = {'current': {'metrics': {metric: {'rows': [
        {'identity': metric + '-id', 'new_nll': 1., 'true_nll': 2.}]}
        for metric in ('RS', 'PS', 'NS')}}, 'N': {'rows': [{'identity': 'n-id', 'true_nll': 3.}]}}
    return value, observed


class IntegrationCPU(unittest.TestCase):
    def test_independent_reducer_covers_native_rephrase_keys(self):
        raw = paired_raw()
        owner = observer.reduce_rpn(raw)
        independent = reduce.independent_pairs(raw)
        self.assertEqual(set(independent), {'RS', 'PS', 'NS'})
        for tag in owner:
            self.assertEqual(independent[tag]['denominator'], owner[tag]['denominator'])
            self.assertEqual(independent[tag]['numerator'], owner[tag]['numerator'])
            self.assertEqual([r['identity'] for r in independent[tag]['rows']],
                             [r['identity'] for r in owner[tag]['rows']])

    def test_independent_reducer_ties_not_success(self):
        raw = paired_raw()
        for rows in raw.values():
            for row in rows:
                row['nll'] = 1.
        reduced = reduce.independent_pairs(raw)
        self.assertEqual({k: v['numerator'] for k, v in reduced.items()}, {'RS': 0, 'PS': 0, 'NS': 0})

    def test_independent_reducer_rejects_cross_endpoint_pair(self):
        raw = paired_raw()
        raw['rewrite_target_true'][0]['endpoint_id'] = 'other-W'
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            reduce.independent_pairs(raw)

    def test_independent_reducer_rejects_cross_prompt_tokens(self):
        raw = paired_raw()
        raw['rewrite_target_true'][0]['full_vocab']['prompt_token_ids_sha256'] = 'other-tokenization'
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            reduce.independent_pairs(raw)

    def test_actual_component_call_panel_contract(self):
        if not INPUT_ROOT.is_dir():
            self.skipTest('sealed design inputs unavailable')
        panels = {key: json.loads((INPUT_ROOT / filename).read_text()) for key, filename in
                  (('geometry', 'geometry-panels.json'), ('history', 'history512.json'), ('neighborhood', 'neighborhood512.json'))}
        parsed = ast.parse(inspect.getsource(writer_runner.run_writers))
        call = next(n for n in ast.walk(parsed) if isinstance(n, ast.Call) and
                    isinstance(n.func, ast.Name) and n.func.id == 'run_components')
        # Evaluate only the argument expression (name/subscript), never runner.
        self.assertIsInstance(call.args[6], (ast.Name, ast.Subscript))
        arg = eval(compile(ast.Expression(call.args[6]), '<panel-binding-fixture>', 'eval'), {}, {'panels': panels})
        ids = [i for cohort in panels['geometry']['cohorts'].values() for i in cohort['calibration_case_ids']]
        fake = types.SimpleNamespace(byid={i: {'case_id': i} for i in ids})
        found, records = component_runner._calibration(fake, arg)
        self.assertEqual(len(found), 512)
        self.assertEqual(set(found), set(ids))
        self.assertEqual([r['case_id'] for r in records], found)

    def test_gate_loads_state_before_g1(self):
        tree = ast.parse(inspect.getsource(runner.main))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        g1 = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'run_g1')
        states = [n for n in calls if isinstance(n.func, ast.Attribute) and
                  isinstance(n.func.value, ast.Name) and n.func.value.id == 'rt' and
                  n.func.attr == 'set_state' and n.lineno < g1.lineno]
        self.assertTrue(states, 'Runtime initializes M=None; G1 requires a loaded W/M entry')

    def test_shared_geometry_statistics_strict_json(self):
        arrays = np.array([[1., 0., 0.], [1., 1., 0.], [0., 0., 0.]], dtype=np.float32)
        summaries = [geometry.geometry_summary(arrays), geometry.geometry_summary(np.zeros_like(arrays))]
        for value in summaries:
            # np.bool/np.float cannot escape create-once serializer.
            json.dumps(common.clean(value), allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            receipt = common.save(Path(directory) / 'fixture.json', {'bool': np.bool_(True), 'float': np.float64(1.5)})
            self.assertEqual(json.loads(Path(receipt['path']).read_text()), {'bool': True, 'float': 1.5})

    def test_missing_ready_is_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'READY.json'
            common.save(file, {'status': 'NOT_OBSERVED'})
            with self.assertRaises(RuntimeError):
                runner.require_ready(file)

    def test_native_loader_returns_records_not_tuple(self):
        path = ROOT / 'scripts/fixed_counterfact.py'
        spec = importlib.util.spec_from_file_location('alpha_causal_red_fixed_data', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        root = Path('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')
        if not root.is_dir():
            self.skipTest('fixed10k read-only input unavailable')
        rows = module.load_prefix(root, 2)
        self.assertIsInstance(rows, list)
        self.assertEqual(len(rows), 2)
        self.assertIn('requested_rewrite', rows[0])
        self.assertEqual(int(rows[0]['case_id']), 16186)

    def test_actual_h512_order_joins_received_ledger(self):
        if not INPUT_ROOT.is_dir():
            self.skipTest('sealed design inputs unavailable')
        root = Path('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')
        if not root.is_dir():
            self.skipTest('fixed10k read-only input unavailable')
        records = json.loads((root / 'counterfact.json').read_text())
        panel = json.loads((INPUT_ROOT / 'history512.json').read_text())
        mask = observer.history_masks(panel, records[:5000])
        selection = technical.timestamp_selection(panel, records)
        self.assertEqual(mask['statistics_weight_sum'], 512)
        self.assertEqual(sum(map(len, selection.values())), 512)
        self.assertEqual(mask['functional_current_valid'] + mask['functional_superseded'], 512)

    def test_scope_no_followup_phase(self):
        self.assertEqual(runner.PHASES, ('gate', 'geometry', 'writers', 'reduce'))
        self.assertEqual(writer_runner.ENTRIES, (50, 70, 80, 90))
        self.assertEqual(writer_runner.BRANCHES, ('NATIVE', 'SHAM', 'H5', 'H6', 'H56', 'MASS56'))

    def test_protected_action_all_valid_coefficients_and_nonmutation(self):
        model = ActionFixtureModel()
        rt = types.SimpleNamespace(model=model, weights={4: model.model.layers[4].mlp.down_proj.weight})
        factor = {'C': torch.tensor([[1., -2.], [0., 1.], [2., 0.]]),
                  'delta': torch.tensor([[1., 0., -1.], [0., 2., 1.]])}
        ids = torch.tensor([[1, 2, 99], [99, 3, 4]])
        mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
        before = model(input_ids=ids, attention_mask=mask)
        weight = rt.weights[4].detach().clone()
        rng = torch.get_rng_state().clone()
        with writer_runner.ProtectedAction(rt, 4, factor) as observed:
            result = model(input_ids=ids, attention_mask=mask)
            # Repeated forwards are separately identified, not merged as one input.
            model(input_ids=ids + 1, attention_mask=mask)
        self.assertTrue(torch.equal(before, result))
        self.assertTrue(torch.equal(weight, rt.weights[4]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len(observed.rows), 4)
        self.assertEqual(len(observed.binding.records), 2)
        self.assertEqual(observed.binding.frames, [])
        for row, keys in zip(observed.rows[:2], model.keys(ids), strict=True):
            valid = keys[mask[row['batch_row']].bool()]
            c = (valid @ factor['C']).double()
            a = torch.nn.functional.linear(valid, factor['delta']).double()
            self.assertEqual(row['valid_tokens'], 2)
            self.assertEqual(row['forward_index'], 0)
            self.assertEqual(row['request_coefficient_sum'], c.sum(0).tolist())
            self.assertEqual(row['request_coefficient_energy'], c.square().sum(0).tolist())
            self.assertEqual(row['last_valid_input_coefficients'], c[-1].tolist())
            self.assertEqual(row['local_action_energy'], float(a.square().sum()))
            self.assertEqual(row['last_valid_input_action_energy'], float(a[-1].square().sum()))
        self.assertEqual([x['forward_index'] for x in observed.rows], [0, 0, 1, 1])
        self.assertFalse(model._forward_pre_hooks or model._forward_hooks)
        self.assertFalse(model.model.layers[4].mlp.down_proj._forward_pre_hooks)

    def test_protected_action_exception_cleans_input_and_layer_hooks(self):
        model = ActionFixtureModel()
        rt = types.SimpleNamespace(model=model, weights={4: model.model.layers[4].mlp.down_proj.weight})
        factor = {'C': torch.ones(3, 2), 'delta': torch.ones(2, 3)}
        with self.assertRaisesRegex(RuntimeError, 'CPU_FIXTURE_AFTER_HOOK'):
            with writer_runner.ProtectedAction(rt, 4, factor) as observed:
                model(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2), raise_after=True)
        self.assertEqual(observed.binding.frames, [])
        self.assertFalse(model._forward_pre_hooks or model._forward_hooks)
        self.assertFalse(model.model.layers[4].mlp.down_proj._forward_pre_hooks)

    def test_sham_exact_control_and_unequal_diagnostics_fail_closed(self):
        original, observed = sham_fixture()
        with tempfile.TemporaryDirectory() as directory:
            okay = Path(directory) / 'exact.json'
            writer_runner.verify_sham(original, copy.deepcopy(original), observed, copy.deepcopy(observed), okay)
            self.assertEqual(json.loads(okay.read_text())['status'], 'PASS')
            for change in ('factor', 'current_nll', 'N512_nll'):
                candidate, observation = copy.deepcopy(original), copy.deepcopy(observed)
                if change == 'factor':
                    candidate['factors'][5]['delta'][0, 0] += .000001
                elif change == 'current_nll':
                    observation['current']['metrics']['PS']['rows'][0]['new_nll'] += .000001
                else:
                    observation['N']['rows'][0]['true_nll'] += .000001
                target = Path(directory) / (change + '.json')
                with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, 'SHAM_NUMERICAL_CONTROL_NOT_ESTABLISHED'):
                    writer_runner.verify_sham(original, candidate, observed, observation, target)
                saved = json.loads(target.read_text())
                self.assertEqual(saved['status'], 'NUMERICAL_CONTROL_NOT_ESTABLISHED')
                self.assertEqual(saved['predeclared_tolerance'], 0)
                self.assertFalse(saved['scientific_failure'])
                self.assertTrue(any(not x['exact'] for x in saved['differences']))

    def test_sham_control_is_inside_branch_loop_before_downstream_promotion(self):
        tree = ast.parse(inspect.getsource(writer_runner.run_writers))
        branch = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'branch')
        guard = next(n for n in branch.body if isinstance(n, ast.If) and
                     ast.unparse(n.test) == "branch == 'SHAM'")
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'verify_sham'
                            for n in ast.walk(guard)))
        guarded = next(n for n in ast.walk(guard) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'verify_sham')
        next_family = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run_components')
        self.assertLess(guarded.lineno, next_family.lineno)

    def test_control_static_resource_allowlist_and_dependency_graph(self):
        # Read and parse only: no control import, freeze, submit, or scheduler call.
        path = Path(writer_runner.__file__).with_name('control.py')
        source = path.read_text()
        tree = ast.parse(source)
        freeze = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'freeze')
        submit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'submit')
        locked = next(n.value for n in ast.walk(freeze) if isinstance(n, ast.Assign) and
                      any(isinstance(x, ast.Name) and x.id == 'lock' for x in n.targets))
        keywords = {k.arg: k.value for k in locked.keywords}
        for name in ('project_gpu_cap', 'task_gpu_cap'):
            self.assertEqual(ast.literal_eval(keywords[name]), 2)
        self.assertEqual(ast.literal_eval(keywords['allowed_phases']), ['gate', 'geometry', 'writers', 'reduce'])
        self.assertEqual(ast.literal_eval(keywords['followup_submissions']), [])
        self.assertIs(ast.literal_eval(keywords['save_new_resume_checkpoints']), False)
        self.assertEqual(ast.literal_eval(keywords['python']), '/data/janghj/EasyEdit/.venv/bin/python')
        resource = {k.arg: ast.literal_eval(k.value) for k in keywords['resources'].keywords}
        self.assertEqual({k: resource[k] for k in ('gpus_each', 'cpus', 'mem_MiB', 'export', 'requeue')},
                         {'gpus_each': 1, 'cpus': 8, 'mem_MiB': 60416, 'export': 'NONE', 'requeue': False})
        old = json.loads((INPUT_ROOT / 'evidence/audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-execution.lock.json').read_text())
        self.assertTrue(Path(old['dependencies']).is_dir())
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'old':
                key = ast.literal_eval(node.slice)
                self.assertIn(key, old, 'control reads nonexistent historical lock key')
        for value in ('--hold', '--export=NONE', '--no-requeue', '--gres=gpu:1', '--kill-on-invalid-dep=yes'):
            self.assertIn(value, source)
        calls = [n for n in ast.walk(submit) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'command']
        commands = [ast.literal_eval(n.args[0].elts[0]) for n in calls if isinstance(n.args[0], ast.List)]
        self.assertNotIn('scancel', commands)
        self.assertNotIn('sacct', commands)
        # Only known arithmetic expressions are interpreted; the join/values
        # expression is decoded manually. No launcher body is executed.
        conditional = [n for n in ast.walk(submit) if isinstance(n, ast.If) and any(
            isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'dependency' for t in x.targets)
            for x in n.body)]
        jobs, deps = {}, {}
        for phase in ('gate', 'geometry', 'writers', 'reduce'):
            dependency = None
            for condition in conditional:
                if not eval(compile(ast.Expression(condition.test), '<static-condition>', 'eval'), {}, {'phase': phase}):
                    continue
                assignment = next(x for x in condition.body if isinstance(x, ast.Assign))
                expression = ast.unparse(assignment.value)
                self.assertIn(expression, ("'afterok:' + jobs['gate']", "'afterany:' + ':'.join(jobs.values())"))
                dependency = ('afterok:' + jobs['gate'] if expression.startswith("'afterok:'")
                              else 'afterany:' + ':'.join(jobs.values()))
            deps[phase] = dependency
            jobs[phase] = str(100 + len(jobs))
        self.assertEqual(deps, {'gate': None, 'geometry': 'afterok:100', 'writers': 'afterok:100', 'reduce': 'afterany:100:101:102'})


if __name__ == '__main__':
    unittest.main()
