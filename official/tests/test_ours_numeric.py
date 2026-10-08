import copy
import importlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from official.ours.config import ROOT, metadata, plain, resolve
from official.ours.core.jlz_interference_l1.cap_controller import RequestController
from official.ours.core.jlz_interference_l1.cap_optimizer import EfficiencyAdamAbs
from official.ours.core.jlz_interference_l1 import cap_price, cap_projection
from official.ours.core.jlz_v12r.optimizer import analytic_norm
from official.ours.core.jlz_v12r import subject
from official.ours.core.jlz_realized_subject.subject import coefficients, row_terms
from official.ours.core.jlz_pilot.prompts import native_loss
from official.tests.price_oracle import load


def configured(model="qwen25", **changes):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "ours"; shutil.copytree(ROOT, root)
        p = root / model / "price.json"; data = json.loads(p.read_text()); data.update(changes)
        p.write_text(json.dumps(data))
        return resolve(model, root=root)


def subject_fixture(config):
    class Adapter:
        profile = config
        sites = (4,)
        device = "cpu"

        def masked(self, group, values, capture=False):
            rows = [r["global_row"] for r in group["rows"]]
            hidden = values[4][rows].unsqueeze(1)
            return hidden, hidden, {}, {4: hidden[:, 0]}

        def head(self, selected): return selected

    groups = []
    for owner in range(2):
        rows = [dict(request=owner, kind=kind, global_row=2*owner+i,
                     target=torch.tensor([0 if kind == "rewrite" else -100]), lookup=0)
                for i, kind in enumerate(("rewrite", "kl"))]
        groups.append(dict(rows=rows, tokens=dict(input_ids=torch.zeros(2, 1, dtype=torch.long),
                      attention_mask=torch.ones(2, 1)), cache=dict(key=torch.zeros(2, 1), residual=torch.zeros(2, 1))))
    entry = dict(groups=groups, pack=dict(n_requests=2, n_rw=1, key_context_weights=[1., 1.]),
                 teachers=torch.tensor([[0., 0.], [1., 0.]]).log_softmax(-1),
                 anchors={4: torch.tensor([2., 3.])})
    built = dict(v={4: torch.tensor([[3.5, 0.], [.1, .2], [1., 0.], [0., 1.]])})
    return Adapter(), entry, built


def price_fixture(config):
    generator = torch.Generator().manual_seed(16)
    layers = config["eligible_layers"]
    adapter = SimpleNamespace(profile=config, sites=layers)
    entry = dict(pack=dict(n_requests=2), anchors={}, factors={}, entry_weights={})
    built = dict(candidate=0, entry_id=id(entry), weights={}, K={}, P={}, mean_M={},
                 rows=[dict(case_id=0), dict(case_id=1)])
    for layer in layers:
        K = torch.rand(2, 2, dtype=torch.float64, generator=generator) / 5
        A = torch.eye(2, dtype=torch.float64)
        P = torch.linalg.solve(A + K @ K.T, K)
        entry["anchors"][layer] = torch.tensor([2., 3.])
        entry["factors"][layer] = dict(A=A, asymmetry_max=0.)
        entry["entry_weights"][layer] = built["weights"][layer] = torch.zeros(2, 2)
        built["K"][layer], built["P"][layer], built["mean_M"][layer] = K, P, P.T @ K
    return adapter, entry, built


class NumericTests(unittest.TestCase):
    def bits(self, left, right):
        self.assertEqual(left.dtype, right.dtype)
        self.assertEqual(left.shape, right.shape)
        self.assertTrue(torch.equal(left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8)))

    def test_controller_default_trajectory_matches_frozen_bitwise(self):
        old_type = load("jlz_interference_l1.cap_controller").RequestController
        anchors = torch.tensor([[2., 3., 4.], [5., 6., 7.]])
        prices = torch.tensor([[1., 2., 4.], [3., 1., 2.]])
        for model in ("llama3", "qwen25", "gptj"):
            config = resolve(model)
            old = old_type(anchors, anchors[-1], (4, 8), prices)
            new = RequestController(anchors, anchors[-1], (4, 8), prices, config)
            for candidate in range(config["K_eval"]):
                F = torch.tensor([.06 if candidate < 3 else .01, .02, .08])
                am, at = old.observe(F, candidate); bm, bt = new.observe(F, candidate)
                self.bits(am, bm); self.assertEqual(at, bt)
                if not at:
                    self.bits(old.before_update(F), new.before_update(F))
                    old.record_update(am); new.record_update(bm)
                for field in ("t", "expansion", "beta", "maximum", "caps", "weights", "active"):
                    self.bits(getattr(old, field), getattr(new, field))
            self.assertEqual(old.terminal_states(F), new.terminal_states(F))

    def test_projection_and_adam_default_outputs_and_moments_match_frozen_bits(self):
        old_projection = load("jlz_interference_l1.cap_projection")
        for caps in ([.5, 2.], None):
            self.assertEqual(cap_projection.lengths([3., 2.], [1., 2.], caps, 2.),
                             old_projection.lengths([3., 2.], [1., 2.], caps, 2.))
        generator = torch.Generator().manual_seed(67)
        old_type = load("jlz_interference_l1.cap_optimizer").EfficiencyAdamAbs
        for model in ("llama3", "qwen25", "gptj"):
            config = resolve(model)
            left = {4: torch.zeros(3, 3), 8: torch.zeros(3, 3)}
            right = copy.deepcopy(left)
            old = old_type(left, lr=config["lr"], eps=config["eps"], betas=config["betas"])
            new = EfficiencyAdamAbs(right, config)
            for step in range(4):
                gradient = {l: torch.randn(3, 3, generator=generator, dtype=torch.float64) for l in left}
                mask = torch.tensor([True, False, True])
                args = dict(active_mask=mask, caps=torch.ones(2, 3)*100,
                            weights=torch.ones(2, 3)/100, beta=torch.ones(3)*config["beta_base"])
                left, _ = old.step(left, gradient, **args); right, _ = new.step(right, gradient, **args)
                for layer in left:
                    self.bits(left[layer], right[layer]); self.bits(old.m[layer], new.m[layer])
                    self.bits(old.v[layer], new.v[layer])
                self.bits(old.s, new.s); self.bits(old.t, new.t)

    def test_analytic_norm_and_loss_coefficients_match_frozen_bits(self):
        old = load("jlz_v12r.optimizer").analytic_norm
        old_coef = load("jlz_realized_subject.subject").COEF
        R = {4: torch.tensor([[1., 0., 3.], [2., 0., -4.]]), 8: torch.tensor([[4., 0., 2.], [1., 0., 5.]])}
        anchor = torch.tensor([2., 3., 4.]); active = torch.tensor([True, True, False])
        for model in ("llama3", "qwen25", "gptj"):
            config = resolve(model)
            a, ga = old(R, anchor, active); b, gb = analytic_norm(R, anchor, active, config=config)
            self.bits(a, b)
            for layer in ga: self.bits(ga[layer], gb[layer])
            self.assertEqual(coefficients(config), old_coef)
        adapter, entry, built = subject_fixture(resolve("qwen25"))
        v = {4: built["v"][4].clone().requires_grad_()}
        before = load("jlz_realized_subject.subject").row_terms(adapter, entry, entry["groups"][0], v)[0]
        after = row_terms(adapter, entry, entry["groups"][0], v)[0]
        old_loss = sum(old_coef[k]*value for k, value in before.items())
        new_loss = sum(coefficients(adapter.profile)[k]*value for k, value in after.items())
        self.bits(old_loss, new_loss)
        self.bits(torch.autograd.grad(old_loss, v[4], retain_graph=True)[0], torch.autograd.grad(new_loss, v[4])[0])

    def test_subject_and_native_helper_match_frozen_forward_and_backward(self):
        old_subject = load("jlz_v12r.subject")
        adapter, entry, built = subject_fixture(resolve("qwen25"))
        old = old_subject.evaluate(adapter, copy.deepcopy(entry), built, backward=True, capture=True)
        new = subject.evaluate(adapter, entry, built, backward=True, capture=True)
        for field in ("F", "nll", "kl", "active_mask"):
            self.bits(old[field], new[field])
        self.bits(old["adjoint"][4], new["adjoint"][4])
        self.assertEqual(old["masked_backward_sum"], new["masked_backward_sum"])
        self.assertEqual([r["role_weight"] for g in entry["groups"] for r in g["rows"]], [1., adapter.profile["lambda_KL"]]*2)
        spec = dict(specs=[dict(target=torch.tensor([0]))]*2, n_rw=1,
                    row_request=[0, 0, 1, 1], row_kind=["rewrite", "kl"]*2,
                    targets=torch.tensor([[0], [-100], [0], [-100]]), lookup=[0]*4)
        logits = built["v"][4].unsqueeze(1).clone().requires_grad_()
        mask = torch.tensor([True, True])
        left = load("jlz_pilot.prompts").native_loss(logits, spec, entry["teachers"], mask)
        right = native_loss(logits, spec, entry["teachers"], mask, config=adapter.profile)
        for a, b in zip(left, right): self.bits(a, b)
        self.bits(torch.autograd.grad(left[0], logits, retain_graph=True)[0], torch.autograd.grad(right[0], logits)[0])

    def test_override_controls_subject_mask_controller_norm_and_learning_rate(self):
        base = resolve("qwen25"); arm = resolve("qwen25", "qwen25-tau002-grace8")
        observed = []
        for config in (base, arm):
            adapter, entry, built = subject_fixture(config)
            result = subject.evaluate(adapter, entry, built, backward=True)
            control = RequestController(torch.ones(1, 2), torch.ones(2), (4,), torch.ones(1, 2), config)
            mask, _ = control.observe(result["F"], 0)
            self.bits(mask, result["active_mask"]); observed.append(mask)
        self.assertFalse(observed[0][0]); self.assertTrue(observed[1][0])
        config = resolve("qwen25", "qwen25-beta300-free")
        control = RequestController(torch.ones(1, 2), torch.ones(2), (4,), torch.ones(1, 2)*2, config)
        self.assertIsNone(control.caps)
        self.assertEqual(control.maximum.tolist(), [6., 6.])
        R = {4: torch.ones(2, 2)}
        losses, _ = analytic_norm(R, torch.ones(2), config=config)
        self.bits(losses, config["lambda_N"]*R[4].double().norm(dim=0))
        self.assertEqual(EfficiencyAdamAbs(R, config).lr, .25)

    def test_entry_price_default_numerics_match_and_override_records_real_values(self):
        old_price = load("jlz_interference_l1.cap_price")
        for model in ("llama3", "qwen25", "gptj"):
            config = resolve(model); adapter, entry, built = price_fixture(config)
            left = old_price.initialize(adapter, entry, built, "CAP075")
            right = cap_price.initialize(adapter, entry, built, config["arm"])
            for key in ("computed_pi", "effective_pi", "anchors"): self.bits(left[key], right[key])
            for key, value in left["record"].items():
                if key not in ("price_seconds", "loo"):
                    self.assertEqual(value, right["record"][key], key)
            self.assertEqual(right["record"]["config_sha256"], config["config_sha256"])
        for arm in ("qwen25-beta150", "qwen25-beta300-free"):
            config = resolve("qwen25", arm); adapter, entry, built = price_fixture(config)
            record = cap_price.initialize(adapter, entry, built, config["arm"])["record"]
            self.assertEqual(record["native_c"], config["c"])
            self.assertEqual(record["beta_max_native_scale"], config["beta_max_scale"])
            self.assertEqual(record["config_arm"], arm)
            if config["cap_mode"] == "none": self.assertIsNone(record["local_caps"])

    def test_both_fit_loops_honor_nondefault_update_budget_and_record_identity(self):
        class Objective:
            def __init__(self, a, entry, events):
                self.a = a
                self.calls = dict(logical_builds=0, logical_subject_forwards=0)
            def zeros(self): return {8: torch.zeros(2, 1)}
            def build(self, R, candidate):
                self.calls["logical_builds"] += 1
                return dict(candidate=candidate, seconds=0.)
            def evaluate(self, R, k, active_previous, terminal, capture, built, blind):
                self.calls["logical_subject_forwards"] += 1
                F = torch.tensor([.3], dtype=torch.float64)
                observed = dict(nll=F[:, None], kl=F*0, logical_backward=int(not terminal),
                                backward_groups=int(not terminal), seconds=0., masked_backward_sum=float(F.sum()))
                return dict(built=built if built is not None else self.build(R, k), observed=observed, F=F,
                            active_mask=F>=self.a.profile["tau_F"], pullback=None,
                            gradient={8: torch.ones(2, 1, dtype=torch.float64)})
        for name in ("jlz_interference_l1.cap_fit", "jlz_price_gptj.fit"):
            module = importlib.import_module("official.ours.core." + name)
            for budget in (1, 3, 27):
                config = configured(model="gptj" if "gptj" in name else "qwen25", max_updates=budget, K_grace=0)
                adapter = SimpleNamespace(profile=config, sites=(8,))
                entry = dict(anchors={8: torch.ones(1)*1000}, groups=[dict(native_c0_pending=False,native_c0_error_max=0.)])
                price = dict(record=metadata(config), sha256="synthetic-price", effective_pi=torch.ones(1, 1),
                             price_seconds=0., loo_seconds=0.)
                run = Mock()
                with patch.object(module, "CandidateObjective", Objective), patch.object(
                        module, "candidate_telemetry", return_value={"seconds": 0.}):
                    result = module.fit(adapter, entry, price=price, wandb_run=run)
                receipt = result["receipt"]
                self.assertEqual(receipt["updates"], budget)
                self.assertEqual(receipt["candidates"], budget+1)
                self.assertEqual(receipt["logical_subject_backwards"], budget)
                self.assertEqual(receipt["resolved_config"], plain(config))
                self.assertEqual(receipt["config_sha256"], config["config_sha256"])
                self.assertTrue(receipt["terminal_no_backward"])
                self.assertTrue(run.config.update.called)

    def test_gptj_configured_readout_selects_requested_layer(self):
        from official.ours.core.jlz_price_gptj.adapter import Adapter
        class Block(torch.nn.Module):
            def forward(self, x): return (x+1,)
        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.h=torch.nn.ModuleList(Block() for _ in range(28))
            def forward(self, input_ids, use_cache=False):
                x=input_ids.float()
                for block in self.h: x=block(x)[0]
                return x
        transformer=Transformer()
        adapter=Adapter.__new__(Adapter)
        adapter.model=SimpleNamespace(transformer=transformer)
        adapter.blocks=transformer.h;adapter.final_layer=27
        for layer in (8, 20, 27):
            adapter.nll_layer=layer
            nll,final=adapter.full(dict(input_ids=torch.zeros(1, 1, 2)))
            self.bits(nll,torch.ones(1, 1, 2)*(layer+1))
            self.bits(final,torch.ones(1, 1, 2)*28)
            self.assertTrue(all(not block._forward_hooks for block in adapter.blocks))


if __name__ == "__main__": unittest.main()
