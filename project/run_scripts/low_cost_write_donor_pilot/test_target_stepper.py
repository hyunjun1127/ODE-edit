"""CPU deterministic algebra/state fixtures, not model-level native parity."""
import unittest
from types import SimpleNamespace

import torch

from .target_stepper import (NativeTargetStepper, TargetBoundary, advance_adam,
                             native_kl, native_regularizer, rebased_hook_delta, snapshot_state)


class CarriedAdamTests(unittest.TestCase):
    def run_schedule(self, caps):
        u = torch.zeros(4, requires_grad=True)
        opt = torch.optim.Adam([u], lr=.08)
        reports = []
        target = torch.tensor([2., -3., .6, 4.])
        for cap in caps:
            reports.append(advance_adam(u, opt, lambda: (((u-target)**2).sum(), {}),
                           max_updates=cap, max_norm_fn=lambda: torch.tensor(100.)))
        return u, opt.state[u], reports

    def test_pause_resume_keeps_same_actual_optimizer_path(self):
        for caps in ([12, 12], [6, 6, 6, 6]):
            u, adam, reports = self.run_schedule(caps)
            ref_u, ref_adam, _ = self.run_schedule([24])
            self.assertTrue(torch.equal(u, ref_u))
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(adam[key], ref_adam[key]))
            self.assertEqual(sum(r["actual_adam_updates"] for r in reports), 24)
            self.assertEqual(sum(r["target_loss_evaluations"] for r in reports), 24+len(caps))

    def test_zero_step_rechecks_next_chunk_and_quota_not_transferred(self):
        u = torch.zeros(2, requires_grad=True)
        opt = torch.optim.Adam([u], lr=.1)
        first = advance_adam(u, opt, lambda: ((u*u).sum(), {}), max_updates=12,
                             max_norm_fn=lambda: torch.tensor(100.))
        second = advance_adam(u, opt, lambda: (((u-10)**2).sum(), {}), max_updates=12,
                              max_norm_fn=lambda: torch.tensor(100.))
        self.assertEqual(first["actual_adam_updates"], 0)
        self.assertEqual(first["target_loss_evaluations"], 1)
        self.assertEqual(first["unused_quota"], 12)
        self.assertEqual(second["actual_adam_updates"], 12)
        self.assertEqual(int(opt.state[u]["step"]), 12)

    def test_regularizer_not_squared_and_kl_argument_order(self):
        u, a0 = torch.tensor([3., 4.]), torch.tensor([2., 0.])
        self.assertEqual(float(native_regularizer(u, a0, .5)), .625)
        teacher = torch.log(torch.tensor([[.1, .9]]))
        current = torch.log(torch.tensor([[.7, .3]]))
        self.assertTrue(torch.equal(native_kl(teacher, current, 2.),
            2*torch.nn.functional.kl_div(teacher, current, log_target=True, reduction="batchmean")))
        self.assertNotEqual(float(native_kl(teacher, current, 1.)), float(native_kl(current, teacher, 1.)))

    def test_clamp_keeps_moments_and_maximum_loss_order(self):
        u = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([u], lr=10.)
        report = advance_adam(u, opt, lambda: (((u-10)**2).sum(), {}),
                             max_updates=2, max_norm_fn=lambda: torch.tensor(.01))
        self.assertEqual(report["clamp_hits"], 2)
        self.assertEqual(report["target_loss_evaluations"], 3)
        self.assertEqual(int(opt.state[u]["step"]), 2)
        self.assertGreater(float(opt.state[u]["exp_avg"].abs()), 0)

    def test_zero_quota_still_evaluates(self):
        u = torch.zeros(1, requires_grad=True)
        report = advance_adam(u, torch.optim.Adam([u]), lambda: ((u-2).square().sum(), {}),
                             max_updates=0, max_norm_fn=lambda: torch.tensor(1.))
        self.assertEqual(report["target_loss_evaluations"], 1)
        self.assertEqual(report["actual_adam_updates"], 0)

    def test_zero_offset_preserves_native_gradient_accumulation_graph(self):
        torch.manual_seed(73)
        a0 = torch.randn(4096)
        contexts, desired = torch.randn(6, 4096), torch.randn(6, 4096)
        initial = torch.randn(4096)*.02
        results = []
        for mode in ("native", "repaired", "shared_zero_add"):
            u = initial.clone().requires_grad_()
            if mode == "native":
                delta = u
            elif mode == "repaired":
                delta, exact_zero = rebased_hook_delta(u, a0, a0.clone())
                self.assertTrue(exact_zero)
                self.assertIs(delta, u)
            else:
                delta = u+(a0-a0)
            loss = sum((contexts[i]+delta-desired[i]).square().mean() for i in range(6))
            loss = loss + native_regularizer(u, a0, .5)
            loss.backward()
            results.append((loss.detach(), u.grad.clone()))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertTrue(torch.equal(results[0][0], results[2][0]))
        self.assertFalse(torch.equal(results[0][1], results[2][1]))

    def test_nonzero_rebase_keeps_registered_formula(self):
        u = torch.tensor([.2, -.1], requires_grad=True)
        a0, aj = torch.tensor([2., 3.]), torch.tensor([2.1, 2.9])
        delta, exact_zero = rebased_hook_delta(u, a0, aj)
        self.assertFalse(exact_zero)
        self.assertIsNot(delta, u)
        self.assertTrue(torch.equal(delta, u+(a0-aj)))


class TokenBatch(dict):
    def to(self, device):
        return TokenBatch({k: v.to(device) for k, v in self.items()})


class ToyTokenizer:
    padding_side = "right"
    bos_token_id = 0
    unk_token_id = -1

    def __call__(self, prompts, **kwargs):
        prompts = [prompts] if isinstance(prompts, str) else prompts
        ids = torch.tensor([[1, 2, 3] for _ in prompts])
        return TokenBatch(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, values):
        return " target"


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.ones(4))

    def forward(self, value):
        return (value + self.shift,)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(123)
        self.embed = torch.nn.Embedding(5, 4)
        self.block = ToyBlock()
        self.norm = torch.nn.Identity()
        self.head = torch.nn.Linear(4, 5, bias=False)
        self.config = SimpleNamespace(hidden_size=4, vocab_size=5)

    def forward(self, input_ids, attention_mask):
        hidden = self.block(self.embed(input_ids))[0]
        return SimpleNamespace(logits=self.head(self.norm(hidden)))


class ToyTrace:
    def __init__(self, module, layers, edit_output, **kwargs):
        self.module, self.layers, self.edit = module, list(dict.fromkeys(layers)), edit_output
        self.outputs, self.handles = {}, []

    def __enter__(self):
        for name in self.layers:
            def hook(module, args, output, name=name):
                output = self.edit(output, name)
                self.outputs[name] = SimpleNamespace(output=output)
                return output
            self.handles.append(self.module.get_submodule(name).register_forward_hook(hook))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def __getitem__(self, key):
        return self.outputs[key]


class ToyNative:
    @staticmethod
    def parameter(model, name):
        try:
            return model.get_parameter(name)
        except AttributeError as exc:
            raise LookupError(name) from exc

    @staticmethod
    def flags(value, model):
        for p in model.parameters():
            p.requires_grad_(value)

    find_fact_lookup_idx = staticmethod(lambda *args, **kwargs: 0)
    nethook = SimpleNamespace(get_module=lambda model, name: model.get_submodule(name),
                            get_parameter=parameter, set_requires_grad=flags, TraceDict=ToyTrace)


class ModelHookFixtures(unittest.TestCase):
    def stepper(self):
        model = ToyModel().eval()
        hp = SimpleNamespace(layers=[4], blue=True, L2=1, v_num_grad_steps=25,
            lm_head_module="head", ln_f_module="norm", layer_module_tmp="block",
            fact_token="subject_last", v_loss_layer=4, v_lr=.1, kl_factor=.1,
            v_weight_decay=.5, clamp_norm_factor=.2)
        return model, NativeTargetStepper(model, ToyTokenizer(), hp, 4, [["{}"]], ToyNative)

    def test_hook_rebase_anchor_teacher_and_leaf_immutable(self):
        model, stepper = self.stepper()
        request = {"prompt": "{} test", "subject": "subject", "target_new": {"str": "answer"}}
        state = stepper.create_state(request, 0)
        initial_flags = [p.requires_grad for p in model.parameters()]
        first = stepper.run_chunk(state, 2, 0)
        self.assertTrue(first["summary"]["hook_zero_offset_native_leaf"])
        a0, teacher, leaf, optimizer = state.a0.clone(), state.teacher.clone(), id(state.u), id(state.opt)
        moments = state.opt.state[state.u]["exp_avg"].data_ptr()
        with torch.no_grad():
            model.block.shift.add_(.4)
        second = stepper.run_chunk(state, 2, 1)
        self.assertFalse(second["summary"]["hook_zero_offset_native_leaf"])
        self.assertTrue(torch.equal(state.a0, a0))
        self.assertTrue(torch.equal(state.teacher, teacher))
        self.assertTrue(torch.allclose(state.aj, state.a0+.4))
        self.assertEqual(id(state.u), leaf)
        self.assertEqual(id(state.opt), optimizer)
        self.assertEqual(state.opt.state[state.u]["exp_avg"].data_ptr(), moments)
        self.assertEqual(initial_flags, [p.requires_grad for p in model.parameters()])
        self.assertFalse(model.block._forward_hooks)
        self.assertEqual(second["evidence"]["t"], 4)
        self.assertTrue(torch.equal(first["evidence"]["teacher"], second["evidence"]["teacher"]))
        self.assertEqual(request["target_new"]["str"], "answer")
        self.assertEqual(snapshot_state(state)["u"].device.type, "cpu")

    def test_wrong_chunk_order_and_cross_stepper_rejected(self):
        _, stepper = self.stepper()
        _, another = self.stepper()
        state = stepper.create_state({"prompt": "{} test", "subject": "subject", "target_new": {"str": "answer"}}, 0)
        with self.assertRaises(TargetBoundary):
            stepper.run_chunk(state, 2, 1)
        with self.assertRaises(TargetBoundary):
            another.run_chunk(state, 2, 0)


if __name__ == "__main__":
    unittest.main()
