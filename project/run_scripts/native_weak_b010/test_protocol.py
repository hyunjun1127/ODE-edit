import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from .binding import patch_source,loss_line,stop_receipt,finite_scalars,OLD,NEW

NATIVE=Path('/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1/inputs/server4-handoff/native-imports/AlphaEdit/compute_z.py')

def sample_fit():
    nll_loss=1.
    if nll_loss <= 1.0:
        return True

def simulate(values):
    tree=ast.parse(patch_source(NATIVE.read_text()))
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='it')
    stops=[n for n in loop.body if isinstance(n,ast.If) and any(isinstance(x,ast.Break) for x in n.body)]
    assert len(stops)==2
    body=ast.parse('nll_loss = values[it]\nfinite_scalars([nll_loss])\nlosses.append(dict(nll=nll_loss))').body+stops+ast.parse('adam += 1').body
    fake=ast.Module(body=[ast.For(target=ast.Name(id='it',ctx=ast.Store()),iter=ast.parse('range(25)',mode='eval').body,body=body,orelse=[])],type_ignores=[])
    env=dict(values=values,finite_scalars=finite_scalars,losses=[],adam=0,hparams=SimpleNamespace(v_num_grad_steps=25))
    exec(compile(ast.fix_missing_locations(fake),'<actual-stop-asts>','exec'),env)
    return stop_receipt(env['losses'],env['adam']),env['adam']

class Protocol(unittest.TestCase):
    def test_only_predicate_changed(self):
        text=NATIVE.read_text();out=patch_source(text)
        self.assertEqual(out.replace(NEW,OLD),text)
    def test_initial_crossing_no_adam(self):
        r,n=simulate([.7]*25);self.assertEqual((r['reason'],n),('INITIAL_BELOW_THRESHOLD',0))
    def test_equal_threshold(self):
        r,n=simulate([2.,1.]+[0.]*23);self.assertEqual(n,1);self.assertEqual(r['final_nll'],1.)
    def test_undershoot_first_not_best(self):
        r,n=simulate([3.,1.4,.82]+[.01]*22);self.assertEqual(n,2);self.assertEqual(r['final_nll'],.82)
    def test_final_forward_crossing(self):
        r,n=simulate([1.1]*24+[.99]);self.assertEqual((r['reason'],n),('NLL_THRESHOLD_REACHED',24))
    def test_budget_exhaustion(self):
        r,n=simulate([1.1]*25);self.assertEqual((r['reason'],n),('BUDGET_EXHAUSTED_ABOVE_TARGET',24))
    def test_nonfinite(self):
        for v in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(RuntimeError):simulate([v]*25)
    def test_trace_line_detects_nll(self):
        lines,start=inspect.getsourcelines(sample_fit)
        self.assertEqual(loss_line(sample_fit),start+2)
    def test_no_checkpoint_calls(self):
        text=Path(__file__).with_name('runner.py').read_text();tree=ast.parse(text)
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertNotIn('tensor_save',calls);self.assertNotIn('torch.save',calls)
        self.assertIn("out/'final-full-metrics.json'",text)
    def test_does_not_mutate_shared_runtime(self):
        self.assertNotIn('write_text',Path(__file__).with_name('binding.py').read_text())

if __name__=='__main__':unittest.main()
