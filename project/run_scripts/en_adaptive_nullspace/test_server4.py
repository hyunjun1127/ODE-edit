"""S4 migration routing and ideal-vs-actual curvature regression; CPU only."""
from pathlib import Path
import inspect
import unittest
import numpy as np
from . import runner
from .controller import run_controller
from .geometry import build_geometry


class Server4Tests(unittest.TestCase):
    def test_curvature_uses_ideal_not_rounded_slope(self):
        geo=build_geometry(np.eye(1),[1.],[0],np.eye(1))
        wn=np.array([[2**20]],dtype=np.float32)
        g=np.ones((1,1));d=np.array([[-.1]])
        result=run_controller(wn,g,d,1.,lambda w:{'J':1.02},geo,
            native_norm=1.,native_action=100.,input_identity='fixed')
        first=result['ledger'][0]
        self.assertNotEqual(first['actual_gradient_inner_product'],first['ideal_gradient_inner_product'])
        self.assertAlmostEqual(first['quadratic']['curvature'],2*(1.02-1.+.1))
        self.assertAlmostEqual(first['quadratic']['scale'],.1/(2*(1.02-1.+.1)))
        self.assertAlmostEqual(first['armijo_limit'],1.+1e-4*first['actual_gradient_inner_product'])

    def test_single_lane_resources(self):
        text=Path(__file__).with_name('run-server4.sbatch').read_text()
        for flag in ['--mem=60416M','--gres=gpu:1','--cpus-per-task=8','--nodelist=server4','--export=NONE','--no-requeue']:
            self.assertIn(flag,text)
        self.assertNotIn('--array',text)
        self.assertNotIn('ubuntu',text)

    def test_scope_and_no_checkpoint(self):
        source=inspect.getsource(runner)
        self.assertEqual(runner.CHAINS,['N4','EN_EXACT','EN_ADAPT'])
        self.assertIn('range(1,4)',source)
        for forbidden in ['torch.save(', 'np.save(', 'pickle.dump(']:self.assertNotIn(forbidden,source)

    def test_does_not_allocate_default_history_for_existing_entry(self):
        source=inspect.getsource(runner)
        self.assertNotIn('states.get(group,dict(',source)
        self.assertIn("states[group] if group in states",source)

    def test_current_cache_released_before_science_svd(self):
        source=inspect.getsource(runner)
        self.assertLess(source.index("del captured['oracle']"),source.index('geo=build_geometry'))


if __name__=='__main__':unittest.main()
