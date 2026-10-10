"""CPU-only owner storage, queue ordering and shared runtime bindings."""
import ast
import unittest
from pathlib import Path
from official.runners.server2.fe_author_prepare import storage_plan
from official.runners.server2.fe_author_submit import graph, RESOURCES
from official.baselines.fe_author_profile import resolve
from official.baselines import registry
from official.runners import fe_author_history

class OwnerTests(unittest.TestCase):
    def test_checkpoint_overlap_budget(self):
        p=storage_plan(161_362_677_760)
        self.assertTrue(p['sufficient'])
        self.assertEqual(p['selected_weights_history_bytes'],8_535_408_640)
        self.assertEqual(p['concurrent_payloads'],3)
        self.assertFalse(storage_plan(p['required_free_bytes']-1)['sufficient'])

    def test_serial_after_entire_existing_frontier(self):
        rows=[dict(job='10',gpus=1,dependency='(null)'),
              dict(job='11',gpus=1,dependency='afterany:10'),
              dict(job='12',gpus=1,dependency='(null)')]
        parents,width=graph(dict(project=rows),'CF','ZSRE')
        self.assertEqual(set(parents),{'11','12'})
        self.assertEqual(width,2)

    def test_actual_qwen_override_not_default(self):
        base=registry.hparams('MEMIT_FE_HISTORY','qwen25')
        hp=resolve('qwen25',device=0)
        self.assertEqual((base.clamp_norm_factor,base.v_num_grad_steps),(4,25))
        self.assertEqual((hp.clamp_norm_factor,hp.v_num_grad_steps,hp.v_lr,hp.v_weight_decay,hp.v_loss_layer),(1,35,.5,.001,27))
        self.assertEqual((hp.layers,hp.mom2_update_weight,hp.kl_factor),([4,5,6,7,8],15000,.0625))

    def test_shared_runtime_no_local_science_fork(self):
        text=Path(__file__).with_name('fe_author_submit.py').read_text()
        self.assertIn("'official.runners.fe_author_history'",text)
        self.assertNotIn("'--resume'",text)
        tree=ast.parse(Path(fe_author_history.__file__).read_text())
        names=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertIn('resolve',names)

    def test_repaired_history_source_preserved(self):
        from official.baselines import memit_fe_history
        text=Path(memit_fe_history.__file__).read_text()
        self.assertIn('copy=True',text)
        self.assertIn('system.mul_',text)
        self.assertIn("device='cpu'",text)
        self.assertLessEqual(RESOURCES['memory_MiB'],60416)
        self.assertEqual((RESOURCES['CPUs'],RESOURCES['GPUs']),(6,1))

if __name__=='__main__':unittest.main()
