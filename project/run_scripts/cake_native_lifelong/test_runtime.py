"""CPU structural/mocked routing checks, not a numerical/model gate."""
import ast
from pathlib import Path
import unittest
from .runtime import merge
from .preflight import storage_plan


class RuntimeTests(unittest.TestCase):
    def test_user_no_checkpoint_disk(self):
        p=storage_plan(4096,14336,39098310656,False)
        self.assertEqual(p['checkpoints'],0)
        self.assertEqual(p['status'],'DISK_ARITHMETIC_ONLY_PASS')
        self.assertEqual(p['planned_available_required_bytes'],16*1024**3)

    def test_original_apply_once_per_loop_and_no_tensor_save(self):
        source=Path(__file__).with_name('runtime.py').read_text()
        tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertEqual(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='apply_Cake_to_model' for n in calls),1)
        for n in calls:
            if isinstance(n.func,ast.Attribute):
                self.assertNotIn(n.func.attr,['compute_z','compute_optimal_deltas','manual_seed_all'])
                self.assertFalse(n.func.attr=='save' and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch')
        self.assertNotIn('cache_c +=',source)
        self.assertIn('cache_template=None,cache_c=state,P=projector',source)

    def test_fullseen_current_reuse(self):
        def obs(ident):
            return dict(requests=1,metrics={k:dict(rows=[dict(identity=ident+k,success=True)]) for k in ['RS','PS','NS']})
        x=merge(obs('a'),obs('b'),{'state':'actual'})
        self.assertEqual(x['requests'],2)
        self.assertTrue(x['current_rows_reused'])
        self.assertEqual(x['metrics']['NS']['denominator'],2)
        with self.assertRaises(AssertionError):merge(obs('a'),obs('a'),{})


if __name__=='__main__':unittest.main()
