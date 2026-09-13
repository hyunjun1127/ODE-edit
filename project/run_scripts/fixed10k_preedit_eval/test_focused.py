"""Model-free adapter/count/metric fidelity checks; no model or GPU load."""
import importlib.util,sys,unittest
from pathlib import Path

ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1')
sys.path.insert(0,str(ROOT/'helper-source'))
from project.run_scripts.blue_alphaedit_sequential_comparison.evaluation import reduce
spec=importlib.util.spec_from_file_location('observer',Path(__file__).with_name('observe.py'))
observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)

class Fidelity(unittest.TestCase):
    def test_exact_native_microbatch_counts(self):
        self.assertEqual([observer.expected_calls(n) for n in (100,200,1000)],[7,13,63])
        self.assertEqual(2*sum(observer.expected_calls(n) for n in (100,200,1000)),166)
    def test_ties_fail_and_secondary_separate(self):
        base=dict(case_id=1,prompt_index=0,prompt='fixture',target='new',nll=1.,all_tokens_correct=True,token_correct=[True])
        raw={}
        for category in ('rewrite','rephrase','locality'):
            raw[category+'_target_new']=[base.copy()]
            raw[category+'_target_true']=[dict(base,target='true')]
        result=reduce(raw)
        self.assertTrue(all(v['numerator']==0 and v['denominator']==1 for v in result.values()))
        self.assertTrue(all(v['rows'][0]['new_strict'] for v in result.values()))
    def test_canonical_preference_direction(self):
        base=dict(case_id=1,prompt_index=0,prompt='fixture',target='new',nll=.2,all_tokens_correct=False,token_correct=[False])
        raw={}
        for category in ('rewrite','rephrase','locality'):
            raw[category+'_target_new']=[base.copy()]
            raw[category+'_target_true']=[dict(base,target='true',nll=.8)]
        self.assertEqual({k:v['numerator'] for k,v in reduce(raw).items()},dict(RS=1,PS=1,NS=0))

if __name__=='__main__':unittest.main()
