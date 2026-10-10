import unittest
from pathlib import Path
from official.runners.server1 import flucon_eval as e
from official.runners.server1.flucon_submit import cap3_scheduling as scheduling
from official.runners.server1.submit import graph_width
from official.runners.server1.test_flucon_eval import cfg
from official.tracking.schema import config,metrics

class Cap3Tests(unittest.TestCase):
    def test_two_free_lanes_no_full_barrier(self):
        old=[dict(key='cf',gpus=1,parents=[]),dict(key='zsre',gpus=1,parents=['cf'])]
        g,w=scheduling({'jobs':old},['gptj','llama3','qwen25'],3)
        self.assertEqual(g['gptj'],[]);self.assertEqual(g['llama3'],[])
        self.assertEqual(g['qwen25'],['gptj']);self.assertLessEqual(w,3)
        self.assertEqual(graph_width(old+[dict(key=k,gpus=1,parents=v) for k,v in g.items()]),3)
    def test_qwen_eval_only_schema_no_factual_or_fit(self):
        c=dict(cfg(),model='qwen25',model_family='qwen2',writer='memit_fe_history',baseline='MEMIT_FE_HISTORY')
        self.assertEqual(config(c)['model'],'qwen25')
        for v in ({'fit/loss':1.},{'official/all_seen/post/Efficacy':100.,'edits':2000}):
            with self.assertRaises(ValueError):metrics(v,config_values=c)
    def test_no_new_fit_forward_or_W0_in_preparation(self):
        text=(Path(__file__).parent/'flucon_cap3_prepare.py').read_text()
        self.assertNotIn('torch.load',text);self.assertNotIn('from_pretrained',text)
        self.assertIn("states[prior_id].startswith('CANCELLED')",text)
        self.assertIn('PARTIAL_GENERATION_NEEDS_EXACT_REUSE_REVIEW',text)

if __name__=='__main__':unittest.main()
