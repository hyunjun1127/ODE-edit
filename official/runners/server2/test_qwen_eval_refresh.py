import unittest
from unittest.mock import patch
from official.runners.server2.qwen_eval_refresh import public_zsre
from official.runners.server2.qwen_plan import rows
from official.runners.server2.qwen_eval_refresh import planned

class RefreshTests(unittest.TestCase):
    def test_public_api_only(self):
        result={'summary':{'requests':100},'cases':[], 'query_sha256':'a'*64,
                'token_denominators':{'Efficacy':100},'work':{'physical_forward_calls':1}}
        with patch('official.evaluation.zsre_paper.evaluate',return_value=result) as f:
            got=public_zsre(None,None,[],{'source':'new'})
        self.assertEqual(f.call_args.kwargs['model_family'],'qwen25')
        self.assertEqual(got['accuracy'],{})
        self.assertEqual(got['work']['query_sha256'],'a'*64)
        self.assertNotIn('W0_prediction_agreement',got['summary'])

    def test_four_lane_graph_and_no_duplicate_ft(self):
        order=[r for m in ('MEMIT','ALPHAEDIT','ALPHAEDIT_BLUE','MEMIT_FE','SPHERE') for r in rows() if r['config']['method']==m]
        plan=planned(order,{'cf':'61898','zsre':'61900'})
        self.assertEqual(len(plan),10)
        lanes={0:'61898',1:'61900',2:None,3:None};seen=set(lanes.values())
        for row,lane,deps in plan:
            if lanes[lane]:self.assertIn(lanes[lane],deps)
            self.assertTrue(set(deps)<=seen)
            self.assertIn('61898' if row['config']['dataset']=='cf' else '61900',deps)
            self.assertNotEqual(row['config']['method'],'FT')
            lanes[lane]=row['logical_main_row'];seen.add(lanes[lane])
        self.assertEqual(set(lanes.values()),{'qwen25-cf-sphere','qwen25-zsre-sphere','qwen25-cf-memit_fe','qwen25-zsre-memit_fe'})

if __name__=='__main__':unittest.main()
