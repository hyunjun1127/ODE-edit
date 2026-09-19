"""Independent reducer arithmetic/identity fixtures; no model."""
import copy
import unittest
from .review_b1 import reduce_raw


def fixture():
    obs=dict(requests=1,raw={},metrics={},strict=dict(rewrite_strict=1,two_P_strict=1,
        R_two_P_strict=1,R_two_P_NLL_joint=1))
    for metric,group,multiple in [('RS','rewrite',1),('PS','rephrase',2),('NS','locality',10)]:
        for label in ('new','true'):
            nll=(2. if label=='new' else 1.) if metric=='NS' else (1. if label=='new' else 2.)
            obs['raw'][group+'_target_'+label]=[dict(case_id=7,prompt_index=i,prompt='fixture'+str(i),
                target=label,target_token_ids=[1 if label=='new' else 2],token_correct=[True],
                all_tokens_correct=True,nll=nll) for i in range(multiple)]
        obs['metrics'][metric]=dict(numerator=multiple)
    return obs


class Tests(unittest.TestCase):
    def test_raw_signs_and_strict(self):
        summary,rows=reduce_raw(fixture())
        self.assertEqual([summary[k]['count'] for k in ('RS','PS','NS')],[1,2,10])
        self.assertTrue(all(row['desired_margin']>0 for group in rows.values() for row in group))

    def test_tie_is_failure(self):
        obs=fixture();obs['raw']['locality_target_new'][0]['nll']=1.
        obs['metrics']['NS']['numerator']=9
        summary,_=reduce_raw(obs)
        self.assertEqual(summary['NS']['ties'],1);self.assertEqual(summary['NS']['count'],9)

    def test_corruptions_not_imputed(self):
        original=fixture()
        for mode in ('missing','duplicate','nonfinite','wrong_prompt','wrong_count','wrong_strict'):
            obs=copy.deepcopy(original)
            if mode=='missing':obs['raw']['locality_target_new'].pop()
            if mode=='duplicate':obs['raw']['locality_target_new'][1]=obs['raw']['locality_target_new'][0]
            if mode=='nonfinite':obs['raw']['rewrite_target_new'][0]['nll']=float('nan')
            if mode=='wrong_prompt':obs['raw']['rewrite_target_true'][0]['prompt']='different'
            if mode=='wrong_count':obs['metrics']['RS']['numerator']=0
            if mode=='wrong_strict':obs['strict']['two_P_strict']=0
            with self.subTest(mode=mode),self.assertRaises(ValueError):reduce_raw(obs)


if __name__=='__main__':unittest.main()
