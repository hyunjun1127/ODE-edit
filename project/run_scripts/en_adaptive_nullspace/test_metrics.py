"""CPU tests for evaluator alignment and paired denominator semantics."""
import copy
import unittest
from types import SimpleNamespace

from project.run_scripts.en_adaptive_nullspace import metrics as m


def row(case, family='NS', flags=(True,), success=True, index=0):
    result=dict(case_id=case, family=family, prompt_index=index, identity=f'{case}-{family}-{index}', token_identity=str(len(flags)),
                success=success, new_nll=2., true_nll=1., desired_nll=1., desired_margin=1. if success else -1.)
    for side in ['new','true','desired']:
        result[side+'_token_correct']=list(flags)
        result[side+'_token_count']=len(flags)
        result[side+'_strict']=all(flags)
    return result


class MetricsTests(unittest.TestCase):
    def test_micro_macro_strict_differ(self):
        result=m.summarize([row(1,flags=(True,)),row(2,flags=(False,False,True))],[1,2])
        a=result['aggregates']['NS']
        self.assertEqual(a['tf_token_micro']['rate'],.5)
        self.assertAlmostEqual(a['tf_prompt_macro']['value'],2/3)
        self.assertEqual(a['tf_strict']['rate'],.5)

    def test_tie_failure_and_neighborhood_direction(self):
        def raw(side,nll):
            return dict(case_id=1,prompt_index=0,prompt='x',target=side,target_token_ids=[1],nll=nll,token_correct=[True],all_tokens_correct=True)
        inputs={}
        for family,_ in m.FAMILIES:
            inputs[family+'_target_new']=[raw('new',1)]
            inputs[family+'_target_true']=[raw('true',1 if family=='rewrite' else 2)]
        rows=m._compact(inputs)
        self.assertEqual([r['success'] for r in rows],[False,True,False])
        self.assertEqual(rows[2]['desired_side'],'true')
        self.assertEqual(rows[2]['desired_margin'],-1)

    def test_equal_totals_preserve_lost_gained(self):
        a=m.summarize([row(1,success=True),row(2,success=False)],[1,2])
        b=m.summarize([row(1,success=False),row(2,success=True)],[1,2])
        result=m.paired(a,b,bootstrap=False)['families']['NS']
        self.assertEqual(result['lost_ids'],[[1,0]])
        self.assertEqual(result['gained_ids'],[[2,0]])

    def test_identity_mismatch_rejected(self):
        a=m.summarize([row(1)],[1]); b=copy.deepcopy(a)
        b['rows'][0]['token_identity']='other'
        with self.assertRaises(ValueError):m.paired(a,b)

    def test_joint_requires_exact_two_paraphrases(self):
        a=m.summarize([row(1,'RS'),row(1,'PS',index=0),row(1,'PS',index=1),row(2,'RS')],[1,2])
        self.assertEqual(a['joint']['tf_strict']['rate'],1)
        self.assertEqual(a['joint']['omitted_case_ids'],[2])

    def test_request_cluster_does_not_resample_neighbors(self):
        a=m.summarize([row(c,index=i,success=False) for c in [1,2] for i in range(10)],[1,2])
        b=m.summarize([row(c,index=i,success=c==1) for c in [1,2] for i in range(10)],[1,2])
        ci=m.paired(a,b)['families']['NS']['cluster_bootstrap']['preference']
        self.assertEqual(ci['delta'],.5)
        self.assertEqual(ci['percentile95'],[0.,1.])

    def test_retention_denominators(self):
        base=m.summarize([row(1,flags=(True,False)),row(2,success=False)],[1,2])
        before=m.summarize([row(1,flags=(False,False),success=False),row(2,success=False)],[1,2])
        now=m.summarize([row(1,flags=(True,True)),row(2,success=True)],[1,2])
        r=m.retention(base,before,now)
        self.assertEqual(r['w0_correct_preference_retention']['denominator'],1)
        self.assertEqual(r['recovered_ids'],[[1,0]])
        self.assertEqual(r['w0_correct_token_retention']['denominator'],2)

    def test_latest_at_write_and_subset(self):
        a=m.summarize([row(1),row(2)],[1,2])
        b=m.summarize([row(1,success=False)],[1])
        r=m.merge_at_write([a,b])
        self.assertEqual(r['request_ids'],[1,2])
        self.assertFalse(m.subset(r,[1])['rows'][0]['success'])
        with self.assertRaises(ValueError):m.subset(r,[3])

    def test_exact_canonical_forward_path(self):
        import torch
        class Tok:
            padding_side='right'; pad_token_id=0; eos_token_id=0; bos_token_id=1; unk_token_id=7
            def encode(self,text,add_special_tokens=False): return [2+len(w)%4 for w in text.split()]
            def __call__(self,text,add_special_tokens=True):return {'input_ids':[1]+self.encode(text)}
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()));self.calls=[]
            def forward(self,input_ids,attention_mask,use_cache):
                self.calls.append((input_ids.clone(),attention_mask.clone()))
                logits=torch.zeros((*input_ids.shape,8)); logits[:,:,3]=1.
                return SimpleNamespace(logits=logits)
        records=[dict(case_id=1,requested_rewrite=dict(prompt='{} is',subject='A',target_new={'str':'one two'},target_true={'str':'a'}),paraphrase_prompts=['A exists','Where A'],neighborhood_prompts=['Who B'])]
        model=Model();result=m.evaluate(model,Tok(),records)
        self.assertEqual(len(model.calls),6)
        self.assertEqual(result['forwards_for_additional_accuracy'],0)
        self.assertEqual(result['aggregates']['RS']['tf_token_micro']['denominator'],2)
        self.assertEqual(len(result['rows']),4)
        self.assertNotIn('prompt',result['rows'][0])
        self.assertEqual(m.contract_receipt()['contract']['layout']['canonical_microbatch'],16)


if __name__=='__main__':unittest.main()
