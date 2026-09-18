import copy
import unittest
from .reducer import reduce_raw,pair,digest


def fixture():
    records=[dict(case_id=i,requested_rewrite=dict(prompt='{} R',subject=str(i),target_new={'str':'new'},target_true={'str':'true'}),
        paraphrase_prompts=[f'{i} P{j}' for j in range(2)],neighborhood_prompts=[f'{i} N{j}' for j in range(10)]) for i in range(100)]
    raw={};metrics={}
    for panel,prefix,count in (('RS','rewrite',1),('PS','rephrase',2),('NS','locality',10)):
        flags=[]
        for branch in ('new','true'):
            rows=[]
            for record in records:
                prompts=[record['requested_rewrite']['prompt'].format(record['case_id'])] if count==1 else record['paraphrase_prompts' if count==2 else 'neighborhood_prompts']
                for j,prompt in enumerate(prompts):
                    nll=(1. if branch=='new' else 2.) if panel!='NS' else (2. if branch=='new' else 1.)
                    rows.append(dict(case_id=record['case_id'],prompt_index=j,prompt=prompt,target=branch,
                        kind=prefix+'_target_'+branch,nll=nll,target_token_ids=[1],token_predictions=[1],token_correct=[True],all_tokens_correct=True))
                    if branch=='new':flags.append({'success':True})
            raw[prefix+'_target_'+branch]=rows
        metrics[panel]=dict(denominator=count*100,rows=flags)
    value=dict(selection_seal={'status':'SELECTION_SEALED','endpoint_weight_sha256':'fixture'},requests=100,
        request_order=digest(list(range(100))),raw=raw,metrics=metrics)
    return records,value


class ReducerTests(unittest.TestCase):
    def test_counts_strict_and_pair(self):
        records,raw=fixture();r=reduce_raw(raw,records)
        self.assertEqual([r['metrics'][m]['numerator'] for m in ('RS','PS','NS')],[100,200,1000])
        self.assertEqual(r['strict']['R_two_P_NLL_joint'],100)
        self.assertEqual(pair(r,r)['NS']['lost'],0)

    def test_ties_fail_and_NS_direction(self):
        records,raw=fixture()
        raw['raw']['locality_target_new'][0]['nll']=1.
        raw['metrics']['NS']['rows'][0]['success']=False
        r=reduce_raw(raw,records)
        self.assertEqual(r['metrics']['NS']['numerator'],999);self.assertEqual(r['metrics']['NS']['ties'],1)
        _,original=fixture();p=pair(reduce_raw(original,records),r)
        self.assertEqual(p['NS']['lost'],1);self.assertEqual(p['NS']['delta_pp'],-.1)

    def test_schema_identity_finite_and_strict_negative(self):
        records,value=fixture()
        for fault in ('nan','target','prompt','case','missing','strict','duplicate'):
            raw=copy.deepcopy(value);row=raw['raw']['rewrite_target_new'][0]
            if fault=='nan':row['nll']=float('nan')
            if fault=='target':row['target']='other'
            if fault=='prompt':row['prompt']='other'
            if fault=='case':row['case_id']=101
            if fault=='missing':raw['raw']['rewrite_target_new'].pop()
            if fault=='strict':row['all_tokens_correct']=False
            if fault=='duplicate':raw['raw']['rewrite_target_new'][1]=copy.deepcopy(row)
            with self.subTest(fault=fault),self.assertRaises(ValueError):reduce_raw(raw,records)


if __name__=='__main__':unittest.main()
