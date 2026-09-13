import copy
import unittest
from project.run_scripts.baseline_mechanism_first.performance_schema import normalize,subset,from_rows
from project.run_scripts.baseline_mechanism_first.case_population import source_digest
from project.run_scripts.baseline_mechanism_first.contracts import ContractBoundary


def fixture(n=2):
    records=[dict(case_id=i,requested_rewrite=dict(prompt='{} is',subject=str(i),target_new={'str':'new'},target_true={'str':'true'}),
                  paraphrase_prompts=[f'{i} p0',f'{i} p1'],neighborhood_prompts=[f'{i} n{k}' for k in range(10)]) for i in range(n)]
    metrics={}
    for tag in ('RS','PS','NS'):
        rows=[]
        for r in records:
            prompts=[f"{r['case_id']} is"] if tag=='RS' else r['paraphrase_prompts' if tag=='PS' else 'neighborhood_prompts']
            for i,p in enumerate(prompts):rows.append(dict(case_id=r['case_id'],prompt_index=i,
                identity=source_digest([r['case_id'],i,p,'new','true']),new_nll=1.,true_nll=2.,success=tag!='NS'))
        metrics[tag]=dict(rows=rows,numerator=sum(r['success'] for r in rows),denominator=len(rows))
    return records,dict(requests=n,metrics=metrics)


class SchemaTests(unittest.TestCase):
    def test_reproduces_old_failure_and_adapts_exact_rows(self):
        records,doc=fixture();before=copy.deepcopy(doc)
        with self.assertRaises(KeyError):_=doc['request_order']
        out=normalize(doc,records)
        self.assertEqual(out['request_order'],source_digest([0,1]));self.assertEqual(before,doc)
    def test_wrong_existing_order_rejected(self):
        records,doc=fixture();doc['request_order']='wrong'
        with self.assertRaises(ContractBoundary):normalize(doc,records)
    def test_wrong_prompt_hash_rejected(self):
        records,doc=fixture();doc['metrics']['NS']['rows'][0]['identity']='wrong'
        with self.assertRaises(ContractBoundary):normalize(doc,records)
    def test_wrong_row_order_rejected(self):
        records,doc=fixture();doc['metrics']['RS']['rows'].reverse()
        with self.assertRaises(ContractBoundary):normalize(doc,records)
    def test_duplicate_rejected(self):
        records,doc=fixture();doc['metrics']['PS']['rows'][1]=doc['metrics']['PS']['rows'][0]
        with self.assertRaises(ContractBoundary):normalize(doc,records)
    def test_merge_shards_current_exact_denominator(self):
        records,doc=fixture();a=subset(doc,0,1,records);b=subset(doc,1,2,records)
        merged=from_rows([a,b],records)
        for tag,m in (('RS',1),('PS',2),('NS',10)):
            self.assertEqual(merged['metrics'][tag]['denominator'],2*m)
            self.assertEqual(merged['metrics'][tag]['rows'],doc['metrics'][tag]['rows'])
        with self.assertRaises(ContractBoundary):from_rows([a,a,b],records)
    def test_wrong_numerator_rejected(self):
        records,doc=fixture();doc['metrics']['RS']['numerator']=0
        with self.assertRaises(ContractBoundary):normalize(doc,records)
