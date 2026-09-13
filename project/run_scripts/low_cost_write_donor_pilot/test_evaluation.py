"""CPU synthetic guards; no model load or claimed GPU parity."""
import copy
import unittest

from project.run_scripts.baseline_mechanism_first.contracts import ContractBoundary, digest
from project.run_scripts.baseline_mechanism_first.case_population import source_digest
from project.run_scripts.baseline_mechanism_first.performance_schema import normalize
from .evaluation import unique_prediction, mmlu_prompt
from .panels import bind_counterfact_panel, history_status, seal_audit, split_mmlu, validate_wiki


def records(n):
    return [dict(case_id=i, requested_rewrite=dict(subject=f's{i}', relation_id=f'r{i}',
                  prompt='{} fact', target_new={'str':'new'}, target_true={'str':'true'}),
                 paraphrase_prompts=[f'p{i}-{j}' for j in range(2)],
                 neighborhood_prompts=[f'n{i}-{j}' for j in range(10)]) for i in range(n)]


def result(data):
    groups={}
    for tag,mult in [('RS',1),('PS',2),('NS',10)]:
        rows=[]
        for r in data:
            w=r['requested_rewrite']
            prompts=([w['prompt'].format(w['subject'])] if tag=='RS' else
                     r['paraphrase_prompts' if tag=='PS' else 'neighborhood_prompts'])
            for i,p in enumerate(prompts):
                rows.append(dict(case_id=r['case_id'],prompt_index=i,
                    identity=source_digest([r['case_id'],i,p,'new','true']),
                    new_nll=1.,true_nll=2.,success=tag!='NS'))
        groups[tag]=dict(rows=rows,numerator=sum(r['success'] for r in rows),denominator=len(rows))
    return dict(requests=len(data),metrics=groups)


class EvaluationTests(unittest.TestCase):
    def test_missing_order_requires_exact_pairs(self):
        data=records(2);doc=result(data)
        self.assertEqual(normalize(doc,data)['request_order'],source_digest([0,1]))
        doc['metrics']['PS']['rows'].reverse()
        with self.assertRaises(ContractBoundary):normalize(doc,data)

    def test_direction_and_tie(self):
        data=records(1);doc=result(data)
        doc['metrics']['RS']['rows'][0]['true_nll']=1.
        with self.assertRaises(ContractBoundary):normalize(doc,data)
        self.assertEqual(unique_prediction([.1,.2,.2,.1]),-1)
        self.assertEqual(unique_prediction([0.,0.,0.,0.]),-1)
        self.assertEqual(unique_prediction([.1,.3,.2,.1]),1)

    def test_canonical_cardinality(self):
        data=records(128)
        panel=bind_counterfact_panel(data,list(range(128)),expected_count=128)
        self.assertEqual((panel['R'],panel['P'],panel['N']),(128,256,1280))
        data[0]['neighborhood_prompts'].pop()
        with self.assertRaises(ContractBoundary):bind_counterfact_panel(data,list(range(128)),expected_count=128)

    def test_audit_known_overlap(self):
        data=records(140)
        data[1]['requested_rewrite']['subject']='s0'
        data[1]['requested_rewrite']['relation_id']='r0'
        audit=seal_audit(data,[0])
        self.assertNotIn(0,[r['ordinal'] for r in audit['rows']])
        self.assertNotIn(1,[r['ordinal'] for r in audit['rows']])
        self.assertEqual(audit,seal_audit(data,[0]))

    def test_history_status_does_not_drop(self):
        data=records(3)
        data[1]['requested_rewrite']=copy.deepcopy(data[0]['requested_rewrite'])
        data[1]['requested_rewrite']['target_new']['str']='other'
        statuses=history_status(data,[0,1,2],entry_n=3)
        self.assertEqual([r['status'] for r in statuses],['SUPERSEDED','ACTIVE_TARGET','ACTIVE_TARGET'])

    def test_mmlu_split_and_prompt(self):
        data=[dict(question=f'q{i}',choices=['a','b','c','d'],answer=i%4) for i in range(100)]
        split=split_mmlu(data)
        a,b=[split['groups'][x]['indices'] for x in ('development','audit')]
        self.assertEqual((len(a),len(b)),(32,68))
        self.assertEqual(set(a)|set(b),set(range(100)))
        self.assertFalse(set(a)&set(b))
        self.assertEqual(mmlu_prompt(data[0]),'Question: q0\n(A) a\n(B) b\n(C) c\n(D) d\nAnswer:')

    def test_wiki_token_identity_mask(self):
        rows=[dict(ordinal=i,input_ids=[1,2,3],predicted_tokens=2) for i in range(128)]
        doc=dict(rows=rows,row_identity=digest(rows))
        self.assertEqual(validate_wiki(doc)['predicted_tokens'],256)
        rows[0]['input_ids'][1]=4
        with self.assertRaises(ContractBoundary):validate_wiki(doc)


if __name__=='__main__':unittest.main()
