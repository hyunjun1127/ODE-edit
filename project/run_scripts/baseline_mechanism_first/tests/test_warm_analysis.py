import copy
import unittest

from project.run_scripts.baseline_mechanism_first.case_population import source_digest
from project.run_scripts.baseline_mechanism_first.contracts import ContractBoundary
from project.run_scripts.baseline_mechanism_first.warm_analysis import check_link, metric_rows


def evaluation():
    metrics={}
    for metric,n in [('RS',1),('PS',2),('NS',10)]:
        rows=[dict(case_id=10,prompt_index=i,identity=str(i),new_nll=1.,true_nll=2.,success=metric!='NS') for i in range(n)]
        metrics[metric]=dict(rows=rows,numerator=n if metric!='NS' else 0,denominator=n)
    return dict(requests=1,request_order=source_digest([10]),evaluator_controller_influence=0,metrics=metrics)


class WarmAnalysisTests(unittest.TestCase):
    def test_canonical_direction_and_cardinality(self):
        counts,_,_=metric_rows(evaluation(),'current','NATIVE',[10])
        self.assertEqual([r['numerator'] for r in counts],[1,2,0])
        self.assertEqual([r['denominator'] for r in counts],[1,2,10])

    def test_tie_is_failure(self):
        data=evaluation();data['metrics']['RS']['rows'][0].update(true_nll=1.,success=False)
        data['metrics']['RS']['numerator']=0
        self.assertEqual(metric_rows(data,'current','W0',[10])[0][0]['numerator'],0)

    def test_duplicate_rejected(self):
        data=evaluation();data['metrics']['PS']['rows'][1]=copy.deepcopy(data['metrics']['PS']['rows'][0])
        with self.assertRaisesRegex(ContractBoundary,'PROMPT_DENOMINATOR'):metric_rows(data,'x','y',[10])

    def test_wrong_sign_nonfinite_order_rejected(self):
        for kind in ('sign','finite','order'):
            data=evaluation()
            if kind=='sign':data['metrics']['NS']['rows'][0]['success']=True
            elif kind=='finite':data['metrics']['RS']['rows'][0]['new_nll']=float('nan')
            else:data['request_order']='wrong'
            with self.subTest(kind=kind),self.assertRaises(ContractBoundary):metric_rows(data,'x','y',[10])

    def test_history_chain_and_native_call_accounting(self):
        observer=dict(compute_z=100,key_calls=2,solve_calls=1,history_append_exact=True,history_append_passes=1,
                      wrappers_restored=True,native_inputs_replaced=False,observer_rng_calls=0)
        receipt=dict(entry_history_sha256='a',history_sha256='b',request_count=100,layer=4,
                     selected_pointer_exact=True,nonselected_bytes_versions_exact=True)
        obs=dict(receipt=receipt,observer=observer,endpoint={'sha256':'endpoint'})
        commit=dict(batch=11,requests=100,history_append=1,status='FINITE_NATIVE_BATCH',endpoint={'sha256':'endpoint'})
        self.assertEqual(check_link(obs,'a',11,commit),'b')
        with self.assertRaisesRegex(ContractBoundary,'HISTORY_CHAIN_BREAK'):check_link(obs,'c',11,commit)
        observer['compute_z']=99
        with self.assertRaisesRegex(ContractBoundary,'NATIVE_COMPUTE_COUNTS'):check_link(obs,'a',11,commit)


if __name__=='__main__':unittest.main()
