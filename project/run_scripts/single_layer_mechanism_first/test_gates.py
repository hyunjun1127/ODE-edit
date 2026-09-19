import copy
import unittest
from .config import require_stage
from .gates import reduce_observation,compare,b1_gate


def fixture():
    value=dict(requests=2,metrics={},strict=dict(rewrite_strict=2,two_P_strict=2,R_two_P_strict=2,R_two_P_NLL_joint=2))
    for k,m in [('RS',1),('PS',2),('NS',10)]:
        rows=[]
        for c in (10,11):
            for i in range(m):
                rows.append(dict(case_id=c,prompt_index=i,identity=f'{k}:{c}:{i}',
                    new_nll=2. if k=='NS' else 1.,true_nll=1. if k=='NS' else 2.,success=True,new_strict=True))
        value['metrics'][k]=dict(rows=rows,numerator=len(rows),denominator=len(rows))
    return value


class Tests(unittest.TestCase):
    def test_stage_fail_closed(self):
        for stage,arm,batch in [('B1','N4',2),('S3','N4',2),('S10','DEC_MODES_CUM',4),('B1','EXTRA',1)]:
            with self.assertRaises(ValueError):require_stage(stage,arm,batch)
        require_stage('B1','N4',1)
        require_stage('S3','DEC_MODES_STEP',2,dict(name='B1_TO_S3',**{'pass':True}))

    def test_tie_is_failure_not_imputation(self):
        x=fixture();x['metrics']['RS']['rows'][0]['true_nll']=1.
        with self.assertRaises(ValueError):reduce_observation(x)

    def test_identity_mismatch(self):
        x=fixture();y=copy.deepcopy(x);y['metrics']['PS']['rows'][0]['identity']='OTHER'
        with self.assertRaises(ValueError):compare(x,y)

    def test_gate_requires_nonzero_and_full512(self):
        x=fixture();s=dict(accepted=True,actual_delta_norm=1.,current_pass=True,full512=True,
           repaired_choices=1,phi_reference_native=1.,phi_reference_gain=.1,tau_risk=1e-6)
        self.assertTrue(b1_gate(x,x,s)['pass'])
        for field,value in [('accepted',False),('actual_delta_norm',0.),('full512',False)]:
            self.assertFalse(b1_gate(x,x,dict(s,**{field:value}))['pass'])

    def test_PS_loss_cannot_be_offset_by_N(self):
        x=fixture();y=copy.deepcopy(x);r=y['metrics']['PS']['rows'][0]
        r.update(new_nll=3.,success=False);y['metrics']['PS']['numerator']-=1
        y['strict']['R_two_P_NLL_joint']-=1
        s=dict(accepted=True,actual_delta_norm=1.,current_pass=True,full512=True,repaired_choices=99)
        self.assertFalse(b1_gate(x,y,s)['pass'])


if __name__=='__main__':unittest.main()
