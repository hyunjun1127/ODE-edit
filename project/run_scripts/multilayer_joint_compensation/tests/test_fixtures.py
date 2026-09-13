import unittest
from project.run_scripts.multilayer_joint_compensation.banks import latest,select

def records(n):
    return [dict(case_id=i,requested_rewrite=dict(subject=str(i),relation_id='r',prompt='{} is',target_new={'str':'new'},target_true={'str':'old'}),
        paraphrase_prompts=[str(i)+' p1',str(i)+' p2'],neighborhood_prompts=[str(i)+' n'+str(j) for j in range(10)]) for i in range(n)]

class Fixtures(unittest.TestCase):
    def test_latest(self):
        rr=records(4);rr[2]['requested_rewrite']['subject']='  0 '
        self.assertEqual(latest(rr,[0,1,2,3]),[1,2,3])
    def test_disjoint_outcome_blind(self):
        rr=records(1300);panel={'panels':{'Current100':list(range(600,700)),'Fixed100':list(range(100)),'Past100':list(range(100,200))}}
        a=select(rr,600,range(600,700),panel)
        self.assertTrue(all(len(v)==128 for v in a['bank'].values()))
        self.assertEqual(a['nested']['B7'],list(range(600,607)))
        self.assertTrue(all(o['fact_overlap']==0 for o in a['overlaps']))
        for r in rr:r['arbitrary_outcome']=100
        b=select(rr,600,range(600,700),panel)
        self.assertEqual(a['bank'],b['bank'])
    def test_noop_empty_past(self):
        rr=records(300);a=select(rr,0,[],{'panels':{}})
        self.assertEqual(a['bank']['Past'],[]);self.assertEqual(a['current_effective'],[])

if __name__=='__main__':unittest.main()
