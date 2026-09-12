import unittest, random
from metrics import scores, labels_and_predictions, transitions
from sklearn.metrics import f1_score, matthews_corrcoef

class MetricsTest(unittest.TestCase):
    def test_independent_against_sklearn(self):
        r=random.Random(20260912)
        for classes in (2,4):
            for _ in range(40):
                g=[r.randrange(classes) for _ in range(100)];p=[r.randrange(-1,classes) for _ in g]
                s=scores(g,p)
                self.assertAlmostEqual(s['weighted_f1'],f1_score(g,p,average='weighted'),places=14)
                self.assertAlmostEqual(s['mcc'],matthews_corrcoef(g,p),places=14)
    def test_perfect_inverse_invalid(self):
        g=[0,1]*50
        self.assertEqual(scores(g,g)['weighted_f1'],1)
        self.assertEqual(scores(g,[1-x for x in g])['mcc'],-1)
        self.assertEqual(scores(g,[-1]*100)['invalid'],100)
        self.assertEqual(scores(g,[-1]*100)['weighted_f1'],0)
    def test_rte_both_branches_and_invalid(self):
        for a in (-1,0,1):
            for q in (0,1):
                for gold in (0,1):
                    d=dict(sentence1='s',sentence2='h',label=gold)
                    row=dict(sentence1='s',sentence2='h',answer=a,prob_yes=.8 if q else .2,prob_no=.2 if q else .8,highest_probability_answer='True' if q else 'False',correct=a==gold,correct_new=q==gold)
                    v=labels_and_predictions('rte',[row],[d])
                    self.assertEqual(v['generation'],[-1 if a==-1 else 1-a]);self.assertEqual(v['alternative'],[1-q])
    def test_tie_and_identity(self):
        d=dict(question='q',answer=0)
        r=dict(sentence='q',answer=-1,prob_a=.2,prob_b=.2,prob_c=.1,prob_d=.1,highest_probability_answer=None,correct=False,correct_new=False)
        self.assertEqual(labels_and_predictions('mmlu',[r],[d])['alternative'],[-1])
        with self.assertRaises(ValueError):labels_and_predictions('mmlu',[r],[dict(question='different',answer=0)])
    def test_pair_accounting(self):
        g=[0,1,0,1];b=[0,0,0,-1];a=[1,1,0,0];t=transitions(g,b,a)
        self.assertEqual((t['lost'],t['gained'],t['still_correct'],t['still_wrong']),(1,1,1,1))
        self.assertEqual(t['endpoint_correct']-t['W0_correct'],t['gained']-t['lost'])
    def test_missing_denominator(self):
        with self.assertRaises(ValueError):scores([0],[])
        with self.assertRaises(ValueError):scores([],[])

if __name__=='__main__':unittest.main()
