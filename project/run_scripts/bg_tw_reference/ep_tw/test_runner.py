"""Focused CPU state/order/admission tests, not GPU parity evidence."""
import copy
import unittest
from scripts.fixed_counterfact import encoded,sha
from .runner import Chronology,validate_batch
from .operations import count_reservations

class SequentialTests(unittest.TestCase):
    def test_ten_own_state_links(self):
        state={'W4':'w0','M4':'m0','rng':'r0','ledger':'l0'}
        c=Chronology(state)
        for batch in range(1,11):
            c.begin(batch,state)
            state={k:f'{k}-{batch}' for k in state}
            c.commit(state,[{'layer':4,'history_append':1}])
        self.assertEqual(c.history_count,10);self.assertEqual(c.next_batch,11)
        with self.assertRaises(AssertionError):c.begin(11,{'W4':'w0'})

    def test_append_once_no_history_between_fits(self):
        c=Chronology({'M4':'m0'});c.begin(1,{'M4':'m0'})
        with self.assertRaises(AssertionError):c.commit({'M4':'m1'},[{'layer':4,'history_append':2}])
        with self.assertRaises(AssertionError):c.begin(1,{'M4':'m0'})

    def test_sample_first1000_no_warm_suffix(self):
        records=[{'case_id':i,'requested_rewrite':{'x':i}} for i in range(1000)]
        for b in range(1,11):
            cur=records[(b-1)*100:b*100]
            item=dict(batch=b,ordinals=[(b-1)*100,b*100],case_ids=[r['case_id'] for r in cur],
                record_sha256=sha(encoded(cur)),request_target_sha256=sha(encoded([r['requested_rewrite'] for r in cur])),
                inventory={'RS':100,'PS':200,'NS':1000})
            self.assertEqual(validate_batch(records,b,item),cur)
            bad=copy.deepcopy(item);bad['ordinals']=[5000,5100]
            with self.assertRaises(AssertionError):validate_batch(records,b,bad)

    def test_teacher_and_pending_cap_accounting(self):
        recs={'1':'ReqNodeList=server4 NodeList=(null) ReqTRES=cpu=8,gres/gpu=1',
              '2':'ReqNodeList=server4 NodeList=server4 AllocTRES=cpu=8,gres/gpu=1',
              '3':'ReqNodeList=server2 NodeList=server2 ReqTRES=gres/gpu=1'}
        reservations=count_reservations('1|teacher|PENDING\n2|ep|RUNNING\n3|other|RUNNING',recs.__getitem__)
        self.assertEqual(sum(r['gpus'] for r in reservations),2)
        self.assertEqual([r['job_id'] for r in reservations],['1','2'])

if __name__=='__main__':unittest.main()
