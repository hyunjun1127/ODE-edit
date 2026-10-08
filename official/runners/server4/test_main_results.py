import unittest
from official.runners.server4.main_results import identity,event

class MainResultTests(unittest.TestCase):
    def binding(self,**kw):
        return identity({'model':'llama3','method':'ALPHAEDIT','dataset':kw.pop('dataset','cf')},
            {'streams':{'cf':{'sha256':'a'*64},'zsre':{'sha256':'b'*64}}},
            {'code_commit':'c'*40,'config_sha256':'d'*64},'fresh-attempt',
            [{'case_id':i} for i in range(2000)],**kw)
    def test_unsubmitted_blank(self):
        row=event(self.binding(),'NOT_SUBMITTED',reason='INPUT_PENDING')
        self.assertEqual(row['table_status'],'');self.assertIsNone(row['job_id'])
    def test_pending_running_exact_name(self):
        b=self.binding(job_id='12',job_name='cpu-fixture-not-submitted')
        self.assertEqual(event(b,'PENDING')['table_status'],'PENDING: cpu-fixture-not-submitted')
        self.assertEqual(event(b,'RUNNING')['table_status'],'ING: cpu-fixture-not-submitted')
        with self.assertRaises(ValueError):event(self.binding(),'RUNNING')
    def test_non_main_excluded(self):
        for role in ('qualification','W0_preparation','collector','heldout_tuning','historical_replay'):
            with self.assertRaises(ValueError):self.binding(role=role)
    def test_only_fresh_W20_gets_values(self):
        b=self.binding(job_id='12',job_name='cpu-fixture-not-submitted',cold='e'*64)
        for state in ('RUNNING','FAILED','CANCELLED'):
            with self.assertRaises(ValueError):event(b,state,metrics={'Efficacy':100},batch=20)
        with self.assertRaises(ValueError):event(b,'W20_COMPLETE',batch=19)
        row=event(b,'W20_COMPLETE',batch=20,metrics={'Efficacy':99.},generation_deferred=True)
        self.assertEqual(row['generation_status'],'DEFERRED')
    def test_cf_zsre_isolation_and_missing_cold(self):
        b=self.binding(dataset='zsre',job_id='12',job_name='cpu-fixture-not-submitted',cold='e'*64)
        with self.assertRaises(ValueError):event(b,'W20_COMPLETE',batch=20,generation_deferred=True)
        b=self.binding(job_id='12',job_name='cpu-fixture-not-submitted')
        with self.assertRaises(ValueError):event(b,'W20_COMPLETE',batch=20)

if __name__=='__main__':unittest.main()
