import itertools
import unittest
from .generation_common import ARMS
from .generation_plan import dependencies, counts, ready

class GenerationPlanTests(unittest.TestCase):
    def plan(self,cap):
        jobs={};edges={}
        for i,arm in enumerate(ARMS):
            edges[arm]=dependencies(arm,['800','801'],jobs,cap)
            jobs[arm]=str(1000+i)
        return jobs,edges
    def test_cap2_width_and_W0_single_owner(self):
        jobs,edges=self.plan(2)
        self.assertEqual(edges['BASE_MEMIT'],['800','801'])
        self.assertEqual(edges['BASE_ALPHAEDIT'],[jobs['BASE_MEMIT']])
        self.assertEqual(edges['CAKE'],[jobs['BASE_MEMIT']])
        self.assertEqual(edges['ALPHAEDIT_BLUE'],[jobs['BASE_ALPHAEDIT']])
        self.assertEqual(edges['PRUNE'],[jobs['CAKE']])
        self.assertEqual(edges['RECT'],[jobs['ALPHAEDIT_BLUE']])
        # Enumerate downward-closed completed sets; ready jobs are a frontier.
        for subset in itertools.chain.from_iterable(itertools.combinations(ARMS,n) for n in range(7)):
            done=set(subset);done_ids={jobs[a] for a in done}|{'800','801'}
            if not all(set(edges[a])<=done_ids for a in done):continue
            runnable=[a for a in ARMS if a not in done and set(edges[a])<=done_ids]
            self.assertLessEqual(len(runnable),2)
        self.assertEqual(dependencies('collector',[],jobs,2),[jobs[a] for a in ARMS])
    def test_stricter_cap1_serial(self):
        jobs,edges=self.plan(1)
        for i,arm in enumerate(ARMS[1:],1):self.assertEqual(edges[arm],[jobs[ARMS[i-1]]])
    def test_exact_role_order_and_no_guessed_ids(self):
        for role,frontier,jobs,cap in [('CAKE',[],{},2),('BASE_MEMIT',['0'],{},2),
            ('BASE_MEMIT',['800_0'],{},2),('collector',[],{},2),('BASE_MEMIT',[],{},3)]:
            with self.assertRaises(RuntimeError):dependencies(role,frontier,jobs,cap)
    def test_count_plan_no_duplicate_generation(self):
        c=counts()
        self.assertEqual(c['generation_edit_state_case_observations_per_arm'],8600)
        self.assertEqual(c['planned_generation_case_observations'],53600)
        self.assertEqual(c['prompt_count_if_fixed10_per_case'],536000)
        self.assertEqual(c['native_per_arm']['ALPHAEDIT_BLUE']['native_z'],4000)
        self.assertEqual(c['native_per_arm']['CAKE']['history_appends'],120)
        self.assertEqual(c['native_per_arm']['BASE_MEMIT']['history_appends'],0)
        self.assertEqual(c['checkpoint_saves'],0)
    def test_unbound_config_not_submission_ready(self):
        c={'generation':{'common_source_status':'NOT_YET_BOUND','reference_status':'NOT_AVAILABLE'},
           'noCP':True,'z_disk_cache':False}
        with self.assertRaisesRegex(RuntimeError,'SH1_GENERATION_SOURCE_NOT_BOUND'):ready(c)
        c['generation'].update(common_source_status='READY_BOUND',reference_status='READY_VERIFIED',
                               W0_owner='BASE_MEMIT',no_GPU_file_poll=True)
        self.assertTrue(ready(c))

if __name__=='__main__':unittest.main()
