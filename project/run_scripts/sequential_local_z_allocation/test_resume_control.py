import unittest
import json
import tempfile
from pathlib import Path
from .common import save
from .resume_control import capacity_plan,main_pending_classification,verify_initial_gate,inspect_main_fields

class MainGateOverrideTests(unittest.TestCase):
    def test_own_technical_serializes_without_permanent_throttle(self):
        p=capacity_plan([dict(job='49421',capacity=1)])
        self.assertEqual(p['throttle'],2);self.assertEqual(p['dependencies'],['afterok:49421'])
    def test_one_external_slot_remains_one_main_slot(self):
        self.assertEqual(capacity_plan([dict(job='other',capacity=1)])['throttle'],1)
    def test_full_external_admission_serializes_without_job_mutation(self):
        p=capacity_plan([dict(job='other',capacity=2)])
        self.assertEqual(p['throttle'],2);self.assertIn('afterany:other',p['dependencies'])
    def test_overcap_is_blocker_not_gpu_pause(self):
        with self.assertRaises(RuntimeError):capacity_plan([dict(job='other',capacity=3)])
    def classify(self,states,reasons,gpu_free=0,ready=True,registered=6):
        return main_pending_classification([dict(state=s,reason=r) for s,r in zip(states,reasons)],
            dict(gpu_free=gpu_free,gpu_total=4),ready=ready,registered=registered,released=True)
    def test_technical_ready_only_is_not_stop(self):
        self.assertEqual(self.classify(['PENDING'],['Resources'],registered=0),'CONTINUE_NOT_MAIN_READY')
    def test_none_dependency_hold_are_not_resource_evidence(self):
        for r in ('None','Dependency','JobHeldUser','JobArrayTaskLimit'):
            self.assertEqual(self.classify(['PENDING'],[r]),'CONTINUE_PENDING_REASON_NOT_GPU_EVIDENCE')
    def test_running_main_requires_main_gate(self):
        self.assertEqual(self.classify(['RUNNING','PENDING'],['None','JobArrayTaskLimit']),
            'CONTINUE_RUNNING_MAIN_INITIAL_GATE')
    def test_resources_without_gpu_evidence_continues(self):
        self.assertEqual(self.classify(['PENDING'],['Resources'],gpu_free=1),'CONTINUE_PENDING_GPU_EXHAUSTION_NOT_ESTABLISHED')
    def test_verified_gpu_exhaustion_only(self):
        self.assertEqual(self.classify(['PENDING'],['Resources']),'MAIN_GPU_RESOURCE_PENDING_HANDOFF')

class InitialGateBindingTests(unittest.TestCase):
    def fixture(self,p,change=None):
        source='frozen-runtime';state=dict(W={'4':'w4','8':'w8'},M={'4':'m4','8':'m8'},rng='r',contexts='c')
        ready=save(p/'READY.json',dict(status='TECHNICAL_READY'))
        lock=p/'lock.json';save(lock,dict(source_head=source))
        history=dict(rows=[dict(layer=l,history_append=1) for l in range(4,9)],candidate_appends=0)
        if change=='double_history':history['rows'][0]['history_append']=2
        histref=save(p/'history.json',history)
        selected={'gates':[1,0,0,0,0]}
        selref=save(p/'selection.json',dict(selected=selected,final_next_state_decided=True,official_observer_access_before_seal=0))
        commit=dict(source=source,batch=1,next_ordinal=100,next_batch=2,received=list(range(100)),received_count=100,
            state=state,history=histref,history_appends=5,inner_history_appends=0,selection=selref,
            selected=selected,controller_counts=dict(commits=1),checkpoint=None)
        if change=='wrong_source':commit['source']='unbound'
        if change=='wrong_ordinal':commit['next_ordinal']=200
        comref=save(p/'B001/commit.json',commit)
        entry=dict(state=state,previous_commit=comref,ordinals=[100,200])
        if change=='wrong_entry':entry['state']=dict(state,rng='changed')
        save(p/'B002/entry.json',entry)
        gate=dict(status='INITIAL_VALID',arm='C45678',actual_committed_B1=True,technical_ready=ready,
            previous_commit=comref,next_ordinal=100,B2_entry=state,history_appends_B1=5,all_six_actual_PASS=False)
        if change=='overclaim':gate['all_six_actual_PASS']=True
        save(p/'initial-gate.json',gate)
        if change=='altered_commit':(p/'B001/commit.json').write_text('{}')
        return p,lock,p/'READY.json'

    def test_bound_commit_and_next_entry(self):
        with tempfile.TemporaryDirectory() as d:
            result=verify_initial_gate(*self.fixture(Path(d)))
            self.assertEqual(result['status'],'MAIN_INITIAL_VALID')
            self.assertFalse(result['all_six_actual_PASS'])

    def test_nonmatching_evidence_is_not_pass(self):
        for change in ('double_history','wrong_source','wrong_ordinal','wrong_entry','overclaim','altered_commit'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                with self.assertRaises(AssertionError):verify_initial_gate(*self.fixture(Path(d),change))

class HeldInspectionTests(unittest.TestCase):
    def detail(self,script,source,lock):
        return (f'UserId=janghj(1025) JobState=PENDING Priority=0 NumCPUs=8 MinMemoryNode=59G '
            f'ReqNodeList=server4 Requeue=0 Command={script} TresPerNode=gres/gpu:rtx_pro_6000:1 '
            f'ArrayTaskId=0-5 ArrayTaskThrottle=2 TimeLimit=12:00:00 Dependency=afterok:49421 '
            f'SubmitLine=sbatch {script} {source} {lock} science')
    def test_correct_and_resolved_dependency(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);script=p/'run.sbatch';script.write_text('#SBATCH --export=NONE\n')
            detail=self.detail(script,p,p/'lock')
            inspect_main_fields(detail,script,p,p/'lock',12,2,['afterok:49421'])
            inspect_main_fields(detail.replace('afterok:49421','(null)'),script,p,p/'lock',12,2,
                ['afterok:49421'],['afterok:49421'])
    def test_incorrect_held_scope_stops_release(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);script=p/'run.sbatch';script.write_text('#SBATCH --export=NONE\n')
            detail=self.detail(script,p,p/'lock')
            for old,new in [('Priority=0','Priority=1'),('ArrayTaskId=0-5','ArrayTaskId=0-4'),
                ('ArrayTaskThrottle=2','ArrayTaskThrottle=3'),('Requeue=0','Requeue=1'),
                ('afterok:49421','(null)'),(' science',' technical'),('NumCPUs=8','NumCPUs=16')]:
                with self.subTest(old=old),self.assertRaises(AssertionError):
                    inspect_main_fields(detail.replace(old,new),script,p,p/'lock',12,2,['afterok:49421'])

if __name__=='__main__':unittest.main()
