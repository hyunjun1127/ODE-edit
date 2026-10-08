"""CPU-only native repair admission/DAG/immutable-source guards."""
import unittest
from pathlib import Path
from . import repo_native_common as common,repo_native_submit as submit

class Control(unittest.TestCase):
    def test_cap2_exact_three_new_native_jobs_and_no_old_cancelled_frontier(self):
        self.assertEqual(common.ARMS,('BASE_MEMIT','BASE_ALPHAEDIT','ALPHAEDIT_BLUE'))
        self.assertEqual(submit.order(2),dict(BASE_MEMIT=[],BASE_ALPHAEDIT=[],ALPHAEDIT_BLUE=['BASE_MEMIT']))
        for cap in (1,2):
            ids={};jobs=[dict(job='80',gpus=1,resource_detail='Dependency=(null) ')]
            parents=submit.order(cap)
            for index,arm in enumerate(common.ARMS):
                ids[arm]=str(100+index)
                dep=submit.role_dependencies(arm,['80'],parents,ids)
                jobs.append(dict(job=ids[arm],gpus=1,resource_detail='Dependency=afterany:'+':'.join(dep)+' '))
            self.assertEqual(submit.width(jobs),cap)
            self.assertEqual(submit.role_dependencies('collector',[],parents,ids),list(ids.values()))
        self.assertEqual(submit.validate_frontier([],[]),[])
        for job in ('61436','61437','61438','61439'):
            with self.assertRaises(RuntimeError):
                submit.validate_frontier([dict(job=job,gpus=1,resource_detail='Dependency=(null) ')],[job])

    def test_launchers_no_fake_job_identity_or_extra_science(self):
        resources=dict(cpu=8,collector_cpu=8,host_mib=65536,collector_host_mib=24576,
            wall='2-00:00:00',collector_wall='04:00:00')
        for role in submit.ROLES:
            argv=submit.arguments(role,[],Path('/attempt'),resources)
            self.assertEqual('--gres=gpu:1' in argv,role!='collector')
            self.assertIn('--hold',argv);self.assertIn('--export=NONE',argv);self.assertIn('--no-requeue',argv)
            launcher=submit.launcher(Path('/source'),'a'*40,role,Path('/attempt'))
            self.assertIn('repo_native_collect' if role=='collector' else 'repo_native_run',launcher)
            for forbidden in ('WANDB_API_KEY','SLURM_JOB_ID=','W0_READY','--arm PRUNE','--arm RECT','--arm CAKE'):
                self.assertNotIn(forbidden,launcher)
        for arm in ('CAKE','PRUNE','RECT'):
            with self.assertRaises(RuntimeError):submit.arguments(arm,[],Path('/attempt'),resources)

    def test_present_empty_pending_allocation_and_typed_GPU_preserve_original_metadata(self):
        calls=[]
        original=('JobId=61436 JobState=PENDING ReqTRES=cpu=8,gres/gpu:a100=1 '
            'ReqNodeList=devbox NodeList= SchedNodeList=devbox Dependency=(null) ')
        def runner(argv):
            calls.append(argv)
            if argv[0]=='squeue':return '61436|owner|protected-new-name|PENDING|gpu:a100:1|Resources|(null)|devbox'
            return original
        result=submit.resource_inventory(runner=runner,owner='owner')
        self.assertEqual(len(result['jobs']),1)
        row=result['jobs'][0]
        self.assertEqual(row['gpus'],1);self.assertEqual(row['allocated_nodes'],'(null)')
        self.assertEqual(row['node_binding'],'devbox')
        self.assertEqual(row['resource_detail_original'],original)
        self.assertIn('NodeList=(null) SchedNodeList=devbox',row['resource_detail'])
        self.assertTrue(row['allocation_parse_compatibility']['present_empty_not_missing'])
        self.assertEqual([c[3] for c in calls if c[0]=='scontrol'],['61436'])

    def test_missing_allocation_field_is_not_normalized_to_empty(self):
        def runner(argv):
            if argv[0]=='squeue':return '61436|owner|protected|PENDING|gpu:1|Resources|(null)|devbox'
            return 'JobState=PENDING ReqTRES=cpu=8,gres/gpu=1 ReqNodeList=devbox SchedNodeList=devbox Dependency=(null) '
        with self.assertRaisesRegex(RuntimeError,'RESOURCE_NODE_BINDING_UNOBSERVED'):
            submit.resource_inventory(runner=runner,owner='owner')

    def test_empty_allocation_requires_detailed_pending_and_single_field(self):
        for state in ('RUNNING',None):
            def runner(argv):
                if argv[0]=='squeue':return '61436|owner|protected|PENDING|gpu:1|Resources|(null)|devbox'
                return ('' if state is None else 'JobState='+state+' ')+\
                    'ReqTRES=cpu=8,gres/gpu=1 ReqNodeList=devbox NodeList= SchedNodeList=devbox Dependency=(null) '
            with self.subTest(state=state),self.assertRaisesRegex(RuntimeError,'EMPTY_ALLOCATION_PENDING_ONLY'):
                submit.resource_inventory(runner=runner,owner='owner')

    def test_other_node_filter_still_precedes_detail_query(self):
        calls=[]
        def runner(argv):
            calls.append(argv)
            if argv[0]=='squeue':return '\n'.join((
                '61436|owner|protected|PENDING|gpu:1|Resources|(null)|devbox',
                '61420|owner|other-server|PENDING|gpu:1|Resources|(null)|server4',
                '61421|foreign|other-user|PENDING|gpu:1|Resources|(null)|devbox'))
            self.assertEqual(argv[3],'61436')
            return 'JobState=PENDING ReqTRES=cpu=8,gres/gpu=1 ReqNodeList=devbox NodeList= SchedNodeList=devbox Dependency=(null) '
        result=submit.resource_inventory(runner=runner,owner='owner')
        self.assertEqual([r['job'] for r in result['jobs']],['61436'])
        self.assertEqual(result['skipped_explicit_other_node_jobs'],1)
        self.assertEqual([c[3] for c in calls if c[0]=='scontrol'],['61436'])

    def test_detailed_node_changes_still_failclosed(self):
        for requested,allocated,error in (('server4','(null)','NODE_BINDING_CHANGED'),
                ('devbox','server4','NODE_ALLOCATION_CHANGED')):
            def runner(argv):
                if argv[0]=='squeue':return '61436|owner|protected|PENDING|gpu:1|Resources|(null)|devbox'
                return 'JobState=PENDING ReqTRES=cpu=8,gres/gpu=1 ReqNodeList='+requested+\
                    ' NodeList='+allocated+' Dependency=(null) '
            with self.subTest(requested=requested,allocated=allocated),self.assertRaisesRegex(RuntimeError,error):
                submit.resource_inventory(runner=runner,owner='owner')

    def test_implicit_pending_with_empty_allocation_remains_conservatively_counted(self):
        def runner(argv):
            if argv[0]=='squeue':return '61499|owner|unmatched-project-name|PENDING|(null)|Resources|(null)|(null)'
            return 'JobState=PENDING ReqTRES=cpu=8,gres/gpu=1 ReqNodeList=(null) NodeList= Dependency=(null) '
        row=submit.resource_inventory(runner=runner,owner='owner')['jobs'][0]
        self.assertEqual(row['gpus'],1);self.assertTrue(row['implicit_pending_counted'])

    def test_unknown_or_malformed_GPU_request_count_never_becomes_zero(self):
        for token in ('gres/gpu:a100=UNKNOWN','gres/gpu=UNKNOWN','gres/gpu:a100=-1',
                'gres/gpu:a100=1.0','gres/gpu:a100=','gres/gpu:a100'):
            def runner(argv):
                if argv[0]=='squeue':return '61436|owner|protected|PENDING|gpu:1|Resources|(null)|devbox'
                return 'JobState=PENDING ReqTRES=cpu=8,'+token+' ReqNodeList=devbox NodeList= Dependency=(null) '
            with self.subTest(token=token),self.assertRaisesRegex(RuntimeError,'GPU_REQUEST_NUMERIC'):
                submit.resource_inventory(runner=runner,owner='owner')

    def test_generic_typed_conflict_or_duplicate_GPU_request_failsclosed(self):
        for token,error in (('gres/gpu=1,gres/gpu:a100=2','GENERIC_TYPED_COUNT_CONFLICT'),
                ('gres/gpu=1,gres/gpu:a100=1,gres/gpu:a100=1','GPU_REQUEST_DUPLICATE_TYPE'),
                ('gres/gpu=1,gres/gpu=1','GPU_REQUEST_DUPLICATE_TYPE')):
            def runner(argv):
                if argv[0]=='squeue':return '61436|owner|protected|PENDING|gpu:1|Resources|(null)|devbox'
                return 'JobState=PENDING ReqTRES=cpu=8,'+token+' ReqNodeList=devbox NodeList= Dependency=(null) '
            with self.subTest(token=token),self.assertRaisesRegex(RuntimeError,error):
                submit.resource_inventory(runner=runner,owner='owner')

    def test_observed_coarse_GPU_without_positive_detailed_GPU_is_not_skipped(self):
        for requested,error in (('ReqTRES=cpu=8 ','COARSE_GPU_WITHOUT_DETAILED_REQUEST'),
                ('ReqTRES=cpu=8,gres/gpu=0 ','COARSE_GPU_WITHOUT_DETAILED_REQUEST'),
                ('','RESOURCE_GPU_REQUEST_UNOBSERVED')):
            def runner(argv):
                if argv[0]=='squeue':return '61436|owner|protected|PENDING|gpu:1|Resources|(null)|devbox'
                return 'JobState=PENDING '+requested+'ReqNodeList=devbox NodeList= Dependency=(null) '
            with self.subTest(requested=requested),self.assertRaisesRegex(RuntimeError,error):
                submit.resource_inventory(runner=runner,owner='owner')

    def test_consistent_generic_typed_and_typed_sum_count_without_double_count(self):
        for token,count in (('gres/gpu=1,gres/gpu:a100=1',1),
                ('gres/gpu:a100=1,gres/gpu:h100=1',2),('gres/gpu=2,gres/gpu:a100=1,gres/gpu:h100=1',2)):
            def runner(argv):
                if argv[0]=='squeue':return '61436|owner|protected|PENDING|gpu:'+str(count)+'|Resources|(null)|devbox'
                return 'JobState=PENDING ReqTRES=cpu=8,'+token+' ReqNodeList=devbox NodeList= Dependency=(null) '
            with self.subTest(token=token):
                row=submit.resource_inventory(runner=runner,owner='owner')['jobs'][0]
                self.assertEqual(row['gpus'],count)
                self.assertEqual(row['gpu_request_validation']['parsed_GPU_count'],count)


if __name__=='__main__':unittest.main()

