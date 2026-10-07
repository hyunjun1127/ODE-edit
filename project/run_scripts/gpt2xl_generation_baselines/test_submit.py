import shlex
import unittest
from pathlib import Path
from .submit import (order,arguments,launcher,ARMS,SOURCE_ENV,dependencies,width,resource_inventory,
    role_dependencies,dependency_text,parse_dependency,held_dependency_conditions)

class Submission(unittest.TestCase):
    def test_real_six_lane_graph_and_W0_gate(self):
        for cap in (1,2):
            ids={};jobs=[];dag=order(cap)
            for i,role in enumerate(ARMS):
                ids[role]=str(i+1)
                dep=[ids[x] for x in dag[role]]
                jobs.append(dict(job=ids[role],gpus=1,resource_detail='Dependency='+('afterany:'+':'.join(dep) if dep else '(null)')+' '))
            self.assertEqual(width(jobs),cap)
        self.assertEqual(order(2)['BASE_ALPHAEDIT'],['BASE_MEMIT'])
        self.assertEqual(dependencies(jobs),['5','6'])
    def test_existing_frontier_preserved(self):
        jobs=[dict(job=str(i),gpus=1,resource_detail='Dependency='+d+' ') for i,d in
            [(1,'(null)'),(2,'(null)'),(3,'afterany:1'),(4,'afterany:2')]]
        self.assertEqual(dependencies(jobs),['3','4'])
        self.assertEqual(width(jobs),2)
    def test_mixed_AND_technical_and_resource_DAG_stays_cap(self):
        for cap in (1,2):
            ids={};jobs=[dict(job='90',gpus=1,resource_detail='Dependency=(null) ')]
            parents=order(cap)
            for index,role in enumerate(ARMS):
                ids[role]=str(index+1)
                conditions=role_dependencies(role,['90'],parents,ids)
                if role=='BASE_MEMIT':self.assertEqual(conditions['afterok'],[])
                else:self.assertEqual(conditions['afterok'],['1'])
                self.assertIn('90',conditions['afterany'])
                jobs.append(dict(job=ids[role],gpus=1,resource_detail='Dependency='+dependency_text(conditions)+' '))
            self.assertEqual(width(jobs),cap)
            self.assertEqual(dependencies(jobs),['6'] if cap==1 else ['5','6'])
            collector=role_dependencies('collector',[],parents,ids)
            self.assertEqual(collector,dict(afterok=[],afterany=list(ids.values())))

    def test_primary_failure_blocks_all_five_not_the_CPU_failure_collector(self):
        ids={role:str(i+1) for i,role in enumerate(ARMS)};parents=order(2)
        states={job:'COMPLETED' for job in ids.values()};states[ids['BASE_MEMIT']]='FAILED'
        def satisfied(conditions):
            return all(states[job]=='COMPLETED' for job in conditions['afterok']) and all(
                states[job] in ('COMPLETED','FAILED') for job in conditions['afterany'])
        for role in ARMS[1:]:
            self.assertFalse(satisfied(role_dependencies(role,[],parents,ids)))
        self.assertTrue(satisfied(role_dependencies('collector',[],parents,ids)))

    def test_AND_only_dependency_parser_unknown_syntax_no_concurrency_credit(self):
        parsed=parse_dependency('afterok:1(unfulfilled),afterany:2:3(fulfilled)')
        self.assertEqual(parsed,dict(afterok={'1'},afterany={'2','3'}))
        for raw in ('afterok:1?afterany:2','aftercorr:1','afterok:1_0','afterany:1,afterok:'):
            self.assertIsNone(parse_dependency(raw))
            jobs=[dict(job='1',gpus=1,resource_detail='Dependency=(null) '),
                dict(job='2',gpus=1,resource_detail='Dependency='+raw+' ')]
            self.assertEqual(width(jobs),2)

    def test_held_may_not_silently_weaken_primary_afterok(self):
        expected=dict(afterok=['1'],afterany=['90'])
        good=held_dependency_conditions('afterok:1(unfulfilled),afterany:90(unfulfilled)',expected,{'1','2'})
        self.assertEqual(good,dict(afterok={'1'},afterany={'90'}))
        # Slurm may omit a completed external resource edge, never own held
        # primary or its dependency type. Terminal accounting is checked later.
        held_dependency_conditions('afterok:1(unfulfilled)',expected,{'1','2'})
        for raw in ('afterany:1:90','afterok:90,afterany:1','(null)','afterok:1?afterany:90'):
            with self.assertRaisesRegex(RuntimeError,'HELD_DEPENDENCY_TYPES'):
                held_dependency_conditions(raw,expected,{'1','2'})

    def test_mixed_submit_arguments_do_not_turn_CPU_collector_into_success_gate(self):
        resources=dict(cpu=8,collector_cpu=8,host_mib=65536,collector_host_mib=24576,
            wall='2-00:00:00',collector_wall='04:00:00')
        argv=arguments('PRUNE',dict(afterok=['100'],afterany=['101','90']),Path('/attempt'),resources)
        self.assertIn('--dependency=afterok:100,afterany:101:90',argv)
        self.assertIn('--kill-on-invalid-dep=yes',argv)
        collector=arguments('collector',dict(afterok=[],afterany=['100','101']),Path('/attempt'),resources)
        self.assertIn('--dependency=afterany:100:101',collector)
        self.assertNotIn('--kill-on-invalid-dep=yes',collector)
        self.assertNotIn('--kill-on-invalid-dep=yes',arguments('BASE_MEMIT',
            dict(afterok=[],afterany=['90']),Path('/attempt'),resources))
    def test_node_filter_precedes_detail_and_implicit_pending_is_counted(self):
        calls=[]
        def runner(argv):
            calls.append(argv)
            if argv[0]=='squeue':
                return '\n'.join(('900|owner|unmatched-native-name|RUNNING|gpu:1|devbox|devbox|(null)',
                    '901|owner|future-no-name-match|PENDING|(null)|Resources|(null)|(null)',
                    '902|owner|other-server-active|RUNNING|gpu:1|othernode|othernode|othernode',
                    '903|owner|other-server-pending|PENDING|gpu:1|Dependency|(null)|othernode',
                    '904|foreign|foreign|RUNNING|gpu:1|devbox|devbox|devbox'))
            job=argv[3]
            self.assertIn(job,('900','901'))
            return 'ReqTRES=cpu=8,gres/gpu=1 ReqNodeList='+('devbox' if job=='900' else '(null)')+' NodeList='+('devbox' if job=='900' else '(null)')+' Dependency=(null) '
        result=resource_inventory(runner=runner,owner='owner')
        self.assertEqual([j['job'] for j in result['jobs']],['900','901'])
        self.assertTrue(result['jobs'][1]['implicit_pending_counted'])
        self.assertEqual(result['skipped_explicit_other_node_jobs'],2)
        self.assertEqual([call[3] for call in calls if call[0]=='scontrol'],['900','901'])
    def test_actual_launchers_no_fake_job_or_secrets(self):
        r=dict(cpu=8,collector_cpu=8,host_mib=65536,collector_host_mib=24576,wall='2-00:00:00',collector_wall='04:00:00')
        args=arguments('BASE_ALPHAEDIT',['100'],Path('/attempt'),r)
        for value in ('--hold','--export=NONE','--no-requeue','--gres=gpu:1','--dependency=afterany:100'):self.assertIn(value,args)
        self.assertNotIn('--gres=gpu:1',arguments('collector',['100'],Path('/attempt'),r))
        script=launcher(Path('/source'),'a'*40,'CAKE',Path('/raid/attempt'))
        for value in ('TMPDIR=/raid/attempt/tmp/CAKE',SOURCE_ENV,'HF_HUB_OFFLINE=1'):self.assertIn(value,script)
        for value in ('WANDB_API_KEY','WANDB_DISABLED','SLURM_JOB_ID='):self.assertNotIn(value,script)
    def test_sidecar_forwards_only_approved_TMPDIR(self):
        from project.run_scripts.experiment_tracking import client
        import inspect
        source=inspect.getsource(client.Tracker.__init__)
        self.assertIn("'TMPDIR'",source)
        self.assertNotIn('env=os.environ.copy()',source)
    def test_compile_actual_production(self):
        for path in Path(__file__).parent.glob('*.py'):compile(path.read_bytes(),str(path),'exec')

if __name__=='__main__':unittest.main()
