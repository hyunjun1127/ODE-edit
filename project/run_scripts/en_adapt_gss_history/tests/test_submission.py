import unittest
from pathlib import Path
from project.run_scripts.en_adapt_gss_history.freeze import held_checks, project_capacity


class SubmissionContractTests(unittest.TestCase):
    def test_exact_args_memory_and_gpu_rejection(self):
        source=Path('/frozen/source');attempt=Path('/attempt/res');name='odeedit_gss_res_10k_s3'
        script=source/'project/run_scripts/en_adapt_gss_history/run.sbatch'
        cmd=['sbatch','--hold','--export=NONE',str(script),str(attempt),str(source)]
        state=(f'UserId=janghj(1025) JobName={name} JobState=PENDING Reason=JobHeldUser '
               'ReqTRES=cpu=8,mem=119G,node=1,gres/gpu=1 NumCPUs=8 MinMemoryNode=119G '
               'Partition=gpu TimeLimit=7-00:00:00 ReqNodeList=ubuntu Requeue=0 Dependency=(null) '
               f'Command={script} WorkDir={source} SubmitLine={" ".join(cmd)}')
        self.assertTrue(all(held_checks(state,cmd,source,attempt,name).values()))
        for before,after,key in [('gres/gpu=1','gres/gpu=2','GPU'),('119G','59G','memory'),
                                 ('Requeue=0','Requeue=1','requeue'),('/attempt/res','/attempt/gss_rec','args'),
                                 ('UserId=janghj(1025)','UserId=someone(9)','owner')]:
            self.assertFalse(held_checks(state.replace(before,after),cmd,source,attempt,name)[key])

    def test_project_capacity_preserves_unrelated_jobs(self):
        text='12|odeedit_other|RUNNING|gres/gpu:1\n13|not_project|RUNNING|gres/gpu:2\n14|odeedit_gss_res_10k_s3|PENDING|gres/gpu:h200:1'
        capacity=project_capacity(text)
        self.assertEqual([r['job_id'] for r in capacity],['12','14'])
        self.assertEqual(sum(r['GPU'] for r in capacity),2)
        with self.assertRaises(ValueError):project_capacity('15_[1-4]|odeedit_array|PENDING|gres/gpu:1')
        with self.assertRaises(ValueError):project_capacity('16|odeedit_unknown|PENDING|N/A')

    def test_two_held_inspections_precede_any_release(self):
        import json, tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from project.run_scripts.en_adapt_gss_history import freeze as module
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir()
            archive=root/'archive';archive.write_bytes(b'test fixture')
            plan=root/'plan.json';plan.write_text(json.dumps({'storage':{'required_free_bytes':0}}))
            binding={'path':str(plan),'sha256':module.sha(plan)}
            for arm in module.ARMS:
                attempt=root/arm.removeprefix('EN_ADAPT_H_').lower()/'attempt-v1'
                attempt.mkdir(parents=True)
                lock={'arm':arm,'execution':{'source':str(source),'commit':'fixture','members':[],
                    'archive':{'path':str(archive),'sha256':module.sha(archive)}},
                    'input_seals':{},**{key:binding for key in ('resource_plan','reference_stat_binding','map_seal','sequence_identity')}}
                (attempt/'execution.lock.json').write_text(json.dumps(lock))
            calls=[];commands={};released=[]
            def check_output(argv,**kwargs):
                self.assertFalse(released,'query performed after first release')
                calls.append(argv)
                if argv[0]=='squeue':return ''
                if argv[:3]==['scontrol','show','node']:return 'RealMemory=512000 CPUTot=96'
                if argv[0]=='sbatch':
                    job=str(100+len(commands));commands[job]=argv;return job+'\n'
                job=argv[3];cmd=commands[job];name=next(x.split('=',1)[1] for x in cmd if x.startswith('--job-name='))
                return (f'UserId=janghj(1025) JobName={name} JobState=PENDING Reason=JobHeldUser '
                    'ReqTRES=cpu=8,mem=119G,gres/gpu=1 NumCPUs=8 MinMemoryNode=119G '
                    'Partition=gpu TimeLimit=7-00:00:00 ReqNodeList=ubuntu Requeue=0 Dependency=(null) '
                    f'Command={cmd[-3]} WorkDir={source} SubmitLine={" ".join(cmd)}')
            def release(argv,**kwargs):
                self.assertEqual(len([x for x in calls if x[:3]==['scontrol','show','job']]),2)
                self.assertEqual(argv[:2],['scontrol','release']);released.append(argv[-1])
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            with patch.object(module,'BASE',root),patch.object(module.subprocess,'check_output',side_effect=check_output),patch.object(module.subprocess,'run',side_effect=release):
                module.submit()
            self.assertEqual(released,['100','101'])
            for arm in ('res','gss_rec'):
                receipt=json.loads((root/arm/'attempt-v1/release.json').read_text())
                self.assertEqual(receipt['actual_initial'],'NOT_OBSERVED')
                self.assertEqual(receipt['actual_terminal'],'NOT_OBSERVED')


if __name__=='__main__':unittest.main()
