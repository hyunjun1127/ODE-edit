"""CPU fake registration/launcher/count protocol; no actual scheduler calls."""
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

from . import generation_final_submit as submit
from .generation_common import ARMS,SOURCE_ENV
from .generation_final_common import counts
from .generation_plan import dependencies


def config(attempt):
    return dict(task_id=submit.TASK,parent_task_id='parent-fixture',attempt=str(attempt),
        tracking_attempt='final-generation-v1',manual_schedule_authority={'CPU_fixture':True},
        runtime={'python':'/fixture/python','members':[]},assets=[],observer_identity={},
        tracking={'env_file':'/fixture/NONSECRET.env'},
        resources=dict(cpu=6,collector_cpu=6,host_mib=59392,collector_host_mib=24576,
            wall='2-00:00:00',collector_wall='04:00:00',gpu=1),
        generation=dict(package_tree='fixturetree',source_sha='f'*40,
            shared_source_members=[],reference_manifest={'CPU_fixture':True},reference_assets_sha256='a'*64,
            repair=dict(qualification_plan={},qualification_cohort={},qualification_plan_sha256='b'*64,
                shared_qualification_plan={},shared_qualification_plan_sha256='c'*64)),
        arm_configs={arm:{'native':{'closure':[]}} for arm in ARMS})


class FinalSubmitTests(unittest.TestCase):
    def test_final_gpu_launcher_exact_parent_and_collector_explicit_module(self):
        attempt=Path('/fixture/attempt');c=config(attempt)
        for arm in ARMS:
            self.assertEqual(submit.launcher(attempt,arm,c,'a'*40),
                             submit.gpu_launcher(attempt,arm,c,'a'*40))
            self.assertIn('generation_run',submit.launcher(attempt,arm,c,'a'*40))
        text=submit.launcher(attempt,'collector',c,'a'*40)
        self.assertIn('generation_final_collect',text)
        self.assertIn('CUDA_VISIBLE_DEVICES=',text)
        self.assertIn('OMP_NUM_THREADS=6',text)
        self.assertIn(SOURCE_ENV+'='+'a'*40,text)
        self.assertNotIn('API_KEY',text)

    def test_counts_one_final_per_arm_not_old_heavy_schedule(self):
        value=counts()
        self.assertEqual(value['generation_endpoint_calls_per_arm'],1)
        self.assertEqual(value['generation_edit_state_case_observations_per_arm'],2000)
        self.assertEqual(value['planned_generation_case_observations'],12000)
        self.assertEqual(value['cold_W0_generation_case_observations'],0)
        self.assertEqual(value['generation_current_subset_reductions_per_arm'],0)
        self.assertEqual(value['checkpoint_saves'],0)

    def test_cap_two_graph_and_cap_one_stricter_fallback_no_stale_ids(self):
        for cap in (1,2):
            jobs={}
            for index,role in enumerate(ARMS):
                deps=dependencies(role,['99'],jobs,cap)
                if index==0:self.assertEqual(deps,['99'])
                elif cap==1:self.assertEqual(deps,[jobs[ARMS[index-1]]])
                elif role in ('BASE_ALPHAEDIT','CAKE'):self.assertEqual(deps,[jobs['BASE_MEMIT']])
                jobs[role]=str(80000+index)
            self.assertEqual(dependencies('collector',['99'],jobs,cap),list(jobs.values()))

    def test_fake_one_pass_all_seven_held_before_release_then_single_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            local=Path(directory);prep=local/'preparation';prep.mkdir();attempt=local/'attempt'
            c=config(attempt);(prep/'config.json').write_text(json.dumps(c))
            (local/'cpu-integration-final.json').write_text(json.dumps(dict(status='PASS_CPU_INTEGRATION',
                CUDA_initialized=False,source=[],config={'sha256':submit.sha(prep/'config.json')})))
            inspections=[];commands=[];argvs=[]
            def command(argv):
                commands.append(argv)
                if argv[:2]==['git','status']:return ''
                if argv[:2]==['git','rev-parse']:
                    return {'HEAD':'f'*40,'HEAD^{tree}':'t'*40,
                        'HEAD:project/run_scripts/experiment_generation_eval':'fixturetree'}[argv[2]]
                if argv[:2]==['git','archive']:
                    path=next(v.split('=',1)[1] for v in argv if v.startswith('--output='))
                    with tarfile.open(path,'w'):pass
                    return ''
                if argv[:2]==['scontrol','release']:
                    self.assertEqual(len(inspections),7)
                    return ''
                if argv[0]=='squeue':return 'CPU_FIXTURE_NOT_ACTUAL_QUEUE'
                raise AssertionError(argv)
            def sbatch(argv,**kwargs):
                argvs.append(argv)
                self.assertIn('--hold',argv);self.assertIn('--export=NONE',argv)
                self.assertIn('--no-requeue',argv);self.assertIn('--cpus-per-task=6',argv)
                return types.SimpleNamespace(returncode=0,stdout=str(80000+len(argvs)),stderr='')
            def held(job,role,attempt,argv,deps,c):
                inspections.append(role);return {'role':role,'job':job}
            with patch.object(submit,'FINAL_LOCAL',local),patch.object(submit,'PREPARATION',prep),\
                 patch.object(submit,'ATTEMPT',attempt),patch.object(submit,'authority'),\
                 patch.object(submit,'ready'),patch.object(submit,'stat_seal'),\
                 patch.object(submit,'verify'),patch.object(submit,'member',
                    side_effect=lambda p:{'path':str(p),'CPU_fixture':True}),\
                 patch.object(submit,'command',side_effect=command),\
                 patch.object(submit,'admission',return_value=dict(cap=2,frontier=['99'])),\
                 patch.object(submit,'inventory',return_value={'project':[]}),\
                 patch.object(submit,'inspect_held',side_effect=held),\
                 patch.object(submit.subprocess,'run',side_effect=sbatch),\
                 contextlib.redirect_stdout(io.StringIO()):
                result=submit.submit()
            self.assertEqual(len(argvs),7);self.assertEqual(inspections,[*ARMS,'collector'])
            self.assertEqual(len([a for a in commands if a[0]=='squeue']),1)
            self.assertEqual(len([a for a in commands if a[:2]==['scontrol','release']]),7)
            self.assertEqual(result['generation_schedule'],'FINAL_W20_ONLY')
            self.assertEqual(result['W0_generation_calls'],0)
            self.assertEqual(result['actual_qualification'],'NOT_OBSERVED')
            lock=json.loads((attempt/'execution.lock.json').read_text())
            self.assertEqual(lock['qualification_status'],'PLAN_BOUND_NOT_ACTUAL_PASS')
            self.assertEqual(lock['count_plan']['planned_generation_case_observations'],12000)
            self.assertNotIn('old_complete_case_inventory',lock)
            self.assertNotIn('compatibility_manifest',lock)
            self.assertTrue(all('--gres=gpu:1' in argv for argv in argvs[:-1]))
            self.assertNotIn('--gres=gpu:1',argvs[-1])


if __name__=='__main__':unittest.main()
