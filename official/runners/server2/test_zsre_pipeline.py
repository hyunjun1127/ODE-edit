"""CPU only: profile/DAG/call ordering, never claims actual model PASS."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from official.experiments.prepare import read, write_new
from official.runners.server2 import submit, zsre_profile, zsre_pipeline


class ZsrePipelineTests(unittest.TestCase):
    def test_actual_caller_shared_metric_mapping(self):
        from official.runners.server2 import run
        from official.tracking import schema
        # Use the actual production config builder, no invented Slurm ID.
        manifest=dict(tracking={'metric_schema':'official-baselines-scalar-v1'},code_commit='a'*40)
        config=run.tracking_config(manifest,run.configuration('MEMIT','zsre'),'chain',Path('MEMIT'))
        raw=dict(dataset='zsre',summary=dict(Efficacy=50.,Generalization=25.,Specificity=10.,
            Specificity_loc_ans=10.,W0_prediction_agreement=100.,requests=100))
        result=run.evaluate_payload(raw,'W5',500,current=True,config_values=config)
        self.assertEqual(result['zsre/current/post/requests'],100)
        self.assertEqual(result['zsre/current/post/Specificity'],10.)
        self.assertEqual(result['zsre/current/post/W0_prediction_agreement'],100.)
        self.assertEqual(result['zsre/current/post/Specificity_loc_ans'],10.)
        self.assertAlmostEqual(result['zsre/current/post/Score'],3/(1/50+1/25+1/10))
        self.assertEqual(schema.metrics(result,scientific=True,config_values=schema.config(config)),result)
        self.assertFalse(any(k.startswith('official/') or 'generation' in k or 'fluency' in k for k in result))
        raw['summary']['requests']=2000
        w0=run.evaluate_payload(raw,'W0',0,config_values=config)
        self.assertEqual(w0['zsre/W0_first2000/requests'],2000)
        self.assertEqual(w0['post_state_edits'],0)

    def test_exact_profile(self):
        self.assertFalse(zsre_profile.enabled({}))
        p=zsre_profile.profile()
        self.assertTrue(zsre_profile.enabled({'zsre_launch_profile':p}))
        for key,value in [('generation',True),('dataset','cf'),('project_gpu_cap',5),('requests',100)]:
            bad=dict(p);bad[key]=value
            with self.assertRaises(ValueError): zsre_profile.enabled({'zsre_launch_profile':bad})
        with self.assertRaises(ValueError):
            zsre_profile.enabled({'zsre_launch_profile':p,'checkpoint_only_profile':{}})

    def test_cap4_after_existing_frontier_then_model_smoke(self):
        roles=submit.roles('zsre_pipeline');jobs={}
        expected=[['61651','61653','61654','61655'],['700'],['700'],['700'],['700'],['701']]
        for i,role in enumerate(roles):
            self.assertEqual(submit.stage_dependencies(role,'zsre_pipeline',roles,jobs,expected[0],4),expected[i])
            jobs[role]=str(700+i)
        self.assertEqual(submit.stage_dependencies('collector','zsre_pipeline',roles,jobs,[],4),list(jobs.values()))
        # Verify every reachable completed/running set respects <=4, including
        # failures releasing afterany (state does not affect this partial order).
        parents={i:set(int(x)-700 for x in expected[i]) if i else set() for i in range(6)}
        for mask in range(64):
            done={i for i in range(6) if mask>>i&1}
            if not all(parents[i]<=done for i in done): continue
            ready={i for i in range(6) if i not in done and parents[i]<=done}
            self.assertLessEqual(len(ready),4)

    def test_no_main_when_smoke_process_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.json'
            write_new(manifest,dict(zsre_launch_profile=zsre_profile.profile(),registration_stage='zsre_pipeline',
                runtime={'python':'CPU_FIXTURE'},code_commit='a'*40))
            calls=[]
            def invoke(argv,check):
                mode=argv[argv.index('--mode')+1];calls.append(mode)
                return SimpleNamespace(returncode=2 if mode=='smoke' else 0)
            with self.assertRaisesRegex(ValueError,'PHASE_FAILED:smoke'):
                zsre_pipeline.execute(manifest,'MEMIT',root/'MEMIT',invoke=invoke)
            self.assertEqual(calls,['w0','smoke'])
            self.assertEqual(read(root/'MEMIT/pipeline-failure.json')['status'],'TECHNICAL_FAILURE')

    def test_smoke_verified_before_fresh_chain_and_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.json'
            write_new(manifest,dict(zsre_launch_profile=zsre_profile.profile(),registration_stage='zsre_pipeline',
                runtime={'python':'CPU_FIXTURE'},code_commit='a'*40))
            calls=[];proof={'CPU_fixture_only':True}
            def invoke(argv,check):
                mode=argv[argv.index('--mode')+1];calls.append(mode)
                if mode=='chain':
                    self.assertEqual(read(root/'zsre-smoke-verified.json'),proof)
                    write_new(root/'MEMIT/chain/result.json',dict(dataset='zsre',batches=20))
                return SimpleNamespace(returncode=0)
            with patch.object(zsre_pipeline,'verify_smoke',return_value=proof):
                zsre_pipeline.execute(manifest,'MEMIT',root/'MEMIT',invoke=invoke)
                with self.assertRaisesRegex(ValueError,'NO_PIPELINE_AUTOMATIC_RETRY'):
                    zsre_pipeline.execute(manifest,'MEMIT',root/'MEMIT',invoke=invoke)
            self.assertEqual(calls,['w0','smoke','chain'])

    def test_pending_other_arm_without_smoke_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.json'
            write_new(manifest,dict(zsre_launch_profile=zsre_profile.profile(),registration_stage='zsre_pipeline',
                runtime={'python':'CPU_FIXTURE'},code_commit='a'*40))
            with patch.object(zsre_pipeline,'verify_smoke',side_effect=FileNotFoundError('NOT_READY')):
                with self.assertRaises(FileNotFoundError):
                    zsre_pipeline.execute(manifest,'FT',root/'FT',invoke=lambda *a,**k:self.fail('No GPU call permitted'))


if __name__=='__main__': unittest.main()
