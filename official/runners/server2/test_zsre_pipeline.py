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
