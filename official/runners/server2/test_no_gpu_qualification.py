"""CPU-only route, explicit-disabled, source and DAG contract regressions."""
import ast
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch
from official.runners.server2 import no_gpu_qualification as nq, submit, run
from official.tracking import schema


class NoGPUQualification(unittest.TestCase):
    def manifest(self):
        return dict(no_gpu_qualification_profile=nq.profile(), runtime={'python':'python'},
            code_commit='a'*40, tracking={'metric_schema':'official-baselines-scalar-v1'})

    def test_exact_overlay_and_old_unaffected(self):
        self.assertFalse(nq.enabled({}))
        self.assertTrue(nq.enabled(self.manifest()))
        for key in ('qualification_plan','qualification_receipt','native_parity_plans','zsre_smoke_plan'):
            with self.assertRaises(ValueError): nq.enabled(dict(self.manifest(), **{key:{}}))
        for mode in ('qualification','smoke'):
            with self.assertRaises(ValueError): nq.check_mode(self.manifest(),mode)
        for mode in ('w0','chain'): nq.check_mode(self.manifest(),mode)

    def test_actual_subset_no_completed_ft_duplicate(self):
        roles=submit.roles('no_gpu_qual')
        self.assertEqual(roles,nq.ROLES)
        self.assertNotIn('CF_FT',roles)
        self.assertEqual(sum(nq.cell(x)[2]=='chain' for x in roles),11)
        self.assertEqual(sum(nq.cell(x)[2]=='w0' for x in roles),2)
        for role in roles:
            argv=submit.runner_argv(Path('/manifest'),role,'no_gpu_qual',Path('/out'),self.manifest())
            self.assertNotIn('smoke',argv)
            self.assertNotIn('qualification',argv)
            self.assertIn('official.runners.server2.run',argv)
            self.assertNotIn('--resume',argv)

    def test_four_lane_dag_all_possible_ready_antichains(self):
        roles=nq.ROLES
        jobs={}; edges={}
        for role in roles:
            dep=submit.stage_dependencies(role,'no_gpu_qual',roles,jobs,[],4)
            edges[role]=set(dep if isinstance(dep,list) else submit.dependency_ids(dep))
            jobs[role]=str(70000+len(jobs))
        # Every reachable completion set; all runnable allocations <=4.
        pending=[frozenset()]; seen=set(pending)
        while pending:
            done=pending.pop()
            ready=[r for r in roles if jobs[r] not in done and edges[r]<=done]
            self.assertLessEqual(len(ready),4)
            for role in ready:
                next_done=done|{jobs[role]}
                if next_done not in seen:
                    seen.add(next_done); pending.append(next_done)
        self.assertIn(frozenset(jobs.values()),seen)
        for role in roles:
            dataset,_,mode=nq.cell(role)
            if mode=='chain': self.assertIn(jobs['W0_'+dataset.upper()],edges[role])

    def test_tracking_both_datasets_no_generation_payload(self):
        for dataset in ('cf','zsre'):
            value=run.tracking_config(self.manifest(),run.configuration('MEMIT',dataset),'chain',Path('/new'))
            schema.config(value)
            if dataset=='cf': self.assertEqual(value['generation_schedule'],'DEFERRED_CHECKPOINT_EVALUATION')
            else: self.assertFalse(any(k.startswith('generation_') for k in value))
            self.assertNotIn('reference_assets_sha256',value)

    def test_cold_w0_no_oracle_or_generation(self):
        import tempfile
        manifest=dict(self.manifest(),model_revision='r',tokenizer_sha256='t',official_tree_sha256='x',
            base_manifest_sha256='b',streams={'cf':{'lock':{'stream_sha256':'s'}}})
        with tempfile.TemporaryDirectory() as folder, patch.object(run,'cf_native_oracle',side_effect=AssertionError('extra forward')), \
             patch.object(run.generation,'observe',side_effect=AssertionError('generation')), \
             patch.object(run,'factual',return_value=({}, {'path':'actual-factual'})), \
             patch.object(run,'evaluate_payload',return_value={}), patch.object(run,'log'):
            result=run.cold_w0(None,None,manifest,[],'cf',Path(folder),None)
            self.assertEqual(result['original_native_reference']['status'],nq.DISABLED)
            self.assertEqual(result['generation_status'],'DEFERRED_NOT_MEASURED')
            self.assertIsNone(result['generation'])

    def test_pre_model_mode_guard(self):
        source=inspect.getsource(run.main)
        self.assertLess(source.index('noqual.check_mode'),source.index('load_model(manifest)'))
        self.assertIn("if deferred(manifest) and not noqual.enabled(manifest)",source)
        ast.parse(source)


if __name__=='__main__': unittest.main()
