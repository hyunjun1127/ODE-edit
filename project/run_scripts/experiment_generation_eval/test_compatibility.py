"""CPU fake-observer reuse tests; NOT a native GPT2/GPT-J GPU qualification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from .common import GenerationError, digest, immutable_write
from .compatibility import member, verify_member, verified_endpoint_row
from .generator import rng_snapshot, rng_equal, SINGLETON_ROUTE
from .kv_qualification import build_qualification_plan, run_qualification
from .observer import GenerationObserver, runtime_identity
from .test_kv_generator import CacheModel, Tokenizer, IDENTITY
from .test_observer import FakeAssets


def record(i):
    return dict(ordered_occurrence=i,case_id=100+i,generation_prompts=['1 2'],
        requested_rewrite=dict(relation_id='r',target_new=dict(id='t',str='new')))


class MemberIdentityTests(unittest.TestCase):
    """Exact production five-field member regressions, without model loading."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'observer-identity.json'
        immutable_write(self.path, dict(identity='local-scalar-fixture'))
        # Same constructor used by the production GPT2 prepare/common layer.
        from project.run_scripts.jlz_realization.common import member as production_member
        self.production = production_member(self.path)

    def test_production_five_field_member_and_legacy_core_are_both_exact(self):
        self.assertEqual(set(self.production), {'path', 'bytes', 'sha256', 'inode', 'mtime_ns'})
        self.assertEqual(verify_member(self.production), self.path)
        self.assertEqual(verify_member(member(self.path)), self.path)
        for optional in ('inode', 'mtime_ns'):
            selected = dict(member(self.path), **{optional:self.production[optional]})
            self.assertEqual(verify_member(selected), self.path)

    def test_hash_size_and_present_metadata_mismatch_stay_blocking(self):
        for key in ('bytes', 'sha256', 'inode', 'mtime_ns'):
            wrong = dict(self.production)
            wrong[key] = '0'*64 if key == 'sha256' else wrong[key]+1
            reason = 'BYTES_IDENTITY' if key in ('bytes', 'sha256') else 'METADATA_IDENTITY'
            with self.subTest(key=key), self.assertRaisesRegex(GenerationError, reason):
                verify_member(wrong)
        with self.assertRaisesRegex(GenerationError, 'REGULAR_FILE'):
            verify_member(dict(self.production, path=str(self.path.parent/'absent.json')))

    def test_unknown_fields_and_invalid_metadata_types_are_not_ignored(self):
        for wrong in (dict(self.production, ignored=True), dict(self.production, inode=True),
                      dict(self.production, mtime_ns='unknown'),
                      dict(self.production, sha256='z'*64),
                      dict(self.production, path='relative-observer.json')):
            with self.assertRaisesRegex(GenerationError, 'MEMBER_SCHEMA'):
                verify_member(wrong)

    def test_stat_change_during_verification_is_not_accepted(self):
        import os
        expected = dict(self.production)
        original = member
        def changed(path):
            actual = original(path)
            info = Path(path).stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns+1))
            return actual
        with mock.patch('project.run_scripts.experiment_generation_eval.compatibility.member', side_effect=changed):
            with self.assertRaisesRegex(GenerationError, 'MEMBER_CHANGED'):
                verify_member(expected)


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.model=CacheModel();self.tok=Tokenizer()
        self.assets=FakeAssets();self.records=[record(i) for i in range(1,4)]
        self.state=dict(W='same-cold-physical',H={})
        oldcfg=dict(model_identity=IDENTITY,generation_source_sha='old-generation-source',
            profile='cf-cake-prompt-inclusive-total100-eos-corrected-v1',eval_seed=20261007,
            reference_assets_sha256=self.assets.sha)
        self.old=GenerationObserver(self.model,self.tok,self.assets,oldcfg,self.root/'old')
        self.old_endpoint=self.old.observe(self.records[:1],'W0',state_identity=self.state)
        # Immutable source/runtime/reference/phase evidence is explicit, not a
        # forged whole-W0 success receipt. Only the first complete case exists.
        immutable_write(self.root/'reference.json',dict(identity_sha256=self.assets.sha))
        oldcfg['assets_manifest_member']=member(self.root/'reference.json')
        immutable_write(self.root/'old-config.json',dict(generation=oldcfg))
        immutable_write(self.root/'cold-guard.json',dict(source_commit='old-source-commit',
            phase='W0_generation',commits=0,history_appends=0,model_W=self.state['W'],
            old_generation_runtime=self.old.runtime_sha,whole_endpoint_guard_recorded=False,
            proof_basis='FROZEN_SOURCE_CONTROL_FLOW_COLD_RUNTIME_AND_RPN',partial_rows_authorized=True))
        plan=build_qualification_plan(self.tok,self.records,model_identity=IDENTITY)
        actual=run_qualification(self.model,self.tok,self.assets,plan,out=self.root/'qualification')
        self.actual=actual
        self.cfg=dict(model_identity=IDENTITY,generation_source_sha='new-cache-source',
            reference_assets_sha256=self.assets.sha,generation_route=actual['selected_route'],
            generation_microbatch=actual['fixed_microbatch'],qualification_plan_sha256=digest(plan),
            qualification_receipt_member=actual['member'],qualification_allow_cpu_fixture=True,
            old_w0_reuse=dict(observations_root=str(self.root/'old'/'observations'),
                observer_identity_member=member(self.root/'old'/'observer-identity.json'),
                config_member=member(self.root/'old-config.json'),source_commit='old-source-commit',
                allowed_only_state_W=self.state['W'],cold_observation_guard_member=member(self.root/'cold-guard.json')))

    def new_observer(self,name='new',events=None):
        return GenerationObserver(self.model,self.tok,self.assets,self.cfg,self.root/name,
                                  progress_callback=events.append if events is not None else None)

    def test_partial_original_bytes_source_and_runtime_preserved_mixed_endpoint_verified(self):
        original=member(self.old_endpoint['rows'][0]['observation_path'])
        before=rng_snapshot();events=[];new=self.new_observer(events=events)
        observed=new.observe(self.records,'W0',state_identity=self.state)
        self.assertTrue(rng_equal(before))
        self.assertEqual(observed['work']['cached_case_observations'],1)
        self.assertEqual(observed['work']['new_case_observations'],2)
        row=observed['rows'][0]
        self.assertEqual(row['provenance']['raw_member'],original)
        self.assertEqual(row['provenance']['runtime_sha256'],self.old.runtime_sha)
        self.assertEqual(row['provenance']['generation_source_sha'],'old-generation-source')
        self.assertEqual(member(row['observation_path']),original)
        loaded=new.read_observed(observed['rows_path'])
        self.assertEqual(loaded['summary']['planned_count'],3)
        subset=new.subset(loaded,self.records[:1],'W0_CURRENT',cohort='CURRENT')
        self.assertEqual(subset['summary']['planned_count'],1)
        raw=verified_endpoint_row(subset['rows'][0],subset,expected_state=self.state)
        self.assertEqual(raw['identity']['runtime'],self.old.runtime_sha)
        self.assertEqual(events[-1]['generation_progress/reused_cases'],1)
        self.assertEqual(events[-1]['generation_progress/total_cases'],3)
        self.assertNotIn('edits',events[-1])
        later=new.observe(self.records[:1],'W1',state_identity=dict(W='edited',H={}))
        self.assertNotIn('compatibility_member',later)
        self.assertGreater(new._progress_step,events[-1]['generation_progress/step'])

    def test_changed_raw_unknown_is_excluded_and_regenerated_not_relabelled(self):
        path=Path(self.old_endpoint['rows'][0]['observation_path'])
        raw=json.loads(path.read_text());raw['observations']=[]
        raw['payload_sha256']=digest({k:v for k,v in raw.items() if k!='payload_sha256'})
        # Test-only deliberate corruption of a source fixture.
        path.write_text(json.dumps(raw))
        new=self.new_observer();observed=new.observe(self.records,'W0',state_identity=self.state)
        self.assertEqual(observed['work']['cached_case_observations'],0)
        manifest=json.loads(Path(observed['compatibility_member']['path']).read_text())
        self.assertEqual(manifest['identity']['eligible_cases'],0)
        self.assertEqual(manifest['identity']['excluded'][0]['reason'],'OLD_ROW_UNVERIFIED')

    def test_wrong_physicalstate_and_runtime_binding_block_reuse(self):
        new=self.new_observer()
        with self.assertRaisesRegex(GenerationError,'COLD_W0_ONLY'):
            new.observe(self.records,'W0',state_identity=dict(W='wrong',H={}))
        config=copy.deepcopy(self.cfg);config['model_identity']=dict(IDENTITY,revision='other')
        with self.assertRaisesRegex(GenerationError,'QUALIFICATION_MODEL_IDENTITY'):
            GenerationObserver(self.model,self.tok,self.assets,config,self.root/'wrong')

    def test_no_actual_receipt_no_fake_gpu_pass_and_runtime_route_diff(self):
        config=copy.deepcopy(self.cfg);config.pop('qualification_receipt_member')
        with self.assertRaisesRegex(GenerationError,'ACTUAL_QUALIFICATION_REQUIRED'):
            GenerationObserver(self.model,self.tok,self.assets,config,self.root/'missing')
        config=copy.deepcopy(self.cfg);config.pop('qualification_allow_cpu_fixture')
        with self.assertRaisesRegex(GenerationError,'NATIVE_GPU_QUALIFICATION_REQUIRED'):
            GenerationObserver(self.model,self.tok,self.assets,config,self.root/'cpu')
        self.assertNotEqual(digest(runtime_identity(self.cfg)),self.old.runtime_sha)

    def test_collector_rejects_origins_and_actualreceipt_mutation(self):
        new=self.new_observer();observed=new.observe(self.records,'W0',state_identity=self.state)
        wrong=copy.deepcopy(observed['rows'][0]);wrong['provenance']['route']=SINGLETON_ROUTE
        with self.assertRaisesRegex(GenerationError,'ROW_PROVENANCE'):
            verified_endpoint_row(wrong,observed)
        path=Path(self.actual['member']['path'])
        path.write_text(path.read_text()+' ')  # fixture corruption, not implementation
        with self.assertRaisesRegex(GenerationError,'MEMBER_BYTES_IDENTITY'):
            new.read_observed(observed['rows_path'])


if __name__=='__main__':unittest.main()
