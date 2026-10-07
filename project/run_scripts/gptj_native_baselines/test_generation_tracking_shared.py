"""Shared caller CPU fixtures: no SDK startup, network, model or GPU work."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from project.run_scripts.experiment_generation_eval.metrics import PUBLIC_REASONS, generation_payload
from project.run_scripts.experiment_tracking import init as shared_init, schema
from . import generation_tracking as producer


def raw_summary(planned=100, fluency=None, consistency=None):
    fluency=planned if fluency is None else fluency
    consistency=planned if consistency is None else consistency
    result=dict(planned_count=planned,fluency_count=fluency,consistency_count=consistency,
        fluency_sum=3.5*fluency,consistency_sum=.25*consistency,
        missing_reason_counts={reason:0 for reason in PUBLIC_REASONS},
        generation_prompt_count=10*planned,generated_token_count=20*planned,
        length_cap_no_continuation_prompt_count=0,reason_count_unit='request_occurrences_nonexclusive',
        fluency_unit='bits',consistency_unit='cosine_0_to_1')
    if fluency:result['ngram_entropy']=3.5
    if consistency:result['reference_score']=.25
    return result


class FakeTracker:
    def __init__(self,spool=None,accepted=True):
        self.spool=spool
        self.accepted=accepted
        self.points=[]

    def log(self,values):
        self.points.append(copy.deepcopy(values))
        return self.accepted


class SharedGenerationCallerTests(unittest.TestCase):
    def test_actual_raw_summary_uses_exact_shared_mapping(self):
        summary=raw_summary()
        summary.update(raw_text='PRIVATE_TEXT',case_ids=[42],raw_tokens=[1,2])
        before=copy.deepcopy(summary)
        actual=producer.generation_values('current/post',summary,100,0,100)
        expected=dict(generation_payload('current/post',summary),edits=100,
                      pre_state_edits=0,post_state_edits=100)
        self.assertEqual(actual,expected)
        self.assertEqual(summary,before)
        self.assertNotIn('PRIVATE_TEXT',json.dumps(actual))
        self.assertTrue(all(type(v) in (int,float) for v in actual.values()))

    def test_missing_means_remain_omitted_and_literal_reason_is_preserved(self):
        summary=raw_summary(fluency=0,consistency=0)
        summary['missing_reason_counts']['missing_reference']=100
        values=producer.generation_values('current/post',summary,100,0,100)
        self.assertNotIn('current/post/fluency/ngram_entropy',values)
        self.assertNotIn('current/post/consistency/reference_score',values)
        self.assertEqual(values['current/post/generation/missing_missing_reference_count'],100)
        summary['reference_score']=0.
        with self.assertRaisesRegex(ValueError,'MISSING_MEAN_COUNT'):
            producer.generation_values('current/post',summary,100,0,100)

    def test_raw_cosine_endpoint_roundoff_is_not_clamped(self):
        summary=raw_summary()
        summary['reference_score']=math.nextafter(1.,math.inf)
        values=producer.generation_values('current/post',summary,100,0,100)
        self.assertEqual(values['current/post/consistency/reference_score'],summary['reference_score'])
        summary['reference_score']=1.000001
        with self.assertRaisesRegex(ValueError,'RAW_UNIT_RANGE'):
            producer.generation_values('current/post',summary,100,0,100)

    def test_shared_state_axes_and_w0_count_guards(self):
        with self.assertRaisesRegex(ValueError,'POST_STATE_AXIS'):
            producer.generation_values('current/post',raw_summary(),100,0,99)
        with self.assertRaisesRegex(ValueError,'PRE_STATE_REQUIRED'):
            producer.generation_values('current/pre',raw_summary(),100,None,100)
        values=producer.generation_values('W0_first2000',raw_summary(2000),0)
        self.assertEqual(values['W0_first2000/generation/planned_count'],2000)
        with self.assertRaisesRegex(ValueError,'EXACT_FIRST2000'):
            producer.generation_values('W0_first2000',raw_summary(),0)

    def test_legacy_flattened_fixture_still_ends_in_shared_validation(self):
        raw=raw_summary()
        prefix='current/post'
        flattened={key[len(prefix)+1:]:value for key,value in generation_payload(prefix,raw).items()}
        self.assertEqual(producer.generation_values(prefix,flattened,100,0,100),
                         producer.generation_values(prefix,raw,100,0,100))

    def test_shared_startup_and_actual_array_index_zero_identity(self):
        self.assertIs(producer.init,shared_init)
        config=dict(generation=dict(schema='counterfact-cake-generation-metrics-v1',
            profile='cf-cake-prompt-inclusive-total100-eos-corrected-v1',eval_seed=20261007,
            reference_assets_sha256='c'*64,source_sha='d'*40),tracking={'env_file':'/fixture/not-read'})
        lock=dict(source_commit='a'*40,config_sha256='b'*64)
        env=dict(SLURM_JOB_ID='70002',SLURM_ARRAY_JOB_ID='70001',SLURM_ARRAY_TASK_ID='0',
                 SLURM_STEP_ID='batch',PRIVATE='ignored')
        captured={}
        def fake_init(**kwargs):
            captured.update(kwargs)
            bound=schema.bind_job_identity(kwargs['config'],env)
            return types.SimpleNamespace(run_id='fixture',config_values=bound,startup={'status':'READY_ONLINE'})
        with patch.dict(os.environ,env,clear=True),patch.object(producer,'init',side_effect=fake_init),\
                patch.object(producer,'write') as receipt:
            tracker=producer.start_tracking(config,lock,Path('/fixture/no-write'),'CAKE')
        identity=schema.job_identity(tracker.config_values)
        self.assertEqual(identity['job_id'],'70002')
        self.assertEqual(identity['array_task_id'],'0')
        self.assertEqual(identity['job_display_id'],'70001_0')
        self.assertTrue(schema.run_name(tracker.config_values).endswith('-job70001_0'))
        self.assertEqual(captured['env_file'],'/fixture/not-read')
        receipt.assert_called_once()
        self.assertNotIn('PRIVATE',tracker.config_values)

    def test_fake_tracker_receives_scalar_only_payload_without_mutating_summary(self):
        tracker=FakeTracker()
        summary=raw_summary()
        before=copy.deepcopy(summary)
        self.assertTrue(producer.log_generation(tracker,'current/post',summary,100,0,100))
        self.assertEqual(summary,before)
        self.assertEqual(tracker.points,[producer.generation_values('current/post',summary,100,0,100)])
        self.assertFalse(any(key.endswith('_pct') for key in tracker.points[0]))

    def test_repeated_rejections_and_invalid_scalars_do_not_restart_science(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker=FakeTracker(Path(directory),accepted=False)
            with patch('builtins.print'):
                self.assertFalse(producer.log_generation(tracker,'current/post',raw_summary(),100,0,100))
                self.assertFalse(producer.log_generation(tracker,'current/post',raw_summary(),200,100,200))
                invalid=raw_summary()
                invalid['ngram_entropy']=float('nan')
                self.assertFalse(producer.log_generation(tracker,'current/post',invalid,300,200,300))
            self.assertEqual(tracker.caller_dropped_points,3)
            receipt=json.loads((Path(directory)/'caller-receipt.json').read_text())
            self.assertEqual(receipt['status'],'LOGGING_DEGRADED_CALLER')
            self.assertTrue(receipt['scientific_state_unchanged'])
            self.assertEqual(len(tracker.points),2)


if __name__=='__main__':
    unittest.main()
