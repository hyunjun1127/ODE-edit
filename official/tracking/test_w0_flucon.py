"""W0-only authority and metrics, CPU fixtures only (no model/SDK/network)."""
import unittest
from . import schema

def config():
    return dict(server='server4', task_id='gptj-w0-flucon-20261011', arm='gptj-w0', attempt='r1',
        source_sha='a'*40, config_sha='b'*64, model='gptj', model_family='gptj', writer='base',
        baseline='W0', role='eval_only', metric_schema='official-baselines-scalar-v1', dataset='cf',
        instruction_id=schema.W0_FLUCON_INSTRUCTION, evaluation_profile='cf-native-generation-W0-only-v1',
        base_model_sha256='c'*64, evaluator_sha256='d'*64, stream_sha256='e'*64, tokenizer_sha256='f'*64,
        generation_metric_schema='counterfact-cake-generation-metrics-v1',
        generation_profile=schema.NATIVE_GENERATION_PROFILE, generation_eval_seed=20261007,
        reference_assets_sha256='1'*64, generation_source_sha='2'*40,
        generation_repair_instruction=schema.W0_FLUCON_INSTRUCTION,
        generation_schedule=schema.W0_GENERATION_SCHEDULE)

def final():
    return {'edits':0,'W0_first2000/generation/planned_count':2000,
        'W0_first2000/generation/fluency_count':2000,'W0_first2000/fluency/ngram_entropy':6.25,
        'W0_first2000/generation/consistency_count':2000,'W0_first2000/consistency/reference_score':.25}

class Tests(unittest.TestCase):
    def test_config(self):
        self.assertEqual(schema.config(config()),config())
        for key,val in [('server','server2'),('model','llama3'),('dataset','zsre'),('role','scientific'),
                ('evaluation_profile','cf-native-generation-W20-only-v1'),
                ('generation_schedule','W20_ONLY_FIRST2000'),('generation_eval_seed',0),
                ('generation_repair_instruction',schema.OFFICIAL_INSTRUCTION),('checkpoint_sha256','c'*64)]:
            with self.subTest(key=key), self.assertRaises(ValueError):schema.config(dict(config(),**{key:val}))
    def test_base_provenance_required(self):
        for key in ('base_model_sha256','evaluator_sha256','stream_sha256','tokenizer_sha256'):
            c=config();del c[key]
            with self.subTest(key=key), self.assertRaises(ValueError):schema.config(c)
    def test_metrics(self):
        self.assertEqual(schema.metrics(final(),scientific=True,config_values=config()),final())
        for bad in ({'fit/loss':1.},{'official/W0_first2000/Efficacy':1.,'edits':0},
                {'edits':2000},{'pre_state_edits':100},{'post_state_edits':True},
                {k.replace('W0_first2000/','all_seen/post/'):v for k,v in final().items()},
                dict(final(),**{'W0_first2000/generation/planned_count':1999})):
            with self.subTest(bad=bad), self.assertRaises(ValueError):schema.metrics(bad,config_values=config())
    def test_progress_and_identity(self):
        p={'phase':'W0_generation','generation_progress/step':1,
            'generation_progress/completed_cases':1,'generation_progress/total_cases':2000}
        schema.metrics(schema.official_generation_progress(p,endpoint='W0'),config_values=config())
        with self.assertRaises(ValueError):schema.metrics(dict(p,phase='W20_generation'),config_values=config())
        c=schema.bind_job_identity(config(),{'SLURM_JOB_ID':'12345'})
        self.assertTrue(schema.run_name(c).endswith('job12345'))
    def test_old_authorities_not_w0_only(self):
        for authority in (schema.OFFICIAL_INSTRUCTION,schema.FLUCON_REEVAL_INSTRUCTION,schema.ZSRE_REEVAL_INSTRUCTION):
            with self.subTest(authority=authority), self.assertRaises(ValueError):
                schema.config(dict(config(),instruction_id=authority))
    def test_production_worker_fake_sdk(self):
        from .test_transport import execute
        progress={'phase':'W0_generation','generation_progress/step':1,
            'generation_progress/completed_cases':2000}
        sdk,out,cfg=execute([progress,final()],cfg=config())
        self.assertEqual(cfg['job_display_id'],'40_0')
        self.assertTrue(sdk.name.endswith('-job40_0'))
        self.assertEqual(out[-1]['method_readback']['status'],'REMOTE_BOUNDED_ROWS_VERIFIED')
        self.assertFalse(out[-1]['scientific_completion_claim'])
        self.assertEqual(sdk.points[-1]['W0_first2000/fluency/ngram_entropy'],6.25)

if __name__=='__main__':unittest.main()
