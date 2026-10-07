import unittest
from . import schema,method

class Generation(unittest.TestCase):
    def payload(self,p='current/post',planned=100):
        return {p+'/generation/planned_count':planned,p+'/generation/fluency_count':planned,
                p+'/generation/consistency_count':planned,p+'/generation/generation_prompt_count':planned*3,
                p+'/generation/generated_token_count':planned*30,p+'/fluency/ngram_entropy':4.2,
                p+'/consistency/reference_score':.2,'edits':100,'pre_state_edits':0,'post_state_edits':100}
    def test_raw_units_and_axes(self):
        self.assertEqual(schema.metrics(self.payload(),scientific=True)['current/post/fluency/ngram_entropy'],4.2)
    def test_missing_is_not_zero(self):
        p=self.payload();p['current/post/generation/consistency_count']=0;p.pop('current/post/consistency/reference_score')
        schema.metrics(p,scientific=True)
        p['current/post/consistency/reference_score']=0
        with self.assertRaisesRegex(ValueError,'MISSING_MEAN_COUNT'):schema.metrics(p,scientific=True)
    def test_valid_measured_zero(self):
        p=self.payload();p['current/post/consistency/reference_score']=0;p['current/post/fluency/ngram_entropy']=0
        schema.metrics(p,scientific=True)
    def test_native_cosine_endpoint_roundoff_preserved_not_clamped(self):
        import numpy as np
        from project.run_scripts.experiment_generation_eval.metrics import reference_similarity
        class Matrix:
            def toarray(self):return np.ones((2,3),dtype=np.float64)
        class Vectorizer:
            def transform(self,texts):return Matrix()
        score,reason=reference_similarity(['same'],['same'],Vectorizer())
        self.assertIsNone(reason)
        p=self.payload();p['current/post/consistency/reference_score']=score
        got=schema.metrics(p,scientific=True)
        self.assertEqual(got['current/post/consistency/reference_score'],score)
        p['current/post/consistency/reference_score']=1.000001
        with self.assertRaisesRegex(ValueError,'RAW_UNIT_RANGE'):schema.metrics(p,scientific=True)
    def test_generation_only_w0(self):
        p=self.payload('W0_first2000',2000);p.update(edits=0,pre_state_edits=0,post_state_edits=0)
        schema.metrics(p,scientific=True)
        p['W0_first2000/generation/planned_count']=100
        p['W0_first2000/generation/fluency_count']=100
        p['W0_first2000/generation/consistency_count']=100
        with self.assertRaisesRegex(ValueError,'EXACT_FIRST2000'):schema.metrics(p,scientific=True)
    def test_rpn_w0_count_guard_preserved(self):
        p={'W0_first2000/R/count':100,'edits':0}
        with self.assertRaisesRegex(ValueError,'W0_EXACT_FIRST2000_COUNTS'):schema.metrics(p,scientific=True)
    def test_count_unit_and_reason_privacy(self):
        p=self.payload();p['current/post/generation/fluency_count']=True
        with self.assertRaises(ValueError):schema.metrics(p,scientific=True)
        p=self.payload();p['current/post/generation/missing_raw_prompt_count']=1
        with self.assertRaisesRegex(ValueError,'NOT_ALLOWLISTED'):schema.metrics(p,scientific=True)
        p=self.payload();p['current/post/generation/missing_missing_reference_count']=3
        schema.metrics(p,scientific=True)
    def test_no_fluency_percentage_or_nan(self):
        p=self.payload();p['current/post/fluency/ngram_entropy_pct']=420
        with self.assertRaises(ValueError):schema.metrics(p,scientific=True)
        p=self.payload();p['current/post/consistency/reference_score']=float('nan')
        with self.assertRaises(ValueError):schema.metrics(p,scientific=True)
    def test_same_method_state_guard(self):
        p=self.payload();p['post_state_edits']=0
        with self.assertRaisesRegex(ValueError,'POST_STATE_AXIS'):schema.metrics(p,scientific=True)
    def test_config_public_fields(self):
        c=dict(server='server1',task_id='task',arm='MEMIT',attempt='a',source_sha='a'*40,config_sha='b'*64,
               model='gpt2xl',model_family='gpt2',writer='memit',role='scientific',
               metric_schema=method.COMPARISON_SCHEMA,baseline='MEMIT',
               generation_metric_schema='counterfact-cake-generation-metrics-v1',
               generation_profile='cf-cake-prompt-inclusive-total100-eos-corrected-v1',
               generation_eval_seed=20261007,reference_assets_sha256='c'*64,generation_source_sha='d'*64)
        schema.config(c)
        c['generated_text']='PRIVATE'
        with self.assertRaisesRegex(ValueError,'NOT_ALLOWLISTED'):schema.config(c)
    def test_fake_sdk_generation_axes(self):
        class Fake:
            def __init__(self):self.rows=[]
            def define_metric(self,k,**v):self.rows.append((k,v))
        f=Fake();method.define_axes(f)
        self.assertIn(('w0/current/*',dict(step_metric='edits',step_sync=False)),f.rows)
        self.assertIn(('current/post/*',dict(step_metric='edits',step_sync=False)),f.rows)

if __name__=='__main__':unittest.main()
