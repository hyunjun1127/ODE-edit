import unittest
from official.tracking import schema
from official.evaluation.generation.native_profile import PROFILE
from official.evaluation.generation.metrics import generation_payload,reduce_cases
from .flucon_eval import display,INSTRUCTION

def cfg():
    return dict(server='server1',task_id='flucon-paper-scale-20261010',arm='fixture',attempt='r1',source_sha='a'*40,config_sha='b'*64,
        model='llama3',model_family='llama',writer='ft',baseline='FT',role='eval_only',metric_schema='official-baselines-scalar-v1',dataset='cf',
        instruction_id=INSTRUCTION,evaluation_profile='cf-native-generation-W20-only-v1',checkpoint_sha256='c'*64,evaluator_sha256='d'*64,
        stream_sha256='e'*64,tokenizer_sha256='f'*64,source_run_id='61771',generation_metric_schema='counterfact-cake-generation-metrics-v1',
        generation_profile=PROFILE,generation_eval_seed=20261007,reference_assets_sha256='1'*64,generation_source_sha='2'*40,
        generation_repair_instruction=INSTRUCTION,generation_schedule='W20_ONLY_FIRST2000')

class Tests(unittest.TestCase):
    def test_half_up_unrounded(self):
        self.assertEqual(display(6.252105796227186,'bits')['paper_display_x100'],'625.21')
        self.assertEqual(display(.2591242773267912,'cosine')['paper_display_x100'],'25.91')
        self.assertEqual(display(1.23445,'bits')['paper_display_x100'],'123.45')
        self.assertIsNone(display(None,'bits')['paper_display_x100'])
    def test_config(self):
        self.assertEqual(schema.config(cfg())['role'],'eval_only')
        for key,val in [('dataset','zsre'),('generation_schedule','W0_AND_W20_FIRST2000'),('role','scientific')]:
            with self.assertRaises(ValueError):schema.config(dict(cfg(),**{key:val}))
    def test_missing_provenance(self):
        c=cfg();del c['checkpoint_sha256']
        with self.assertRaises(ValueError):schema.config(c)
    def test_progress(self):
        v={'phase':'W20_generation','generation_progress/step':1,'generation_progress/completed_cases':1,'generation_progress/total_cases':2000}
        schema.metrics(v,config_values=cfg())
        with self.assertRaises(ValueError):schema.metrics(dict(v,phase='W0_generation'),config_values=cfg())
    def test_only_final_generation(self):
        v={'edits':2000,'post_state_edits':2000,'all_seen/post/generation/planned_count':2000,
           'all_seen/post/generation/fluency_count':2000,'all_seen/post/fluency/ngram_entropy':6.25,
           'all_seen/post/generation/consistency_count':0}
        schema.metrics(v,config_values=cfg())
        for bad in ({'fit/loss':1.},{'official/all_seen/post/Efficacy':1.,'edits':2000},
                    {'W0_first2000/fluency/ngram_entropy':6.,'edits':0}):
            with self.assertRaises(ValueError):schema.metrics(bad,config_values=cfg())

if __name__=='__main__':unittest.main()
