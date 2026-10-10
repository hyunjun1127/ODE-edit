import unittest
from unittest.mock import patch
from official.runners.server4 import gptj_w0_flucon as runner

class W0Tests(unittest.TestCase):
    def test_truthful_tracking_scope(self):
        c=dict(attempt='r1',source='a'*40,config_sha256='b'*64,evaluator_sha256='c'*64,
               stream={'sha256':'d'*64},tokenizer_sha256='e'*64,reference_identity='f'*64,
               model_members=[dict(path='/base/model.bin',bytes=1,sha256='1'*64)])
        cfg=runner.tracking_values(c)
        self.assertEqual(cfg['evaluation_profile'],'cf-native-generation-W0-only-v1')
        self.assertEqual(cfg['generation_schedule'],'W0_ONLY_FIRST2000')
        self.assertEqual(cfg['role'],'eval_only')
        self.assertEqual(cfg['writer'],'none')
        self.assertNotIn('checkpoint_sha256',cfg)
        self.assertEqual(runner.validate_tracking(cfg),cfg)
    def test_full_stream(self):
        records=[dict(case_id=i,occurrence_index=i+1,generation_prompts=[]) for i in range(2000)]
        c=dict(stream={},ordered_case_ids_sha256=runner.digest(list(range(2000))))
        with patch.object(runner,'verify',return_value='unused'),patch.object(runner,'read',return_value=records):
            self.assertEqual(len(runner.verify_records(c)),2000)
            records[0]['occurrence_index']=0
            with self.assertRaisesRegex(ValueError,'OCCURRENCE'):runner.verify_records(c)
    def test_missing_prompt_rejected(self):
        records=[dict(case_id=i,occurrence_index=i+1) for i in range(2000)]
        c=dict(stream={},ordered_case_ids_sha256=runner.digest(list(range(2000))))
        with patch.object(runner,'verify',return_value='unused'),patch.object(runner,'read',return_value=records):
            with self.assertRaisesRegex(ValueError,'PROMPTS'):runner.verify_records(c)

if __name__=='__main__':unittest.main()
