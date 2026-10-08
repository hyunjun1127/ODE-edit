"""CPU plumbing only: not an actual model/GPU generation PASS."""
import inspect
import unittest
from unittest.mock import patch
from project.run_scripts.experiment_generation_eval import generator as shared
from project.run_scripts.experiment_generation_eval import kv_qualification as qualification
from project.run_scripts.experiment_generation_eval.test_kv_generator import CacheModel,Tokenizer,request
from . import generation_adapter as task


class GenerationRepairTests(unittest.TestCase):
    def test_task_llama_cache_parity_and_no_shared_mutation(self):
        original=shared._generate_kv_bucket
        rows=[request('1 2',0),request('2 2',1),request('1 2',2),request('2 2',3)]
        reference=shared.generate_rows(CacheModel(),Tokenizer(),rows)
        model=CacheModel();model.config.model_type='llama'
        got=task.generate_rows(model,Tokenizer(),rows,route=shared.BATCH_ROUTE,microbatch=4)
        self.assertEqual([r['full_token_ids'] for r in reference],[r['full_token_ids'] for r in got])
        self.assertEqual(model.config.model_type,'llama')
        self.assertIs(shared._generate_kv_bucket,original)
        self.assertNotIn("'llama'",inspect.getsource(original))
        self.assertFalse(model.config.use_cache)
        self.assertEqual(model.calls[0]['batch'],4)
        self.assertTrue(all(r['query']==1 for r in model.calls[1:]))

    def test_unsupported_family_still_rejected(self):
        model=CacheModel();model.config.model_type='unsupported'
        with self.assertRaisesRegex(Exception,'MODEL_FAMILY'):
            task.generate_rows(model,Tokenizer(),[request('1 2',0)],route=shared.SINGLETON_ROUTE)

    def test_post_only_qualification_and_memory_bound(self):
        from . import run
        source=inspect.getsource(run.attach_post_generation)
        self.assertIn("b==first_generation_batch(c)",source)
        self.assertIn("name.startswith('W0')",source)
        self.assertIn('NO_SILENT_SLOW_FALLBACK',source)
        self.assertIn('progress_callback',source)
        self.assertIs(qualification.run_qualification.__globals__['generate_rows'],shared.generate_rows)


if __name__=='__main__':unittest.main()
