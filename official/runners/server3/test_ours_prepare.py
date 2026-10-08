import unittest
from official.ours.config import resolve
from official.runners.server3.ours_prepare import tracking_config
from official.tracking.schema import bind_job_identity, run_name

class PreparationTests(unittest.TestCase):
    def test_default_is_only_provisional_and_hash_bound(self):
        r=resolve('qwen25'); c=tracking_config('a'*40,r)
        self.assertEqual(c['config_sha'],r['config_sha256'])
        self.assertNotIn('ours',c)
        self.assertEqual(c['writer'],'memit')
    def test_local_does_not_invent_job(self):
        c=bind_job_identity(tracking_config('a'*40,resolve('qwen25')),environ={})
        self.assertNotIn('job_id',c)
        self.assertTrue(run_name(c).endswith('-local'))
    def test_future_slurm_array_zero(self):
        c=bind_job_identity(tracking_config('a'*40,resolve('qwen25')),
            environ={'SLURM_JOB_ID':'123','SLURM_ARRAY_JOB_ID':'120','SLURM_ARRAY_TASK_ID':'0'})
        self.assertEqual(c['job_id'],'123')
        self.assertTrue(run_name(c).endswith('-job120_0'))
    def test_bad_source_rejected(self):
        with self.assertRaises(ValueError):tracking_config('not-a-source',resolve('qwen25'))

if __name__=='__main__':unittest.main()
