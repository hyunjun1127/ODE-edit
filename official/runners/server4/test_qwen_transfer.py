import copy
import unittest
from official.experiments.prepare import build_matrix,load_plan
from official.runners.server4 import qwen_plan,qwen_run
from official.runners.server3 import run as original


class TransferTests(unittest.TestCase):
    def test_twelve_exact_and_blue_alpha_l2(self):
        rows=qwen_plan.rows()
        self.assertEqual(len(rows),12)
        for r in rows:
            self.assertEqual(qwen_run.validate_config(r['config']),r['config'])
            if r['config']['method']=='ALPHAEDIT_BLUE':
                self.assertEqual(r['config']['hparams']['L2'],1)
                self.assertEqual(r['config']['hparams']['layers'],[4,8])

    def test_excluded_grid_and_clamp_rejected(self):
        c,p=load_plan()
        extra=[r for r in build_matrix(c,p) if r['model']=='qwen25' and
               (r['run_id'].endswith('-l2-95') or r['run_id'].endswith('-clamp075'))]
        self.assertTrue(extra)
        for r in extra:
            with self.assertRaises(ValueError):qwen_run.validate_config(r)

    def test_native_configs_preserved(self):
        c,p=load_plan();native={r['run_id']:r for r in build_matrix(c,p)}
        for row in qwen_plan.rows():
            r=row['config'];prior=copy.deepcopy(native[r['run_id']])
            if r['dataset']=='zsre' and r['method']=='ALPHAEDIT_BLUE':
                prior['hparams']['L2']=1
                prior['config_sha256']=r['config_sha256']
            self.assertEqual(prior,r)

    def test_dataset_stream_cpu_identity(self):
        from pathlib import Path
        root=Path('/data/janghj/ODE-edit/local/qwen-baselines-12-20261009/preparation-v2/streams')
        for ds in ('cf','zsre'):
            ours=qwen_run.validate_stream(root/(ds+'-stream.json'),ds)
            self.assertEqual(ours,original.validate_stream(root/(ds+'-stream.json'),ds))
            self.assertEqual(len(ours[1]),2000)


if __name__=='__main__':unittest.main()
