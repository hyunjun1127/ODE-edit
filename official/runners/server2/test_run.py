"""CPU source/metadata/RNG fixtures; not actual native GPU qualification."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from official.experiments import checkpoint
from official.runners.server2 import run
from official.tracking import schema


class RunConnectorTests(unittest.TestCase):
    def test_twelve_exact_gptj_configs_not_ours(self):
        configs = [run.configuration(method, dataset) for dataset in ('cf','zsre')
                   for method in run.METHODS]
        self.assertEqual(len(configs), 12)
        self.assertEqual(len({row['config_sha256'] for row in configs}), 12)
        self.assertTrue(all(row['edit_seed'] == 0 and row['stream']['requests'] == 2000
                            and row['stream']['batch_size'] == 100 for row in configs))
        self.assertFalse(any('PRICE' in row['method'] for row in configs))

    def test_checkpoint_identity_is_transport_independent_and_cell_specific(self):
        manifest = dict(streams={name:{'lock':{'stream_sha256':name+'-stream'}}
                        for name in ('cf','zsre')}, code_commit='a'*40, official_tree_sha256='b'*40,
                        model_revision='c'*40, tokenizer_sha256='d'*64, assets_identity_sha256='e'*64)
        identity = run.checkpoint_identity(manifest, 'MEMIT', 'cf')
        checkpoint.validate_identity(identity)
        other = dict(manifest, registration_stage='qualification', Slurm_job='12345',
                     tracking_run_id='not-science', base_manifest_sha256='f'*64)
        self.assertEqual(identity, run.checkpoint_identity(other, 'MEMIT', 'cf'))
        self.assertNotEqual(identity, run.checkpoint_identity(manifest, 'MEMIT', 'zsre'))
        self.assertNotEqual(identity, run.checkpoint_identity(manifest, 'FT', 'cf'))

    def test_rng_exact_comparison_not_numerical_nearness(self):
        before = checkpoint.rng_snapshot()
        self.assertTrue(run.equal(before, deepcopy(before)))
        modified = deepcopy(before)
        modified['torch_cpu'][0] ^= 1
        self.assertFalse(run.equal(before, modified))
        self.assertFalse(run.equal(np.array([1], dtype=np.float32), np.array([1], dtype=np.float64)))
        self.assertFalse(run.equal(torch.tensor([1.]), torch.tensor([1.000001])))

    def test_cpu_tensor_hash_is_shape_dtype_and_bytes_bound(self):
        self.assertEqual(run.tensor_hash(torch.tensor([[1.,2.]])), run.tensor_hash(torch.tensor([[1.,2.]])))
        self.assertNotEqual(run.tensor_hash(torch.tensor([[1.,2.]])), run.tensor_hash(torch.tensor([1.,2.])))
        self.assertNotEqual(run.tensor_hash(torch.tensor([1.])), run.tensor_hash(torch.tensor([1.], dtype=torch.float64)))

    def test_missing_shared_tracking_is_typed_not_offline_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'OFFICIAL_TRACKING_SHARED_API_NOT_BOUND'):
                run.tracker({}, run.configuration('MEMIT','cf'), 'qualification', Path(temp))

    def test_official_macro_payload_not_old_pair_rpn_relabel(self):
        value = run.evaluate_payload({'summary':{'Efficacy':25., 'Specificity':75.,
            'requests':100, 'Specificity_availability':'NOT_MEASURED'}}, 'W5', 500)
        self.assertEqual(value['official/all_seen/post/Efficacy'],25.)
        self.assertEqual(value['edits'],500)
        self.assertNotIn('all_seen/post/R/success_pct',value)
        self.assertNotIn('official/all_seen/post/Specificity_availability',value)

    def test_actual_shared_config_cf_zsre_without_sdk(self):
        manifest = dict(tracking=dict(metric_schema='official-baselines-scalar-v1'), code_commit='a'*40)
        source = dict(profile='cf-cake-native-casebatch-kv-total100-globalrng-v1', eval_seed=20261007,
                      reference_assets_sha256='b'*64, generation_source_sha='c'*64)
        with patch.object(run.generation, 'configuration', return_value=source):
            cf = run.tracking_config(manifest, run.configuration('MEMIT','cf'), 'chain', Path('MEMIT'))
            self.assertEqual(schema.config(cf), cf)
            self.assertEqual(cf['instruction_id'], run.INSTRUCTION)
            self.assertEqual(cf['generation_schedule'], 'W0_AND_W20_FIRST2000')
            self.assertNotIn('execution_mode', cf)
            zsre = run.tracking_config(manifest, run.configuration('ALPHAEDIT_BLUE','zsre'),
                                       'smoke', Path('BLUE'))
            self.assertEqual(schema.config(zsre), zsre)
            self.assertFalse(any(k.startswith('generation_') for k in zsre))
            self.assertNotIn('reference_assets_sha256', zsre)
            w0 = run.tracking_config(manifest, run.configuration('MEMIT','cf'), 'w0', Path('W0'))
            self.assertEqual(w0['writer'], 'none')
            self.assertEqual(w0['baseline'], 'none')
            self.assertEqual(w0['arm'], 'W0_BASE_MODEL')

    def test_variable_cf_pairs_token_micro_and_desired_true_locality(self):
        def pair(true_nll, new_nll, true_correct, new_correct, token_count=2, neighbor=False):
            def target(nll, count):
                return dict(mean_nll=nll, token_count=token_count,
                            token_correct_count=count, strict_correct=count==token_count)
            return dict(target_true=target(true_nll,true_correct), target_new=target(new_nll,new_correct),
                        desired_target='true' if neighbor else 'new')
        case = dict(rewrite_observations=[pair(2.,1.,0,2)],
            paraphrase_observations=[pair(2.,1.,0,1), pair(1.,1.,2,2,token_count=4)],
            neighborhood_observations=[pair(1.,2.,2,0,neighbor=True)])
        value = run.cf_diagnostics([case], 'W0_first2000')
        self.assertEqual(value['W0_first2000/P/count'],2)
        self.assertEqual(value['W0_first2000/P/success_count'],1)  # tie fails
        self.assertEqual(value['W0_first2000/P/token_acc_pct'],50.)
        self.assertEqual(value['W0_first2000/N/token_acc_pct'],100.)
        self.assertEqual(value['W0_first2000/N/margin_true_minus_new'],-1.)
        self.assertNotEqual(value['W0_first2000/P/count'],4000)

    def test_shared_macro_validation_and_smoke_current_not_allseen100(self):
        manifest = dict(tracking=dict(metric_schema='official-baselines-scalar-v1'), code_commit='a'*40)
        cfg = run.tracking_config(manifest, run.configuration('MEMIT','zsre'), 'smoke', Path('MEMIT'))
        observed=dict(summary=dict(Efficacy=50.,Generalization=60.,Specificity=70.,
                                  Specificity_loc_ans=30.,requests=100))
        value=run.evaluate_payload(observed,'W1',100,current=True)
        schema.metrics(value,scientific=True,config_values=cfg)
        self.assertEqual(value['official/current/post/requests'],100)
        self.assertFalse(any(k.startswith('official/all_seen/post/') for k in value))

    def test_subset_occurrence_order_no_extra_forward(self):
        rows = [dict(case_id=i,occurrence_index=i+1,
            rewrite_prompts_correct=[True],paraphrase_prompts_correct=[False],
            neighborhood_prompts_correct=[True],neighborhood_W0_agreement=[False]) for i in range(500)]
        observed=dict(cases=rows,identity_sha256='f'*64)
        records=[dict(case_id=i,occurrence_index=i+1) for i in range(400,500)]
        subset=run.current_subset(observed,records,'zsre')
        self.assertEqual(subset['summary']['requests'],100)
        self.assertEqual(subset['work']['new_model_forward_calls'],0)
        self.assertEqual(subset['parent_identity_sha256'],'f'*64)
        with self.assertRaisesRegex(ValueError,'CURRENT_SUBSET_ORDERED_OCCURRENCE'):
            run.current_subset(observed,list(reversed(records)),'zsre')

    def test_actual_qualification_and_durable_resume_code_are_present(self):
        text = Path(run.__file__).read_text()
        self.assertIn("checkpoint.load(out/'checkpoints', identity)",text)
        self.assertIn("'contexts_equal'", text.replace('contexts_equal=', "'contexts_equal':"))
        self.assertIn("current_commit_receipt",text)
        self.assertIn("metadata_recovered_from_durable_checkpoint",text)
        self.assertIn("ATTEMPT_ALREADY_EXECUTED",text)
        self.assertNotIn('torch.save(model',text)


if __name__ == '__main__':
    unittest.main()
