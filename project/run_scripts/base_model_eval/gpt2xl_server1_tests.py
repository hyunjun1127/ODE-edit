"""CPU scalar/input/identity fixtures only; never a toy model or forward fit."""
import ast
import copy
import types
import unittest
import tempfile
from unittest.mock import patch

from .gpt2xl_server1_common import DENOMINATORS, check_guard, digest, pair_specs, token_identity
from .gpt2xl_server1 import summary_payload, tracking_start, observe
from project.run_scripts.experiment_tracking import schema
from project.run_scripts.jlz_realization.observe import reduce_rows
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    reduce_rows as independent_reduce, compare_summary, validate_rows)


def row(kind, index):
    value = dict(identity=str(index), case_id=index, kind=kind, prompt_index=0, endpoint='W0',
        new_nll=1., true_nll=2., new_token_count=2, new_token_correct=1, new_strict=False,
        true_token_count=1, true_token_correct=1, true_strict=True,
        margin_true_minus_new=1., new_token_identity='a', true_token_identity='b')
    if kind == 'N':
        value.update(new_nll=2., true_nll=1., margin_true_minus_new=-1.)
    return value


class W0CPUContracts(unittest.TestCase):
    def test_exact_w0_schema_denominators_units_axes_and_no_fit(self):
        raw = [row(kind, index) for kind, count in DENOMINATORS.items() for index in
               range(sum(DENOMINATORS[k] for k in DENOMINATORS if k < kind),
                     sum(DENOMINATORS[k] for k in DENOMINATORS if k < kind)+count)]
        groups = reduce_rows(raw)
        compare_summary(independent_reduce(raw), groups)
        payload = summary_payload(groups)
        schema.metrics(payload, scientific=True)
        self.assertEqual(payload['edits'], 0)
        self.assertEqual(payload['pre_state_edits'], payload['post_state_edits'])
        self.assertEqual(payload['W0_first2000/success_harmonic_pct'], 100)
        self.assertFalse(any(key.startswith(('fit/', 'current/', 'all_seen/')) for key in payload))
        self.assertEqual(payload['W0_first2000/N/token_acc_pct'], 100)

    def test_ties_failure_nonfinite_block_and_missing_not_zero(self):
        fixture = [row('N', 0)]
        fixture[0].update(new_nll=1., true_nll=1., margin_true_minus_new=0.)
        self.assertEqual(independent_reduce(fixture)['N']['numerator'], 0)
        self.assertNotIn('R', independent_reduce(fixture))
        fixture[0]['new_nll'] = float('nan')
        with self.assertRaises(ValueError):
            validate_rows(fixture)

    def test_job_identity_array_zero_signed_step_writer_none(self):
        values = dict(server='server1', task_id='base-model-gpt2xl-w0', arm='W0_BASE_MODEL', attempt='fixture',
            source_sha='a'*40,config_sha='b'*64,model='gpt2xl',model_family='gpt2',writer='none',
            role='scientific',metric_schema='price-first2k-scalar-v1')
        cfg = schema.bind_job_identity(values, {'SLURM_JOB_ID':'9','SLURM_ARRAY_JOB_ID':'8',
            'SLURM_ARRAY_TASK_ID':'0','SLURM_STEP_ID':'-2'})
        self.assertEqual(cfg['job_display_id'], '8_0')
        self.assertIn('job8_0', schema.run_name(cfg))
        self.assertEqual(cfg['writer'], 'none')
        with self.assertRaises(ValueError):
            schema.bind_job_identity(dict(values,job_id='10'),{'SLURM_JOB_ID':'9'})

    def test_full_panel_order_targets_and_token_identity_no_feedback(self):
        record = dict(case_id=17,requested_rewrite=dict(prompt='{} works',subject='X',
            target_new={'str':'new'},target_true={'str':'old'}),paraphrase_prompts=['P0','P1'],
            neighborhood_prompts=['N'+str(i) for i in range(10)])
        from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
        bench = CounterFactAdapter(None, [])
        bench.evaluation_ids = lambda prompt,target: ([len(prompt),11],[len(target),12])
        before = copy.deepcopy(record)
        specs,pairs = pair_specs([record],bench)
        self.assertEqual([value['kind'] for value in specs], ['R']+['P']*2+['N']*10)
        self.assertEqual(len(pairs),26)
        self.assertEqual([value[1] for value in pairs],['new','old']*13)
        self.assertEqual(token_identity([record],bench)[0]['new_token_identity'],digest([[7,11],[3,12]]))
        self.assertEqual(before,record)

    def test_weight_pointer_version_guard(self):
        before = {'projection':(123,0,(6400,1600),'torch.float32')}
        check_guard(before,dict(before))
        with self.assertRaisesRegex(RuntimeError,'MUTATION'):
            check_guard(before,{'projection':(123,1,(6400,1600),'torch.float32')})

    def test_production_tracking_start_immutable_scalar_config_before_model(self):
        from pathlib import Path
        fake = types.SimpleNamespace(run_id='cpu-fixture-not-uploaded',
            startup=dict(url='https://wandb.ai/fixture/runs/cpu-fixture-not-uploaded'), config_values={})
        received = []
        def init(**kwargs):
            received.append(kwargs)
            fake.config_values = schema.bind_job_identity(kwargs['config'])
            return fake
        config = dict(run_instance=dict(attempt='cpu-fixture'),tracking=dict(env_file='no-auth-read-fixture'))
        lock = dict(source_commit='a'*40,config_sha256='b'*64)
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ',{'SLURM_JOB_ID':'9'}), \
                patch('project.run_scripts.experiment_tracking.init',init):
            self.assertIs(tracking_start(config,lock,Path(folder)),fake)
            import json
            receipt = json.loads((Path(folder)/'tracking-identity.json').read_text())
        self.assertTrue(receipt['startup_before_model_load'])
        self.assertEqual(receipt['config']['writer'],'none')
        self.assertEqual(receipt['config']['job_id'],'9')
        self.assertFalse(receipt['scientific_complete'])
        self.assertEqual(set(received[0]),{'env_file','spool','config'})

    def test_production_observer_detects_mutation_without_any_model(self):
        # Pure descriptor and scorer-output mock, not a tiny model/forward pilot.
        from pathlib import Path
        record = dict(case_id=17,requested_rewrite=dict(prompt='{} works',subject='X',
            relation_id='fixture',target_new={'str':'new'},target_true={'str':'old'}),
            paraphrase_prompts=['P0','P1'],neighborhood_prompts=['N'+str(i) for i in range(10)])
        from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
        bench = CounterFactAdapter(None,[])
        bench.evaluation_ids = lambda prompt,target: ([len(prompt),11],[len(target),12])
        expected = token_identity([record],bench)
        descriptor = {'version':0}
        view = types.SimpleNamespace(guard=lambda:dict(descriptor),selected_state=lambda:{'fixture':'hash'},
                                     hooks=lambda:{'fixture':((),(),())})
        def mutated_score(*args):
            descriptor['version'] += 1
            return [dict(nll=1.,token_count=2,token_correct=1,strict=False,
                token_identity=expected[index//2][('new' if index%2==0 else 'true')+'_token_identity'])
                for index in range(26)]
        with tempfile.TemporaryDirectory() as folder, patch(
                'project.run_scripts.jlz_price_gpt2xl.scores.scores',mutated_score):
            with self.assertRaisesRegex(RuntimeError,'MUTATION'):
                observe(view,bench,[record],expected,Path(folder))

    def test_source_only_no_scientific_or_slurm_calls(self):
        from pathlib import Path
        source = Path(__file__).with_name('gpt2xl_server1.py').read_text()
        tree = ast.parse(source)
        names = {node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}
        self.assertFalse(names & {'backward','step','solve','apply_memit_to_model','apply_AlphaEdit_to_model','save_pretrained'})
        self.assertNotIn('sbatch',source)
        self.assertNotIn('torch.load',source)

    def test_production_submit_argv_scoped_exception_and_actual_collector(self):
        from pathlib import Path
        from .gpt2xl_server1_submit import argv_for
        resources = dict(cpu=8, host_mib=65536, wall='04:00:00',
            collector_cpu=8, collector_host_mib=24576, collector_wall='02:00:00')
        gpu = argv_for('GPU',Path('/fixture'),resources)
        self.assertIn('--hold',gpu)
        self.assertIn('--gres=gpu:1',gpu)
        self.assertIn('--export=NONE',gpu)
        self.assertIn('--no-requeue',gpu)
        self.assertFalse(any(value.startswith('--dependency=') for value in gpu))
        cpu = argv_for('collector',Path('/fixture'),resources,'9')
        self.assertIn('--dependency=afterany:9',cpu)
        self.assertIn('--mem=24576M',cpu)
        self.assertFalse(any(value.startswith('--gres=') for value in cpu))
        with self.assertRaisesRegex(RuntimeError,'ACTUAL_COLLECTOR_DEPENDENCY'):
            argv_for('collector',Path('/fixture'),resources)

    def test_frozen_closure_contains_profile_submit_and_real_reducer(self):
        from .gpt2xl_server1_prepare import source_files
        files = source_files()
        self.assertIn('project/run_scripts/base_model_eval/gpt2xl_server1_submit.py',files)
        self.assertIn('project/run_scripts/jlz_realization/observe.py',files)
        self.assertIn('project/run_scripts/jlz_price_gpt2xl/scores.py',files)
        self.assertIn('messages/head/2026-10-07-base-model-w0-server1.json',files)


if __name__ == '__main__':
    unittest.main()
