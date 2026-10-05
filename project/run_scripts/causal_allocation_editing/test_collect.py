"""Independent CPU arithmetic/cardinality tests, no model/tokenizer calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import write,member,digest,NONCE,TASK
from .collect import validate_solver,collect,independent_rows,reduce_rows,paired,batch_diagnostics

def row(kind='R',identity='one'):
    value=dict(case_id=1,kind=kind,prompt_index=0,identity=identity,endpoint='W1',active_at_endpoint=True,
               new_nll=1.,true_nll=1.,margin_true_minus_new=0.)
    for label in ('new','true'):
        value.update({label+'_token_identity':label+'tok',label+'_token_count':2,
                      label+'_token_correct':1,label+'_strict':False})
    return value

def solver_receipt():
    return dict(status='LINE_SEARCH_BUDGET',B=100,logical_evaluations=3,accepted_updates=1,full_gradients=2,
        rejected_trials=1,accepted_candidate_id=1,last_evaluated_candidate_id=2,terminal_gradient_reused=True,
        terminal_extra_gradient=0,terminal_extra_evaluation=0,logical_gradient_reduction='REQUEST_SUM',
        norm_gradient_in_smooth=False,norm_prox_once=True,shared_sum_cap=None,per_block_radius=.75,
        eta0=.1,tau=.1,J_sum=20.,J_mean=.2,events=[dict(candidate_id=0,accepted=True,full_gradient=True),
        dict(candidate_id=1,accepted=True,full_gradient=True),dict(candidate_id=2,accepted=False,full_gradient=False)])

class CollectorTests(unittest.TestCase):
    def test_realization_scalar_package_and_nested_cost(self):
        from project.run_scripts.jlz_realized_writer_sequential.review_completed import Reader
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);view=dict(columns=100,norm_ratio=None,directional_ratio=None,cosine=None,relative_error=None)
            layer=dict(mean=view,canonical=view,rewrite=view,KL=view,effective_Q_status='MEASURED',per_layer_cap=.75,
                actual_update='double(W_candidate_FP32)-double(W_entry_FP32)',ideal_Q=3.,ideal_Q_C0=2.,ideal_Q_H=1.,
                effective_Q=4.,effective_Q_C0=2.5,effective_Q_H=1.5)
            telemetry=dict(candidate=2,terminal=True,B=100,layers={'4':layer},no_extra_model_forward=True,
                no_extra_backward=True,no_durable_tensors=True,seconds=.2,ideal_Q_sum=3.,native_NLL_sum=1.,
                native_KL_sum=.2,norm_sum=.3,scope='local own-write')
            write(folder/'realization.json',telemetry)
            write(folder/'entry-capture.json',dict(fresh_capture=True,pack='pack',H_entry={'4':'beforeH'},seconds=.5))
            for name in ('pre','post'):write(folder/name/'summary.json',dict(seconds=1.))
            write(folder/'writer.json',dict(seconds=2.))
            commit=dict(batch=1,accepted_candidate=2,native_pack='pack',before={'H':{'4':'beforeH'}},
                after={'W':{'4':'weight'}},fit={'seconds':5.,'engine_calls':{'logical_candidates':3}},
                seconds=9.,writer=member(folder/'writer.json'))
            realization,cost=batch_diagnostics(Reader(),folder,commit)
            self.assertEqual(realization['accepted_candidate'],2)
            self.assertIn('NOT additive',cost['time_policy'])
            commit['accepted_candidate']=3
            with self.assertRaisesRegex(RuntimeError,'TERMINAL_REALIZATION_IDENTITY'):
                batch_diagnostics(Reader(),folder,commit)
    def test_tie_fails_every_preference(self):
        result=reduce_rows([row(k,k) for k in ('R','P','N')])
        self.assertTrue(all(v['numerator']==0 for v in result.values()))
    def test_token_micro_and_strict_are_distinct(self):
        result=reduce_rows([row()])['R'];self.assertEqual(result['token_micro'],.5);self.assertEqual(result['strict_numerator'],0)
    def test_duplicate_identity_blocks(self):
        with self.assertRaises(ValueError):independent_rows([row(),row()])
    def test_nonfinite_blocks(self):
        r=row();r['new_nll']=float('nan')
        with self.assertRaises(ValueError):independent_rows([r])
    def test_paired_identity_set_required(self):
        with self.assertRaises(ValueError):paired([row()],[row(identity='other')])
    def test_last_rejected_never_commit(self):
        r=solver_receipt();self.assertEqual(validate_solver(r)['accepted_candidate'],1)
        r['accepted_candidate_id']=2
        with self.assertRaises(RuntimeError):validate_solver(r)
    def test_rejected_backward_is_illegal(self):
        r=solver_receipt();r['events'][2]['full_gradient']=True
        with self.assertRaises(RuntimeError):validate_solver(r)
    def test_native_common_stop_old_policy_not_required(self):
        r=solver_receipt();self.assertEqual(validate_solver(r)['full_gradients'],2)
    def test_gradient_count_includes_initial_and_terminal(self):
        r=solver_receipt();r['full_gradients']=1
        with self.assertRaises(RuntimeError):validate_solver(r)
    def test_bounds_and_fixed_tau(self):
        for key,value in [('logical_evaluations',51),('accepted_updates',25),('full_gradients',26),('tau',.2)]:
            r=solver_receipt();r[key]=value
            with self.assertRaises(RuntimeError):validate_solver(r)
    def test_terminal_name_alone_never_proves_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            attempt=Path(d)/'attempt';attempt.mkdir();identity=attempt/'observer.json';write(identity,dict(rows=[]))
            records=[dict(case_id=i) for i in range(2000)];ordered=digest(list(range(2000)))
            c=dict(task_id=TASK,instruction_id=NONCE,stream='/dataset/counterfact.json',ordered_ids_sha256=ordered,
                   observer_identity=member(identity),cold_W0_H0=dict(W={},H={}))
            write(attempt/'config.json',c)
            from . import sha
            write(attempt/'execution.lock.json',dict(instruction_id=NONCE,config_sha256=sha(attempt/'config.json'),source_commit='frozen'))
            write(attempt/'main/terminal.json',dict(status='W20_COMPLETE'))
            with patch('project.run_scripts.causal_allocation_editing.collect.load_prefix',return_value=records),patch(
                'project.run_scripts.causal_allocation_editing.collect.ORDERED_SHA',ordered):
                result=collect(attempt,accounting=False)
            self.assertEqual(result['status'],'PARTIAL_OR_TECHNICAL_BLOCKED');self.assertEqual(result['commits'],0)
            self.assertTrue((attempt/'collector/report-ko.md').is_file())
            self.assertTrue((attempt/'collector/inventory.json').is_file())
    def test_structural_reducer_fault_still_publishes_failure(self):
        with tempfile.TemporaryDirectory() as d:
            attempt=Path(d)/'attempt';attempt.mkdir()
            # Malformed existing config is preserved; it cannot turn into a
            # COMPLETE or a silent collector exit with no report.
            write(attempt/'config.json',dict(task_id='foreign'))
            result=collect(attempt,accounting=False)
            self.assertEqual(result['status'],'CPU_REVIEW_TECHNICAL_BLOCKED')
            for name in ('reducer-first-error.json','report-ko.md','inventory.json','terminal.json'):
                self.assertTrue((attempt/'collector'/name).is_file())
            self.assertEqual(json.loads((attempt/'config.json').read_text()),dict(task_id='foreign'))

if __name__=='__main__':unittest.main()
