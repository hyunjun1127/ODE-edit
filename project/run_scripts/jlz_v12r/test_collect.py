import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from .controller import RequestController
from .profile import arm_profile
from . import TASK,write
from .collect import collect,finite,validate_fit,accounting_snapshot
from project.run_scripts.jlz_realized_writer_sequential.review_completed import validate_rows,reduce_rows,paired

def fit_fixture(arm='MAIN',B=3,noop=False):
    import torch
    p=arm_profile({},arm);layers=p['eligible_layers']
    ctrl=RequestController(torch.tensor([[1.]*B for _ in layers]),torch.tensor([2.]*B),layers,n_exp=p['n_exp'],base_multiplier=p['base_multiplier'])
    events=[];updates=0;backwards=0
    for k in range(1 if noop else 25):
        F=[.01]*B if noop else [.08 if r==0 or (r==1 and k>=3) else .01 for r in range(B)]
        mask,terminal=ctrl.observe(torch.tensor(F,dtype=torch.float64),k);mask=mask.tolist()
        row=dict(candidate=k,ordinal=k+1,F=F,active_mask=mask,controller=ctrl.receipt(),terminal=terminal,
            backward=not terminal,full_task_sum=sum(F),masked_backward_sum=sum(f for f,m in zip(F,mask) if m),
            gradient_status='NO_BACKWARD_TERMINAL' if terminal else 'MASKED_REQUEST_SUM_MEASURED')
        if not terminal:
            ctrl.before_update(torch.tensor(F));ctrl.record_update(mask);updates+=1;backwards+=1
            row.update(post_update_controller=ctrl.receipt(),projection=dict(local_excess_max=0.,radius_excess_max=0.,postcast_shrink=False,
                lr=.1,eps=1e-8,moment_reset=False,adam_updates=ctrl.receipt()['update_counts'],theta=[0.]*B,post_norm=[[0.]*B for _ in layers]))
        events.append(row)
    return p,dict(events=events,candidates=len(events),updates=updates,logical_builds=len(events),logical_subject_forwards=len(events),
        logical_subject_backwards=backwards,terminal_candidate=len(events)-1,terminal_no_backward=True,terminal_extra_forward=0,
        norm_analytic_once=True,production_builder_reverse=0,production_solve_VJP=0,requested_gradient='same_owner_adjoint_sum' if p['blind'] else 'FP64_Lambda_M_T',
        controller=ctrl.receipt(),request_updates=sum(ctrl.receipt()['update_counts']),terminal_states=ctrl.terminal_states(torch.tensor(F)))

def rawrow(kind='R',true=2.,new=1.,identity='row'):
    return dict(identity=identity,case_id=1,kind=kind,prompt_index=0,new_token_identity='new',true_token_identity='true',endpoint='W1',
        new_nll=new,true_nll=true,margin_true_minus_new=true-new,new_token_count=2,new_token_correct=2,new_strict=True,
        true_token_count=3,true_token_correct=2,true_strict=False)

class CollectFixtures(unittest.TestCase):
    def test_all_profiles_mask_counter_grace_and_terminal(self):
        for arm in ('MAIN','BLIND','L4-ONLY','BASE-2X','NO-EXPAND'):
            p,f=fit_fixture(arm);v=validate_fit(f,p,3)
            self.assertEqual((v['builds'],v['subject_backwards'],v['request_updates']),(25,24,45))
            self.assertEqual(f['controller']['expansion'][0],0 if p['n_exp']==0 else 4)
    def test_noop_zero_step_has_one_forward_no_update(self):
        p,f=fit_fixture(noop=True);v=validate_fit(f,p,3)
        self.assertEqual((v['builds'],v['subject_backwards'],v['request_updates']),(1,0,0))
        self.assertEqual(f['terminal_states'],['ZERO_STEP']*3)
    def test_terminal_backward_and_controller_tampering_block(self):
        p,f=fit_fixture();bad=copy.deepcopy(f);bad['events'][-1]['backward']=True
        with self.assertRaisesRegex(RuntimeError,'NO_BACKWARD'):validate_fit(bad,p,3)
        bad=copy.deepcopy(f);bad['events'][12]['post_update_controller']['expansion'][0]=0
        with self.assertRaisesRegex(RuntimeError,'GRACE12'):validate_fit(bad,p,3)
    def test_identity_order_nonfinite_and_ties_fail(self):
        rows=[rawrow('R',1.,1.,'r'),rawrow('P',2.,1.,'p'),rawrow('N',1.,2.,'n')]
        metrics=reduce_rows(rows);self.assertEqual(metrics['R']['numerator'],0);self.assertEqual(metrics['N']['numerator'],1)
        self.assertEqual(metrics['N']['token_micro'],2/3);self.assertEqual(metrics['N']['strict_numerator'],0)
        with self.assertRaisesRegex(ValueError,'EXACT_IDENTITY'):validate_rows(list(reversed(rows)),rows)
        bad=copy.deepcopy(rows);bad[0]['new_nll']=float('nan')
        with self.assertRaisesRegex(ValueError,'NONFINITE'):reduce_rows(bad)
    def test_paired_same_identity_and_signed_margin(self):
        pre=[rawrow('R',2.,1.,'r')];post=[rawrow('R',1.,2.,'r')]
        pair=paired(pre,post);self.assertEqual(pair['R']['preference']['lost'],1)
        bad=copy.deepcopy(post);bad[0]['margin_true_minus_new']=1.
        with self.assertRaisesRegex(ValueError,'MARGIN'):reduce_rows(bad)
    def test_scalar_nonfinite_stops_not_zero_imputed(self):
        with self.assertRaisesRegex(RuntimeError,'NONFINITE'):finite({'unmeasured':float('inf')})
        finite({'unmeasured':None})
    def test_independent_scalar_postcast_budget_blocks_false_saved_summary(self):
        p,f=fit_fixture();f['events'][0]['projection']['post_norm'][0][0]=2.
        with self.assertRaisesRegex(RuntimeError,'INDEPENDENT_POSTCAST'):validate_fit(f,p,3)
    def test_missing_accounting_parent_is_unavailable_not_zero(self):
        import getpass
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);write(path/'submission.json',{'jobs':{'MAIN':1,'collector':2}})
            raw=f'1|{getpass.getuser()}|{TASK}-MAIN|FAILED|100|gres/gpu=1|00:01:00|1:0\n'
            with patch('project.run_scripts.jlz_v12r.collect.subprocess.run',return_value=SimpleNamespace(stdout=raw)):
                value=accounting_snapshot(path)
            self.assertFalse(value['cost_complete']);self.assertIsNone(value['allocated_GPU_seconds'])
            self.assertEqual(value['missing_job_ids'],['2']);self.assertEqual(value['observed_allocated_GPU_seconds'],100)
    def test_failed_reducer_writes_failure_package_before_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);attempt=root/'attempt';attempt.mkdir();out=root/'review'
            value=collect(attempt,out,accounting=False)
            self.assertEqual(value['status'],'CPU_REVIEW_TECHNICAL_BLOCKED')
            self.assertTrue((out/'report-ko.md').exists());self.assertTrue((out/'failure-inventory.json').exists())
            terminal=json.loads((out/'terminal.json').read_text());self.assertFalse(terminal['scientific_coverage_complete'])
    def test_terminal_filename_is_not_fullcoverage_proof(self):
        source=Path(__file__).with_name('collect.py').read_text()
        self.assertIn("complete=len(commits)==20 and 20 in prefix",source)
        self.assertIn("dict(R=2000,P=4000,N=20000)",source)
        self.assertIn("expected=dict(commits=100,history_appends=420)",source)

if __name__=='__main__':unittest.main()
