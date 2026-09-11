import unittest
import torch
from project.run_scripts.multilayer_joint_compensation.functional import OutputBatch,FunctionalPanel,context_nll
from project.run_scripts.multilayer_joint_compensation.track_b.protocol import BProblem,run


def make_problem(support=(8,),empty=False):
    weights=(torch.tensor([[.15,-.12],[.09,.21]],dtype=torch.float32),)
    x=torch.tensor([[.2,.7],[.3,.1]],dtype=torch.float32)
    logits=lambda ws:x@ws[0]
    teacher=logits((weights[0]-.08,)).detach().log_softmax(-1)
    current_teacher=logits(weights).detach().log_softmax(-1)
    def panel(role):
        lp=current_teacher if role=='current' else teacher
        b=OutputBatch(logits,torch.tensor([0,1]),torch.tensor([0,1]),torch.ones(2),
                      torch.full((2,),.5),lp,-lp.diag(),identity=role,input_tokens=4)
        return FunctionalPanel([] if empty else [b],role)
    base,past,current=[panel(r) for r in ('base','past','current')]
    reference=current.observe(weights)['mean_nll'];nr=(base.value(weights),past.value(weights))
    guard=lambda ws:dict(selected_fp32_shape_valid=ws[0].dtype==torch.float32 and ws[0].shape==weights[0].shape,
                         nonselected_unchanged=True,compute_z_count=0,history_append_count=0)
    return BProblem(weights,support,base,past,current,lambda v:v,lambda v:v,lambda v:v,nr,reference,guard,dict(cpu='toy'))


class TestBProtocol(unittest.TestCase):
    def test_bf_fixed_reference_and_frozen_derivatives(self):
        for arm in ('B-BF4','B-Frozen-BF4'):
            p=make_problem();before=p.wn[0].clone();out,rows=run(p,arm)
            torch.testing.assert_close(p.wn[0],before,rtol=0,atol=0)
            self.assertEqual(len(rows),4);self.assertEqual(out[0].dtype,torch.float32)
            self.assertEqual([r['s'] for r in rows],[0.,.25,.5,.75])
            self.assertEqual(len({r['current_reference_nll'] for r in rows}),1)
            for row in rows:
                self.assertEqual(row['compute_z'],0);self.assertEqual(row['history_append'],0)
                self.assertEqual(row['inner_physical_assignment'],0)
                self.assertEqual(row['normalized_budget'][1],p.native_risks[1]/max(p.native_risks[1],1e-3))
                self.assertGreaterEqual(min(row['solver']['dual']),0.)
            if 'Frozen' in arm:
                self.assertEqual([r['counts']['current']['gradient_backward'] for r in rows],[2,0,0,0])
            else:self.assertEqual([r['counts']['current']['gradient_backward'] for r in rows],[2,2,2,2])

    def test_os_control_support_and_empty_noop(self):
        p=make_problem((4,));_,rows=run(p,'N4-L4-FUNCTIONAL-OS')
        self.assertEqual(rows[0]['support'],[4]);self.assertEqual(rows[0]['h'],1.)
        self.assertIsNone(rows[0]['actual_budget_violation'])
        p=make_problem(empty=True);out,rows=run(p,'B-BF4')
        torch.testing.assert_close(out[0],p.wn[0],rtol=0,atol=0)
        self.assertTrue(all(r['actual_delta_norm']==0 for r in rows))

    def test_invalid_support_and_state_guard(self):
        with self.assertRaisesRegex(ValueError,'SUPPORT'):run(make_problem((4,)),'B-OS')
        p=make_problem();p.validate_state=lambda w:dict(selected_fp32_shape_valid=True,nonselected_unchanged=False)
        with self.assertRaisesRegex(RuntimeError,'GUARD'):run(p,'B-OS')


if __name__=='__main__':unittest.main()
