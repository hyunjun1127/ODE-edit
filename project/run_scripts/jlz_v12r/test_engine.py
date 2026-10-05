"""CPU tiny random Llama/analytic fixtures, never pretrained GPU evidence."""
import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from project.run_scripts.jlz_native_writer_aware.test_cpu import fixture as inherited_fixture
from project.run_scripts.jlz_native_writer_aware.physical import Adapter
from project.run_scripts.jlz_realized_subject.writer import Transaction
from project.run_scripts.jlz_native_writer_aware.common import state
from .entry import prepare_entry
from .engine import CandidateObjective
from .subject import evaluate,pullback
from .qualification import qualify,comparison,MB2_entry
from .fit import fit
from .telemetry import terminal


def fixture(root,B=3,single=False):
    a,old,H=inherited_fixture(root,B,2)
    bench=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    if single:
        a=Adapter(a.model,dict(a.profile,eligible_layers=[0],anchor_layer=2))
        H={0:H[0]}
    stats={str(l):str(root/f'stats{l}.npz') for l in a.sites}
    entry=prepare_entry(a,bench,old['pack'],H,stats)
    return a,entry,H


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_full_native_owner_entry_and_external_anchor(self):
        for single in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                a,e,H=fixture(Path(folder),3,single)
                self.assertEqual(e['anchors'][2].shape,(3,))
                self.assertEqual(len(e['groups']),3)
                self.assertEqual(set(e['entry_weights']),set(a.sites))
                for group in e['groups']:
                    self.assertEqual(len({r['request'] for r in group['rows']}),1)
                    self.assertEqual(len(group['rows']),3)
                    for row in group['rows']:
                        self.assertEqual(len(row['tokens']['input_ids']),5)

    def test_whole_response_off_owner_pullback_and_blind(self):
        torch.manual_seed(2);B=3;N=7;d=4;n=5
        raw=torch.randn(N,n);P=torch.randn(n,B,dtype=torch.float64)
        adj=torch.randn(N,d);R={0:torch.randn(d,B)}
        rows=[dict(request=i%B) for i in range(N)]
        built=dict(raw={0:raw},P={0:P},rows=rows)
        g,_=pullback(built,{0:adj},R)
        target=adj.double().T@raw.double()@P
        torch.testing.assert_close(g[0],target,atol=1e-12,rtol=1e-12)
        blind,_=pullback(built,{0:adj},R,blind=True)
        ref=torch.zeros(d,B,dtype=torch.float64)
        for i,row in enumerate(rows):ref[:,row['request']]+=adj[i].double()
        torch.testing.assert_close(blind[0],ref,atol=0,rtol=0)
        self.assertGreater(float((target-ref).norm()),1.)
        # This fixed-M quadratic surrogate has a whole-B finite difference.
        M=P.T@raw.double().T;x=R[0].double();eps=1e-6
        for index in ((0,0),(2,1),(3,2)):
            direction=torch.zeros_like(x);direction[index]=eps
            fd=float((((x+direction)@M)*adj.double().T).sum()-(((x-direction)@M)*adj.double().T).sum())/(2*eps)
            self.assertAlmostEqual(fd,float(target[index]),places=7)

    def test_same_mask_inactive_forward_no_loss_backward(self):
        with tempfile.TemporaryDirectory() as folder:
            a,e,H=fixture(Path(folder),3);engine=CandidateObjective(a,e);R=engine.zeros();built=engine.build(R,0)
            mask=torch.tensor([True,False,True]);value=evaluate(a,e,built,backward=True,fixed_mask=mask)
            self.assertTrue(all(value['complete_owner']));self.assertEqual(value['forward_groups'],3)
            self.assertEqual(value['backward_groups'],2)
            inactive=[r['global_row'] for r in built['rows'] if r['request']==1]
            for adj in value['adjoint'].values():self.assertEqual(float(adj[inactive].norm()),0.)
            self.assertGreater(float(value['F'][1]),0.)
            terminal_value=evaluate(a,e,built,backward=True,terminal=True,fixed_mask=mask)
            self.assertEqual(terminal_value['backward_groups'],0);self.assertIsNone(terminal_value['adjoint'])

    def test_three_candidate_no_fit_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            a,e,H=fixture(Path(folder),3);before=state(a,H)
            result=qualify(a,e,max_candidates=3)
            self.assertEqual(result['fixed_candidates'],3);self.assertEqual(result['optimizer_updates'],0)
            self.assertEqual(state(a,H),before)
            self.assertEqual(result['permanent_commit'],0)
            projected=result['candidate_checks'][2]['FP32_feasible_boundary_projection']
            self.assertIsNotNone(projected)
            self.assertEqual([v[0] for v in projected['post_norm']],[0.]*len(a.sites))
            self.assertTrue(all(x<1. for x in projected['theta'][1:]))

    def test_MB2_cache_position_is_time_not_batch(self):
        count=3;time=torch.arange(count)
        group=dict(rows=[dict(global_row=i) for i in range(count)],tokens=dict(input_ids=torch.ones(count,count)),
            cache=dict(key=torch.zeros(count,count,4),residual=torch.zeros(count,count,2),
                kwargs=dict(cache_position=time,attention_mask=torch.zeros(count,1,count,count))))
        split=MB2_entry(dict(groups=[group]))
        self.assertEqual([len(g['rows']) for g in split['groups']],[2,1])
        for g in split['groups']:
            self.assertTrue(torch.equal(g['cache']['kwargs']['cache_position'],time))
            self.assertEqual(g['cache']['kwargs']['attention_mask'].shape[0],len(g['rows']))

    def test_fit_25_24_no_terminal_backward_and_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            a,e,H=fixture(Path(folder),2);before=state(a,H)
            plan=fit(a,e,dict(arm='MAIN',n_exp=4,base_multiplier=1.))
            self.assertEqual(plan['receipt']['candidates'],25);self.assertEqual(plan['receipt']['updates'],24)
            self.assertEqual(plan['receipt']['logical_subject_backwards'],24)
            self.assertEqual(plan['built']['candidate'],24)
            self.assertIsNone(plan['observed']['adjoint']);self.assertEqual(state(a,H),before)
            diag=terminal(a,e,plan['R'],plan['built'],plan['observed'])
            self.assertEqual(diag['terminal_extra_planner_backward'],0)
            self.assertEqual(set(diag['layers']),set(map(str,a.sites)))
            # Exact RAM copy and final rewrite-only keys; transaction rollback
            # proves no durable model state is required by this engine fixture.
            with Transaction(a,H) as tx:
                from .run import commit_measure
                receipt=commit_measure(a,e,H,plan,Path(folder)/'writer')
                self.assertEqual(receipt['history_appends'],len(a.sites))
                self.assertTrue(receipt['no_resolve'] and receipt['no_double_add'])
                for l,w in a.weights.items():
                    self.assertTrue(torch.equal(w,plan['built']['weights'][l]))
                    self.assertEqual(receipt['history'][str(l)]['appends'],1)
            self.assertTrue(tx.rollback_verified);self.assertEqual(state(a,H),before)

    def test_comparison_nonfinite_is_not_pass(self):
        for number in (float('nan'),float('inf'),-float('inf')):
            for side in (0,1):
                values=[torch.tensor([0.]),torch.tensor([0.])];values[side][0]=number
                with self.assertRaisesRegex(RuntimeError,'QUALIFICATION_NONFINITE'):comparison(*values)

    def test_production_no_reverse_or_checkpoint_calls(self):
        root=Path(__file__).parent
        for name in ('engine.py','subject.py','entry.py','fit.py','telemetry.py'):
            calls=[ast.unparse(node.func) for node in ast.walk(ast.parse((root/name).read_text())) if isinstance(node,ast.Call)]
            self.assertNotIn('diagnostic_reverse',calls);self.assertNotIn('torch.save',calls)
            self.assertNotIn('cached_vjp',calls)


if __name__=='__main__':unittest.main()
