"""CPU repair policy/control tests; no Slurm/model/GPU execution."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from .technical_repair import assess_hook, checked_tensor, LENIENT_GATE_ID
from .submit_program import inspection, other_capacity
from .z_hook_parity import _compare
from .z_hook import ZHookBoundary
from .test_program_integration import ProgramHarness


def comparison():
    rows=[dict(request_index=i,loss_steps_equal=True,adam_steps_equal=True,
        max_NLL_abs=2e-5,max_total_loss_abs=2e-5,max_gradient_relative=1.4033e-4,
        z_max_abs=1e-6,z_l2=2e-6,z_relative=1e-7) for i in range(4)]
    return dict(rows=rows,finite_checked=True,pass_inherited_NLL_gradient_and_stop_gate=False)


class RepairTests(unittest.TestCase):
    def policy(self):
        return dict(id=LENIENT_GATE_ID,user_quote='explicit USER leniency')

    def test_only_trajectory_ceiling_is_diagnostic(self):
        value=assess_hook(comparison(),{'pass_':True},self.policy())
        self.assertTrue(value['pass_'])
        self.assertFalse(value['historical_strict_gate_pass'])
        self.assertEqual(value['trajectory_gradient'],'WARN_DIAGNOSTIC_ONLY')
        self.assertFalse(value['method_acceptance_changed'])

    def test_no_silent_policy_inheritance(self):
        for policy in ({},{'id':LENIENT_GATE_ID},dict(id='wrong',user_quote='yes')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_USER'):
                assess_hook(comparison(),{'pass_':True},policy)

    def test_actual_write_failure_still_blocks(self):
        self.assertFalse(assess_hook(comparison(),{'pass_':False},self.policy())['pass_'])

    def test_native_nll_and_stops_still_block(self):
        for key,value in [('max_NLL_abs',1.001e-4),('loss_steps_equal',False),('adam_steps_equal',False)]:
            cmp=comparison();cmp['rows'][1][key]=value
            self.assertFalse(assess_hook(cmp,{'pass_':True},self.policy())['pass_'])

    def test_nonfinite_and_incomplete_are_not_warnings(self):
        for key in ('max_NLL_abs','max_gradient_relative','z_max_abs'):
            cmp=comparison();cmp['rows'][0][key]=float('nan')
            with self.assertRaisesRegex(ValueError,'NONFINITE'):
                assess_hook(cmp,{'pass_':True},self.policy())
        cmp=comparison();cmp['rows'].pop()
        with self.assertRaises(ValueError):assess_hook(cmp,{'pass_':True},self.policy())

    def test_finite_check_is_required_not_inferred(self):
        cmp=comparison();cmp.pop('finite_checked')
        with self.assertRaises(ValueError):assess_hook(cmp,{'pass_':True},self.policy())

    def test_new_compare_rejects_corrupt_trace(self):
        trace=[dict(losses=[dict(nll=.2,loss=.3)],gradients=[])]
        targets=torch.ones(2,1)
        for kind in ('target','loss','gradient'):
            bad=deepcopy(trace);z=targets.clone()
            if kind=='target':z[0,0]=float('nan')
            elif kind=='loss':bad[0]['losses'][0]['loss']=float('nan')
            else:
                trace[0]['gradients']=[torch.ones(2)]
                bad[0]['gradients']=[torch.tensor([float('nan'),0.])]
            with self.assertRaisesRegex(ZHookBoundary,'NONFINITE'):_compare(trace,bad,targets,z)

    def test_corrupt_retained_reference_prevents_load(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.pt';p.touch()
            with patch('torch.load',side_effect=AssertionError('must not load')):
                with self.assertRaisesRegex(ValueError,'ARTIFACT_CHANGED'):
                    checked_tensor(dict(path=str(p),bytes=0,sha256='0'*64))

    def test_inline_repair_runs_before_full_T0_no_old_dependency(self):
        h=ProgramHarness(b1=False);h.rt.lock['hook_repair']={'kind':'CPU_MOCK'}
        with patch('project.run_scripts.single_layer_mechanism_first.technical_repair.run_hook_repair',
                   return_value=({'weight':h.rt.W0.clone()},h.rt.records[:4])) as repaired:
            h.run()
        repaired.assert_called_once()
        self.assertEqual(len(h.batch_calls),1)
        self.assertFalse(h.rt.lock['save_checkpoints'])

    def test_inline_failure_does_not_enter_science(self):
        h=ProgramHarness();h.rt.lock['hook_repair']={'kind':'CPU_MOCK'}
        with patch('project.run_scripts.single_layer_mechanism_first.technical_repair.run_hook_repair',
                   side_effect=RuntimeError('HOOK_FAILED')):
            with self.assertRaisesRegex(RuntimeError,'HOOK_FAILED'):h.run()
        self.assertEqual(h.batch_calls,[])

    def test_repair_inspection_requires_no_old_failed_dependency(self):
        source=Path('/task/source');lock=Path('/task/lock.json')
        command=['sbatch',str(source/'project/run_scripts/single_layer_mechanism_first/run.sbatch'),str(source),str(lock)]
        text=('UserId=janghj(1001) JobName=odeedit_slmf_repair_s4 JobState=PENDING Reason=JobHeldUser '
              'TRES=cpu=8,mem=59G,gres/gpu=1 NumCPUs=8 ReqNodeList=server4 Requeue=0 '
              'WorkDir=/task/source Dependency=(null) Command='+' '.join(command[-3:])+' StdOut=/task/log')
        kwargs=dict(job_name='odeedit_slmf_repair_s4',dependency=None)
        self.assertTrue(all(inspection(text,command,source,lock,**kwargs).values()))
        self.assertFalse(inspection(text.replace('Dependency=(null)','Dependency=afterok:50974(failed)'),
                                   command,source,lock,**kwargs)['dependency'])

    def test_old_pending_conservatively_counted(self):
        rows=other_capacity('50983|old|PENDING|gpu:1',excluded_jobs=())
        self.assertEqual(sum(x['GPU'] for x in rows)+1,2)


if __name__=='__main__':unittest.main()
