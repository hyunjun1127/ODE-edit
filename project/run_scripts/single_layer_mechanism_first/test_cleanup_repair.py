"""CPU regression for the actual finally bug and completed-component reuse."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import torch
from . import technical_repair as repair
from .reuse_completed_hook import assert_cleanup_only_change
from .test_repair import comparison
from .test_program_integration import ProgramHarness


class CleanupRepairTests(unittest.TestCase):
    def run_hook(self,*,actual_write_bad=False):
        native={'weight':torch.zeros(2,3)};records=list(range(4));oracle=object()
        rt=SimpleNamespace(lock=dict(save_checkpoints=False,editor_sha256='CPU_MOCK',
            hook_repair={'original_execution':'MOCK_PRIOR'},
            hook_gate_policy=dict(id=repair.LENIENT_GATE_ID,user_quote='USER')),
            oracles=[],model=object(),tok=object(),module=object(),hp=object(),context=[],M=object(),P=object(),
            requests=lambda x:x,guard=Mock())
        def reset():rt.oracles=[]
        def protected(records):rt.oracles.append(oracle);return oracle,[],None,{}
        rt.reset=Mock(side_effect=reset);rt.protected_oracle=protected
        receipt={'batch1_comparison':comparison()}
        vectors={'cache_batch1_targets':torch.zeros(2,4),'cache_batched_targets':torch.zeros(2,4)}
        def score(oracle,w,rows):
            nll=.1+(.0002 if actual_write_bad and bool(w.sum()) else 0.)
            return {'r:new':dict(kind='canonical',branch='new',nll=nll,strict=True),
                    'r:old':dict(kind='canonical',branch='old',nll=.8,strict=False)}
        writes=[]
        with ExitStack() as stack:
            stack.enter_context(patch.object(repair,'validate_reference',return_value=(native,records,{'binding':{}})))
            stack.enter_context(patch.object(repair,'compare_native_z_paths',return_value=(receipt,vectors)))
            fitter=stack.enter_context(patch.object(repair,'_SavedTargetsFitter'))
            fitter.return_value.fit.return_value={'weight':torch.ones(2,3),'receipt':{}}
            stack.enter_context(patch.object(repair,'score_rows',side_effect=score))
            stack.enter_context(patch.object(repair,'save_tensor'))
            stack.enter_context(patch.object(repair,'write',side_effect=lambda p,v:writes.append((str(p),v))))
            if actual_write_bad:
                with self.assertRaisesRegex(RuntimeError,'T0_NATIVE_Z_HOOK_PARITY_FAILED'):
                    repair.run_hook_repair(rt,Path('/CPU_ONLY'))
            else:
                self.assertEqual(repair.run_hook_repair(rt,Path('/CPU_ONLY'))[1],records)
        self.assertEqual(rt.oracles,[])
        self.assertEqual(rt.reset.call_count,4)
        return writes

    def test_reset_clears_registry_no_double_unregister(self):
        writes=self.run_hook()
        self.assertTrue(writes[-1][1]['pass_'])

    def test_cleanup_does_not_mask_original_method_failure(self):
        writes=self.run_hook(actual_write_bad=True)
        self.assertFalse(writes[-1][1]['pass_'])

    def test_ast_proves_only_cleanup_changed(self):
        new=Path(repair.__file__).read_text();prefix,suffix=new.rsplit('rt.reset()',1)
        old=prefix+'rt.reset();rt.oracles.remove(oracle)'+suffix
        assert_cleanup_only_change(old,new)
        changed=new.replace("r['max_NLL_abs']<=NUMERIC['current_nll']","r['max_NLL_abs']<=.01")
        with self.assertRaisesRegex(ValueError,'NUMERICAL_CODE_CHANGED'):
            assert_cleanup_only_change(old,changed)

    def test_unknown_old_cleanup_cannot_reuse(self):
        new=Path(repair.__file__).read_text()
        with self.assertRaisesRegex(ValueError,'CLEANUP_DELTA_NOT_EXACT'):
            assert_cleanup_only_change(new,new)

    def test_reuse_route_does_not_refit_hook(self):
        h=ProgramHarness(b1=False);h.rt.lock['completed_hook_reuse']={'job_id':'51055'}
        with (patch('project.run_scripts.single_layer_mechanism_first.reuse_completed_hook.reuse_completed_hook',
                   return_value=({'weight':h.rt.W0},h.rt.records[:4])) as reused,
             patch.object(repair,'run_hook_repair',side_effect=AssertionError('DUPLICATE_HOOK_FIT'))):
            h.run()
        reused.assert_called_once();self.assertEqual(len(h.batch_calls),1)

    def test_reuse_component_does_not_bypass_remaining_T0(self):
        h=ProgramHarness(technical=False);h.rt.lock['completed_hook_reuse']={'job_id':'51055'}
        with patch('project.run_scripts.single_layer_mechanism_first.reuse_completed_hook.reuse_completed_hook',
                   return_value=({'weight':h.rt.W0},h.rt.records[:4])):
            with self.assertRaisesRegex(RuntimeError,'T0_FULL_CHECKS_NOT_ESTABLISHED'):h.run()
        self.assertEqual(h.batch_calls,[])


if __name__=='__main__':unittest.main()
