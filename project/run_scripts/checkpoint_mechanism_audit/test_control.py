"""No scheduler invocation: configuration and create-once fixtures only."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import control
from .common import write_json
from .model_gate import repeat_norm
from . import submit
import torch


class ControlTests(unittest.TestCase):
    def test_closed_runner_inventory(self):
        self.assertEqual(set(control.MODULES),{'gate','keys','operator','activation'})
        self.assertNotIn('edit',control.MODULES)

    def test_execution_checks_exact_module_and_preserves_argv(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'lock.json';out=str(Path(root)/'result dir')
            write_json(p,dict(mode='operator',runner_module='project.run_scripts.checkpoint_mechanism_audit.operator_lane',
                             expected_output=out,runner_args=['--histories','0,10','--gate','path with spaces']))
            with patch.object(control,'verify') as verify,patch.object(control.subprocess,'run') as run:
                control.execute(p);verify.assert_called_once_with(p)
                args=run.call_args.args[0]
                self.assertEqual(args[-1],'path with spaces');self.assertIn(out,args)
                self.assertEqual(run.call_args.kwargs,{'check':True})

    def test_write_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'x.json';write_json(p,{'a':1})
            with self.assertRaises(FileExistsError):write_json(p,{'a':2})

    def test_repeat_zero_reference_absolute_threshold(self):
        ref=torch.zeros(2,3,dtype=torch.float64)
        self.assertTrue(repeat_norm(torch.full_like(ref,1e-9),ref,1e-4)['passed'])
        self.assertFalse(repeat_norm(torch.full_like(ref,1e-8),ref,1e-4)['passed'])

    def test_pause_guard_precedes_scheduler_access(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            write_json(root/'receipts/user-implementation-only-20260920.json',{'status':'PAUSED'})
            with patch.object(submit,'ATTEMPT',root),patch.object(submit.subprocess,'run') as scheduler:
                with self.assertRaisesRegex(RuntimeError,'USER_RECALL_REQUIRED'):
                    submit.submit('/not-read')
                scheduler.assert_not_called()


if __name__=='__main__':unittest.main()
