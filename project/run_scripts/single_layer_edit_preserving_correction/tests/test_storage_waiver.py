"""CPU routing/IO error propagation only; no model and no real ENOSPC fill."""
import errno
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from project.run_scripts.single_layer_edit_preserving_correction.common import write
from project.run_scripts.single_layer_edit_preserving_correction.storage_waiver import admission,verify,STATUS,NONCE

VALUE=dict(nonce=NONCE,reserve_bytes_original=72*(1<<30),reserve_is_submission_block=False,
    storage_status=STATUS,space_cleanup_owner='USER',SH4_delete_move_authorized=False,
    actual_IO_error_detection=True,atomic_endpoint_integrity=True,M_final_endpoints_retained=80,
    M_episodes=10,GPU_cap=2,T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',
    M_waits_for_T=False,S_R_L=False)


class StorageWaiver(unittest.TestCase):
    def test_low_free_admission_no_write_delete_or_rename(self):
        with tempfile.TemporaryDirectory() as d:
            ref=write(Path(d)/'waiver.json',VALUE)
            with patch.object(Path,'unlink',side_effect=AssertionError('delete forbidden')), \
                 patch.object(Path,'rename',side_effect=AssertionError('move forbidden')):
                for free in (0,1,71*(1<<30)):
                    result=admission(free,72*(1<<30),ref)
                    self.assertEqual(result['status'],STATUS)
                    self.assertEqual(result['user_cleanup'],'PLANNED_NOT_VERIFIED')
            with self.assertRaises(ValueError):admission(0,72*(1<<30))

    def test_guards_endpoints_and_original_estimate_unchanged(self):
        for k,v in [('M_final_endpoints_retained',0),('actual_IO_error_detection',False),
                    ('SH4_delete_move_authorized',True),('reserve_bytes_original',1)]:
            with self.assertRaises(ValueError):verify(dict(VALUE,**{k:v}))

    def test_actual_create_once_error_propagates(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(Path,'open',side_effect=OSError(errno.ENOSPC,'fixture disk full')):
                with self.assertRaises(OSError) as ctx:write(Path(d)/'receipt.json',{})
            self.assertEqual(ctx.exception.errno,errno.ENOSPC)


if __name__=='__main__':unittest.main()
