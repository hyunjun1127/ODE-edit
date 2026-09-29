"""CPU-only S2 routing/cancellation preconditions; never submits jobs."""
import unittest
from unittest.mock import patch
from . import server2_entry as entry
entry.bind()
from . import common
from .server2_launch import cancellation

class Server2(unittest.TestCase):
    def test_platform_binding(self):
        self.assertEqual(common.NONCE,entry.NONCE)
        for k in ('ROOT','DEPS','NATIVE','MODEL','DATA','PYTHON'):
            self.assertTrue(str(getattr(common,k)).startswith('/mnt/raid5/janghj/'))
    def test_handoff_exact(self):
        p=entry.ROOT/'inputs/server4-handoff/receiver-exact-allowlist-s2.json'
        self.assertEqual(common.sha(p),'dbc3180d680a5a534b19de2084147f280f217adca949f68bc774bcc03005ccf2')
    def test_actual_cancel_and_mapping(self):
        cfg=common.read(entry.ROOT/'preflight/configuration.json')
        self.assertEqual(cancellation(cfg)['sha256'],'2f934836622526bc5555ede5ea54e185323968a8f6f3cba507dd07f5c54f683d')
    def test_active_receipt_blocks_release(self):
        cfg=common.read(entry.ROOT/'preflight/configuration.json')
        original=common.read
        def changed(path):
            result=original(path)
            if str(path)==cfg['cancellation']['path']:result['jobs'][0]['state']='RUNNING'
            return result
        with patch('project.run_scripts.joint_multilayer_bs10.server2_launch.read',changed):
            with self.assertRaisesRegex(RuntimeError,'S4_ACTIVE'):cancellation(cfg)
    def test_scope_and_source_unchanged(self):
        cfg=common.read(entry.ROOT/'preflight/configuration.json');common.validate_execution(cfg)
        self.assertEqual(cfg['contract']['geometry']['max_rank_per_layer'],1)
        self.assertEqual(cfg['contract']['weight_snapshots']['total_tensor_bytes'],42278584320)
        from pathlib import Path
        source=Path(__file__).parent
        old=entry.ROOT/'inputs/server4-handoff/execution-code-reference'
        for f in ('runtime.py','observations.py','common.py'):
            self.assertEqual(common.sha(source/f),common.sha(old/f))

if __name__=='__main__':unittest.main()
