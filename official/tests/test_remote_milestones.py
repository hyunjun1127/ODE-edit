"""Real native serialization, two-phase publication and interrupted-ACK recovery."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from official.experiments import checkpoint as cp, remote_milestones as remote


class RemoteMilestoneTests(unittest.TestCase):
    def command(self,archive,spec):
        return [sys.executable,'-c',remote.RECEIVER,json.dumps(spec)]

    def fetch(self,archive,ref,folder):
        shutil.copy2(Path(archive['root'])/'checkpoint'/ref['file'],folder/ref['file'])

    def args(self,root):
        return dict(archive=dict(host='test',python=sys.executable,root=str(root/'remote'),reserve_bytes=0),
            lock_path=root/'io.lock',weights={'W':torch.ones(2,3)},cache_c={'4':torch.eye(3)},
            contexts=[['{}']],evaluation_cursor={},identity={k:'bound-'+k for k in cp.IDENTITY_FIELDS},
            method='CAKE',evaluation_complete=True)

    def test_native_roundtrip_four_milestones_no_local_payload_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);local=root/'local';args=self.args(root)
            with patch.object(remote,'_command',self.command),patch.object(remote,'_fetch',self.fetch):
                direct=cp.save(root/'native',batch=0,interval=5,**{k:v for k,v in args.items() if k not in ('archive','lock_path')})
                remote.save(local,batch=0,**args)
                self.assertEqual(json.loads((local/'checkpoint/latest.json').read_text()),direct)
                for b in [5,10,15,20]:
                    args['weights']['W'].add_(b);args['cache_c']['4'].add_(b)
                    remote.save(local,batch=b,**args)
                    loaded=cp.load(root/'remote/milestones'/f'edit-{b*100:04d}',args['identity'])
                    self.assertTrue(torch.equal(loaded['weights']['W'],args['weights']['W']))
                    self.assertTrue(torch.equal(loaded['cache_c']['4'],args['cache_c']['4']))
                    self.assertTrue(torch.equal(loaded['rng']['torch_cpu'],torch.get_rng_state()))
                    self.assertEqual(list(local.rglob('*.pt')),[])
                loaded=remote.load(local,args['identity'],args['archive'])
                self.assertEqual(loaded['batch'],20)
                self.assertEqual(list(local.rglob('*.pt')),[])
            self.assertEqual(len(list((root/'remote/milestones').glob('edit-*'))),5)
            self.assertEqual(cp.load(root/'remote/checkpoint',args['identity'])['batch'],20)

    def test_corrupt_transfer_keeps_previous_and_can_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);local=root/'local';args=self.args(root)
            with patch.object(remote,'_command',self.command):
                remote.save(local,batch=0,**args);remote.save(local,batch=5,**args)
                original=remote._stream
                def corrupt(a,spec,payload):
                    result=original(a,spec,payload)
                    p=Path(a['root'])/spec['stage']/'payload.pt'
                    data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
                    return result
                with patch.object(remote,'_stream',corrupt):
                    with self.assertRaisesRegex(RuntimeError,'ARCHIVE_PAYLOAD_SHA_MISMATCH'):
                        remote.save(local,batch=10,**args)
                self.assertEqual(cp.load(root/'remote/checkpoint',args['identity'])['batch'],5)
                self.assertFalse((root/'remote/milestones/edit-1000').exists())
                self.assertFalse((local/'milestones/edit-1000/latest.json').exists())
                remote.save(local,batch=10,**args)
                self.assertEqual(cp.load(root/'remote/checkpoint',args['identity'])['batch'],10)

    def test_lost_ack_recovers_remote_commit_and_rejects_wrong_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);local=root/'local';args=self.args(root)
            with patch.object(remote,'_command',self.command),patch.object(remote,'_fetch',self.fetch):
                remote.save(local,batch=0,**args)
                with patch.object(remote,'_mirror',side_effect=OSError('lost ACK')):
                    with self.assertRaisesRegex(OSError,'lost ACK'):remote.save(local,batch=5,**args)
                self.assertEqual(json.loads((local/'checkpoint/latest.json').read_text())['batch'],0)
                # Also exercise crash after milestone publication, before rolling pointer.
                (root/'remote/checkpoint/latest.json').write_text((local/'checkpoint/latest.json').read_text())
                restored=remote.load(local,args['identity'],args['archive'])
                self.assertEqual(restored['batch'],5)
                self.assertEqual(json.loads((local/'checkpoint/latest.json').read_text())['batch'],5)
                with self.assertRaisesRegex(RuntimeError,'ARCHIVE_IDENTITY_MISMATCH'):
                    remote.load(local,dict(args['identity'],code_commit='wrong'),args['archive'])
                remote.save(local,batch=10,**args)


if __name__=='__main__':unittest.main()
