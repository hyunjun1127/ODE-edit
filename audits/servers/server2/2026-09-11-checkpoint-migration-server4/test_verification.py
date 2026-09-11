"""수신 검증기 최소 CPU fixtures. 실제 checkpoint 변경 없음."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from verify_initial import save,stable_verify
from seal_bundle import rename_noreplace

class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='s2-migration-test-')
        self.root=Path(self.tmp.name);self.p=self.root/'file'
        self.p.write_bytes(b'exact bytes\r\n')
        self.e=dict(bytes=self.p.stat().st_size,sha256=hashlib.sha256(self.p.read_bytes()).hexdigest())
    def tearDown(self):self.tmp.cleanup()
    def test_exact(self):self.assertEqual(stable_verify(self.p,self.e)['sha256'],self.e['sha256'])
    def test_wrong_sha(self):
        with self.assertRaises(AssertionError):stable_verify(self.p,dict(self.e,sha256='0'*64))
    def test_wrong_size(self):
        with self.assertRaises(AssertionError):stable_verify(self.p,dict(self.e,bytes=1))
    def test_symlink(self):
        q=self.root/'link';q.symlink_to(self.p)
        with self.assertRaises(AssertionError):stable_verify(q,self.e)
    def test_hardlink(self):
        os.link(self.p,self.root/'hardlink')
        with self.assertRaises(AssertionError):stable_verify(self.p,self.e)
    def test_create_once(self):
        p=self.root/'receipt';save(p,{'a':1})
        before=p.read_bytes()
        with self.assertRaises(FileExistsError):save(p,{'a':2})
        self.assertEqual(before,p.read_bytes())
    def test_atomic_no_replace(self):
        a=self.root/'a';b=self.root/'b';a.mkdir();b.mkdir()
        with self.assertRaises(OSError):rename_noreplace(a,b)
        self.assertTrue(a.is_dir());self.assertTrue(b.is_dir())
        c=self.root/'c';rename_noreplace(a,c)
        self.assertTrue(c.is_dir());self.assertFalse(a.exists())
    def test_mutation_during_hash(self):
        def changed(_):
            self.p.write_bytes(b'changed and different size')
            return self.e['sha256']
        with patch('verify_initial.sha',side_effect=changed):
            with self.assertRaises(AssertionError):stable_verify(self.p,self.e)

if __name__=='__main__':unittest.main()
