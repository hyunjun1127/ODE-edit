import json,tempfile,unittest
from pathlib import Path
from .common import canonical_hash,member
from .git_checkout_modes import restore

class GitModeTests(unittest.TestCase):
    def fixture(self,root):
        p=root/'x.txt';p.write_text('sealed scalar');p.chmod(0o644)
        rows=[member(p,relative_to=root)]
        (root/'analysis-manifest.json').write_text(json.dumps({'members':rows,'member_root':canonical_hash(rows)}))
        return p
    def test_mode_only_restore(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=self.fixture(r);before=p.read_bytes();p.chmod(0o664)
            self.assertEqual(len(restore(r)['changes']),1)
            self.assertEqual(len(restore(r,True)['changes']),1)
            self.assertEqual(p.read_bytes(),before);self.assertEqual(p.stat().st_mode&0o777,0o644)
            self.assertEqual(restore(r)['changes'],[])
    def test_content_change_fails_before_chmod(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=self.fixture(r);p.chmod(0o664);p.write_text('tampered')
            with self.assertRaises(RuntimeError):restore(r,True)
            self.assertEqual(p.stat().st_mode&0o777,0o664)
    def test_executable_mode_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=self.fixture(r);p.chmod(0o755)
            with self.assertRaises(RuntimeError):restore(r,True)

if __name__=='__main__':unittest.main()
