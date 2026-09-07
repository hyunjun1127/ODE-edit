import json,tempfile,unittest
from pathlib import Path
from .publish import verify,safe_copy
from .performance import hash_json
from ..reporting import sha


class PackageTests(unittest.TestCase):
    def test_member_rehash_fail_close(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);report=p/'factual-report-ko.md';report.write_text('fixture')
            members=[dict(path=report.name,sha256=sha(report),bytes=report.stat().st_size)]
            m={'members':members,'member_root':hash_json(members)}
            (p/'analysis-manifest.json').write_text(json.dumps(m))
            r=dict(manifest_sha256=sha(p/'analysis-manifest.json'),root_sha256=hash_json(m),member_root=m['member_root'],report_sha256=sha(report))
            (p/'rooted-receipt.json').write_text(json.dumps(r));self.assertTrue(verify(p))
            report.write_text('tampered')
            with self.assertRaises(AssertionError):verify(p)

    def test_copy_create_once(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'src').write_text('one');safe_copy(p/'src',p/'dst')
            with self.assertRaises(FileExistsError):safe_copy(p/'src',p/'dst')


if __name__=='__main__':unittest.main()
