import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location("review_package",Path(__file__).parents[1]/"completed_review_package.py")
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class PackageTests(unittest.TestCase):
    def test_canonical_order(self):
        self.assertEqual(p.canonical({"b":1,"a":2}),p.canonical({"a":2,"b":1}))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):p.canonical({"value":float("nan")})

    def test_create_once(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"receipt.json";p.save_once(f,{"ok":True})
            with self.assertRaises(FileExistsError):p.save_once(f,{"ok":False})
            self.assertEqual(json.loads(f.read_text()),{"ok":True})

    def test_parent_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"real").mkdir();p.save_once(root/"real/file.json",{})
            (root/"link").symlink_to(root/"real",target_is_directory=True)
            with self.assertRaises(ValueError):p.member(root/"link/file.json")

    def test_content_hash_detects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"x.json";p.save_once(f,{"a":1});before=p.member(f)
            f.write_bytes(b'{"a":2}\n')
            self.assertNotEqual(before["sha256"],p.member(f)["sha256"])


if __name__=="__main__":unittest.main()
