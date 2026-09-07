import tempfile
from pathlib import Path
import unittest
from .publication import write_once,seal_package,verify_package,member


class PublicationTests(unittest.TestCase):
    def test_create_once_and_full_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_once(root/'facts.json',{'status':'CPU_ONLY'},root=root)
            with self.assertRaises(FileExistsError):write_once(root/'facts.json',{},root=root)
            sealed=seal_package(root)
            self.assertEqual(sealed,verify_package(root))
            self.assertEqual(sealed['member_count'],1)

    def test_parent_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'outside').mkdir();(root/'link').symlink_to(root/'outside',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'SYMLINK'):
                write_once(root/'link'/'escape.json',{},root=root)
            self.assertFalse((root/'outside'/'escape.json').exists())


if __name__=='__main__':unittest.main()
