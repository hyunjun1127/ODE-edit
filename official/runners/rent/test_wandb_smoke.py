"""CPU only: HEAD resolution without git; no SDK, auth or network."""
from pathlib import Path
import tempfile
import unittest

from .wandb_smoke import git_head

SHA = "a" * 40


class GitHead(unittest.TestCase):
    def repo(self, head, files=None):
        root = Path(tempfile.mkdtemp())
        (root / ".git/refs/heads").mkdir(parents=True)
        (root / ".git/HEAD").write_text(head + "\n")
        for name, text in (files or {}).items():
            (root / ".git" / name).write_text(text)
        return root

    def test_loose_packed_and_detached_heads(self):
        self.assertEqual(git_head(self.repo("ref: refs/heads/main", {"refs/heads/main": SHA + "\n"})), SHA)
        packed = "# pack-refs with: peeled\n%s refs/heads/main\n" % SHA
        self.assertEqual(git_head(self.repo("ref: refs/heads/main", {"packed-refs": packed})), SHA)
        self.assertEqual(git_head(self.repo(SHA)), SHA)

    def test_unknown_ref_fails(self):
        with self.assertRaisesRegex(ValueError, "SOURCE_SHA_UNRESOLVED"):
            git_head(self.repo("ref: refs/heads/main", {"packed-refs": SHA + " refs/heads/other\n"}))


if __name__ == "__main__":
    unittest.main()
