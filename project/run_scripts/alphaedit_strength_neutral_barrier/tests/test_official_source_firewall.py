from __future__ import annotations

import os
import unittest
from pathlib import Path

from project.run_scripts.alphaedit_strength_neutral_barrier.firewall import (
    verify_stock_easyedit,
)


class OfficialSourceFirewallTests(unittest.TestCase):
    def test_stock_easyedit_identity_and_hook_boundary(self) -> None:
        easyedit_root = Path(
            os.environ.get(
                "EASYEDIT_STOCK_ROOT",
                "/mnt/raid5/janghj/.codex/worktrees/easyeditsh1-official-readonly-v1",
            )
        )
        package = Path(__file__).resolve().parents[1]
        receipt = verify_stock_easyedit(
            easyedit_root, package / "official-source-lock.json"
        )
        self.assertTrue(receipt["tracked_clean"])
        self.assertEqual(
            receipt["implementation_boundary"],
            "ODE_EDIT_HOOK_STOCK_EASYEDIT_READ_ONLY",
        )
        self.assertEqual(len(receipt["members"]), 4)


if __name__ == "__main__":
    unittest.main()
