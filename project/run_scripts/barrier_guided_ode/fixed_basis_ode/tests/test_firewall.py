from __future__ import annotations

import ast
import unittest
from pathlib import Path


class FixedBasisFirewallTests(unittest.TestCase):
    def test_euler_has_no_physical_or_factor_builder_calls(self) -> None:
        root = Path(__file__).parents[1]
        tree = ast.parse((root / "euler_writer.py").read_text(encoding="utf-8"))
        names = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        self.assertTrue(names.isdisjoint({"apply", "propose_ordered", "propose_validated_ordered", "compute_z"}))

    def test_functional_observer_has_no_physical_writer(self) -> None:
        root = Path(__file__).parents[1]
        source = (root / "functional_observer.py").read_text(encoding="utf-8")
        self.assertNotIn("AtomicWeightTrajectory", source)
        self.assertNotIn("propose_ordered", source)
        self.assertNotIn("heldout", source)


if __name__ == "__main__":
    unittest.main()
