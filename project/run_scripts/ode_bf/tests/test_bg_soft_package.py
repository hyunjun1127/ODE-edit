from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts import (
    session05_ode_bf_bg_soft_missing_cell_package as package,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError


class BgSoftPackageTests(unittest.TestCase):
    def test_raw_free_gate_rejects_raw_private_and_absolute_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            safe = root / "safe.json"
            safe.write_text(
                json.dumps(
                    {
                        "request_sha256": "a" * 64,
                        "target_new_nll": 1.25,
                        "relative_receipt": "raw/stepwise/item.json",
                    }
                ),
                encoding="utf-8",
            )
            package.validate_raw_free_json(safe)
            for key, value in (
                ("subject", "hidden"),
                ("token_ids", [1, 2]),
                ("credential", "hidden"),
                ("safe_key", "/absolute/private/path"),
            ):
                malformed = root / f"malformed-{key}.json"
                malformed.write_text(
                    json.dumps({key: value}), encoding="utf-8"
                )
                with self.assertRaises(ODEBFContractError):
                    package.validate_raw_free_json(malformed)

    def test_frozen_reference_inventory_is_raw_free_and_minimal(self) -> None:
        rows = package.reference_inventory()
        names = [item[0] for item in rows]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(any("r10-report" in item for item in names))
        for alias in package.ALIASES:
            self.assertTrue(
                any(item.endswith(f"{alias}/terminal.json") for item in names)
            )
            self.assertTrue(
                any(
                    f"{alias}/raw/fixed-e8/BG-NEUTRAL/field-0000.json"
                    in item
                    for item in names
                )
            )
        self.assertFalse(any("RS-NEUTRAL" in item for item in names))
        self.assertFalse(any("RS-SOFT" in item for item in names))
        self.assertTrue(all(not path.is_symlink() for _, path, _ in rows))

    def test_bundle_header_requires_exact_child_and_r10_parent(self) -> None:
        child = "1" * 40
        parent = package.BG_SOFT_PARENT_HEAD
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.bundle"
            path.write_bytes(
                (
                    "# v2 git bundle\n"
                    f"-{parent} prerequisite\n"
                    f"{child} HEAD\n\n"
                ).encode("utf-8")
                + b"PACK"
            )
            value = package._parse_bundle_header(path)
            self.assertEqual(value["prerequisites"], [parent])
            self.assertEqual(
                value["heads"], [{"commit": child, "name": "HEAD"}]
            )

    def test_entry_manifest_and_archive_are_sorted_hash_exact_and_no_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            inventory = (
                ("z/second", second, "TEST"),
                ("a/first", first, "TEST"),
            )
            entries, digest = package._entry_manifest(inventory)
            self.assertEqual(
                [item["path"] for item in entries],
                [
                    package.PACKAGE_ID + "/a/first",
                    package.PACKAGE_ID + "/z/second",
                ],
            )
            self.assertEqual(len(digest), 64)
            archive = root / "package.tar"
            package._write_deterministic_tar(archive, inventory)
            package._verify_deterministic_tar(archive, entries)

    def test_packager_has_no_transport_or_model_action(self) -> None:
        source = inspect.getsource(package)
        for forbidden in (
            "rsync",
            "scp",
            "ssh",
            "sbatch",
            "srun",
            "transformers",
            "from_pretrained",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('"transport_executed_by_packager": False', source)
        self.assertIn('"transfer_executed": False', source)
        self.assertIn('"^" + BG_SOFT_PARENT_HEAD', source)


if __name__ == "__main__":
    unittest.main()
