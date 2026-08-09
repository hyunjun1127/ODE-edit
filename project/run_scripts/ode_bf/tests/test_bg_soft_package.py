from __future__ import annotations

import inspect
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from project.run_scripts import (
    session05_ode_bf_bg_soft_missing_cell_package as package,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError


class BgSoftPackageTests(unittest.TestCase):
    def test_r1_identity_and_source_manifest_object_contract(self) -> None:
        self.assertEqual(
            package.PACKAGE_ID, "BGSOFT_R10_FROZEN_BUNDLE_A1_R1"
        )
        path = (
            package.REPO_ROOT
            / "project/run_scripts/ode_bf/locks/"
            / package.BG_SOFT_SOURCE_MANIFEST_FILE
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["frozen_bundle_id"], package.PACKAGE_ID)
        self.assertEqual(
            value["package_repair_instruction_id"],
            package.PACKAGE_REPAIR_INSTRUCTION_ID,
        )
        self.assertEqual(value["entry_count"], len(value["entries"]))
        for entry in value["entries"]:
            self.assertIsInstance(entry["mode"], int)
            self.assertEqual(len(entry["object_id"]), 40)
            self.assertTrue(entry["role"])

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

    def test_case_id_exception_scanner_allows_ids_and_hash_only_metadata(self) -> None:
        value = {
            "case_id": 17,
            "case_ids": ["18", 19],
            "target_span_sha256": "a" * 64,
            "direct_z_tensor": "b" * 64,
            "status": "SEALED",
        }
        receipt = package._case_id_content_scan(value)
        self.assertEqual(receipt["case_id_value_count"], 3)
        self.assertTrue(receipt["content_scanner_pass"])
        self.assertEqual(receipt["prompt_or_target_content_match_count"], 0)
        self.assertEqual(receipt["tensor_payload_match_count"], 0)

    def test_case_id_exception_scanner_rejects_non_id_leakage(self) -> None:
        malformed = (
            {"case_id": -1},
            {"case_id": "not-decimal"},
            {"case_id": 1, "prompt": "raw"},
            {"case_id": 1, "target_new": "raw"},
            {"case_id": 1, "direct_z_tensor": [1.0, 2.0]},
            {"case_id": 1, "credential": "secret"},
        )
        for value in malformed:
            with self.subTest(value_keys=sorted(value)):
                with self.assertRaises(ODEBFContractError):
                    package._case_id_content_scan(value)

    def test_reachable_source_scan_inventories_committed_case_ids_only(self) -> None:
        source_head = package._run(["git", "rev-parse", "HEAD"]).stdout.strip()
        receipt = package._source_bundle_content_scan(source_head)
        self.assertGreater(
            receipt["tracked_case_id_only_exception_count"], 0
        )
        self.assertEqual(receipt["forbidden_path_match_count"], 0)
        self.assertEqual(
            receipt["raw_prompt_or_target_content_match_count"], 0
        )
        self.assertEqual(receipt["tensor_payload_match_count"], 0)
        self.assertEqual(
            receipt["credential_private_config_runtime_log_match_count"], 0
        )
        for item in receipt["tracked_case_id_only_exception"]:
            self.assertEqual(len(item["owning_commit"]), 40)
            self.assertEqual(len(item["owning_tree"]), 40)
            self.assertEqual(len(item["blob_object_id"]), 40)
            self.assertEqual(len(item["sha256"]), 64)
            self.assertTrue(
                item["field_name_classification"]["content_scanner_pass"]
            )

    def test_frozen_reference_inventory_is_raw_free_and_complete(self) -> None:
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
        self.assertTrue(any("RS-NEUTRAL" in item for item in names))
        self.assertTrue(any("RS-SOFT" in item for item in names))
        self.assertTrue(all(not path.is_symlink() for _, path, _ in rows))
        closure = package.reference_closure_receipt()
        self.assertTrue(closure["all_arm_path_counts_nonzero"])
        self.assertTrue(closure["link_rehash_pass"])
        for alias in package.ALIASES:
            for arm in package.REFERENCE_ARMS:
                receipt = closure["aliases"][alias]["arms"][arm]
                self.assertGreater(receipt["path_count"], 0)
                self.assertEqual(
                    receipt["accepted_prefix_count"],
                    receipt["accepted_to_transition_link_rehash_count"],
                )

    def test_bundle_header_requires_self_contained_advertised_ref(self) -> None:
        child = "1" * 40
        advertised = "refs/heads/" + package.EXECUTION_BRANCH
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.bundle"
            path.write_bytes(
                (
                    "# v2 git bundle\n"
                    f"{child} {advertised}\n\n"
                ).encode("utf-8")
                + b"PACK"
            )
            value = package._parse_bundle_header(path)
            self.assertEqual(value["prerequisites"], [])
            self.assertEqual(
                value["heads"], [{"commit": child, "name": advertised}]
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
            source_head = package._run(
                ["git", "rev-parse", "HEAD"]
            ).stdout.strip()
            entries, digest = package._entry_manifest(
                inventory, source_head=source_head
            )
            self.assertEqual(
                [item["path"] for item in entries],
                [
                    package.PACKAGE_ID + "/a/first",
                    package.PACKAGE_ID + "/z/second",
                ],
            )
            self.assertEqual(len(digest), 64)
            for entry in entries:
                self.assertEqual(entry["mode"], 0o600)
                self.assertEqual(entry["archive_kind"], "REGULAR_FILE")
                self.assertEqual(entry["object_algorithm"], "GIT_BLOB_SHA1")
                self.assertEqual(len(entry["object_id"]), 40)
                self.assertFalse(entry["committed_membership"])
                self.assertEqual(
                    entry["uncommitted_content_class"],
                    "GENERATED_PACKAGE_MEMBER",
                )
            for missing in ("mode", "object_id", "role"):
                malformed = dict(entries[0])
                malformed.pop(missing)
                with self.assertRaisesRegex(
                    ODEBFContractError, "manifest entry"
                ):
                    package._validate_manifest_entries([malformed])
            malformed = dict(entries[0])
            malformed.pop("uncommitted_content_class")
            with self.assertRaisesRegex(
                ODEBFContractError, "uncommitted manifest"
            ):
                package._validate_manifest_entries([malformed])
            malformed = dict(entries[0])
            malformed["committed_membership"] = True
            malformed.pop("uncommitted_content_class")
            with self.assertRaisesRegex(
                ODEBFContractError, "committed manifest"
            ):
                package._validate_manifest_entries([malformed])
            malformed = dict(entries[0])
            malformed["path"] = package.PACKAGE_ID + "/a//first"
            with self.assertRaisesRegex(
                ODEBFContractError, "manifest entry"
            ):
                package._validate_manifest_entries([malformed])
            malformed = dict(entries[0])
            malformed["path"] = "WRONG_PACKAGE/a/first"
            with self.assertRaisesRegex(
                ODEBFContractError, "manifest entry"
            ):
                package._validate_manifest_entries([malformed])
            with self.assertRaisesRegex(ODEBFContractError, "path repeats"):
                package._validate_manifest_entries([entries[0], entries[0]])
            archive = root / "package.tar"
            package._write_deterministic_tar(archive, inventory)
            package._verify_deterministic_tar(archive, entries)

    def test_tracked_source_metadata_binds_ls_tree_mode_object_and_tree(self) -> None:
        relative = "project/run_scripts/ode_bf/contracts.py"
        source_head = package._run(["git", "rev-parse", "HEAD"]).stdout.strip()
        source_tree = package._run(
            ["git", "rev-parse", source_head + "^{tree}"]
        ).stdout.strip()
        row = package._run(
            ["git", "ls-tree", source_head, "--", relative]
        ).stdout.rstrip("\n")
        object_id = row.split("\t", 1)[0].split(" ", 2)[2]
        value = package._tracked_source_metadata(
            package.REPO_ROOT / relative,
            source_head,
            source_tree,
            object_id,
        )
        self.assertTrue(value["committed_membership"])
        self.assertEqual(value["committed_path"], relative)
        self.assertEqual(value["committed_object_id"], object_id)
        with self.assertRaisesRegex(ODEBFContractError, "source object"):
            package._tracked_source_metadata(
                package.REPO_ROOT / relative,
                source_head,
                source_tree,
                "0" * 40,
            )

    def test_reference_snapshot_is_private_hash_exact_and_reclosed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root = Path(temporary) / "snapshot"
            snapshot_root.mkdir(mode=0o700)
            external_rows = package.reference_inventory()
            expected = {
                relative: (role, path.stat().st_size, package._sha256_file(path))
                for relative, path, role in external_rows
            }
            rows, closure = package._snapshot_reference_inventory(snapshot_root)
            self.assertEqual(closure, package.reference_closure_receipt())
            self.assertTrue(closure["all_arm_path_counts_nonzero"])
            self.assertTrue(closure["link_rehash_pass"])
            self.assertEqual(
                [relative for relative, _path, _role in rows],
                sorted(expected),
            )
            for relative, path, role in rows:
                self.assertTrue(path.is_relative_to(snapshot_root))
                self.assertFalse(path.is_symlink())
                self.assertTrue(path.is_file())
                self.assertEqual(
                    (role, path.stat().st_size, package._sha256_file(path)),
                    expected[relative],
                )

    def test_nofollow_reader_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_bytes(b"safe")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(
                ODEBFContractError, "stable regular file"
            ):
                package._read_regular_file_nofollow(link)

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
        self.assertNotIn('"^" + BG_SOFT_PARENT_HEAD', source)
        self.assertIn('"complete_history": True', source)
        self.assertIn("empty_bare_repository_bundle_verify", source)

    def test_bundle_uses_advertised_head_and_exact_repair_chain(self) -> None:
        create_source = inspect.getsource(package._create_complete_bundle)
        validate_source = inspect.getsource(package._validate_source_head)
        self.assertIn("EXECUTION_BRANCH", create_source)
        self.assertNotIn('"^" + BG_SOFT_PARENT_HEAD', create_source)
        self.assertIn("_verify_self_contained_bundle", create_source)
        self.assertIn("BG_SOFT_PACKAGE_REPAIR_PARENT", validate_source)
        self.assertIn("BG_SOFT_EXECUTION_REPAIR_PARENT", validate_source)
        self.assertIn("BG_SOFT_IMPLEMENTATION_PARENT", validate_source)
        self.assertIn('["git", "rev-parse", "HEAD^^^^"]', validate_source)
        self.assertIn("BG_SOFT_PARENT_HEAD", validate_source)

        child = "1" * 40
        outputs = {
            ("git", "rev-parse", "HEAD"): child + "\n",
            ("git", "rev-parse", "HEAD^"): (
                package.BG_SOFT_PACKAGE_REPAIR_PARENT + "\n"
            ),
            ("git", "rev-parse", "HEAD^^"): (
                package.BG_SOFT_EXECUTION_REPAIR_PARENT + "\n"
            ),
            ("git", "rev-parse", "HEAD^^^"): (
                package.BG_SOFT_IMPLEMENTATION_PARENT + "\n"
            ),
            ("git", "rev-parse", "HEAD^^^^"): package.BG_SOFT_PARENT_HEAD + "\n",
            ("git", "branch", "--show-current"): package.EXECUTION_BRANCH + "\n",
            (
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ): "",
        }

        def run(args: list[str], *, check: bool = True) -> SimpleNamespace:
            del check
            return SimpleNamespace(stdout=outputs[tuple(args)])

        with mock.patch.object(package, "_run", side_effect=run):
            package._validate_source_head(child)
        outputs[("git", "rev-parse", "HEAD^")] = package.BG_SOFT_PARENT_HEAD + "\n"
        with mock.patch.object(package, "_run", side_effect=run):
            with self.assertRaisesRegex(ODEBFContractError, "provenance"):
                package._validate_source_head(child)

    def test_submit_and_sbatch_fail_closed_on_exact_child_chain(self) -> None:
        from project.run_scripts import (
            session05_ode_bf_submit_bg_soft_missing_cell as submit,
        )

        provenance_source = inspect.getsource(submit._execution_provenance_gate)
        self.assertIn("BG_SOFT_PACKAGE_REPAIR_PARENT", provenance_source)
        self.assertIn("BG_SOFT_EXECUTION_REPAIR_PARENT", provenance_source)
        self.assertIn("BG_SOFT_IMPLEMENTATION_PARENT", provenance_source)
        self.assertIn('["git", "rev-parse", "HEAD^^^^"]', provenance_source)
        sbatch = (
            package.REPO_ROOT
            / "project/run_scripts/session05_ode_bf_bg_soft_missing_cell.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'EXPECTED_PACKAGE_REPAIR_PARENT="'
            + package.BG_SOFT_PACKAGE_REPAIR_PARENT
            + '"',
            sbatch,
        )
        self.assertIn(
            'EXPECTED_EXECUTION_REPAIR_PARENT="'
            + package.BG_SOFT_EXECUTION_REPAIR_PARENT
            + '"',
            sbatch,
        )
        self.assertIn('git rev-parse HEAD^^', sbatch)
        self.assertIn('git rev-parse HEAD^^^', sbatch)
        self.assertIn('git rev-parse HEAD^^^^', sbatch)
        self.assertIn('EXPECTED_IMPLEMENTATION_PARENT', sbatch)
        self.assertIn('EXPECTED_SCIENTIFIC_PARENT', sbatch)

    def test_empty_bare_materialization_uses_imported_repository(self) -> None:
        source = inspect.getsource(package._verify_self_contained_bundle)
        self.assertIn('str(bare), str(checkout)', source)
        self.assertNotIn('str(path), str(checkout)', source)
        self.assertIn('"fsck", "--full"', source)
        self.assertIn('"checkout", "--detach", source_head', source)

    def test_operational_bundle_empty_bare_proof_and_thin_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "source"
            subprocess.run(
                ["git", "init", "-b", "exec", str(repository)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            commits: list[str] = []
            for ordinal in range(5):
                (repository / "source.txt").write_text(
                    f"{ordinal}\n", encoding="utf-8"
                )
                subprocess.run(
                    ["git", "-C", str(repository), "add", "source.txt"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repository),
                        "-c",
                        "user.name=ODEEdit Test",
                        "-c",
                        "user.email=odeedit-test@example.invalid",
                        "commit",
                        "-m",
                        f"commit-{ordinal}",
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                commits.insert(
                    0,
                    subprocess.run(
                        ["git", "-C", str(repository), "rev-parse", "HEAD"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    ).stdout.strip(),
                )
            complete = root / "complete.bundle"
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "bundle",
                    "create",
                    str(complete),
                    "refs/heads/exec",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            original_run = package._run

            def run(
                args: list[str] | tuple[str, ...], *, check: bool = True
            ) -> subprocess.CompletedProcess[str]:
                if list(args[:2]) == ["git", "rev-parse"]:
                    return subprocess.run(
                        ["git", "-C", str(repository), *list(args[1:])],
                        check=check,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                return original_run(args, check=check)

            with (
                mock.patch.object(package, "_run", side_effect=run),
                mock.patch.object(
                    package, "BG_SOFT_PACKAGE_REPAIR_PARENT", commits[1]
                ),
                mock.patch.object(
                    package, "BG_SOFT_EXECUTION_REPAIR_PARENT", commits[2]
                ),
                mock.patch.object(
                    package, "BG_SOFT_IMPLEMENTATION_PARENT", commits[3]
                ),
                mock.patch.object(package, "BG_SOFT_PARENT_HEAD", commits[4]),
            ):
                proof = package._verify_self_contained_bundle(
                    complete, commits[0], "refs/heads/exec"
                )
            self.assertTrue(proof["empty_bare_repository_bundle_verify"])
            self.assertTrue(proof["empty_bare_repository_fsck_full"])
            self.assertTrue(proof["materialized_checkout_clean"])
            self.assertEqual(proof["exact_lineage"], commits)

            thin = root / "thin.bundle"
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "bundle",
                    "create",
                    str(thin),
                    "refs/heads/exec",
                    "^" + commits[4],
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertTrue(package._parse_bundle_header(thin)["prerequisites"])

    def test_lock_hash_index_excludes_raw_seals_but_binds_their_hashes(self) -> None:
        self.assertIn(
            "project/run_scripts/ode_bf/locks/p1r10_common_coldcoord_cf_b10_seal.json",
            package.LOCK_RELATIVES,
        )
        self.assertNotIn(
            "project/run_scripts/ode_bf/locks/p1r10_common_coldcoord_cf_b10_seal.json",
            package.RAW_FREE_LOCK_RELATIVES,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / package.LOCK_HASH_INDEX
            def indexed(args: list[str], *, check: bool = True) -> SimpleNamespace:
                del check
                relative = args[-1]
                source = package.REPO_ROOT / relative
                object_id = package._git_blob_oid(source.read_bytes())
                return SimpleNamespace(
                    stdout=f"100644 {object_id} 0\t{relative}\n"
                )

            with mock.patch.object(package, "_run", side_effect=indexed):
                value = package._create_lock_hash_index(path)
            package.validate_raw_free_json(path)
            self.assertEqual(value["entry_count"], len(package.LOCK_RELATIVES))
            self.assertEqual(value["raw_id_or_content_count"], 0)
            indexed = {item["path"]: item for item in value["entries"]}
            seal = indexed[
                "project/run_scripts/ode_bf/locks/"
                "p1r10_common_coldcoord_cf_b10_seal.json"
            ]
            self.assertFalse(seal["content_included"])
            self.assertEqual(len(seal["sha256"]), 64)
            self.assertIsInstance(seal["mode"], int)
            self.assertEqual(len(seal["object_id"]), 40)
            self.assertEqual(
                seal["role"], "HASH_ONLY_RAW_OR_PRIVATE_LOCK"
            )

    def test_lock_index_rejects_index_blob_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / package.LOCK_HASH_INDEX

            def mismatched(
                args: list[str], *, check: bool = True
            ) -> SimpleNamespace:
                del check
                relative = args[-1]
                return SimpleNamespace(
                    stdout=f"100644 {'0' * 40} 0\t{relative}\n"
                )

            with mock.patch.object(package, "_run", side_effect=mismatched):
                with self.assertRaisesRegex(ODEBFContractError, "lock object"):
                    package._create_lock_hash_index(path)

    def test_missing_linked_arm_receipt_fails_before_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = root / "raw/fixed-e8/RS-NEUTRAL"
            fixed.mkdir(parents=True)
            accepted = fixed / "accepted-0000.json"
            accepted.write_text(
                json.dumps({"transition_sha256": "a" * 64}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ODEBFContractError, "linked receipt"):
                package._arm_closure_receipt(
                    root, "RS-NEUTRAL", {accepted}
                )

    def test_recursive_arm_links_rehash_and_fail_on_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = root / "raw/fixed-e8/RS-SOFT/nested"
            fixed.mkdir(parents=True)
            trial = fixed / "trial-0000.json"
            trial.write_text(json.dumps({"category": "trial"}), encoding="utf-8")
            transition = fixed / "transition-0000.json"
            transition.write_text(
                json.dumps(
                    {
                        "trial_receipt_sha256": package._sha256_file(trial),
                    }
                ),
                encoding="utf-8",
            )
            accepted = fixed / "accepted-0000.json"
            accepted.write_text(
                json.dumps(
                    {
                        "transition_sha256": package._sha256_file(transition),
                    }
                ),
                encoding="utf-8",
            )
            selected = {accepted, transition, trial}
            receipt = package._arm_closure_receipt(
                root, "RS-SOFT", selected
            )
            self.assertEqual(receipt["accepted_prefix_count"], 1)
            self.assertEqual(
                receipt["accepted_to_transition_link_rehash_count"], 1
            )
            self.assertEqual(
                receipt["transition_to_trial_link_rehash_count"], 1
            )
            trial.write_text(
                json.dumps({"category": "trial", "changed": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ODEBFContractError, "link differs"):
                package._arm_closure_receipt(root, "RS-SOFT", selected)


if __name__ == "__main__":
    unittest.main()
