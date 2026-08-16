from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts.ode_alloc.contracts import ODEAllocContractError
from project.run_scripts.ode_alloc.firewall import (
    assert_dry_launcher_ast,
    assert_inner_payload_schema,
    assert_p0_runtime_firewall_ast,
)
from project.run_scripts.ode_alloc.selection import (
    EXPLICIT_EXCLUSIONS,
    assert_seal_source_current,
    build_seal_candidate,
    load_and_verify_seal_candidate,
    project_request_identity,
    load_projected_request,
    scan_tracked_prior_case_ids,
    write_canonical_json,
)
from project.run_scripts.ode_alloc import selection as selection_module


def _row(case_id: int, *, duplicate: int | None = None) -> dict:
    subject_id = case_id if duplicate is None else duplicate
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "prompt": "{} has relation",
            "relation_id": "P1",
            "subject": f"Subject {subject_id}",
            "target_new": {"str": f"New {case_id}", "id": f"N{case_id}"},
            "target_true": {"str": f"Old {case_id}", "id": f"O{case_id}"},
        },
        "paraphrase_prompts": [f"evaluation secret {case_id}"],
        "neighborhood_prompts": [f"heldout secret {case_id}"],
    }


class FirewallSelectionTests(unittest.TestCase):
    def test_schema_rejects_any_row_evaluation_surface(self) -> None:
        row = _row(42)
        projected = {
            "case_id": row["case_id"],
            "requested_rewrite": row["requested_rewrite"],
        }
        assert_inner_payload_schema(projected)
        with self.assertRaises(ODEAllocContractError):
            assert_inner_payload_schema(row)

    def test_request_projection_deserializes_only_the_rewrite_slice(self) -> None:
        blob = json.dumps(_row(42), ensure_ascii=False).encode("utf-8")
        real_loads = json.loads

        def guarded_loads(value, *args, **kwargs):
            raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
            self.assertNotIn(b"evaluation secret", raw)
            self.assertNotIn(b"heldout secret", raw)
            return real_loads(value, *args, **kwargs)

        with mock.patch.object(selection_module.json, "loads", side_effect=guarded_loads):
            identity = project_request_identity(blob)
        self.assertEqual(identity.case_id, 42)
        self.assertEqual(len(identity.request_hash), 64)

    def test_p0_loader_returns_only_approved_rewrite_fields(self) -> None:
        row = _row(42)
        identity = project_request_identity(json.dumps(row).encode("utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rows.json"
            source.write_text(json.dumps([row]), encoding="utf-8")
            request = load_projected_request(
                source,
                case_id=42,
                expected_request_hash=identity.request_hash,
            )
        self.assertEqual(
            set(request),
            {"case_id", "prompt", "relation_id", "subject", "target_new", "target_old"},
        )
        self.assertNotIn("paraphrase", json.dumps(request))

    def test_ast_firewall_is_fail_closed(self) -> None:
        launcher = Path(__file__).resolve().parents[2] / "session04_ode_alloc_dry_plan.py"
        assert_dry_launcher_ast(launcher)
        with tempfile.TemporaryDirectory() as directory:
            for index, source in enumerate(
                (
                    "import transformers\n",
                    "from project.run_scripts import session03_private\n",
                    "paraphrase_prompts = []\n",
                )
            ):
                bad = Path(directory) / f"bad-{index}.py"
                bad.write_text(source, encoding="utf-8")
                with self.assertRaises(ODEAllocContractError):
                    assert_dry_launcher_ast(bad)

    def test_p0_runtime_firewall_allows_runtime_but_rejects_heldout_keys(self) -> None:
        runtime = Path(__file__).resolve().parents[1] / "p0_runtime.py"
        launcher = Path(__file__).resolve().parents[2] / "session04_ode_alloc_p0.py"
        assert_p0_runtime_firewall_ast([runtime, launcher])
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.py"
            bad.write_text("secret = 'paraphrase_prompts'\n", encoding="utf-8")
            with self.assertRaises(ODEAllocContractError):
                assert_p0_runtime_firewall_ast([bad])

    def test_deterministic_split_excludes_prior_and_never_reads_session03_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            tracked = repo / "prior.json"
            tracked.write_text(json.dumps({"case_ids": [1001, 1002]}), encoding="utf-8")
            forbidden_dir = repo / "project" / "run_scripts"
            forbidden_dir.mkdir(parents=True)
            forbidden = forbidden_dir / "session03_private.json"
            forbidden.write_text(json.dumps({"case_ids": [1003]}), encoding="utf-8")
            subprocess.run(
                ["git", "add", "prior.json", "project/run_scripts/session03_private.json"],
                cwd=repo,
                check=True,
            )
            dataset = root / "counterfact.json"
            rows = [_row(case_id) for case_id in range(1000, 1060)]
            rows.append(_row(1060, duplicate=1004))
            dataset.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            prior, _, count = scan_tracked_prior_case_ids(repo)
            self.assertEqual(prior, {1001, 1002})
            self.assertEqual(count, 1)
            first = build_seal_candidate(dataset, repo)
            second = build_seal_candidate(dataset, repo)
            self.assertEqual(first, second)
            self.assertEqual(first["root_digest"], second["root_digest"])
            self.assertEqual(first["tracked_prior_collision_scan"]["session03_paths_read"], 0)
            pools = (
                first["p0_identity"],
                first["p1_order"],
                first["pretrained_anchor"],
            )
            selected = [item["case_id"] for pool in pools for item in pool]
            self.assertEqual(len(selected), 21)
            self.assertEqual(len(selected), len(set(selected)))
            self.assertTrue(set(selected).isdisjoint({1001, 1002}))
            self.assertTrue(set(selected).isdisjoint(EXPLICIT_EXCLUSIONS))
            self.assertTrue(set(selected).isdisjoint({1004, 1060}))
            self.assertEqual(
                first["duplicate_policy"]["excluded_case_id_count"], 2
            )
            manifest = root / "seal.json"
            write_canonical_json(manifest, first)
            loaded = load_and_verify_seal_candidate(manifest)
            self.assertEqual(loaded, first)
            assert_seal_source_current(loaded, dataset)
            dataset.write_text("[]", encoding="utf-8")
            with self.assertRaises(ODEAllocContractError):
                assert_seal_source_current(loaded, dataset)


if __name__ == "__main__":
    unittest.main()
