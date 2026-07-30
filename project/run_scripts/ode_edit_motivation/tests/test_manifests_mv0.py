import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_edit_motivation.contracts import ExpectedFileIdentity
from project.run_scripts.ode_edit_motivation.manifests import (
    FIXED_FILE_IDENTITIES,
    MODEL_SPECS,
    build_counterfact_selection,
    load_counterfact_requests,
    scan_counterfact_case_ids,
    write_selection_manifest,
)


class MV0ManifestTests(unittest.TestCase):
    def test_fixed_manifest_has_every_required_artifact(self):
        for spec in MODEL_SPECS.values():
            self.assertEqual(len(spec.covariance_paths), 5)
            for path in (*spec.covariance_paths, spec.hparams_path, spec.projector_path):
                identity = FIXED_FILE_IDENTITIES[path]
                self.assertEqual(len(identity.sha256), 64)
                self.assertGreater(identity.size, 0)
        self.assertEqual(len(MODEL_SPECS), 2)

    def test_selection_depends_only_on_case_ids(self):
        ids = tuple(str(index) for index in range(12))
        identity = ExpectedFileIdentity(
            sha256=hashlib.sha256(b"source").hexdigest(),
            size=6,
        )
        first = build_counterfact_selection(
            ids,
            source_identity=identity,
            seed="unit-test",
            split_counts=(2, 3, 1),
        )
        second = build_counterfact_selection(
            tuple(reversed(ids)),
            source_identity=identity,
            seed="unit-test",
            split_counts=(2, 3, 1),
        )
        self.assertEqual(first, second)
        self.assertEqual(
            [len(first.calibration), len(first.confirmatory), len(first.untouched)],
            [2, 3, 1],
        )
        self.assertEqual(len(first.order_hash), 64)
        self.assertEqual(len(first.split_hash), 64)

    def test_selected_request_loader_drops_all_eval_fields(self):
        secret = "DO-NOT-PERSIST-EVALUATION-SECRET"
        rows = [
            {
                "case_id": index,
                "requested_rewrite": {
                    "prompt": "{} lives in",
                    "subject": f"Person {index}",
                    "target_new": {"str": f"City {index}", "id": 10 + index},
                    "target_true": {"str": secret},
                },
                "paraphrase_prompts": [secret],
                "neighborhood_prompts": [secret],
                "generation_prompts": [secret],
            }
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data/counterfact/counterfact.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(scan_counterfact_case_ids(source), ("0", "1", "2", "3"))
            requests = load_counterfact_requests(root, ("3", "1"))
            self.assertEqual([request.case_id for request in requests], ["3", "1"])
            serialized = json.dumps(
                [request.to_dict() for request in requests],
                sort_keys=True,
            )
            self.assertNotIn(secret, serialized)
            self.assertEqual(
                set(requests[0].to_dict()),
                {"case_id", "prompt", "subject", "target_new"},
            )

    def test_selection_artifact_contains_ids_and_hashes_not_rows(self):
        identity = ExpectedFileIdentity(sha256="a" * 64, size=123)
        manifest = build_counterfact_selection(
            ("a", "b", "c", "d"),
            source_identity=identity,
            split_counts=(1, 2, 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            write_selection_manifest(path, manifest)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(payload["case_ids"]), {"calibration", "confirmatory", "untouched"})
        self.assertNotIn("requests", payload)
        self.assertNotIn("prompt", path.name)


if __name__ == "__main__":
    unittest.main()
