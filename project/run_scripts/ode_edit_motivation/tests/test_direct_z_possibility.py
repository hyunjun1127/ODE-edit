"""CPU-only contract tests for the direct-z possibility runner helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from project.run_scripts.ode_edit_motivation import direct_z_possibility as direct_z


def _rank_case_ids(case_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            case_ids,
            key=lambda case_id: (
                hashlib.sha256(
                    direct_z.DEFAULT_SELECTION_SEED.encode("utf-8")
                    + b"\0"
                    + case_id.encode("utf-8")
                ).digest(),
                case_id,
            ),
        )
    )


class DirectZPossibilityRunnerTests(unittest.TestCase):
    def test_selection_uses_locked_fresh_eight_case_slice(self) -> None:
        case_ids = tuple(f"case-{index:03d}" for index in range(160))
        ranked = _rank_case_ids(case_ids)
        canonical = SimpleNamespace(
            source_sha256="0" * 64,
            source_size=123,
            source_row_count=len(case_ids),
            ordered_case_ids=ranked[:100],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    direct_z,
                    "generate_counterfact_selection",
                    return_value=canonical,
                ) as generate,
                mock.patch.object(
                    direct_z,
                    "scan_counterfact_case_ids",
                    return_value=case_ids,
                ),
            ):
                selection = direct_z.generate_directz_selection(root)
        generate.assert_called_once_with(
            root.resolve(),
            seed=direct_z.DEFAULT_SELECTION_SEED,
        )
        self.assertEqual(direct_z.DIRECTZ_CASE_COUNT, 8)
        self.assertEqual(
            (direct_z.DIRECTZ_RANK_START, direct_z.DIRECTZ_RANK_STOP),
            (132, 140),
        )
        self.assertEqual(selection["case_ids"], list(ranked[132:140]))
        self.assertTrue(set(selection["case_ids"]).isdisjoint(ranked[:132]))
        self.assertEqual(
            selection["prior_order_hash"],
            direct_z.sha256_bytes(
                direct_z.canonical_json(list(ranked[:132])).encode("utf-8")
            ),
        )

    def test_branch_order_is_locked_six_arm_order(self) -> None:
        self.assertEqual(
            direct_z.DIRECTZ_BRANCH_ORDER,
            (
                direct_z.NO_OP,
                direct_z.ORACLE_DO_Z,
                direct_z.NATIVE_ORDERED,
                direct_z.NATIVE_C_MATCHED,
                direct_z.BF_REFRESHED_K4,
                direct_z.SYNC_Z_CONE,
            ),
        )
        self.assertEqual(len(set(direct_z.DIRECTZ_BRANCH_ORDER)), 6)

    def test_frobenius_norm_uses_each_low_rank_weight_update(self) -> None:
        first = SimpleNamespace(
            left=torch.tensor([[3.0], [4.0]]),
            right=torch.tensor([[-12.0], [5.0]]),
        )
        second = SimpleNamespace(
            left=torch.tensor([[1.0]]),
            right=torch.tensor([[2.0]]),
        )
        proposal = SimpleNamespace(factors=(first, second))
        self.assertEqual(direct_z._proposal_frobenius_norm(None), 0.0)
        self.assertAlmostEqual(
            direct_z._proposal_frobenius_norm(proposal),
            math.sqrt(65.0**2 + 2.0**2),
        )

    def test_heldout_loader_selects_fixed_text_and_hashes_only_it(self) -> None:
        request = SimpleNamespace(
            case_id="42",
            prompt="{} is a fictional character",
            subject="Alice",
            target_new="Wonder",
        )
        row = {
            "case_id": "42",
            "requested_rewrite": {
                "prompt": request.prompt,
                "subject": request.subject,
                "target_new": {"str": request.target_new},
            },
            "paraphrase_prompts": ["para 1", "para 2", "unused para"],
            "neighborhood_prompts": ["near 1", "near 2", "near 3", "near 4"],
            "generation_prompts": ["gen 1", "gen 2", "gen 3", "gen 4"],
        }
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                direct_z,
                "_iter_top_level_json_objects",
                return_value=iter((json.dumps(row),)),
            ):
                heldout = direct_z.load_heldout_evaluation_case(directory, request)
        self.assertEqual(heldout.paraphrase_prompts, ("para 1", "para 2"))
        self.assertEqual(
            heldout.preservation_prompts,
            (
                "near 1",
                "near 2",
                "near 3",
                "near 4",
                "gen 1",
                "gen 2",
                "gen 3",
                "gen 4",
            ),
        )
        expected = {
            "case_id": request.case_id,
            "paraphrase_prompts": ["para 1", "para 2"],
            "preservation_prompts": list(heldout.preservation_prompts),
        }
        self.assertEqual(
            heldout.payload_hash,
            direct_z.sha256_bytes(
                direct_z.canonical_json(expected).encode("utf-8")
            ),
        )

    def test_cli_and_runtime_envelopes_reject_unlocked_identity(self) -> None:
        parser = direct_z.build_parser()
        args = parser.parse_args(
            [
                "--easyedit-root",
                "/fixed/easyedit",
                "--model",
                "llama3-8b-inst",
                "--run-id",
                direct_z.DIRECTZ_RUN_IDS["llama3-8b-inst"],
            ]
        )
        self.assertEqual(args.run_id, direct_z.DIRECTZ_RUN_IDS[args.model])
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--easyedit-root",
                    "/fixed/easyedit",
                    "--model",
                    "not-a-fixed-model",
                    "--run-id",
                    "anything",
                ]
            )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                direct_z._slurm_state(
                    "llama3-8b-inst",
                    direct_z.DIRECTZ_RUN_IDS["llama3-8b-inst"],
                ),
                {"under_slurm": False},
            )
        with self.assertRaises(direct_z.MV1Error):
            direct_z._execution_envelope("llama3-8b-inst", "wrong-run-id")
        slurm = {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NAME": direct_z.DIRECTZ_JOB_NAME,
            "SLURMD_NODENAME": "devbox",
        }
        with mock.patch.dict(os.environ, slurm, clear=True):
            self.assertEqual(
                direct_z._slurm_state(
                    "qwen2.5-7b-inst",
                    direct_z.DIRECTZ_RUN_IDS["qwen2.5-7b-inst"],
                ),
                {
                    "under_slurm": True,
                    "job_id": "12345",
                    "job_name": direct_z.DIRECTZ_JOB_NAME,
                    "node": "devbox",
                },
            )


if __name__ == "__main__":
    unittest.main()
