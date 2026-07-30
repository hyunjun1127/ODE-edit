import os
import unittest
from types import SimpleNamespace
from unittest import mock

from project.run_scripts.ode_edit_motivation.contracts import (
    ContractError,
    ExpectedFileIdentity,
)
from project.run_scripts.ode_edit_motivation.manifests import (
    DEFAULT_SELECTION_SEED,
    CounterFactSelectionManifest,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_confirmatory import (
    CONFIRMATORY_OUTCOME_ACTIONS,
    CONFIRMATORY_POLICY_PATHS,
    CONFIRMATORY_WAVE_LOCK,
    SCORE_MIX_Q,
    ScoreMixWaveLock,
    _run_confirmatory_event,
    run_score_mix_wave,
    select_confirmatory_fold_cases,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix_followup import (
    FOLLOWUP_JOB_NAMES,
    FOLLOWUP_MODES,
    FOLLOWUP_POLICY_PATHS,
    FOLLOWUP_RUN_IDS,
    FOLLOWUP_RUN_SEED,
    FOLLOWUP_WAVE_LOCKS,
    _followup_execution_envelope,
    _followup_slurm_state,
    build_parser,
    followup_wave_lock,
    run_mv1_score_mix_followup,
    select_followup_cases,
)


class FollowupRunnerContractTests(unittest.TestCase):
    @staticmethod
    def selection(*, seed=DEFAULT_SELECTION_SEED, overlap=False):
        calibration = tuple(f"cal-{index:02d}" for index in range(20))
        confirmatory = tuple(f"conf-{index:02d}" for index in range(60))
        untouched = tuple(f"unt-{index:02d}" for index in range(20))
        if overlap:
            untouched = (confirmatory[0], *untouched[1:])
        return CounterFactSelectionManifest(
            seed=seed,
            source_sha256=ExpectedFileIdentity(
                sha256="d" * 64,
                size=1,
            ).sha256,
            source_size=1,
            source_row_count=100,
            calibration=calibration,
            confirmatory=confirmatory,
            untouched=untouched,
        )

    def test_only_two_modes_and_all_execution_constants_are_precommitted(self):
        self.assertEqual(FOLLOWUP_MODES, ("fold1", "untouched"))
        self.assertEqual(FOLLOWUP_RUN_SEED, 17)
        self.assertEqual(
            dict(FOLLOWUP_JOB_NAMES),
            {
                "fold1": "odeedit_mv1mix_fold1_pair_v1",
                "untouched": "odeedit_mv1mix_untouched_pair_v1",
            },
        )
        self.assertEqual(
            {mode: dict(run_ids) for mode, run_ids in FOLLOWUP_RUN_IDS.items()},
            {
                "fold1": {
                    "llama3-8b-inst": "mv1mix_llama_fold1_v1",
                    "qwen2.5-7b-inst": "mv1mix_qwen_fold1_v1",
                },
                "untouched": {
                    "llama3-8b-inst": "mv1mix_llama_untouched_v1",
                    "qwen2.5-7b-inst": "mv1mix_qwen_untouched_v1",
                },
            },
        )
        self.assertEqual(SCORE_MIX_Q, 1.0 / 256.0)
        self.assertEqual(
            CONFIRMATORY_OUTCOME_ACTIONS,
            (
                "score_mix",
                "frozen_static_mix",
                "uniform",
                "ordered_global_alpha",
                "native_memit_full",
                "no_op_replay",
            ),
        )
        self.assertEqual(
            {
                model: dict(paths)
                for model, paths in FOLLOWUP_POLICY_PATHS.items()
            },
            CONFIRMATORY_POLICY_PATHS,
        )
        with self.assertRaises(Exception):
            followup_wave_lock("c1")
        with self.assertRaises(Exception):
            followup_wave_lock("fold2")

    def test_fold1_and_untouched_are_exact_deterministic_selection_ids(self):
        selection = self.selection()
        fold0, fold_manifest0 = select_confirmatory_fold_cases(selection)
        fold1, fold_manifest1 = select_followup_cases(selection, mode="fold1")
        repeated, repeated_manifest = select_followup_cases(
            selection,
            mode="fold1",
        )
        untouched, untouched_manifest = select_followup_cases(
            selection,
            mode="untouched",
        )

        self.assertEqual(len(fold0), 12)
        self.assertEqual(len(fold1), 12)
        self.assertEqual(fold1, repeated)
        self.assertEqual(fold_manifest0.manifest_id, fold_manifest1.manifest_id)
        self.assertEqual(
            fold_manifest1.manifest_id,
            repeated_manifest.manifest_id,
        )
        self.assertEqual(
            fold1,
            tuple(
                case_id
                for case_id in selection.confirmatory
                if fold_manifest1.fold_for(case_id) == 1
            ),
        )
        self.assertFalse(set(fold0).intersection(fold1))
        self.assertEqual(untouched, selection.untouched)
        self.assertEqual(
            untouched_manifest.manifest_id,
            fold_manifest1.manifest_id,
        )

    def test_selection_seed_and_split_overlap_fail_closed(self):
        with self.assertRaises(ContractError):
            select_followup_cases(
                self.selection(seed="not-the-canonical-seed"),
                mode="fold1",
            )
        for mode in FOLLOWUP_MODES:
            with self.subTest(mode=mode):
                with self.assertRaises(ContractError):
                    select_followup_cases(
                        self.selection(overlap=True),
                        mode=mode,
                    )

    def test_forged_wave_identity_job_run_or_seed_is_rejected(self):
        valid = FOLLOWUP_WAVE_LOCKS["fold1"]
        mutations = (
            {
                "job_name": "forged",
                "run_ids": valid.run_ids,
                "run_seed": valid.run_seed,
            },
            {
                "job_name": valid.job_name,
                "run_ids": {
                    **dict(valid.run_ids),
                    "llama3-8b-inst": "forged",
                },
                "run_seed": valid.run_seed,
            },
            {
                "job_name": valid.job_name,
                "run_ids": valid.run_ids,
                "run_seed": 18,
            },
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ContractError):
                    ScoreMixWaveLock(
                        label="fold1",
                        selected_split="confirmatory",
                        fold=1,
                        case_count=12,
                        **mutation,
                    )
        for field, value in (("fold", True), ("case_count", 12.0)):
            arguments = {
                "label": "fold1",
                "selected_split": "confirmatory",
                "fold": 1,
                "case_count": 12,
                "job_name": valid.job_name,
                "run_ids": valid.run_ids,
                "run_seed": valid.run_seed,
            }
            arguments[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ContractError):
                    ScoreMixWaveLock(**arguments)
        self.assertIsNone(CONFIRMATORY_WAVE_LOCK.run_seed)

    def test_run_job_and_mode_are_exact_fail_closed(self):
        for mode in FOLLOWUP_MODES:
            for model_alias, run_id in FOLLOWUP_RUN_IDS[mode].items():
                self.assertEqual(
                    _followup_execution_envelope(
                        model_alias,
                        run_id,
                        mode=mode,
                    ),
                    mode,
                )
            with self.assertRaises(Exception):
                _followup_execution_envelope(
                    "llama3-8b-inst",
                    FOLLOWUP_RUN_IDS[mode]["qwen2.5-7b-inst"],
                    mode=mode,
                )
            with mock.patch.dict(
                os.environ,
                {
                    "SLURM_JOB_ID": "50001",
                    "SLURM_JOB_NAME": FOLLOWUP_JOB_NAMES[mode],
                    "SLURMD_NODENAME": "devbox",
                },
                clear=True,
            ):
                state = _followup_slurm_state(
                    "llama3-8b-inst",
                    FOLLOWUP_RUN_IDS[mode]["llama3-8b-inst"],
                    mode=mode,
                )
                self.assertEqual(state["slice_label"], mode)
                self.assertEqual(state["fold"], 1 if mode == "fold1" else None)
            with mock.patch.dict(
                os.environ,
                {
                    "SLURM_JOB_ID": "50001",
                    "SLURM_JOB_NAME": "forged",
                    "SLURMD_NODENAME": "devbox",
                },
                clear=True,
            ):
                with self.assertRaises(Exception):
                    _followup_slurm_state(
                        "llama3-8b-inst",
                        FOLLOWUP_RUN_IDS[mode]["llama3-8b-inst"],
                        mode=mode,
                    )

    def test_wrapper_thinly_delegates_to_c1_core_with_locked_seed(self):
        sentinel = {"all_pass": True}
        with mock.patch(
            "project.run_scripts.ode_edit_motivation."
            "mv1_score_mix_followup.run_score_mix_wave",
            return_value=sentinel,
        ) as delegated:
            observed = run_mv1_score_mix_followup(
                mode="fold1",
                easyedit_root="/not-opened",
                model_alias="llama3-8b-inst",
                run_id=FOLLOWUP_RUN_IDS["fold1"]["llama3-8b-inst"],
                static_policy_path="/not-opened/static.json",
                forecast_policy_path="/not-opened/forecast.json",
            )
        self.assertIs(observed, sentinel)
        kwargs = delegated.call_args.kwargs
        self.assertIs(kwargs["wave"], FOLLOWUP_WAVE_LOCKS["fold1"])
        self.assertEqual(kwargs["seed"], FOLLOWUP_RUN_SEED)
        self.assertIs(kwargs["event_runner"], _run_confirmatory_event)

    def test_generic_core_rejects_followup_seed_before_any_file_access(self):
        def injected_loader(_model_alias):
            return object()

        def injected_event(**_kwargs):
            return {}

        for invalid_seed in (18, 17.0, True):
            with self.subTest(seed=invalid_seed):
                with self.assertRaisesRegex(Exception, "seed differs"):
                    run_score_mix_wave(
                        wave=FOLLOWUP_WAVE_LOCKS["fold1"],
                        easyedit_root="/not-opened",
                        model_alias="llama3-8b-inst",
                        run_id=FOLLOWUP_RUN_IDS["fold1"][
                            "llama3-8b-inst"
                        ],
                        static_policy_path="/not-opened/static.json",
                        forecast_policy_path="/not-opened/forecast.json",
                        seed=invalid_seed,
                        model_loader=injected_loader,
                        event_runner=injected_event,
                    )

        fake_wave = SimpleNamespace(
            label="fold2",
            selected_split="confirmatory",
            fold=2,
            case_count=12,
            job_name="forged",
            run_ids={"llama3-8b-inst": "forged"},
            run_seed=None,
        )
        with self.assertRaisesRegex(ContractError, "runtime type"):
            run_score_mix_wave(
                wave=fake_wave,
                easyedit_root="/not-opened",
                model_alias="llama3-8b-inst",
                run_id="forged",
                static_policy_path="/not-opened/static.json",
                forecast_policy_path="/not-opened/forecast.json",
                model_loader=injected_loader,
                event_runner=injected_event,
            )

    def test_cli_exposes_no_seed_or_selection_retuning(self):
        parsed = build_parser().parse_args(
            [
                "--mode",
                "untouched",
                "--easyedit-root",
                "/tmp/easyedit",
                "--model",
                "llama3-8b-inst",
                "--run-id",
                FOLLOWUP_RUN_IDS["untouched"]["llama3-8b-inst"],
                "--static-policy",
                "/tmp/static.json",
                "--forecast-policy",
                "/tmp/forecast.json",
            ]
        )
        self.assertEqual(parsed.mode, "untouched")
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(
                    [
                        "--mode",
                        "fold1",
                        "--easyedit-root",
                        "/tmp/easyedit",
                        "--model",
                        "llama3-8b-inst",
                        "--run-id",
                        FOLLOWUP_RUN_IDS["fold1"]["llama3-8b-inst"],
                        "--static-policy",
                        "/tmp/static.json",
                        "--forecast-policy",
                        "/tmp/forecast.json",
                        "--seed",
                        "18",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
