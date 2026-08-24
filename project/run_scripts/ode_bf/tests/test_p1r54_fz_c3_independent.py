from __future__ import annotations

import inspect
import hashlib
import unittest

import torch

from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE
from project.run_scripts.ode_bf.p1r54_fz_c3_independent import (
    P1R54FZExecutionMode,
    RESULT_NAMES,
    ROLES,
    STREAM_ORDER,
    STREAM_ROOT,
    atomic_source_equivalence,
    bind_independent_batch_view,
    cell_config,
    expected_result_name,
)
from project.run_scripts.ode_bf.scalable_batched_runtime import (
    scalable_ordered_request_digest,
)


def _stream_fixture() -> tuple[tuple[tuple[dict[str, str], ...], ...], dict[str, object]]:
    batches = tuple(
        tuple(
            {
                "request_sha256": hashlib.sha256(
                    f"{batch:02d}-{request:03d}".encode()
                ).hexdigest()
            }
            for request in range(100)
        )
        for batch in range(10)
    )
    stream = {
        "root_digest": STREAM_ROOT,
        "all_request_order_sha256": STREAM_ORDER,
        "batch_ordered_request_digest_v1": [
            scalable_ordered_request_digest(
                [str(item["request_sha256"]) for item in batch]
            )
            for batch in batches
        ],
    }
    return batches, stream


class P1R54FZC3IndependentTests(unittest.TestCase):
    def test_typed_cells_are_unique_independent_namespaces(self) -> None:
        configs = [cell_config(cell) for cell in range(10)]
        self.assertEqual([item.batch_index for item in configs], list(range(1, 11)))
        self.assertEqual([item.role for item in configs], list(ROLES))
        self.assertEqual(
            [item.result_name for item in configs],
            [RESULT_NAMES[role] for role in ROLES],
        )
        self.assertEqual(len({item.result_name for item in configs}), 10)
        self.assertTrue(
            all(
                item.execution_mode is P1R54FZExecutionMode.INDEPENDENT_BATCH
                and item.weight_entry_version == 0
                and item.alpha_cache_entry_width == 0
                and item.cross_batch_state_consumption_count == 0
                for item in configs
            )
        )

    def test_selector_binds_each_canonical_slice_without_payload_copy(self) -> None:
        batches, stream = _stream_fixture()
        for batch_index in range(1, 11):
            selected = bind_independent_batch_view(
                batches,
                stream,
                batch_index=batch_index,
            )
            self.assertEqual(len(selected), 10)
            self.assertIs(selected[0], batches[batch_index - 1])
            self.assertEqual({id(item) for item in selected}, {id(item) for item in batches})

    def test_parameterized_cpu_w0_and_cold_cache_entry_fixture(self) -> None:
        base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        expected_bytes = tensor_sha256(base)
        cold_cache_identity = "COLD_ALPHA_CACHE_ENTRY_V0_WIDTH0"
        observed_entries = []
        for cell in range(10):
            parameter = torch.nn.Parameter(base.clone())
            pointer = int(parameter.data_ptr())
            with torch.no_grad():
                parameter.add_(float(cell + 1))
                parameter.copy_(base)
            observed_entries.append(
                {
                    "cell": cell,
                    "W0_sha256": tensor_sha256(parameter),
                    "W0_pointer_restored": int(parameter.data_ptr()) == pointer,
                    "weight_entry_version": 0,
                    "cache_identity": cold_cache_identity,
                    "cache_entry_version": 0,
                    "cache_entry_width": 0,
                    "state_namespace": cell_config(cell).result_name,
                }
            )
        self.assertEqual({item["W0_sha256"] for item in observed_entries}, {expected_bytes})
        self.assertTrue(all(item["W0_pointer_restored"] for item in observed_entries))
        self.assertEqual({item["weight_entry_version"] for item in observed_entries}, {0})
        self.assertEqual({item["cache_identity"] for item in observed_entries}, {cold_cache_identity})
        self.assertEqual({item["cache_entry_width"] for item in observed_entries}, {0})
        self.assertEqual(len({item["state_namespace"] for item in observed_entries}), 10)

    def test_atomic_source_equivalence_and_runtime_result_binding(self) -> None:
        for batch_index, role in enumerate(ROLES, start=1):
            receipt = atomic_source_equivalence(batch_index)
            self.assertEqual(receipt["canonical_batch_index"], batch_index)
            self.assertEqual(receipt["different_scientific_input_count"], 0)
            self.assertEqual(receipt["cross_batch_state_consumption_count"], 0)
            self.assertEqual(receipt["sample_payload_copy_count"], 0)
            self.assertEqual(receipt["heldout_k_indices"], [7])
            self.assertEqual(
                expected_p1r52_sequential_result_name(
                    "llama3-8b-inst", role, scale=P1R52_B100X10_SCALE
                ),
                expected_result_name(role),
            )

    def test_adapter_does_not_reimplement_science_or_mutate_sequential_binding(self) -> None:
        from project.run_scripts.ode_bf import p1r54_fz_c3_independent as module

        source = inspect.getsource(module)
        for prohibited in (
            "run_official_native_apply",
            "KStepBatchEntryCachePolicy(",
            "_fp32_target_and_j0(",
            "prepare_p1r52_target_proposal(",
            "loss.backward(",
        ):
            self.assertNotIn(prohibited, source)
        self.assertIn("run_target_timescale_b100(", source)
        self.assertIn("atomic_b1_source_equivalence()", source)


if __name__ == "__main__":
    unittest.main()
