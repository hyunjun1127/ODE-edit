import unittest

import torch

from project.run_scripts.ode_edit_motivation.alphaedit_history import (
    AlphaEditHistoryBank,
    AlphaEditHistoryError,
)


class AlphaEditHistoryBankTests(unittest.TestCase):
    def test_append_once_and_matrix_preserve_canonical_cache_equivalence(self) -> None:
        bank = AlphaEditHistoryBank((4, 5))
        first = {
            4: torch.tensor([[1.0, 0.0], [0.5, 1.0], [0.0, -0.5]]),
            5: torch.tensor([[0.2], [0.4], [0.6]]),
        }
        second = {
            4: torch.tensor([[0.1], [0.2], [0.3]]),
            5: torch.tensor([[1.0, -1.0], [0.0, 0.5], [0.5, 0.25]]),
        }
        bank.append_post_edit_keys(edit_id="edit-1", keys_by_layer=first)
        bank.append_post_edit_keys(edit_id="edit-2", keys_by_layer=second)

        for layer in bank.layers:
            observed = bank.matrix(layer, width=3, device="cpu", dtype=torch.float64)
            expected = torch.cat((first[layer], second[layer]), dim=1).double()
            torch.testing.assert_close(observed, expected)
            torch.testing.assert_close(
                observed @ observed.T,
                first[layer].double() @ first[layer].double().T
                + second[layer].double() @ second[layer].double().T,
            )
        self.assertEqual(bank.edit_count, 2)
        self.assertEqual(bank.rank(4), 3)
        self.assertEqual(bank.rank(5), 3)
        self.assertEqual(len(bank.history_id), 64)

    def test_empty_bank_returns_zero_rank_matrix_without_mutation(self) -> None:
        bank = AlphaEditHistoryBank((4, 5))
        observed = bank.matrix(4, width=7, device="cpu", dtype=torch.float32)
        self.assertEqual(tuple(observed.shape), (7, 0))
        self.assertEqual(bank.edit_count, 0)

    def test_duplicate_or_partial_append_fails_before_mutation(self) -> None:
        bank = AlphaEditHistoryBank((4, 5))
        keys = {4: torch.ones((3, 1)), 5: torch.ones((3, 1))}
        bank.append_post_edit_keys(edit_id="edit-1", keys_by_layer=keys)
        with self.assertRaises(AlphaEditHistoryError):
            bank.append_post_edit_keys(edit_id="edit-1", keys_by_layer=keys)
        with self.assertRaises(AlphaEditHistoryError):
            bank.append_post_edit_keys(edit_id="edit-2", keys_by_layer={4: torch.ones((3, 1))})
        self.assertEqual(bank.edit_count, 1)

    def test_width_change_fails_closed(self) -> None:
        bank = AlphaEditHistoryBank((4,))
        bank.append_post_edit_keys(edit_id="edit-1", keys_by_layer={4: torch.ones((3, 1))})
        with self.assertRaises(AlphaEditHistoryError):
            bank.append_post_edit_keys(edit_id="edit-2", keys_by_layer={4: torch.ones((4, 1))})
        self.assertEqual(bank.edit_count, 1)


if __name__ == "__main__":
    unittest.main()
