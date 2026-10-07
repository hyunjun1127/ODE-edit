"""CPU scalar/terminal/mask reducer checks; no live job or model access."""
import copy
import unittest
from . import collect


class CollectTests(unittest.TestCase):
    def test_native_count_budget_no_history_noBLUE(self):
        for arm in ('PRUNE','RECT'):
            self.assertEqual(collect.EXPECTED[arm],dict(native_z=100,write_keys=5,
                history_keys=0,solves=5,history_appends=0))

    def test_rect_actual_tie_support_not_forced_to40percent(self):
        total=6400*1600;support=total//2
        rows=[dict(weight=f'transformer.h.{l}.mlp.c_proj.weight',denominator=total,
            support_count=support,support_pct=50.,threshold_ties=total//4,
            strictly_below_threshold_count=total-support,threshold=.125,kth_index=int(total*.6),
            native_k_percent=40,native_epsilon=1e-8,native_comparison='>=',
            no_mask_tensor_saved=True,no_additional_model_forward=True) for l in (13,14,15,16,17)]
        result=collect.rect_masks(dict(rect_mask_rows=rows),'RECT',5)
        self.assertEqual(len(result),5);self.assertEqual(result[0]['support_pct'],50.)
        broken=copy.deepcopy(rows);broken[0]['support_count']-=1
        with self.assertRaisesRegex(RuntimeError,'SUPPORT_TIE'):
            collect.rect_masks(dict(rect_mask_rows=broken),'RECT',5)

    def test_nonterminal_dense_identity_and_PRUNE_onlyterminal(self):
        state=dict(W={'13':'hash'},H={})
        commit=dict(prune_applied=False,terminal_transform=dict(prune_applied=False),
                    native_after=state,after=state)
        self.assertIsNone(collect.transform_guard(commit,'PRUNE',5))
        with self.assertRaisesRegex(RuntimeError,'STAGE'):
            collect.transform_guard(commit,'PRUNE',20)

    def test_terminal_repair_fields_and_nativefive_spectral_rows(self):
        row=dict(stored_weight_shape=[6400,1600],dtype='torch.float32',singular_count=1600,
            exact_copy_verified=True,transformed_singular_count=0,max_sigma=1.,delta_norm=.1,
            compressed_delta_norm=.1,dense_weight_norm=2.,final_weight_norm=2.,
            dense_weight_sha256='a'*64,cold_weight_sha256='b'*64,
            compressed_delta_sha256='c'*64,final_weight_sha256='d'*64)
        value=dict(prune_applied=True,repair='PRUNE_TERMINAL_BASE_FIX',repair_authorized=True,
            upstream_bitwise_equivalence=False,terminal_transforms=1,final_base='SAVED_COLD_W0',
            spectrum_formula_changed=False,checkpoint_saved=False,
            original_line='adjusted_weight = original_weight + upd_matrix[k]',
            repaired_line='adjusted_weight = saved_cold_weight + upd_matrix[k]',
            layers={f'transformer.h.{l}.mlp.c_proj.weight':dict(row) for l in (13,14,15,16,17)},seconds=2.)
        cold=dict(W={str(l):'b'*64 for l in (13,14,15,16,17)},H={})
        commit=dict(prune_applied=True,terminal_transform=value,
            native_after=dict(W={str(l):'a'*64 for l in (13,14,15,16,17)},H={}),
            after=dict(W={str(l):'d'*64 for l in (13,14,15,16,17)},H={}))
        self.assertEqual(collect.transform_guard(commit,'PRUNE',20,cold)['repair'],'PRUNE_TERMINAL_BASE_FIX')
        value['final_base']='ALREADY_UPDATED_CURRENT_W'
        with self.assertRaisesRegex(RuntimeError,'EXPLICIT_BASE_FIX'):
            collect.transform_guard(commit,'PRUNE',20,cold)

    def test_terminal_exact_layer_names_and_three_state_hashes_reject_mismatch(self):
        row=dict(stored_weight_shape=[6400,1600],dtype='torch.float32',singular_count=1600,
            exact_copy_verified=True,transformed_singular_count=0,max_sigma=1.,delta_norm=.1,
            compressed_delta_norm=.1,dense_weight_norm=2.,final_weight_norm=2.,
            dense_weight_sha256='a'*64,cold_weight_sha256='b'*64,
            compressed_delta_sha256='c'*64,final_weight_sha256='d'*64)
        value=dict(prune_applied=True,repair='PRUNE_TERMINAL_BASE_FIX',repair_authorized=True,
            upstream_bitwise_equivalence=False,terminal_transforms=1,final_base='SAVED_COLD_W0',
            spectrum_formula_changed=False,checkpoint_saved=False,
            original_line='adjusted_weight = original_weight + upd_matrix[k]',
            repaired_line='adjusted_weight = saved_cold_weight + upd_matrix[k]',
            layers={f'transformer.h.{l}.mlp.c_proj.weight':dict(row) for l in (13,14,15,16,17)},seconds=2.)
        cold=dict(W={str(l):'b'*64 for l in (13,14,15,16,17)},H={})
        base=dict(prune_applied=True,terminal_transform=value,
            native_after=dict(W={str(l):'a'*64 for l in (13,14,15,16,17)},H={}),
            after=dict(W={str(l):'d'*64 for l in (13,14,15,16,17)},H={}))
        name='transformer.h.13.mlp.c_proj.weight'
        for key in ('dense_weight_sha256','cold_weight_sha256','final_weight_sha256'):
            bad=copy.deepcopy(base);bad['terminal_transform']['layers'][name][key]='f'*64
            with self.assertRaisesRegex(RuntimeError,'HASH_BINDING'):
                collect.transform_guard(bad,'PRUNE',20,cold)
        bad=copy.deepcopy(base)
        bad['terminal_transform']['layers']['transformer.h.12.mlp.c_proj.weight']=\
            bad['terminal_transform']['layers'].pop(name)
        with self.assertRaisesRegex(RuntimeError,'PARAMETER_NAMES'):
            collect.transform_guard(bad,'PRUNE',20,cold)


if __name__=='__main__':unittest.main()
