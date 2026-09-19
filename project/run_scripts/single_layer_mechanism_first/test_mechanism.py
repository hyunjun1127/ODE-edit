"""Independent small CPU algebra fixtures; never actual model evidence."""
import unittest
from unittest.mock import patch

import torch

from .mechanism import (MechanismTechnicalError, extract_native_capture,
                        analyze_writer, stream_reference_action,
                        global_interventions, workspace_estimate)


class MechanismTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.g=torch.Generator().manual_seed(9183)

    def matrix(self,m,n):
        return torch.randn((m,n),generator=self.g,dtype=torch.float32)

    def fixture(self,history=False):
        n,m,b=11,7,4
        k,r=self.matrix(n,b),self.matrix(m,b)
        u=torch.eye(n,dtype=torch.float64)[:,:9]
        p=(u@u.T).float()
        keys=self.matrix(n,3) if history else None
        ch=keys@keys.T if history else torch.zeros((n,n))
        entry=self.matrix(m,n)
        system=p@(k@k.T+ch)+torch.eye(n)
        update=torch.linalg.solve(system,(p@k)@r.T).T
        native=entry+update
        return dict(entry_weight=entry,native_weight=native,keys=k,residuals=r,
                    projector=p,history=ch,allowed_basis=u,history_keys=keys),update

    def test_source_order_replay_and_fp32_add(self):
        args,update=self.fixture()
        result=analyze_writer(**args,replay_dense=True)
        receipt=result.receipt["source_order"]
        self.assertTrue(receipt["replay_endpoint_equal"])
        self.assertEqual(receipt["diagnostic_dense_rhs_solve_calls"],1)
        self.assertEqual(receipt["map_diagnostic_solve_calls"],1)
        self.assertLess(receipt["map_relative_solve_residual"],1e-6)
        self.assertLess(receipt["dense_relative_solve_residual"],1e-6)
        self.assertTrue(receipt["algebra_association_not_bitexact_native_rhs"])
        self.assertEqual(result.receipt["new_z_calls"],0)

    def test_reused_update_no_second_dense_solve(self):
        args,update=self.fixture()
        original=torch.linalg.solve
        calls=[]
        def counted(a,b):
            calls.append(b.shape)
            return original(a,b)
        with patch("project.run_scripts.single_layer_mechanism_first.mechanism.torch.linalg.solve",counted):
            result=analyze_writer(**args,native_update=update)
        self.assertEqual(calls,[(11,4)])
        self.assertEqual(result.receipt["source_order"]["diagnostic_dense_rhs_solve_calls"],0)
        self.assertTrue(result.receipt["source_order"]["replay_endpoint_equal"])

    def test_cold_ideal_gain_matches_independent_svd(self):
        args,update=self.fixture()
        result=analyze_writer(**args,native_update=update)
        x=args["allowed_basis"].T@args["keys"].double()
        _,singular,vh=torch.linalg.svd(x,full_matrices=False)
        actual=torch.tensor([row["sigma"] for row in result.mode_rows],dtype=torch.float64)
        torch.testing.assert_close(actual,singular,rtol=1e-12,atol=1e-12)
        loading=(args["residuals"].double()@vh.T).square().sum(0)
        torch.testing.assert_close(torch.tensor([row["target_loading_squared"] for row in result.mode_rows],dtype=torch.float64),loading,rtol=1e-12,atol=1e-12)
        self.assertEqual(result.receipt["ideal_metric"]["C_inverse_action"]["iterations"],0)

    def test_history_ideal_uses_actual_dense_history_not_key_approximation(self):
        args,update=self.fixture(history=True)
        # Recorded keys are deliberately an imperfect preconditioner only.
        args["history_keys"]=args["history_keys"]*.7
        result=analyze_writer(**args,native_update=update)
        u,k,h=args["allowed_basis"],args["keys"].double(),args["history"].double()
        x=u.T@k
        c=torch.eye(u.shape[1],dtype=torch.float64)+u.T@h@u
        inverse=torch.linalg.solve(c,x)
        expected=u@inverse@torch.linalg.inv(torch.eye(x.shape[1],dtype=torch.float64)+x.T@inverse)
        torch.testing.assert_close(result.ideal_map,expected,rtol=1e-9,atol=1e-10)
        receipt=result.receipt["ideal_metric"]["C_inverse_action"]
        self.assertFalse(receipt["history_factor_used_as_operator"])
        self.assertLessEqual(max(receipt["relative_residual_per_rhs"]),1e-10)

    def test_no_svd_on_writer_sized_matrix(self):
        args,update=self.fixture(history=True)
        with patch("project.run_scripts.single_layer_mechanism_first.mechanism.torch.linalg.svd",side_effect=AssertionError("SVD forbidden")):
            result=analyze_writer(**args,native_update=update)
        self.assertFalse(result.receipt["ideal_metric"]["dense_full_SVD"])
        self.assertEqual(result.receipt["ideal_metric"]["request_Gram_dimension"],4)

    def test_raw_modes_reconstruct_algebra_not_claim_actual_exactness(self):
        args,update=self.fixture()
        result=analyze_writer(**args,native_update=update)
        total=sum(mode.materialize() for mode in result.mode_factors)
        algebra=args["residuals"].double()@result.algebra_map.double().T
        torch.testing.assert_close(total,algebra,rtol=1e-12,atol=1e-12)
        self.assertIn("actual FP32 discrepancy",result.receipt["mode_sum_target"])

    def test_R_permutation_fixed_key_norm(self):
        args,update=self.fixture()
        result=analyze_writer(**args,native_update=update)
        row=result.receipt["R_column_permutation_control"]
        self.assertEqual(row["original_R_norm"],row["permuted_R_norm"])
        self.assertTrue(row["fixed_K_and_map"])
        self.assertFalse(row["new_model_or_edit_arm"])

    def test_extract_native_fp32_subtract_then_repeat(self):
        z=[self.matrix(5,1).flatten() for _ in range(3)]
        current=self.matrix(3,5)
        k=self.matrix(6,7)
        result=extract_native_capture({"captures":{"compute_z":z,"compute_ks":[k],
                    "get_module_input_output_at_words":[current]},"receipt":{"source":{"sha":"fixture"}}})
        torch.testing.assert_close(result["R"],(torch.stack(z,1)-current.T).repeat_interleave(2,1),rtol=0,atol=0)
        self.assertEqual(result["receipt"]["repeat_factor"],2)
        self.assertEqual(result["receipt"]["new_z_calls"],0)

    def test_exact_cross_terms_all_documents_and_factors(self):
        e,d=self.matrix(3,5),self.matrix(3,5)
        keys=[self.matrix(5,t) for t in [2,7,3]]
        factors=[self.matrix(3,k.shape[1]) for k in keys]
        result=stream_reference_action(e,d,keys,expected_documents=3,native_gradient_factors=factors)
        self.assertEqual(result["documents"],3)
        self.assertEqual(result["valid_input_tokens"],12)
        self.assertLess(result["identity_max_relative_error"],1e-14)
        for row,k,a in zip(result["rows"],keys,factors):
            expected=float((a.double()*(d.double()@k.double())).sum())
            self.assertEqual(row["signed_native_scalar_derivative"],expected)
        expected_mean=sum(float((d.double()@k.double()).square().sum())/k.shape[1] for k in keys)/3
        self.assertAlmostEqual(result["document_normalized_means"]["native_step_energy"],expected_mean,places=12)

    def test_cross_terms_b1_zero_entry_displacement(self):
        d=self.matrix(3,5)
        row=stream_reference_action(torch.zeros_like(d),d,[self.matrix(5,2)],expected_documents=1)["rows"][0]
        self.assertEqual(row["entry_energy"],0)
        self.assertEqual(row["twice_entry_step_inner"],0)
        self.assertEqual(row["native_step_energy"],row["native_net_energy"])

    def test_missing_excess_documents_and_factors(self):
        e,d=self.matrix(3,5),self.matrix(3,5)
        for count in (1,3):
            with self.assertRaises(MechanismTechnicalError):
                stream_reference_action(e,d,[self.matrix(5,2)]*count,expected_documents=2)
        with self.assertRaises(MechanismTechnicalError):
            stream_reference_action(e,d,[self.matrix(5,2)],expected_documents=1,native_gradient_factors=[])

    def test_four_global_interventions_norm_rank_and_replay(self):
        args,update=self.fixture()
        diagnostics=analyze_writer(**args,native_update=update)
        interventions,receipt=global_interventions(diagnostics,selection_seal="fixture-sealed",native_parity_confirmed=True,allowed_basis=args["allowed_basis"])
        repeat,_=global_interventions(diagnostics,selection_seal="fixture-sealed",native_parity_confirmed=True,allowed_basis=args["allowed_basis"])
        self.assertEqual(receipt["interventions"],4)
        for index in [0,2]:
            source,control=interventions[index:index+2]
            self.assertAlmostEqual(float(source.materialize().norm()),float(control.materialize().norm()),places=10)
            self.assertEqual(int(torch.linalg.matrix_rank(control.materialize())),1)
            k=self.matrix(11,3).double()
            torch.testing.assert_close(source.action(k),source.materialize()@k,rtol=1e-12,atol=1e-12)
        for one,two in zip(interventions,repeat):
            torch.testing.assert_close(one.left,two.left,rtol=0,atol=0)
            torch.testing.assert_close(one.right,two.right,rtol=0,atol=0)

    def test_intervention_requires_postseal_and_parity(self):
        args,update=self.fixture()
        diagnostics=analyze_writer(**args,native_update=update)
        for seal,parity in [("",True),("sealed",False)]:
            with self.assertRaises(MechanismTechnicalError):
                global_interventions(diagnostics,selection_seal=seal,native_parity_confirmed=parity)

    def test_invalid_dtype_nonfinite_and_double_replay(self):
        args,update=self.fixture()
        with self.assertRaises(MechanismTechnicalError):
            analyze_writer(**args,native_update=update,replay_dense=True)
        args["keys"]=args["keys"].double()
        with self.assertRaises(MechanismTechnicalError):analyze_writer(**args)
        args["keys"]=args["keys"].float();args["keys"][0,0]=float("nan")
        with self.assertRaises(MechanismTechnicalError):analyze_writer(**args)

    def test_workspace_estimate_not_measured_peak(self):
        result=workspace_estimate(14336,4096,100,14326)
        self.assertEqual(result["dense_RHS_optional_FP32_bytes"],234881024)
        self.assertFalse(result["full_svd"])
        self.assertIn("NOT_MEASURED",result["status"])


if __name__ == "__main__":unittest.main()
