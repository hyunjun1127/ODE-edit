"""Synthetic CPU report-contract tests. No model, figures or real results read."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from . import reporting as r


class ReportingTests(unittest.TestCase):
    def test_absent_optional_is_status_not_zero(self):
        tables={x:pd.DataFrame() for x in ("fixed_probe_history.csv","native_write_modes.csv","reconstruction.csv","counterfactuals.csv","activation_margin.csv")}
        out=r.summarize_optional(tables)
        self.assertTrue(all(df.to_dict("records")==[{"status":"NOT_MEASURED"}] for df in out.values()))

    def test_optional_raw_probe_not_clipped(self):
        tables={x:pd.DataFrame() for x in ("fixed_probe_history.csv","native_write_modes.csv","reconstruction.csv","counterfactuals.csv","activation_margin.csv")}
        tables["fixed_probe_history.csv"]=pd.DataFrame([{"history_batch":0,"raw_score":-2.,"nu":-3.},{"history_batch":0,"raw_score":4.,"nu":5.}])
        x=r.summarize_optional(tables)["fixed_probe_summary.csv"].iloc[0]
        self.assertEqual(x.nu_mean,1.);self.assertEqual(x.negative_raw_count,1);self.assertEqual(x.nu_outside_unit_count,2)

    def test_sensitive_context_rejected_recursively(self):
        with self.assertRaises(ValueError):r.validate_public_context({"runtime":{"access_token":"synthetic"}})
        r.validate_public_context({"runtime":{"tokenizers":"0.19.1"},"source":{"sha256":"a"}})

    def test_activation_source_fields_preserved_and_aliased(self):
        tables={x:pd.DataFrame() for x in ("fixed_probe_history.csv","native_write_modes.csv","reconstruction.csv","counterfactuals.csv","activation_margin.csv")}
        tables["activation_margin.csv"]=pd.DataFrame([{"DK_squared_norm":3.,"EK_squared_norm":4.,"EK_DK_signed_cross_term":-2.,"nonlinear_remainder":.1}])
        x=r.summarize_optional(tables)["activation_margin.csv"].iloc[0]
        self.assertEqual(x.D_energy,3.);self.assertEqual(x.cross_term,-2.);self.assertEqual(x.EK_DK_signed_cross_term,-2.)

    def test_hypothesis_unmeasured_remains_unresolved(self):
        rows=[dict(cohort="at_write",end_batch=100,view=view,metric_tag=tag,loss_numerator=n,loss_denominator=100) for view in ("original_all_case","interval_target_conflict_free") for tag,n in (("RS",1),("PS",2),("NS",10))]
        x=r.hypotheses(pd.DataFrame(rows),pd.DataFrame(),{})
        self.assertEqual(x["H1"]["status"],"SUPPORTED")
        self.assertTrue(all(x[h]["status"]=="UNRESOLVED" for h in ("H2","H3","H4")))
        self.assertTrue(x["H1"]["provisional"])

    def test_hypothesis_override_requires_evidence(self):
        t=pd.DataFrame(columns=["cohort","end_batch","view","metric_tag"])
        with self.assertRaises(ValueError):r.hypotheses(t,pd.DataFrame(),{"hypotheses":{"H2":{"status":"SUPPORTED"}}})

    def test_terminal_csv_rejects_unbound_member(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/"x.csv").write_text("a\n1\n");(p/"terminal.json").write_text(json.dumps({"status":"PASS","members":[]}))
            with self.assertRaises(ValueError):r.terminal_csvs(p,"x.csv")

    def test_terminal_csv_keeps_failure_status(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);csv=p/"x.csv";csv.write_text("a\n1\n")
            (p/"terminal.json").write_text(json.dumps({"status":"FAILED","members":[{"path":str(csv),"bytes":csv.stat().st_size,"sha256":r.sha(csv)}]}))
            data,_=r.terminal_csvs(p,"x.csv")
            self.assertEqual(data.iloc[0].receipt_status,"FAILED")

    def test_final_requires_explicit_all31(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"cells.json";p.write_text(json.dumps({"A01":"PASS","B00":"PASS"}))
            with self.assertRaises(ValueError):r.cells_from_input(p,"PASS","PASS",True)

    def test_draft_not_automatic_all_pass(self):
        cells=r.cells_from_input(None,"PASS","PASS",False)
        self.assertEqual(len(cells),31)
        self.assertEqual(set(cells[cells.status=="PASS"].cell_id),{"A01","B00"})
        self.assertEqual(cells[cells.cell_id=="C01"].iloc[0].status,"NOT_MEASURED")

    def test_final_explicit_nonterminal_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"cells.json";cells={x:"BLOCKED" for x in pd.read_csv(r.CELLS).cell_id};cells["C01"]="RUNNING";p.write_text(json.dumps(cells))
            with self.assertRaises(ValueError):r.cells_from_input(p,"PASS","PASS",True)

    def test_functional_counts_strict_and_nll(self):
        records=[dict(checkpoint_batch=100,arrival_batch=1,metric_tag="RS",success=s,target_strict=t,target_token_correct=int(t),target_token_count=1,case_id=i,new_nll=float(i),true_nll=3.,safety_margin=3.-i) for i,s,t in ((1,True,True),(2,True,False))]
        with patch.object(r,"CP",(100,)):
            out=r.functional_summary(pd.DataFrame(records))
        self.assertEqual(set(out.cohort),{"all_seen","first100","first1000"})
        self.assertEqual(out.iloc[0].strict_numerator,1);self.assertEqual(out.iloc[0].numerator,2);self.assertEqual(out.iloc[0].new_nll_mean,1.5)

    def test_markdown_missing_not_numerical_zero(self):
        text=r.markdown(pd.DataFrame([{"x":None,"y":0.}]))
        self.assertIn("NOT_MEASURED",text);self.assertIn("|0|",text.replace("NOT_MEASURED|",""))


if __name__=="__main__":unittest.main()
