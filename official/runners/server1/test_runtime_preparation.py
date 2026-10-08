"""Real published row construction/validation without assets, SDK or GPU."""
import copy
import unittest

from official.experiments.prepare import build_matrix, digest, load_plan
from official.runners.server1.common import METHODS, validate_config
from official.runners.server1.prepare_runtime import QUALIFICATION_PLAN


class RuntimePreparationTests(unittest.TestCase):
    def configs(self):
        contract, profiles = load_plan()
        for row in build_matrix(contract, profiles):
            if row["model"] == "llama3" and row["method"] in METHODS:
                value = dict(row, schema="official-server1-runtime-v1",
                             qualification_plan=copy.deepcopy(QUALIFICATION_PLAN))
                value.pop("config_sha256", None)
                value["config_sha256"] = digest(value)
                yield value

    def test_six_actual_rows_accept_exact_native_oracle_plan(self):
        rows = list(self.configs())
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertIs(validate_config(row), row)

    def test_old_or_modified_oracle_plan_rejected(self):
        original = next(self.configs())
        for change in ("missing", "tolerance"):
            row = copy.deepcopy(original)
            if change == "missing":
                row["qualification_plan"].pop("native_reference_matched_B3")
            else:
                row["qualification_plan"]["native_reference_matched_B3"]["nll_abs_nats"] = 1e-3
            row.pop("config_sha256")
            row["config_sha256"] = digest(row)
            with self.assertRaisesRegex(ValueError, "QUALIFICATION_PLAN_CHANGED"):
                validate_config(row)

    def test_deferred_generation_requires_explicit_CF_override_and_exact_schedule(self):
        from official.runners.server1.common import DEFERRED_W20, CF_CHECKPOINT_AUTHORITY
        row = next(value for value in self.configs() if value["dataset"] == "cf")
        row.update(cf_W20_generation=DEFERRED_W20, scope_override=CF_CHECKPOINT_AUTHORITY)
        row["evaluation"]["generation"]["edited_endpoints"] = []
        row["evaluation"]["generation"]["deferred_to_checkpoint"] = 20
        row.pop("config_sha256"); row["config_sha256"] = digest(row)
        validate_config(row)
        for key,value in (("scope_override","wrong"),("dataset","zsre"),("cf_W20_generation","UNKNOWN")):
            wrong=copy.deepcopy(row); wrong[key]=value
            wrong.pop("config_sha256"); wrong["config_sha256"]=digest(wrong)
            with self.assertRaises(ValueError):validate_config(wrong)


if __name__ == "__main__":
    unittest.main()
