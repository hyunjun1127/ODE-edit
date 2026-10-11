"""CPU only: rent Qwen BLUE L2 selection from a synthetic completed grid."""
import json
from pathlib import Path
import tempfile
import unittest

from official.experiments.prepare import digest, file_sha
from official.runners.rent import blue_selection as sel

GRID = (1, 10, 95)


def metrics(score, spe=60.0, eff=99.0, gen=97.0):
    return dict(Score=score, Specificity=spe, Efficacy=eff, Generalization=gen)


class BlueSelection(unittest.TestCase):
    def grid_root(self, scores, on_main=True):
        root = Path(tempfile.mkdtemp())
        source = dict(code_commit="c" * 40, official_tree_sha256="t" * 64, on_main=on_main)
        (root / "source.json").write_text(json.dumps(source))
        _, matrix = sel.rows()
        for l2, values in scores.items():
            run_id = sel.GRID_RUN.format(l2)
            out = root / "runs" / "qwen25" / run_id / "chain-job111111"
            (out / "factual").mkdir(parents=True)
            w20 = out / "factual" / "W20.json"
            w20.write_text(json.dumps(dict(summary=dict(values, requests=2000))))
            result = dict(status="SCIENTIFIC_COMPLETE", run_id=run_id, dataset="cf", batches=20,
                          code_commit=source["code_commit"], official_tree_sha256=source["official_tree_sha256"],
                          checkpoint_identity=dict(config_sha256=matrix[run_id]["config_sha256"],
                                                   code_commit=source["code_commit"]),
                          factual_endpoints=dict(W20=sel.member(w20)))
            (out / "result.json").write_text(json.dumps(result))
        return root

    def test_winner_follows_contract_order(self):
        rows = [dict(l2=l2, metrics=metrics(s)) for l2, s in ((1, 67.7), (10, 79.9), (95, 81.2))]
        self.assertEqual(sel.choose_winner(rows, GRID)["l2"], 95)
        tie = [dict(l2=1, metrics=metrics(80.0, spe=60.0)), dict(l2=10, metrics=metrics(80.0, spe=61.0)),
               dict(l2=95, metrics=metrics(80.0, spe=61.0))]
        self.assertEqual(sel.choose_winner(tie, GRID)["l2"], 10)  # equal through Generalization: the lowest L2
        with self.assertRaisesRegex(ValueError, "BLUE_GRID_INCOMPLETE"):
            sel.choose_winner(rows[:2], GRID)

    def test_select_writes_derived_zsre_row_and_proof(self):
        root = self.grid_root({1: metrics(67.76), 10: metrics(79.91), 95: metrics(83.0)})
        out = Path(tempfile.mkdtemp()) / "selected-configs"
        proof = sel.select(root, out)
        self.assertEqual((proof["chosen_L2"], proof["winner_run_id"]), (95, "qwen25-cf-alphaedit_blue-l2-95"))
        path = out / (sel.ZSRE_RUN + ".json")
        config = sel.selected_configuration(path)
        _, matrix = sel.rows()
        original = matrix[sel.ZSRE_RUN]
        self.assertEqual(config["hparams"], dict(original["hparams"], L2=95))
        self.assertEqual(config["config_sha256"], digest({k: v for k, v in config.items() if k != "config_sha256"}))
        self.assertEqual(proof["selected_zsre_config_sha256"], file_sha(path))

    def test_selection_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "BLUE_GRID_SOURCE_NOT_ON_MAIN"):
            sel.select(self.grid_root({1: metrics(1), 10: metrics(2), 95: metrics(3)}, on_main=False),
                       Path(tempfile.mkdtemp()))
        with self.assertRaisesRegex(ValueError, "BLUE_GRID_ATTEMPT_NOT_UNIQUE"):
            sel.select(self.grid_root({1: metrics(1), 10: metrics(2)}), Path(tempfile.mkdtemp()))
        root = self.grid_root({1: metrics(1), 10: metrics(2), 95: metrics(3)})
        out = Path(tempfile.mkdtemp())
        sel.select(root, out)
        path = out / (sel.ZSRE_RUN + ".json")
        config = json.loads(path.read_text())
        tampered = dict(config, hparams=dict(config["hparams"], L2=50))
        tampered["config_sha256"] = digest({k: v for k, v in tampered.items() if k != "config_sha256"})
        path.write_text(json.dumps(tampered))
        with self.assertRaisesRegex(ValueError, "BLUE_SELECTION_PROOF"):
            sel.selected_configuration(path)
        proof = json.loads((out / sel.PROOF_NAME).read_text())
        (out / sel.PROOF_NAME).write_text(json.dumps(dict(proof, chosen_L2=50, selected_zsre_config_sha256=file_sha(path))))
        with self.assertRaisesRegex(ValueError, "CONFIG_NOT_CANONICAL"):
            sel.selected_configuration(path)
        other = dict(config, dataset="cf")
        other["config_sha256"] = digest({k: v for k, v in other.items() if k != "config_sha256"})
        path.write_text(json.dumps(other))
        (out / sel.PROOF_NAME).write_text(json.dumps(dict(proof, selected_zsre_config_sha256=file_sha(path))))
        with self.assertRaisesRegex(ValueError, "DERIVED_BLUE_CONFIG_CHANGED_OTHER_FIELDS"):
            sel.selected_configuration(path)


if __name__ == "__main__":
    unittest.main()
