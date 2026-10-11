"""Qwen AlphaEdit-BLUE L2 selection on rent: the CF grid winner becomes the zsRE row's L2.

contract.json ``qwen_blue_selection``: the CF grid rows ``qwen25-cf-alphaedit_blue-l2-{L2}`` run as ordinary chains;
the winner has the highest unrounded CF W20 Score, then Specificity, Efficacy, Generalization, then the lowest L2
(server3 submit.choose_blue_winner). ``select`` reads the completed rent grid chains of one frozen on-main commit and
writes the derived zsRE config (the canonical ``qwen25-zsre-alphaedit_blue`` row with only ``hparams.L2`` filled) and
the selection proof. ``selected_configuration`` accepts that derived config only with its proof (server4
qwen_run._original_validate_config rules). No model, GPU or tracker.
"""
import argparse
import copy
import json
import math
from pathlib import Path

from official.experiments.prepare import build_matrix, digest, file_sha, load_plan, read, write_new

ZSRE_RUN = "qwen25-zsre-alphaedit_blue"
GRID_RUN = "qwen25-cf-alphaedit_blue-l2-{}"
SELECTION_DEPENDENCY = "qwen25-cf-blue-l2-grid"
KEYS = ("Score", "Specificity", "Efficacy", "Generalization")
PROOF_NAME = "qwen25-blue-selection.json"


def require(value, code):
    if not value:
        raise ValueError(code)


def member(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path))


def rows():
    contract, profiles = load_plan()
    return contract, {row["run_id"]: row for row in build_matrix(contract, profiles)}


def choose_winner(scored, grid):
    require({row["l2"] for row in scored} == set(grid) and len(scored) == len(grid), "BLUE_GRID_INCOMPLETE")
    for row in scored:
        require(all(type(row["metrics"].get(k)) in (int, float) and math.isfinite(row["metrics"][k]) for k in KEYS),
                "BLUE_GRID_METRICS_INVALID")
    return max(scored, key=lambda row: tuple(row["metrics"][k] for k in KEYS) + (-row["l2"],))


def grid_result(grid_root, run_id, expected_sha, source):
    """The one completed chain of a grid row under ``grid_root`` and its unrounded W20 CF summary."""
    attempts = sorted((Path(grid_root) / "runs" / "qwen25" / run_id).glob("chain-job*/result.json"))
    complete = [path for path in attempts if read(path).get("status") == "SCIENTIFIC_COMPLETE"]
    require(len(complete) == 1, "BLUE_GRID_ATTEMPT_NOT_UNIQUE:" + run_id)
    result = read(complete[0])
    identity = result["checkpoint_identity"]
    require(result["run_id"] == run_id and result["dataset"] == "cf" and result["batches"] == 20
            and identity["config_sha256"] == expected_sha
            and result["code_commit"] == identity["code_commit"] == source["code_commit"]
            and result["official_tree_sha256"] == source["official_tree_sha256"], "BLUE_GRID_RESULT_IDENTITY:" + run_id)
    w20 = result["factual_endpoints"]["W20"]
    require(member(w20["path"]) == w20, "BLUE_GRID_W20_CHANGED:" + run_id)
    summary = read(w20["path"])["summary"]
    return dict(run_id=run_id, result=member(complete[0]), w20=w20, metrics={k: summary[k] for k in KEYS})


def select(grid_root, output):
    """Choose the CF winner from the completed rent grid and write the derived zsRE config and its proof."""
    contract, matrix = rows()
    source = read(Path(grid_root) / "source.json")
    require(source["on_main"] is True, "BLUE_GRID_SOURCE_NOT_ON_MAIN")
    grid = contract["qwen_blue_selection"]["L2_grid"]
    scored = []
    for l2 in grid:
        run_id = GRID_RUN.format(l2)
        row = matrix[run_id]
        require(row["hparams"]["L2"] == l2 and row["selection_dependency"] is None, "BLUE_GRID_ROW:" + run_id)
        scored.append(dict(grid_result(grid_root, run_id, row["config_sha256"], source), l2=l2))
    winner = choose_winner(scored, grid)
    original = copy.deepcopy(matrix[ZSRE_RUN])
    require(original["hparams"]["L2"] is None and original["selection_dependency"] == SELECTION_DEPENDENCY,
            "ZSRE_BLUE_ALREADY_SELECTED")
    original["hparams"]["L2"] = winner["l2"]
    original["config_sha256"] = digest({k: v for k, v in original.items() if k != "config_sha256"})
    output = Path(output)
    config_path = output / (ZSRE_RUN + ".json")
    write_new(config_path, original)
    proof = dict(status="CF_BLUE_GRID_SELECTED", server="rent", criteria=list(KEYS) + ["lowest_L2"],
                 grid_source=dict(code_commit=source["code_commit"], official_tree_sha256=source["official_tree_sha256"],
                                  on_main=True, root=str(Path(grid_root).resolve())),
                 grid=scored, winner_run_id=winner["run_id"], chosen_L2=winner["l2"],
                 selected_zsre_config=str(config_path.resolve()), selected_zsre_config_sha256=file_sha(config_path))
    write_new(output / PROOF_NAME, proof)
    return proof


def selected_configuration(path):
    """The derived zsRE BLUE config, accepted only next to its selection proof."""
    path = Path(path)
    config = read(path)
    proof = read(path.parent / PROOF_NAME)
    require(proof["status"] == "CF_BLUE_GRID_SELECTED" and proof["selected_zsre_config_sha256"] == file_sha(path)
            and proof["chosen_L2"] == config["hparams"].get("L2"), "BLUE_SELECTION_PROOF")
    contract, matrix = rows()
    original = matrix[ZSRE_RUN]
    require(config["run_id"] == ZSRE_RUN and original["hparams"]["L2"] is None
            and config["hparams"].get("L2") in contract["qwen_blue_selection"]["L2_grid"], "CONFIG_NOT_CANONICAL")
    require(config["config_sha256"] == digest({k: v for k, v in config.items() if k != "config_sha256"}), "CONFIG_DIGEST")
    require({k for k in original if original[k] != config.get(k)} | (set(config) - set(original))
            <= {"hparams", "config_sha256"}, "DERIVED_BLUE_CONFIG_CHANGED_OTHER_FIELDS")
    require({k: v for k, v in config["hparams"].items() if k != "L2"} ==
            {k: v for k, v in original["hparams"].items() if k != "L2"}, "DERIVED_BLUE_HPARAMS_CHANGED")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, required=True,
                        help="frozen on-main run root holding the three CF grid chains")
    parser.add_argument("--output", type=Path, required=True, help="selected-configs folder of the zsRE run root")
    args = parser.parse_args()
    proof = select(args.grid_root, args.output)
    print(json.dumps({k: proof[k] for k in ("status", "winner_run_id", "chosen_L2", "selected_zsre_config_sha256")}))


if __name__ == "__main__":
    main()
