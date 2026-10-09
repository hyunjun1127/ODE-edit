"""CPU-only AST replay of pinned public loader + evaluator query construction.

Only public pure query generation is executed. The public inference callback
is replaced by a recorder; no LM forward, cache, edit or checkpoint load occurs.
"""
import ast
from copy import deepcopy
import hashlib
from itertools import chain
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .factual import _digest, _require
from .zsre_paper import build_queries

SOURCES = Path(__file__).with_name("zsre_public_sources")


def verify_sources():
    lock = json.loads((SOURCES / "lock.json").read_text())
    for name, member in lock["sources"].items():
        data = (SOURCES / name).read_bytes()
        _require(len(data) == member["bytes"] and
                 hashlib.sha256(data).hexdigest() == member["sha256"],
                 "ZSRE_PUBLIC_SOURCE_SHA_MISMATCH")
    return lock


def compare_queries(tokenizer, records, *, model_family):
    """Assert every native prompt string, target, input ID and scored target.

Records are model-independent official stream rows, not already token-expanded.
This proves CPU query/input parity only, never pretrained output parity.
"""
    lock = verify_sources()
    plan = build_queries(tokenizer, records, model_family=model_family)
    filename = "memit_eval_utils_zsre.py" if model_family == "gptj" else "alphaedit_eval_utils_zsre.py"
    module = ast.parse((SOURCES / filename).read_text())
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef)
              and n.name == "compute_rewrite_quality_zsre")
    loader = ast.parse((SOURCES / "zsre_dataset.py").read_text())
    neighbor_expr = next(value for node in ast.walk(loader) if isinstance(node, ast.Dict)
                         for key, value in zip(node.keys, node.values)
                         if isinstance(key, ast.Constant) and key.value == "neighborhood_prompts")
    loader_code = compile(ast.Expression(neighbor_expr), "pinned-public-loader", "eval")
    captured = []

    def capture(model, tok, prompts, targets):
        captured.append((prompts, targets))
        return [False] * len(prompts)

    namespace = dict(np=np, chain=chain, test_batch_prediction_acc=capture)
    exec(compile("from __future__ import annotations\n" + ast.unparse(fn),
                 "pinned-public-evaluator", "exec"), namespace)
    model = SimpleNamespace(config=SimpleNamespace(
        _name_or_path="llama" if model_family == "llama3" else model_family))
    cursor = 0
    for rec in records:
        neighbor = rec["neighborhood_prompts"][0]
        source_record = dict(loc=neighbor["prompt"][:-1], loc_ans=neighbor["target"])
        ans_toks = tokenizer(" " + source_record["loc_ans"])["input_ids"]
        native = deepcopy(rec)
        native["neighborhood_prompts"] = eval(loader_code, dict(record=source_record,
                                                               tok=tokenizer, ans_toks=ans_toks))
        captured.clear()
        namespace["compute_rewrite_quality_zsre"](model, tokenizer, native, None, None)
        _require(len(captured) == 2, "ZSRE_PUBLIC_CALLBACK_COUNT")
        for prompts, targets in captured:
            # Original test_batch_prediction_acc tokenizes the entire group;
            # check batch-tokenizer equivalence as well as pure text equality.
            inputs = tokenizer(prompts, padding=True, return_tensors="pt")
            correct = tokenizer(targets, padding=True, return_tensors="pt")["input_ids"]
            for i, (prompt, target) in enumerate(zip(prompts, targets)):
                query = plan["queries"][cursor]
                length = int(inputs["attention_mask"][i].sum())
                reference_ids = inputs["input_ids"][i, :length].tolist()
                reference_target = int(correct[i, 1 if model_family == "llama3" else 0])
                _require(query["prompt"] == prompt and query["target_text"] == target
                         and query["input_ids"] == reference_ids
                         and query["target_token_id"] == reference_target,
                         "ZSRE_PUBLIC_QUERY_MISMATCH_AT_" + str(cursor))
                cursor += 1
    _require(cursor == len(plan["queries"]), "ZSRE_PUBLIC_QUERY_COVERAGE")
    return dict(schema="zsre-public-query-parity-v1", status="PASS_CPU_QUERY_ONLY",
                model_family=model_family, requests=len(records), queries=cursor,
                token_denominators=plan["token_denominators"],
                stream_sha256=plan["stream_sha256"], query_sha256=plan["query_sha256"],
                oracle_lock_sha256=_digest(lock), model_forward_calls=0,
                input_mismatches=0, target_mismatches=0,
                actual_model_output_parity="NOT_MEASURED")
