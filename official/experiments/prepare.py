"""Build reviewable baseline configs and immutable input streams; never submit jobs."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "hparams"
MODELS = ("llama3", "qwen25", "gptj")
METHODS = ("FT", "MEMIT", "ALPHAEDIT", "ALPHAEDIT_BLUE", "MEMIT_FE", "SPHERE")
DATASETS = ("cf", "zsre")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Identical repeat preparation is allowed; changed sealed bytes are refused."""
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f"REFUSE_OVERWRITE: {path}")


def load_plan(path=DEFAULT_PLAN):
    path = Path(path)
    contract, profiles, sources = (read(path / name) for name in
                                   ("contract.json", "profiles.json", "sources.lock.json"))
    if set(contract["models"]) != set(MODELS) or tuple(contract["methods"]) != METHODS:
        raise ValueError("BASELINE_SCOPE_CHANGED")
    if set(profiles) != {f"{model}/{method}" for model in MODELS for method in METHODS}:
        raise ValueError("PROFILE_MATRIX_INCOMPLETE")
    if contract["stream"]["requests"] != 2000 or contract["stream"]["batch_size"] != 100:
        raise ValueError("STREAM_CONTRACT_CHANGED")
    if contract["qwen_blue_selection"]["L2_grid"] != [1, 10, 95]:
        raise ValueError("QWEN_GRID_CHANGED")
    for source in sources["files"]:
        if "local_copy" in source and file_sha(path / source["local_copy"]) != source["sha256"]:
            raise ValueError("UPSTREAM_HPARAMS_CHANGED: " + source["path"])
    for model in MODELS:
        for method in METHODS:
            p = profiles[f"{model}/{method}"]
            if read(path / method / (model + ".json")) != p["hparams"]:
                raise ValueError("EFFECTIVE_HPARAMS_CHANGED")
            if p["hparams"]["model_name"] != contract["models"][model]["model_id"]:
                raise ValueError("MODEL_VARIANT_MISMATCH")
            for source_id in p["source_ids"]:
                if source_id not in {s["id"] for s in sources["files"]}:
                    raise ValueError("PROFILE_SOURCE_MISSING")
    return contract, profiles


def build_matrix(contract, profiles):
    rows = []

    def cell(model, dataset, method, group="main", suffix="", overrides=None):
        hp = copy.deepcopy(profiles[f"{model}/{method}"]["hparams"])
        hp.update(overrides or {})
        value = dict(run_id=f"{model}-{dataset}-{method.lower()}{suffix}", group=group,
                     model=model, dataset=dataset, method=method, hparams=hp,
                     profile_sha256=digest(profiles[f"{model}/{method}"]),
                     model_identity=contract["models"][model],
                     stream=contract["stream"], precision=contract["precision"],
                     edit_seed=contract["edit_seed"], evaluation=contract["evaluation"],
                     checkpoint=contract["checkpoint"], contract_sha256=digest(contract),
                     execution_kind="new_chain", ready_to_submit=False)
        value["required_state"] = ["weights", "rng", "context_templates", "completed_batch",
                                   "evaluation_cursor", "identity"]
        if method in ("ALPHAEDIT", "ALPHAEDIT_BLUE", "SPHERE"):
            value["required_state"].append("cache_c")
        value["selection_dependency"] = None
        if model == "qwen25" and method == "ALPHAEDIT_BLUE" and hp["L2"] is None:
            value["selection_dependency"] = "qwen25-cf-blue-l2-grid"
            if group == "main" and dataset == "cf":
                value["execution_kind"] = "selected_grid_alias"
        value["config_sha256"] = digest(value)
        rows.append(value)

    for dataset in DATASETS:
        for model in MODELS:
            for method in METHODS:
                cell(model, dataset, method)
    for l2 in contract["qwen_blue_selection"]["L2_grid"]:
        cell("qwen25", "cf", "ALPHAEDIT_BLUE", "qwen_grid", f"-l2-{l2}", {"L2": l2})
    for method in ("ALPHAEDIT", "ALPHAEDIT_BLUE"):
        cell("qwen25", "cf", method, "qwen_control", "-clamp075", {"clamp_norm_factor": 0.75})
    return rows


def materialize_matrix(plan, output):
    contract, profiles = load_plan(plan)
    rows = build_matrix(contract, profiles)
    output = Path(output)
    for row in rows:
        write_new(output / "configs" / f"{row['run_id']}.json", row)
    fields = ["run_id", "group", "model", "dataset", "method", "execution_kind",
              "selection_dependency", "config_sha256"]
    # This CSV is a derived index, not an execution receipt.
    output.mkdir(parents=True, exist_ok=True)
    with (output / "matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    summary = dict(main_rows=36, qwen_grid_rows=3, qwen_control_rows=2,
                   logical_rows=len(rows), physical_chains=40,
                   selected_grid_aliases=1, W0_model_dataset_evaluations=6,
                   ready_to_submit=False, status="CONFIGURATION_PREPARED")
    write_new(output / "matrix-summary.json", summary)
    return summary


def normalize_records(raw, dataset, count=2000):
    if not isinstance(raw, list) or len(raw) < count:
        raise ValueError("INSUFFICIENT_STREAM_RECORDS")
    records = []
    for index, source in enumerate(raw[:count]):
        if dataset == "cf":
            row = copy.deepcopy(source)
            rewrite = row["requested_rewrite"]
            if not all(row.get(key) for key in ("paraphrase_prompts", "neighborhood_prompts")):
                raise ValueError("MISSING_CF_EVALUATION_PROMPTS")
        elif dataset == "zsre":
            subject, prompt = source["subject"], source["src"]
            if not subject or prompt.count(subject) != 1:
                raise ValueError(f"AMBIGUOUS_ZSRE_SUBJECT_AT_ROW_{index}")
            if not source["answers"] or not source["rephrase"] or not source["loc_ans"]:
                raise ValueError("MISSING_ZSRE_TARGET_OR_EVALUATION")
            if not source["loc"].startswith("nq question: "):
                raise ValueError("ZSRE_NEIGHBORHOOD_FORMAT")
            rewrite = dict(prompt=prompt.replace(subject, "{}"), subject=subject,
                           target_new={"str": source["answers"][0]},
                           target_true={"str": "<|endoftext|>"})
            # Token-level expansion and W0 predictions are model-dependent.
            # Do not disguise loc_ans as the unobserved W0 prediction.
            row = dict(case_id=index, requested_rewrite=rewrite,
                       paraphrase_prompts=[source["rephrase"]],
                       neighborhood_prompts=[dict(prompt=source["loc"] + "?",
                                                  target=source["loc_ans"])],
                       neighborhood_expansion="TEACHER_FORCED_TOKEN_PREFIXES_AT_EVALUATION",
                       specificity_W0_predictions=None, attribute_prompts=[], generation_prompts=[])
        else:
            raise ValueError("UNKNOWN_DATASET")
        if not isinstance(row["case_id"], int) or isinstance(row["case_id"], bool):
            raise ValueError("CASE_ID_TYPE")
        if rewrite["prompt"].count("{}") != 1 or not rewrite["subject"]:
            raise ValueError("SUBJECT_PLACEHOLDER")
        if not rewrite["target_new"]["str"]:
            raise ValueError("EMPTY_TARGET")
        row["occurrence_index"] = index + 1
        records.append(row)
    if len({r["case_id"] for r in records}) != len(records):
        raise ValueError("DUPLICATE_CASE_IDS")
    return records


def prepare_stream(source, dataset, output, expected_source_sha=None, count=2000, batch_size=100):
    if count <= 0 or batch_size <= 0 or count % batch_size:
        raise ValueError("INVALID_BATCH_BOUNDARIES")
    before = file_sha(source)
    if expected_source_sha and before != expected_source_sha:
        raise ValueError("SOURCE_SHA_MISMATCH")
    records = normalize_records(read(source), dataset, count)
    if file_sha(source) != before:
        raise ValueError("SOURCE_CHANGED_WHILE_READING")
    output = Path(output)
    write_new(output / f"{dataset}-stream.json", records)
    ids = [r["case_id"] for r in records]
    receipt = dict(dataset=dataset, source_sha256=before,
                   stream_sha256=file_sha(output / f"{dataset}-stream.json"),
                   ordered_case_ids_sha256=digest(ids), requests=count, batch_size=batch_size,
                   order="FIRST_N_IN_EXISTING_FILE_ORDER_NO_SHUFFLE_NO_FILTER",
                   target_mapping="requested_rewrite.target_new" if dataset == "cf" else "answers[0]",
                   batches=[dict(batch=i // batch_size + 1, start_inclusive=i,
                                 end_exclusive=i + batch_size,
                                 ordered_case_ids_sha256=digest(ids[i:i + batch_size]))
                            for i in range(0, count, batch_size)])
    write_new(output / f"{dataset}-stream.lock.json", receipt)
    return receipt


def audit_tokenizer(stream, tokenizer, output, model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True, use_fast=True)
    if not tok.is_fast:
        raise ValueError("OFFSET_AUDIT_REQUIRES_FAST_TOKENIZER")
    records = read(stream)
    rows = []
    for record in records:
        rw = record["requested_rewrite"]
        template, subject = rw["prompt"], rw["subject"]
        begin = template.index("{}")
        prompt = template.format(subject)
        encoded = tok(prompt, return_offsets_mapping=True)
        indices = [i for i, (start, end) in enumerate(encoded["offset_mapping"])
                   if end > begin and start < begin + len(subject) and end > start]
        if not indices:
            raise ValueError("SUBJECT_TOKEN_SPAN_EMPTY")
        # Native repr_tools locates the last subject token in prefix+subject.
        # Some byte/BPE tokenizers merge subject suffix and following punctuation;
        # keep the offset span separately and report that disagreement explicitly.
        prefix_ids = tok(template[:begin] + subject)["input_ids"]
        lookup = len(prefix_ids) - 1
        rows.append(dict(case_id=record["case_id"], occurrence_index=record["occurrence_index"],
                         batch=(record["occurrence_index"] - 1) // 100 + 1,
                         lookup=lookup, offset_lookup=indices[-1],
                         lookup_offset_agree=lookup == indices[-1],
                         subject_token_count=len(indices),
                         target_token_count=len(tok(" " + rw["target_new"]["str"],
                                                    add_special_tokens=False)["input_ids"])))
    zero = [row for row in rows if row["lookup"] == 0]
    identity_files = {p.name: file_sha(p) for p in Path(tokenizer).iterdir()
                      if p.is_file() and p.name in ("tokenizer.json", "tokenizer_config.json",
                         "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json")}
    receipt = dict(model=model, stream_sha256=file_sha(stream), requests=len(rows),
                   lookup_zero_count=len(zero),
                   lookup_zero_batches=sorted({r["batch"] for r in zero}),
                   lookup_offset_disagreements=sum(not r["lookup_offset_agree"] for r in rows),
                   multiple_target_tokens=sum(r["target_token_count"] > 1 for r in rows),
                   add_bos_token=getattr(tok, "add_bos_token", None),
                   bos_token_id=tok.bos_token_id, padding_side=tok.padding_side,
                   tokenizer_class=type(tok).__name__, model_forward_calls=0)
    receipt["tokenizer_files_sha256"] = identity_files
    receipt["tokenizer_sha256"] = digest(identity_files)
    write_new(output, dict(summary=receipt, rows=rows))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "matrix"):
        sub = commands.add_parser(name)
        sub.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
        if name == "matrix":
            sub.add_argument("--output", type=Path, required=True)
    stream = commands.add_parser("stream")
    stream.add_argument("--source", type=Path, required=True)
    stream.add_argument("--dataset", choices=DATASETS, required=True)
    stream.add_argument("--output", type=Path, required=True)
    stream.add_argument("--expected-source-sha")
    audit = commands.add_parser("audit-tokenizer")
    for key in ("stream", "tokenizer", "output"):
        audit.add_argument("--" + key, type=Path, required=True)
    audit.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        load_plan(args.plan)
        result = dict(status="CONFIGURATION_VALID", ready_to_submit=False)
    elif args.command == "matrix":
        result = materialize_matrix(args.plan, args.output)
    elif args.command == "stream":
        expected = args.expected_source_sha or read(
            DEFAULT_PLAN / f"{args.dataset}-stream.lock.json")["source_sha256"]
        result = prepare_stream(args.source, args.dataset, args.output, expected)
    else:
        result = audit_tokenizer(args.stream, args.tokenizer, args.output, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
