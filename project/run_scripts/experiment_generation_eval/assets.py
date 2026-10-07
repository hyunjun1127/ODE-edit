"""Frozen public CounterFact generation references, CPU only and no downloads.

The caller obtains the three policy-authorized originals separately. Preparation
validates them in place and writes create-once local metadata/READY. Nothing here
fits TF-IDF, loads a model, edits an existing cache or uploads reference text.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA = "counterfact-generation-reference-assets-v1"
FILES = ("attribute_snippets.json", "idf.npy", "tfidf_vocab.json")
URLS = {name: "https://memit.baulab.info/data/dsets/" + name for name in FILES}
AUTHORIZED_DOWNLOAD_SIZES = dict(zip(FILES, (926285719, 11042168, 31639072)))
TAB_FILES = ("collocations.tab", "sent_starters.txt", "abbrev_types.txt", "ortho_context.tab")
NATIVE_ROOT = Path("/mnt/raid5/janghj/CAKE")


class AssetError(RuntimeError):
    """Technical prerequisite failure, not a numerical score or fabricated zero."""
    def __init__(self, code, detail=""):
        self.code = code
        self.status = "ASSET_NOT_AVAILABLE"
        self.reason = "tokenizer_not_available" if code.startswith("TOKENIZER_") else "asset_not_available"
        super().__init__(code + (":" + detail if detail else ""))


def require(condition, code):
    if not condition:
        raise AssetError(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def member(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "ASSET_REGULAR_FILE")
    before = path.stat()
    value = sha(path)
    after = path.stat()
    require((before.st_size, before.st_ino, before.st_mtime_ns) ==
            (after.st_size, after.st_ino, after.st_mtime_ns), "ASSET_CHANGED_DURING_HASH")
    return dict(path=str(path), bytes=after.st_size, sha256=value)


def verify(row, *, overrides=None, name=None):
    require(isinstance(row, dict) and type(row.get("bytes")) is int
            and row["bytes"] >= 0 and type(row.get("sha256")) is str
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]), "ASSET_MEMBER_SCHEMA")
    path = Path(overrides[name] if overrides and name in overrides else row["path"])
    actual = member(path)
    require(actual["bytes"] == row["bytes"] and actual["sha256"] == row["sha256"],
            "ASSET_MEMBER_IDENTITY")
    return path


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "ASSET_DUPLICATE_JSON_KEY")
        value[key] = item
    return value


def json_read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_pairs)


def _array_items(path, chunk_chars=1 << 20):
    """Stream the large top-level snippets list; no whole-GB JSON string copy.

    JSONDecoder still parses each original entry completely and in order. This
    is just framing, not a lenient parser or a schema/byte substitution.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_pairs)
    with Path(path).open(encoding="utf-8") as stream:
        buffer, pos, eof = "", 0, False

        def more():
            nonlocal buffer, pos, eof
            added = stream.read(chunk_chars)
            buffer = buffer[pos:] + added
            pos = 0
            eof = not added or stream.tell() == os.fstat(stream.fileno()).st_size

        def ws():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                more()

        more(); ws()
        require(pos < len(buffer) and buffer[pos] == "[", "SNIPPETS_TOP_LEVEL_LIST")
        pos += 1
        first = True
        while True:
            ws()
            require(pos < len(buffer), "SNIPPETS_JSON_TRUNCATED")
            if buffer[pos] == "]":
                pos += 1
                break
            if not first:
                require(buffer[pos] == ",", "SNIPPETS_JSON_SEPARATOR")
                pos += 1; ws()
                require(pos < len(buffer) and buffer[pos] != "]", "SNIPPETS_JSON_TRAILING_COMMA")
            while True:
                try:
                    value, stop = decoder.raw_decode(buffer, pos)
                    pos = stop
                    break
                except json.JSONDecodeError as error:
                    if eof:
                        raise AssetError("SNIPPETS_JSON_INVALID") from error
                    more()
            yield value
            first = False
        ws()
        require(pos == len(buffer) and eof, "SNIPPETS_JSON_TRAILING_DATA")


def read_snippets(path):
    snippets, entries, samples = {}, 0, 0
    for entry in _array_items(path):
        require(isinstance(entry, dict) and all(type(entry.get(k)) is str and entry[k]
                for k in ("relation_id", "target_id")) and type(entry.get("samples")) is list,
                "SNIPPETS_RELATION_TARGET_SAMPLES_SCHEMA")
        key = (entry["relation_id"], entry["target_id"])
        texts = snippets.setdefault(key, [])
        for sample in entry["samples"]:
            require(isinstance(sample, dict) and type(sample.get("text")) is str,
                    "SNIPPETS_SAMPLE_TEXT_SCHEMA")
            texts.append(sample["text"])
            samples += 1
        entries += 1
    return snippets, dict(entries=entries, relation_target_pairs=len(snippets), samples=samples,
                         selection="all samples per exact relation_id/target_new.id, original order")


def read_tfidf(idf_path, vocab_path):
    vocabulary = json_read(vocab_path)
    require(type(vocabulary) is dict and bool(vocabulary)
            and all(type(token) is str and token and type(index) is int
                    for token, index in vocabulary.items()), "TFIDF_TOKEN_INDEX_SCHEMA")
    require(set(vocabulary.values()) == set(range(len(vocabulary))), "TFIDF_DENSE_UNIQUE_INDICES")
    try:
        idf = np.load(idf_path, allow_pickle=False)
    except (ValueError, OSError) as error:
        raise AssetError("TFIDF_IDF_SAFE_NPY") from error
    require(isinstance(idf, np.ndarray) and idf.ndim == 1 and len(idf) == len(vocabulary)
            and np.issubdtype(idf.dtype, np.number) and not np.iscomplexobj(idf)
            and np.isfinite(idf).all(), "TFIDF_FINITE_1D_LENGTH")
    return vocabulary, idf


def fixed_vectorizer(vocabulary, idf):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    # sklearn's public setter initializes its transformer and fixed vocabulary.
    # No fit()/fit_transform(), even a dummy fit, is needed in this runtime.
    vectorizer.idf_ = idf
    require(vectorizer.fixed_vocabulary_ and vectorizer.vocabulary_ == vocabulary
            and np.array_equal(vectorizer.idf_, idf), "TFIDF_FIXED_BINDING")
    return vectorizer


def dependency_versions():
    return {key: importlib.metadata.version(distribution) for key, distribution in
            (("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("nltk", "nltk"))}


def nltk_binding():
    import nltk
    version = importlib.metadata.version("nltk")
    names = nltk.tokenize.sent_tokenize.__code__.co_names
    require("_get_punkt_tokenizer" in names or "load" in names, "TOKENIZER_IMPLEMENTATION_UNSUPPORTED")
    kind = "punkt_tab" if "_get_punkt_tokenizer" in names else "punkt"
    identifier = "tokenizers/punkt_tab/english/" if kind == "punkt_tab" else "tokenizers/punkt/english.pickle"
    try:
        located = Path(str(nltk.data.find(identifier)))
        required = [member(located / name) for name in TAB_FILES] if kind == "punkt_tab" else [member(located)]
    except (LookupError, OSError, AssetError) as error:
        raise AssetError("TOKENIZER_RESOURCE_NOT_AVAILABLE") from error
    # Inventory already installed resources only; old NLTK's find() rewrites
    # punkt_tab incorrectly, so identify its physical sibling without loading it.
    resource_root = Path(*located.parts[:located.parts.index("tokenizers")])
    inactive = []
    other = resource_root / "tokenizers" / "punkt_tab" / "english"
    if kind == "punkt" and all((other / name).is_file() for name in TAB_FILES):
        inactive = [member(other / name) for name in TAB_FILES]
    modules = ("__init__.py", "data.py", "compat.py", "tokenize/__init__.py",
               "tokenize/punkt.py", "tokenize/destructive.py", "tokenize/treebank.py")
    package = Path(nltk.__file__).parent
    sources = [dict(member(package / name), relative=name) for name in modules if (package / name).is_file()]
    return dict(nltk_version=version, language="english", required_family=kind,
                identifier=identifier, required_resources=required,
                available_inactive_resources=inactive, source_members=sources,
                word_tokenize="nltk.word_tokenize(text,language='english',preserve_line=False)",
                replacement_regex=False, downloaded_resources=False)


def _nltk_callable(binding):
    import nltk
    require(importlib.metadata.version("nltk") == binding["nltk_version"]
            and binding["language"] == "english", "TOKENIZER_RUNTIME_VERSION_IDENTITY")
    actual = nltk_binding()
    require(actual["required_family"] == binding["required_family"]
            and [(r["bytes"], r["sha256"]) for r in actual["required_resources"]] ==
                [(r["bytes"], r["sha256"]) for r in binding["required_resources"]],
            "TOKENIZER_RESOLVED_RESOURCE_IDENTITY")
    # Peer hosts resolve their own exact same public runtime/resource content;
    # a server1 absolute path is provenance, not a required peer mount alias.
    source_key = lambda row: (row["relative"], row["bytes"], row["sha256"])
    require(sorted(map(source_key, actual["source_members"])) ==
            sorted(map(source_key, binding["source_members"])), "TOKENIZER_SOURCE_IDENTITY")
    return lambda text: nltk.word_tokenize(text, language="english", preserve_line=False)


def _key(record):
    rewrite = record.get("requested_rewrite", record)
    target = rewrite.get("target_new", {})
    return rewrite.get("relation_id"), target.get("id") if isinstance(target, dict) else None


@dataclass
class Assets:
    vectorizer: object
    snippets: dict
    manifest: dict
    sha: str
    word_tokenize: object

    def snippets_for(self, relation_id, target_id):
        if type(relation_id) is not str or type(target_id) is not str:
            return []
        return list(self.snippets.get((relation_id, target_id), ()))

    def reference_for(self, relation_id, target_id):
        texts = self.snippets_for(relation_id, target_id)
        return dict(texts=texts, reason=None if texts else "missing_reference")

    def coverage(self, records):
        rows = []
        for ordinal, record in enumerate(records):
            relation, target = _key(record)
            texts = self.snippets_for(relation, target)
            prompts = record.get("generation_prompts")
            available_prompts = type(prompts) is list and bool(prompts) and all(type(p) is str for p in prompts)
            rows.append(dict(occurrence=ordinal, case_id=record.get("case_id"), relation_id=relation,
                target_new_id=target, reference_sample_count=len(texts),
                reference_reason=None if texts else "missing_reference",
                generation_prompt_count=len(prompts) if available_prompts else 0,
                generation_reason=None if available_prompts else "missing_generation_prompts"))
        summary = dict(planned_count=len(rows), reference_available_count=sum(r["reference_reason"] is None for r in rows),
            missing_reference_count=sum(r["reference_reason"] is not None for r in rows),
            generation_prompts_available_count=sum(r["generation_reason"] is None for r in rows),
            missing_generation_prompts_count=sum(r["generation_reason"] is not None for r in rows),
            ordered_case_ids_sha256=digest([r["case_id"] for r in rows]),
            occurrence_identity_sha256=digest([[r[k] for k in ("occurrence", "case_id", "relation_id", "target_new_id")] for r in rows]),
            no_model_observation=True, coverage_not_metric_validity=True)
        return dict(summary=summary, rows=rows)


def _identity(manifest):
    return dict(schema=SCHEMA,
        files={name: dict(bytes=manifest["files"][name]["bytes"], sha256=manifest["files"][name]["sha256"])
               for name in FILES}, versions=manifest["versions"],
        tokenizer=dict(nltk_version=manifest["tokenizer"]["nltk_version"],
            required_family=manifest["tokenizer"]["required_family"], language="english",
            required_resources=[dict(bytes=r["bytes"], sha256=r["sha256"])
                for r in manifest["tokenizer"]["required_resources"]]),
        fixed_vectorizer=manifest["fixed_vectorizer"])


def _manifest(value):
    overrides = {}
    if isinstance(value, (str, Path)):
        value = json_read(value)
    if isinstance(value, dict) and value.get("schema") != SCHEMA:
        selected = value.get("generation_assets", value.get("reference_assets", value.get("manifest")))
        overrides = value.get("asset_paths", {})
        require(selected is not None, "ASSET_CONFIG_MANIFEST_REQUIRED")
        value = json_read(verify(selected)) if isinstance(selected, dict) and "path" in selected else _manifest(selected)[0]
    require(isinstance(value, dict) and value.get("schema") == SCHEMA
            and value.get("status") == "READY" and set(value.get("files", {})) == set(FILES), "ASSET_MANIFEST_READY_SCHEMA")
    require(value["identity_sha256"] == digest(_identity(value)), "ASSET_MANIFEST_IDENTITY")
    return value, overrides


def load_assets(config_or_manifest):
    manifest, overrides = _manifest(config_or_manifest)
    require(dependency_versions() == manifest["versions"], "ASSET_SCORING_RUNTIME_VERSIONS")
    paths = {name: verify(manifest["files"][name], overrides=overrides, name=name) for name in FILES}
    vocabulary, idf = read_tfidf(paths["idf.npy"], paths["tfidf_vocab.json"])
    snippets, schema = read_snippets(paths["attribute_snippets.json"])
    require(schema == manifest["snippet_schema"] and len(vocabulary) == manifest["vocabulary_size"]
            and list(idf.shape) == manifest["idf_shape"] and str(idf.dtype) == manifest["idf_dtype"],
            "ASSET_LOADED_SCHEMA_IDENTITY")
    return Assets(fixed_vectorizer(vocabulary, idf), snippets, manifest,
                  manifest["identity_sha256"], _nltk_callable(manifest["tokenizer"]))


def _write_new(path, value):
    path = Path(path)
    encoded = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def prepare_assets(source_dir, output_dir, records=None, *, expected_sizes=None):
    """Adopt downloaded originals in place; no implicit copy/download/refit."""
    source, out = Path(source_dir).absolute(), Path(output_dir).absolute()
    require(not out.exists(), "ASSET_PREPARATION_CREATE_ONCE")
    files = {name: dict(member(source / name), source_url=URLS[name], copied=False) for name in FILES}
    if expected_sizes is not None:
        require(set(expected_sizes) == set(FILES) and all(files[name]["bytes"] == expected_sizes[name] for name in FILES),
                "ASSET_AUTHORIZED_DOWNLOAD_SIZES")
    vocabulary, idf = read_tfidf(source / "idf.npy", source / "tfidf_vocab.json")
    snippets, schema = read_snippets(source / "attribute_snippets.json")
    binding = nltk_binding()
    versions = dependency_versions()
    manifest = dict(schema=SCHEMA, status="READY", files=files, versions=versions,
        tokenizer=binding, snippet_schema=schema, vocabulary_size=len(vocabulary),
        idf_shape=list(idf.shape), idf_dtype=str(idf.dtype),
        fixed_vectorizer=dict(vocabulary="original dense fixed token->index", idf="original finite saved idf",
            fit_calls=0, refit=False, class_name="sklearn.feature_extraction.text.TfidfVectorizer",
            initialization="public idf_ setter, no dummy fit", defaults="native sklearn TfidfVectorizer defaults"),
        source_members=[member(Path(__file__))], prepared_python=dict(executable=sys.executable, version=sys.version.split()[0]),
        native_reference_sources=[member(NATIVE_ROOT / "dsets" / name) for name in
                                  ("attr_snippets.py", "tfidf_stats.py")],
        downloads_by_loader=False, raw_reference_Git_W_B=False, no_model=True)
    manifest["identity_sha256"] = digest(_identity(manifest))
    assets = Assets(fixed_vectorizer(vocabulary, idf), snippets, manifest,
                    manifest["identity_sha256"], _nltk_callable(binding))
    coverage = assets.coverage(records) if records is not None else None
    if coverage is not None:
        manifest["coverage"] = coverage["summary"]
    out.mkdir(parents=True)
    if coverage is not None:
        _write_new(out / "coverage-local.json", coverage)
    _write_new(out / "manifest.json", manifest)
    _write_new(out / "READY.json", dict(schema=SCHEMA, status="READY", identity_sha256=assets.sha,
        manifest=member(out / "manifest.json"), coverage=None if coverage is None else coverage["summary"],
        no_model_observation=True, no_tfidf_refit=True, no_raw_reference_in_receipt=True))
    return out / "manifest.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--cases", type=int, default=2000)
    args = parser.parse_args()
    records = json_read(args.dataset)[:args.cases] if args.dataset else None
    manifest = prepare_assets(args.source, args.out, records, expected_sizes=AUTHORIZED_DOWNLOAD_SIZES)
    ready = json_read(args.out / "READY.json")
    print(json.dumps(dict(manifest=str(manifest), status=ready["status"], identity_sha256=ready["identity_sha256"],
                         coverage=ready["coverage"]), sort_keys=True))


if __name__ == "__main__":
    main()
