"""Create-once, outcome-blind existing-Wikipedia diagnostic input (local raw).

No model weights, download, editing, or dataset replacement. The first 128
physical Arrow rows are selected BEFORE tokenization; short rows are boundaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import MODEL
from .contracts import ContractBoundary, digest, member, save

ARROW = Path('/mnt/raid5/janghj/.cache/huggingface/datasets/wikipedia/20200501.en/1.0.0/009f923d9b6dd00c00c8cdc7f408f2b47f45dd4f5fb7982a21f9448f4afbe475/wikipedia-train.arrow')


def select_rows(rows, tokenizer, *, count=128, max_tokens=256):
    """Tokenize a fixed prefix, never replace a selected short/invalid row."""
    if count < 1 or max_tokens < 2:
        raise ContractBoundary('GENERAL_PANEL_DIMENSION')
    selected = []
    for ordinal, row in enumerate(rows):
        if ordinal == count:
            break
        if not isinstance(row.get('text'), str):
            raise ContractBoundary('GENERAL_TEXT_SCHEMA', ordinal=ordinal)
        ids = tokenizer.encode(row['text'], add_special_tokens=True)[:max_tokens]
        if len(ids) < 2:
            raise ContractBoundary('GENERAL_SELECTED_ROW_TOO_SHORT', ordinal=ordinal, tokens=len(ids))
        selected.append(dict(ordinal=ordinal, source_id=row.get('id'),
                             source_record_sha256=digest(row), text_sha256=digest(row['text']),
                             input_ids=[int(x) for x in ids], tokens=len(ids),
                             predicted_tokens=len(ids)-1))
    if len(selected) != count:
        raise ContractBoundary('GENERAL_CORPUS_TOO_SHORT', actual=len(selected), expected=count)
    return selected


def _arrow_rows(path):
    import pyarrow as pa
    with pa.memory_map(str(path), 'r') as handle:
        reader = pa.ipc.open_stream(handle)
        for batch in reader:
            yield from batch.to_pylist()


def build_general_manifest(destination, *, arrow=ARROW, snapshot=MODEL):
    """Hashes full Arrow once; tokenizer-only local load, no remote code."""
    import transformers
    import pyarrow
    from transformers import AutoTokenizer
    arrow, snapshot = Path(arrow), Path(snapshot)
    if Path(destination).exists():
        raise ContractBoundary('GENERAL_CREATE_ONCE_DESTINATION_EXISTS')
    source = member(arrow)
    info = member(arrow.parent/'dataset_info.json')
    tokenizer_members = []
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json'):
        path = snapshot/name
        if path.exists():
            item = member(path.resolve(strict=True))
            item['consumed_path'] = str(path)
            tokenizer_members.append(item)
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    rows = select_rows(_arrow_rows(arrow), tok)
    result = dict(schema='E01_GENERAL_EXISTING_WIKIPEDIA_V1', source=source, dataset_info=info,
                  tokenizer_members=tokenizer_members, snapshot=str(snapshot),
                  transformers=transformers.__version__, pyarrow=pyarrow.__version__,
                  tokenizer_class=type(tok).__name__, add_special_tokens=True,
                  source_selection='PHYSICAL_ARROW_ROWS_0_127_NO_REPLACEMENT',
                  max_tokens=256, rows=rows, row_count=128,
                  row_identity=digest(rows), raw_text_or_tokens_git_allowed=False,
                  outcome_selection=0, writer_influence=0, external_download=0,
                  scalar='MEAN_NEXT_TOKEN_NLL_PER_SEQUENCE; GENERAL_SEPARATE_FROM_RPN')
    return save(destination, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    parser.add_argument('--arrow', default=str(ARROW))
    parser.add_argument('--snapshot', default=str(MODEL))
    args = parser.parse_args()
    print(json.dumps(build_general_manifest(args.destination, arrow=args.arrow, snapshot=args.snapshot)))
