"""User-approved exact public GPT-J asset acquisition; no model load or inference."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = '47e169305d2e8376be1d31e765533382721b2cc1'
DESTINATION = Path('/var/tmp/janghj-price-gptj-model-Wv5YCJTj')
FILES = ['pytorch_model.bin', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'added_tokens.json', 'merges.txt', 'vocab.json']
WEIGHT_SHA = '0e183edc2025ecfdba4429ba43c960224103b3c3dc26616503cdc2158a3d6c93'
WEIGHT_BYTES = 24207819307


def main():
    assert DESTINATION.is_dir() and not DESTINATION.is_symlink()
    assert DESTINATION.stat().st_uid == os.getuid()
    assert shutil.disk_usage(DESTINATION).free >= WEIGHT_BYTES + 4 * 1024**3
    path = Path(snapshot_download(repo_id='EleutherAI/gpt-j-6b', revision=REVISION,
        allow_patterns=FILES, cache_dir=str(DESTINATION / 'cache'), token=False,
        max_workers=2))
    members = []
    for name in FILES:
        source = path / name
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(chunk)
        row = dict(name=name, path=str(source), realpath=str(source.resolve()),
                   bytes=source.stat().st_size, sha256=digest.hexdigest())
        if name == 'pytorch_model.bin':
            assert row['bytes'] == WEIGHT_BYTES and row['sha256'] == WEIGHT_SHA
        members.append(row)
    receipt = dict(model='EleutherAI/gpt-j-6b', revision=REVISION, snapshot=str(path),
        members=members, model_loads=0, GPU=0, user_download_authorized=True,
        location_policy='별도 filesystem /var/tmp; 영구 보존 보장 아님, 매 실행 stat/identity 검사 필수')
    with (DESTINATION / 'download-receipt.json').open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status='DOWNLOADED_SHA_VERIFIED', snapshot=str(path),
        receipt=str(DESTINATION / 'download-receipt.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
