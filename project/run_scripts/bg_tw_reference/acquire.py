"""Only the two approved pinned C4 shards, pinned README and PSL metadata."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request

from .control import identity, save


def download(url, dst, expected_bytes=None):
    dst = Path(dst)
    if dst.exists() or dst.with_suffix(dst.suffix + '.partial').exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_suffix(dst.suffix + '.partial')
    started = time.monotonic()
    request = urllib.request.Request(url, headers={'User-Agent': 'ODE-edit-BG1-C4/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('xb') as f:
        if response.status != 200:
            raise RuntimeError(('FULL_DOWNLOAD_REQUIRES_HTTP200', response.status))
        for block in iter(lambda: response.read(1 << 20), b''):
            f.write(block)
        headers = {k:response.headers.get(k) for k in ('Content-Length', 'ETag', 'Last-Modified')}
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise RuntimeError(('FULL_BYTES_MISMATCH', partial.stat().st_size, expected_bytes))
    # No-overwrite hardlink seal; partial retained as provenance, no deletion.
    import os
    os.link(partial, dst)
    result = dict(**identity(dst), source_url=url, expected_bytes=expected_bytes,
                  headers=headers, seconds=time.monotonic()-started, status='FULL_FILE_DOWNLOADED',
                  gzip_crc_and_rows='PENDING_BUILDER_FULL_SCAN', partial_path=str(partial))
    save(dst.with_suffix(dst.suffix + '.receipt.json'), result)
    return result


def run(contract, output):
    c = json.loads(Path(contract).read_text())
    assert c['contract_id'] == 'C4-WebRef-v2'
    assert c['corpus']['revision'] == '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
    assert len(c['corpus']['source_files']) == 2
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    jobs = [(r['url']+'?download=true', out/Path(r['path']).name,
             r['expected_bytes_from_content_range']) for r in c['corpus']['source_files']]
    jobs += [('https://huggingface.co/datasets/allenai/c4/raw/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/README.md',out/'README.md',None),
             ('https://publicsuffix.org/list/public_suffix_list.dat',out/'public_suffix_list.dat',None)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda item: download(*item), jobs))
    return save(out/'acquisition-manifest.json', dict(members=records,
        contract=identity(contract), permitted_downloads_only=True,
        psl_policy='Exact retrieved bytes fixed by SHA before sampling; composition only, not selection',
        full_c4_or_pile_download=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.contract,a.output)))
