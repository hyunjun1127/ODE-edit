"""Prepare the author's FE main.py entrypoint, changing target supply only.

The FE->MEMIT filename replacement is the author's documented installation step.
This is an offline patch generator, never an editing runner.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

AUTHOR_COMMIT = '478134dfb24b43f4e18b47e8500893ce3f9cc50f'
FE_SHA256 = '6934f007ef6066690933a38a5a8d5bfde1e17d5ebbb5879210e4f6d037d26f9d'


def function_text(source, name):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def patched_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != FE_SHA256:
        raise ValueError('UNPINNED_AUTHOR_FE_SOURCE')
    start = source.index('    # Load z cache once for all batches')
    end = source.index('    return model\n', start)
    replacement = '''    # Sequential diagnostic: only target supply changes. The author writer below is untouched.
    from .native_sequential_z import TargetSupplier
    from .native_checkpoint_observer import snapshot
    context_templates = get_context_templates(model, tok)
    target_supplier = TargetSupplier(model, cfg)
    snapshot(cfg, model, cache_c, context_templates, 0, requests)
    for batch_index, requests_chunks in enumerate(chunks(requests, cfg.bs), start=1):
        print(f"NATIVE_FE_BATCH_START {batch_index} case_ids={[r['case_id'] for r in requests_chunks]}", flush=True)
        for local_index, request in enumerate(requests_chunks):
            request["sample_idx"] = local_index
        zs_all = target_supplier(model, tok, requests_chunks, cfg, context_templates,
            compute_z=compute_z, find_fact_lookup_idx=find_fact_lookup_idx, TraceDict=nethook.TraceDict)
        batch_edit(cfg,model,tok,requests_chunks,device,cache_c,zs_all)
        snapshot(cfg, model, cache_c, context_templates, batch_index, requests)
        print(f"NATIVE_FE_BATCH_COMPLETE {batch_index}", flush=True)
'''
    result = source[:start] + replacement + source[end:]
    for name in ('batch_edit', 'load_cov', 'get_context_templates', 'upd_matrix_match_shape'):
        if function_text(source, name) != function_text(result, name):
            raise ValueError(f'AUTHOR_FUNCTION_CHANGED_{name}')
    return result


def install(root):
    root = Path(root)
    source = (root/'algs/memit/FE-memit_main.py').read_text()
    target = root/'algs/memit/memit_main.py'
    main_before = hashlib.sha256((root/'main.py').read_bytes()).hexdigest()
    result = patched_source(source)
    target.write_text(result)
    here = Path(__file__).parent
    for src, dst in (('sequential_z.py', 'native_sequential_z.py'),
                     ('checkpoint_observer.py', 'native_checkpoint_observer.py')):
        (target.parent/dst).write_bytes((here/src).read_bytes())
    patch = ''.join(difflib.unified_diff(source.splitlines(True), result.splitlines(True),
                                        fromfile='author/FE-memit_main.py', tofile='execution/memit_main.py'))
    if hashlib.sha256((root/'main.py').read_bytes()).hexdigest() != main_before:
        raise ValueError('MAIN_ENTRYPOINT_CHANGED')
    return dict(author_commit=AUTHOR_COMMIT, main_sha256=main_before,
                upstream_fe_sha256=FE_SHA256, patched_fe_sha256=hashlib.sha256(result.encode()).hexdigest(),
                scientific_change='current-batch-only firstforward target supply; batch-entry default, explicit initial-model causal control available',
                observation_change='W0 and 500-edit weights/history/RNG snapshots', patch=patch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('author_root')
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    result = install(args.author_root)
    with Path(args.receipt).open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
