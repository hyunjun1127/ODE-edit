"""Thin runner for the unmodified repository BLUE AlphaEdit callable."""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from .integrity import checkpoint, content, digest, file_sha, restore, save, signature, tensor_sha
from .evaluation import evaluate
from .observer import observe


def run(lock_path, output, mode):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    lock = json.loads(Path(lock_path).read_text())
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    stage = 'SOURCE_BINDING'
    model = weights = cache = w0 = None
    committed = []
    started = time.monotonic()
    try:
        for x in lock['members']:
            if file_sha(x['path']) != x['sha256']:
                raise RuntimeError('LOCKED_MEMBER_CHANGED:' + x['path'])
        assert transformers.__version__ == '4.44.2'
        assert subprocess.check_output(['git', '-C', lock['source_root'], 'rev-parse', 'HEAD'], text=True).strip() == lock['source_head']
        blue = Path(lock['blue_root'])
        sys.path.insert(0, str(blue))
        os.chdir(blue)  # globals.yml is read by the native module; no native CLI writes.
        module = importlib.import_module('AlphaEdit.AlphaEdit_main')
        hp = importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams.from_json(lock['config'])
        assert hp.blue and hp.layers == [4, 8]
        np.random.seed(lock['seed']); random.seed(lock['seed']); torch.manual_seed(lock['seed'])
        torch.set_num_threads(8)
        stage = 'MODEL_LOAD'
        # Default model dtype is FP32, as in the original CLI (no dtype override).
        # Eager is the original README-era attention implementation; memory loading only is changed.
        model = AutoModelForCausalLM.from_pretrained(lock['snapshot'], local_files_only=True,
            low_cpu_mem_usage=True, attn_implementation='eager').cuda()
        model.eval()
        tok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        tok.add_bos_token = False; tok.pad_token_id = tok.eos_token_id
        assert tok.padding_side == 'right'
        evaltok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        evaltok.pad_token_id = evaltok.eos_token_id
        assert evaltok.padding_side == 'right'
        assert {p.dtype for p in model.parameters()} == {torch.float32}
        weights = {hp.rewrite_module_tmp.format(layer)+'.weight': module.nethook.get_parameter(model, hp.rewrite_module_tmp.format(layer)+'.weight') for layer in hp.layers}
        w0 = {k: v.detach().cpu().clone() for k, v in weights.items()}
        pointers0 = {k: v.data_ptr() for k, v in weights.items()}
        allp = torch.load(lock['projector'], map_location='cpu', weights_only=True)
        assert isinstance(allp, torch.Tensor) and allp.shape[0] == 5
        # Cached native .02-threshold projector inventory is indexed by physical L4..L8.
        projector = allp[[0, 4]].clone(); del allp
        cache = torch.zeros_like(projector)
        assert cache.dtype == torch.float32 and cache.shape[1] == next(iter(weights.values())).shape[1]
        w0sig = signature(weights, cache)
        dataset = json.loads(Path(lock['dataset']).read_text())
        byid = {int(r['case_id']): r for r in dataset}
        sample = json.loads(Path(lock['sample']).read_text())
        if mode == 'smoke':
            batches = [[byid[i]] for i in sample['smoke_ids'][:2]]
        else:
            rows = [byid[r['case_id']] for r in sample['records']]
            batches = [rows[i:i+100] for i in range(0, 1000, 100)]
        del dataset, byid
        runtime = dict(mode=mode, lock_sha256=file_sha(lock_path), blue_head=lock['blue_head'],
            source_head=lock['source_head'], source_tree=lock['source_tree'], hparams=vars(hp),
            model_revision=lock['revision'], model_load_seconds=time.monotonic()-started,
            torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
            parameter_elements=sum(p.numel() for p in model.parameters()), dtype='torch.float32',
            attention=model.config._attn_implementation, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32, autocast=torch.is_autocast_enabled(),
            writer_tokenizer=dict(add_bos_token=tok.add_bos_token, padding=tok.padding_side, pad=tok.pad_token_id, eos=tok.eos_token_id),
            evaluator_tokenizer=dict(add_bos_token=evaltok.add_bos_token, padding=evaltok.padding_side, pad=evaltok.pad_token_id),
            context_policy='native get_context_templates at first batch; retained across sequential chain',
            W0=w0sig, projector_layers=[4,8], projector_sha256=tensor_sha(projector),
            scientific_promotion=False, model_training=False, slurm_job=os.environ.get('SLURM_JOB_ID'))
        save(output/'runtime.json', runtime)
        stage = 'PRE_EDIT'
        allrows = [r for batch in batches for r in batch]
        before_eval = time.monotonic()
        pre = evaluate(model, evaltok, allrows, weights, cache)
        save(output/'pre-edit.json', pre)
        save(output/'pre-edit-timing.json', dict(seconds=time.monotonic()-before_eval))
        seen = []
        prior = content(w0sig)
        for bi, rows in enumerate(batches, 1):
            stage = f'B{bi}_ENTRY'
            root = output/f'B{bi:02d}'
            entry = signature(weights, cache)
            assert content(entry) == prior
            entry_w = {k:v.detach().cpu().clone() for k,v in weights.items()}
            entry_m = cache.clone()
            save(root/'entry.json', dict(signature=entry, seen_before=len(seen),
                request_ids=[r['case_id'] for r in rows], request_hashes=[digest(r['requested_rewrite']) for r in rows],
                cache_entries= len(seen), context_hash=digest(module.CONTEXT_TEMPLATES_CACHE)))
            requests = [dict(r['requested_rewrite'], case_id=int(r['case_id'])) for r in rows]
            edit_start = time.monotonic()
            try:
                stage = f'B{bi}_NATIVE_WRITE'
                with observe(module, hp, weights, cache, requests) as counters:
                    returned, returned_cache = module.apply_AlphaEdit_to_model(model, tok, requests, hp,
                        cache_template=None, cache_c=cache, P=projector)
                assert returned is model and returned_cache is cache
                if not all(torch.isfinite(v).all() for v in weights.values()) or not torch.isfinite(cache).all():
                    raise RuntimeError('NONFINITE_ENDPOINT')
                edit_seconds = time.monotonic()-edit_start
                endpoint = signature(weights, cache)
                assert all(v.data_ptr() == pointers0[k] for k,v in weights.items())
                save(root/'native-observation.json', counters)
                save(root/'contexts.json', module.CONTEXT_TEMPLATES_CACHE)
                update = {}
                for k, v in weights.items():
                    d = v.detach().cpu().double()-entry_w[k].double()
                    update[k] = dict(norm=float(d.norm()), squared_norm=float(d.square().sum()),
                        relative_norm=float(d.norm()/entry_w[k].double().norm()))
                total = sum(v['norm'] for v in update.values())
                for v in update.values(): v['magnitude_share'] = v['norm']/total if total else None
                stage = f'B{bi}_ENDPOINT_EVAL'
                eval_start = time.monotonic()
                current = evaluate(model, evaltok, rows, weights, cache)
                save(root/'current.json', current)
                prior_rewrite = evaluate(model, evaltok, seen, weights, cache, full=False) if seen else None
                if prior_rewrite is not None: save(root/'prior-rewrite.json', prior_rewrite)
                # Current rewrite is reused, not reevaluated, for all-seen RS.
                rewrite_rows = ([] if prior_rewrite is None else prior_rewrite['metrics']['RS']['rows']) + current['metrics']['RS']['rows']
                save(root/'seen-rewrite.json', dict(requests=len(seen)+len(rows), rows=rewrite_rows,
                    numerator=sum(r['success'] for r in rewrite_rows), denominator=len(rewrite_rows), state=content(endpoint)))
                full = None
                if bi in (1,5,10) or mode == 'smoke':
                    # Same endpoint; current batch already evaluated once.
                    if seen:
                        past = evaluate(model, evaltok, seen, weights, cache)
                        merged = {}
                        for tag in ('RS','PS','NS'):
                            a,b=past['metrics'][tag],current['metrics'][tag]
                            rr=a['rows']+b['rows']; n=a['numerator']+b['numerator']; den=a['denominator']+b['denominator']
                            merged[tag]=dict(rows=rr,numerator=n,denominator=den,rate=n/den)
                        full = dict(requests=len(seen)+len(rows), metrics=merged, state=content(endpoint))
                    else: full = current
                    save(root/'seen-full.json', full)
                eval_seconds = time.monotonic()-eval_start
                assert signature(weights, cache) == endpoint
                cp = None
                if bi in (1,5,10) or mode == 'smoke':
                    cp = checkpoint(root/'W-M.pt', weights, cache, dict(batch=bi, request_ids=[r['case_id'] for r in seen+rows],
                        contexts=module.CONTEXT_TEMPLATES_CACHE, source_head=lock['source_head'], state=content(endpoint)))
                receipt = dict(status='BATCH_COMMITTED', batch=bi, requests=len(rows), seen_requests=len(seen)+len(rows),
                    entry=content(entry), endpoint=content(endpoint), W_pointer_exact=True, history_append_passes=1,
                    history_entries_in=len(seen), history_entries_out=len(seen)+len(rows), compute_z=counters['compute_z'],
                    context_hash=digest(module.CONTEXT_TEMPLATES_CACHE), layer_updates=update,
                    edit_seconds=edit_seconds, target_seconds=counters['target_seconds'], key_seconds=counters['key_seconds'],
                    evaluation_seconds=eval_seconds, checkpoint=cp, nonfinite=0, evaluator_mutation=0,
                    current={k:{a:b for a,b in v.items() if a!='rows'} for k,v in current['metrics'].items()},
                    peak_gpu_bytes=torch.cuda.max_memory_allocated())
                save(root/'commit.json', receipt)
                committed.append(receipt); prior=content(endpoint); seen.extend(rows)
                print('BLUE_BATCH_COMMITTED', bi, len(seen), edit_seconds, flush=True)
            except BaseException:
                restore(weights, entry_w, cache, entry_m)
                assert content(signature(weights, cache)) == content(entry)
                save(root/'rollback.json', dict(entry_exact=True, stage=stage, accepted_denominator=0))
                raise
            del entry_w, entry_m
        stage = 'FINAL_W0_RESTORE'
        restore(weights, w0, cache, torch.zeros_like(cache))
        assert content(signature(weights, cache)) == content(w0sig)
        assert all(v.data_ptr()==pointers0[k] for k,v in weights.items())
        save(output/'terminal.json', dict(status='TERMINAL_VALID', mode=mode, batches=len(committed),
            requests=len(seen), compute_z=sum(r['compute_z'] for r in committed), W_chain_links=len(committed)-1,
            history_append_passes=len(committed), W0_cache_bytes_pointer_restore=True,
            version_restore='NOT_CLAIMED: in-place copy increments version; evaluation versions unchanged',
            cold_reset_count=1, nonfinite=0, failure=0, seconds=time.monotonic()-started,
            source_head=lock['source_head'], manifest_members=[dict(path=str(p.relative_to(output)),sha256=file_sha(p),bytes=p.stat().st_size)
                for p in sorted(output.rglob('*')) if p.is_file()], scientific_promotion=False))
    except BaseException as e:
        restore_error=None
        if weights is not None and w0 is not None and cache is not None:
            try: restore(weights,w0,cache,torch.zeros_like(cache))
            except BaseException as re: restore_error=repr(re)
        save(output/'failure.json', dict(status='HOLD', stage=stage, error=repr(e), traceback=traceback.format_exc(),
            prior_committed_batches=len(committed), prior_committed_requests=sum(r['requests'] for r in committed),
            final_denominator=0, W0_restore_error=restore_error, seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['smoke','main'],required=True)
    a=p.parse_args();run(a.lock,a.output,a.mode)
