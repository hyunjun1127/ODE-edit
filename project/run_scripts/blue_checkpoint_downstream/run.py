"""Inference-only fixed checkpoint queue; original BLUE class methods reused."""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

from evaluator_adapter import DatasetBinding, TASKS, CANONICAL_MODEL_NAME, bind_classes, evaluate_task, canonical
from restoration import load_reference, W0Transaction
from verify_ready import file_sha, write_once


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    lock=json.loads(args.lock.read_text())
    args.output.mkdir(mode=0o700,parents=False,exist_ok=False)
    start=time.monotonic()
    model=None
    guard=None
    completed=[]
    try:
        for member in lock['members']:
            path=Path(member['path'])
            assert path.is_file() and path.stat().st_size == member['bytes']
            assert file_sha(path) == member['sha256'], path
        import torch
        import transformers
        torch.set_num_threads(8)
        torch.manual_seed(lock['seed'])
        assert torch.cuda.device_count() == 1
        imports=Path(lock['imports'])
        expected=json.loads(Path(lock['dataset_audit']).read_text())
        binding=DatasetBinding(lock['dataset_root'],expected)
        classes=bind_classes(imports/'evaluator-source',binding)
        entries=json.loads((imports/'source/checkpoint-manifest.json').read_text())['checkpoints']
        assert len(entries)==72
        snapshot=lock['snapshot']
        tokenizer=transformers.AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token=tokenizer.eos_token
        model=transformers.AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
                                                               attn_implementation='eager')
        assert all(p.dtype==torch.float32 for p in model.parameters())
        actual_name=model.config._name_or_path
        # Portable metadata binding preserves source's Llama BOS/context branch.
        model.config._name_or_path=CANONICAL_MODEL_NAME
        model.requires_grad_(False).eval().to('cuda')
        reference=load_reference(imports/'source/closure/003-checkpoint_loader.py')
        guard=W0Transaction(model,reference,entries)
        write_once(args.output/'model-entry.json',dict(status='PASS',source_head=lock['source_head'],
            snapshot=snapshot,revision=lock['revision'],actual_name_before_binding=actual_name,
            evaluator_name=CANONICAL_MODEL_NAME,parameter_hashes=guard.w0_hashes,
            dtype='torch.float32',attention='eager',transformers=transformers.__version__,
            torch=torch.__version__,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32,tokenizer_padding_side=tokenizer.padding_side,
            tokenizer_pad_id=tokenizer.pad_token_id,tokenizer_eos_id=tokenizer.eos_token_id,
            lock_sha256=file_sha(args.lock),elapsed_seconds=time.monotonic()-start))
        ledger=dict(forwards=0,input_tokens=0)
        def before_forward(module,args,kwargs):
            ids=kwargs.get('input_ids')
            if ids is None and args:
                ids=args[0]
            ledger['forwards']+=1
            if ids is not None:
                ledger['input_tokens']+=ids.numel()
        def after_forward(module,args,result):
            assert torch.isfinite(result.logits).all().item(), 'NONFINITE_LOGITS'
        pre=model.register_forward_pre_hook(before_forward,with_kwargs=True)
        post=model.register_forward_hook(after_forward)
        queue=[None]+entries
        for ordinal,entry in enumerate(queue):
            state_name='W0' if entry is None else f"{entry['arm']}-edits{entry['editcount']:05d}"
            state=args.output/state_name
            state.mkdir(mode=0o700,exist_ok=False)
            state_start=time.monotonic()
            identity=dict(state=state_name,checkpoint=None if entry is None else entry['file'],
                          model_revision=lock['revision'],source_head=lock['source_head'])
            if entry is not None:
                path=imports/entry['file']['destination_relative']
                checkpoint=reference.read_selected(path,entry)
                assert all(torch.isfinite(t).all().item() for t in checkpoint['weights'].values())
                assert torch.isfinite(checkpoint['cache_c']).all().item()
                assert checkpoint['metadata']['base_model_revision']==lock['revision']
                assert checkpoint['metadata']['sample_root']==entry['sample_root']
                assert checkpoint['metadata']['batch']==entry['batch']
                guard.apply(checkpoint,entry)
                del checkpoint
                gc.collect()
            identity['parameter_root']=hashlib.sha256(canonical(guard.expected)).hexdigest()
            write_once(state/'entry.json',dict(**identity,restoration='PASS',cache_apply=0,
                                             full_nonselected_W0_byte_identity='PASS'))
            summaries={}
            for task in TASKS:
                task_start=time.monotonic()
                before=dict(ledger)
                # Original source prints prompts even with print_logs=False;
                # preserve them only inside private local artifacts, never Git.
                with (state/f'{task}.source.log').open('x') as log, contextlib.redirect_stdout(log):
                    with torch.inference_mode():
                        summary,rows=evaluate_task(task,classes,binding,model,tokenizer)
                write_once(state/f'{task}.rows.json',rows)
                summary.update(state_identity=identity,task=task,eval_slice=[10,110],
                               data_order=expected['datasets'][task]['eval100_ordered_row_hash'],
                               fewshot=0,gen_len=5,wall_seconds=time.monotonic()-task_start,
                               forwards=ledger['forwards']-before['forwards'],
                               input_tokens=ledger['input_tokens']-before['input_tokens'],
                               scientific_promotion=False)
                write_once(state/f'{task}.metrics.json',summary)
                summaries[task]=dict(path=str(state/f'{task}.metrics.json'),
                                     sha256=file_sha(state/f'{task}.metrics.json'))
            guard.verify_endpoint()
            guard.restore_w0()
            write_once(state/'terminal.json',dict(status='TERMINAL_VALID',**identity,tasks=summaries,
                request_count=600,nonmutation='ALL_PARAMETER_POINTER_VERSION_AND_BYTES_PASS',
                W0_restore='ALL_PARAMETER_BYTES_PASS',history_apply=0,edit=0,backward=0,
                wall_seconds=time.monotonic()-state_start,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_host_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            completed.append(state_name)
            print(json.dumps(dict(stage='STATE_TERMINAL_VALID',state=state_name,
                                  completed=len(completed),total=73)),flush=True)
            if ordinal==1:
                write_once(args.output/'initial-valid.json',dict(status='INITIAL_VALID',
                    W0_tasks=6,checkpoint_tasks=6,examples_per_task=100,full_evaluation_complete=False,
                    restoration='PASS',nonmutation='PASS',W0_restore='PASS',
                    completed_states=completed,remaining_states=71,
                    observed_seconds=time.monotonic()-start,
                    extrapolated_remaining_seconds=(time.monotonic()-state_start)*71,
                    external_monitoring_policy='MONITORING_PAUSED_AWAITING_GH'))
        pre.remove();post.remove()
        write_once(args.output/'terminal.json',dict(status='TERMINAL_VALID',states=73,checkpoints=72,
            W0_once=True,task_evaluations=438,examples=43800,completed=completed,
            forwards=ledger['forwards'],input_tokens=ledger['input_tokens'],
            model_load=1,edit=0,backward=0,history_apply=0,scientific_promotion=False,
            wall_seconds=time.monotonic()-start))
    except BaseException as exc:
        write_once(args.output/'failure.json',dict(status='TECHNICAL_FAILURE',
            exception_type=type(exc).__name__,error=str(exc),completed=completed,
            scores_imputed=0,scientific_promotion=False))
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
