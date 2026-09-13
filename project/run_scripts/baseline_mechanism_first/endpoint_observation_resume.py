"""Resume only missing full-seen evaluation from an immutable native endpoint.

No editor/native builder import; no compute-z, update or history append. Past
chunks128 are aligned to canonical16 for every RS/PS/NS candidate stream. The
last Current100 is the already completed same-state observation, as in source.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback
from types import SimpleNamespace

from .contracts import member,save,ContractBoundary,digest
from .performance_schema import normalize,subset,from_rows
from .terminal_performance import observe_guarded,paired
from .cold_analysis import table


def execute(lock_path,output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    from .evaluation import bind_evaluation_sources,evaluate_records
    from .fixtures import FixtureTransaction,SingletonSpec,restore_checkpoint,tensor_sha
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    stage='INPUT_VERIFY';tx=None;start=time.monotonic();completed=[]
    try:
        lock=json.loads(Path(lock_path).read_text())
        for ref in lock['members']:member(ref['path'],expected=ref['sha256'])
        parent=json.loads(Path(lock['parent_input']['path']).read_text())
        if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=lock['source_head']:
            raise ContractBoundary('EXECUTION_SOURCE_MISMATCH')
        for ref in parent['source_members']:member(ref['path'],expected=ref['sha256'])
        assert torch.__version__==parent['torch'] and transformers.__version__==parent['transformers']
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=parent['tf32_matmul'];torch.backends.cudnn.allow_tf32=parent['tf32_cudnn']
        records=load_prefix(parent['dataset_root'],10000)[:lock['terminal_batch']*100];past_n=len(records)-100
        baseline=normalize(json.loads(Path(parent['terminal_performance']['seen-full']['path']).read_text()),records)
        current=normalize(json.loads(Path(lock['reuse_current']['path']).read_text()),records[-100:])
        original_current=normalize(json.loads(Path(parent['terminal_performance']['current']['path']).read_text()),records[-100:])
        metrics=paired(original_current,current,'current_REUSED_NO_FORWARD')
        cp=torch.load(lock['endpoint']['path'],map_location='cpu',weights_only=False,mmap=True)
        spec=SingletonSpec(parent['cell']['layer'])
        assert tensor_sha(cp['weights'][spec.weight_name])==lock['expected_W']
        assert tensor_sha(cp['cache_c'])==lock['expected_M']
        binding=bind_evaluation_sources(parent['historical_root'],helper_root=parent['helper_root'])
        stage='MODEL_LOAD';t=time.monotonic()
        model=AutoModelForCausalLM.from_pretrained(parent['snapshot'],local_files_only=True,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        tok=AutoTokenizer.from_pretrained(parent['snapshot'],local_files_only=True);tok.pad_token_id=tok.eos_token_id
        assert tok.padding_side=='right' and all(p.dtype==torch.float32 for p in model.parameters())
        load_seconds=time.monotonic()-t
        weight=dict(model.named_parameters())[spec.weight_name];history=torch.zeros_like(cp['cache_c'])
        # A fixture-owned inert namespace, not the native editor module.
        context=SimpleNamespace(CONTEXT_TEMPLATES_CACHE=None,COV_CACHE={})
        tx=FixtureTransaction(model,context,history)
        with tx:
            stage='EXACT_ENDPOINT_RESTORE'
            restored=restore_checkpoint(model,context,history,cp,spec,expected_model_revision=parent['model_revision'],expected_seen_ids=[r['case_id'] for r in records])
            save(root/'restore.json',dict(receipt=restored,source_endpoint=lock['endpoint'],new_native_batches=0,new_z=0,new_history_append=0))
            for offset in range(0,past_n,128):
                end=min(offset+128,past_n);panel=records[offset:end];stage=f'FULLSEEN_SHARD_{offset:05d}_{end:05d}'
                result,guard=observe_guarded(model,weight,history,lock['expected_W'],lock['expected_M'],lambda:evaluate_records(model,tok,panel))
                result=normalize(result,panel)
                comparison=paired(subset(baseline,offset,end,records),result,f'seen:{offset}:{end}')
                raw=save(root/f'shards/{offset:05d}-{end:05d}.json',result)
                receipt=save(root/f'shards/{offset:05d}-{end:05d}.receipt.json',dict(raw=raw,guard=guard,paired=comparison,
                    endpoint=lock['endpoint'],begin=offset,end=end,requests=len(panel),prompt_pairs=len(panel)*13))
                completed.append(receipt)
                if offset==0:
                    save(root/'INITIAL_VALID.json',dict(status='ACTUAL_FULLSEEN_SCHEMA_REPAIR_INITIAL_VALID',
                        fullseen_source_schema='VERIFIED_ORDERED_ROW_IDENTITIES_NO_REQUEST_ORDER_HEADER',
                        actual_guard=guard,actual_fullseen_shard=receipt,endpoint=lock['endpoint'],
                        prior_native_batches_reused=10,current_requests_reused=100,new_native_batches=0,new_z=0,new_history_append=0,
                        fullseen_complete=False,elapsed_seconds=time.monotonic()-start,after_initial='MONITORING_PAUSED_AWAITING_USER'))
                    print('E01_FULLSEEN_REPAIR_INITIAL_VALID',flush=True)
                print('E01_FULLSEEN_SHARD_COMMITTED',offset,end,flush=True)
            stage='MERGE_UNIQUE_PAST_AND_REUSED_CURRENT'
            groups=[]
            for receipt in completed:
                meta=json.loads(Path(receipt['path']).read_text());member(meta['raw']['path'],expected=meta['raw']['sha256'])
                groups.append(json.loads(Path(meta['raw']['path']).read_text()))
            merged=from_rows(groups+[current],records)
            metrics+=paired(baseline,merged,'seen-full')
            full=save(root/'seen-full.json',dict(merged,current_rows_reused=True,evaluator_controller_influence=0,
                endpoint_weight_sha256=lock['expected_W'],endpoint_history_sha256=lock['expected_M']))
            summary=table(root/'paired-performance.csv',metrics)
            assert tensor_sha(weight)==lock['expected_W'] and tensor_sha(history)==lock['expected_M']
        save(root/'terminal.json',dict(status='ENDPOINT_CURRENT_FULLSEEN_OBSERVATION_COMPLETE',source_head=lock['source_head'],
            original_native_attempt=lock['parent_attempt'],endpoint=lock['endpoint'],reuse_current=lock['reuse_current'],
            baseline=parent['terminal_performance'],fullseen=full,paired_summary=summary,shards=completed,restore=tx.receipt,
            new_native_batches=0,new_z=0,new_history_append=0,unique_seen_requests=len(records),reused_current_requests=100,
            evaluated_past_requests=past_n,model_load_seconds=load_seconds,elapsed_seconds=time.monotonic()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            evaluator_binding=binding,full_E01_complete=False,trajectory_equivalence=False,scientific_promotion=False))
    except BaseException as exc:
        save(root/'failure.json',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc(),receipt=getattr(exc,'receipt',{}),
            completed_shards=completed,restore=None if tx is None else tx.receipt,elapsed_seconds=time.monotonic()-start,
            new_native_batches=0,new_z=0,new_history_append=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();execute(a.lock,a.output)
