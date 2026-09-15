"""Single persistent original CAKE W0 B100x100, user-directed no CP storage."""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


def rng_state():
    import numpy as np
    import torch
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])],
                torch=torch.get_rng_state().tolist(), cuda=[v.tolist() for v in torch.cuda.get_rng_state_all()])


def merge(past, current, endpoint):
    from project.run_scripts.blue_alphaedit_sequential_comparison.integrity import digest
    metrics = {}
    for tag in ('RS','PS','NS'):
        rows = (past['metrics'][tag]['rows'] if past else []) + current['metrics'][tag]['rows']
        assert len({r['identity'] for r in rows}) == len(rows), 'DUPLICATE_PROMPT_IDENTITY'
        n = sum(r['success'] for r in rows)
        metrics[tag] = dict(rows=rows, numerator=n, denominator=len(rows), rate=n/len(rows),
                            bit_order_sha256=digest([(r['identity'],r['success']) for r in rows]))
    return dict(requests=(past['requests'] if past else 0)+current['requests'], metrics=metrics,
                state=endpoint, current_rows_reused=True, evaluation_type='ACTUAL_WK_FULL_SEEN_PREFIX')


def run(lock_path):
    from project.run_scripts.blue_alphaedit_sequential_comparison.integrity import content, digest, file_sha, restore, save, signature, tensor_sha
    from project.run_scripts.blue_alphaedit_sequential_comparison.evaluation import evaluate
    from .observation import observe, nonselected
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lock_path = Path(lock_path).resolve()
    lock = json.loads(lock_path.read_bytes())
    assert lock['method'] == 'CAKE_NATIVE' and lock['policies'] == ['CAKE_NATIVE']
    assert lock['checkpoint_storage'] == 'DISABLED_USER_DIRECTED_NO_W_M_TENSORS'
    assert lock['batches'] == 100 and lock['batch_size'] == 100
    assert os.environ.get('SLURM_RESTART_COUNT','0') == '0', 'REQUEUE_NOT_ALLOWED'
    output = Path(lock['output'])
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    stage = 'SOURCE_BINDING'
    committed, seen = [], []
    started = time.monotonic()
    try:
        for x in lock['members']:
            assert Path(x['path']).stat().st_size == x['bytes'] and file_sha(x['path']) == x['sha256'], 'SOURCE_INPUT_DRIFT:' + x['path']
        assert transformers.__version__ == '4.44.2' and torch.__version__ == lock['torch_version']
        assert os.statvfs(output).f_bavail * os.statvfs(output).f_frsize >= lock['raw_reserve_bytes'], 'INSUFFICIENT_DISK_RESERVE'
        stage = 'FIXED_DATASET_BEFORE_MODEL'
        spec = importlib.util.spec_from_file_location('fixed_counterfact', lock['fixed_loader'])
        fixed = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixed)
        records = fixed.load_prefix(lock['dataset_root'], 10000)
        assert digest([r['case_id'] for r in records]) == lock['case_order_sha256']
        sys.path.insert(0, lock['cake_root']); os.chdir(lock['cake_root'])
        module = importlib.import_module('Cake.Cake_main')
        hp = importlib.import_module('Cake.Cake_hparams').CakeHyperParams.from_json(lock['hparams_path'])
        assert vars(hp) == lock['hparams'] and hp.layers == [4,5,6,7,8]
        random.seed(lock['seed']); np.random.seed(lock['seed']); torch.manual_seed(lock['seed'])
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        stage = 'MODEL_LOAD'
        model = AutoModelForCausalLM.from_pretrained(lock['snapshot'], local_files_only=True,
                    low_cpu_mem_usage=True, torch_dtype=torch.float32, attn_implementation='eager').cuda().eval()
        tok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        tok.add_bos_token = False; tok.pad_token_id = tok.eos_token_id
        evaltok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        evaltok.pad_token_id = evaltok.eos_token_id
        assert tok.padding_side == evaltok.padding_side == 'right'
        assert {p.dtype for p in model.parameters()} == {torch.float32}
        stage = 'P_AND_COLD_HISTORY'
        projector = torch.load(lock['projector']['path'], map_location='cpu', weights_only=True)
        assert tuple(projector.shape) == (5,14336,14336) and projector.dtype == torch.float32
        assert tensor_sha(projector) == lock['projector_tensor_sha256'], 'P_MAPPING_OR_CONTENT'
        state = torch.zeros_like(projector)
        module.CONTEXT_TEMPLATES_CACHE = json.loads(Path(lock['contexts']['path']).read_bytes())
        assert digest(module.CONTEXT_TEMPLATES_CACHE) == lock['context_digest']
        weights = {hp.rewrite_module_tmp.format(l)+'.weight': module.nethook.get_parameter(model, hp.rewrite_module_tmp.format(l)+'.weight') for l in hp.layers}
        assert all(tuple(w.shape) == (4096,14336) for w in weights.values())
        prior = content(signature(weights,state)); initial_nonselected = nonselected(model, weights)
        save(output/'runtime.json',dict(method='CAKE_NATIVE', lock_sha256=file_sha(lock_path), source=lock['source'],
             upstream=lock['upstream'], actual_imports={n:dict(path=m.__file__,sha256=file_sha(m.__file__)) for n,m in list(sys.modules.items()) if getattr(m,'__file__',None) and any(str(m.__file__).startswith(p) for p in [lock['cake_root'],lock['helper_root'],lock['source_root']]) and Path(m.__file__).is_file()},
             torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(), dtype='torch.float32',
             attention=model.config._attn_implementation, tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
             model_revision=lock['revision'], seed=lock['seed'], W0=prior, cold_history=True,
             context_sha256=lock['contexts']['sha256'], context_digest=lock['context_digest'],
             writer_tokenizer=dict(add_bos_token=tok.add_bos_token,padding=tok.padding_side,pad=tok.pad_token_id),
             evaluator_tokenizer=dict(padding=evaltok.padding_side,manual_batch_padding='left; original canonical source'),
             hparams=vars(hp), projector_mapping=lock['projector_mapping'], projector_tensor_sha256=lock['projector_tensor_sha256'],
             slurm_job=os.environ.get('SLURM_JOB_ID'), load_setup_seconds=time.monotonic()-started,
             checkpoint_storage=lock['checkpoint_storage'], exact_restart_available=False, scientific_promotion=False))
        for bi in range(1,101):
            rows = records[(bi-1)*100:bi*100]
            root = output/f'B{bi:03d}'
            stage = f'B{bi}_ENTRY'
            entry = signature(weights,state)
            assert content(entry) == prior, 'W_M_CHAIN'
            assert initial_nonselected == nonselected(model,weights), 'NONSELECTED_STATE_CHANGED'
            # In-memory rollback only; never written as a tensor artifact.
            ew = {k:v.detach().cpu().clone() for k,v in weights.items()}; em = state.clone()
            save(root/'entry.json',dict(batch=bi,seen_before=len(seen),entry=content(entry),rng=rng_state(),
                 case_ids=[r['case_id'] for r in rows],request_hashes=[digest(r['requested_rewrite']) for r in rows],
                 context_digest=lock['context_digest'],history_request_events=len(seen)))
            try:
                stage=f'B{bi}_NATIVE_WRITE'; t0=time.monotonic()
                requests=[dict(r['requested_rewrite'],case_id=int(r['case_id'])) for r in rows]
                with observe(module,hp,weights,requests,model) as counters:
                    returned, returned_state = module.apply_Cake_to_model(model,tok,requests,hp,cache_template=None,cache_c=state,P=projector)
                assert returned is model and returned_state is state, 'NATIVE_RETURN_STATE'
                assert all(torch.isfinite(v).all() for v in weights.values()) and torch.isfinite(state).all(), 'NONFINITE_NATIVE_ENDPOINT'
                assert digest(module.CONTEXT_TEMPLATES_CACHE) == lock['context_digest'], 'CONTEXT_MUTATION'
                edit_seconds=time.monotonic()-t0
                endpoint=signature(weights,state)
                save(root/'native-observation.json',counters)
                updates={}
                for k,w in weights.items():
                    d=w.detach().cpu().double()-ew[k].double()
                    updates[k]=dict(norm=float(d.norm()),relative_norm=float(d.norm()/ew[k].double().norm()))
                del d
                stage=f'B{bi}_EVALUATION'; ev0=time.monotonic(); rng_before=rng_state()
                current=evaluate(model,evaltok,rows,weights,state)
                save(root/'current.json',current)
                if bi in lock['evaluation_batches']:
                    past=evaluate(model,evaltok,seen,weights,state) if seen else None
                    full=merge(past,current,content(endpoint))
                    assert full['requests']==bi*100
                    assert {k:v['denominator'] for k,v in full['metrics'].items()} == dict(RS=bi*100,PS=bi*200,NS=bi*1000)
                    save(root/'seen-full.json',full)
                    save(root/'seen-rewrite.json',dict(requests=bi*100,state=content(endpoint),metrics={'RS':full['metrics']['RS']},reused_from='seen-full.json'))
                    del past,full
                else:
                    past=evaluate(model,evaltok,seen,weights,state,full=False) if seen else None
                    rw=(past['metrics']['RS']['rows'] if past else [])+current['metrics']['RS']['rows']
                    assert len({r['identity'] for r in rw})==bi*100
                    save(root/'seen-rewrite.json',dict(requests=bi*100,state=content(endpoint),metrics={'RS':dict(rows=rw,numerator=sum(r['success'] for r in rw),denominator=len(rw))},current_rows_reused=True))
                    del past,rw
                assert signature(weights,state)==endpoint and rng_state()==rng_before, 'EVALUATION_STATE_OR_RNG_MUTATION'
                assert initial_nonselected == nonselected(model,weights)
                receipt=dict(status='BATCH_COMMITTED',batch=bi,requests=100,seen_requests=bi*100,next_batch=bi+1,next_ordinal=bi*100,
                     entry=content(entry),endpoint=content(endpoint),history_append_passes=1,layer_history_updates=5,
                     history_entries_in=(bi-1)*100,history_entries_out=bi*100,compute_z=counters['compute_z'],solve_calls=counters['solve_calls'],
                     context_digest=lock['context_digest'],rng=rng_state(),layer_updates=updates,
                     edit_seconds=edit_seconds,target_seconds=counters['target_seconds'],key_seconds=counters['key_seconds'],solve_seconds=counters['solve_seconds'],
                     evaluation_seconds=time.monotonic()-ev0,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                     checkpoint_storage=lock['checkpoint_storage'],saved_weight_tensor_count=0,saved_history_tensor_count=0,
                     resume='METADATA_ONLY_NOT_RECONSTRUCTABLE; process has live W/M',evaluator_mutation=0,nonfinite=0,
                     current={k:{a:b for a,b in v.items() if a!='rows'} for k,v in current['metrics'].items()})
                save(root/'commit.json',receipt);committed.append(receipt);prior=content(endpoint);seen.extend(rows)
                if bi==1:
                    save(output/'initial-execution.json',dict(status='FIRST_NATIVE_BATCH_COMMITTED',job_id=os.environ.get('SLURM_JOB_ID'),batch=1,next_ordinal=100,
                         history_append_passes=1,layer_history_updates=5,execution_continues_without_agent=True,no_tensor_checkpoint=True))
                print('CAKE_NATIVE_LIFELONG_BATCH_COMMITTED',bi,bi*100,edit_seconds,flush=True)
            except BaseException:
                restore(weights,ew,state,em)
                save(root/'rollback.json',dict(stage=stage,entry_exact=content(signature(weights,state))==content(entry),accepted_denominator=0,
                                             persistent_resume_available=False))
                raise
            del ew,em,current
        stage='TERMINAL_PUBLICATION'
        assert tensor_sha(projector)==lock['projector_tensor_sha256'], 'PROJECTOR_MUTATION'
        assert initial_nonselected==nonselected(model,weights)
        save(output/'terminal.json',dict(status='TERMINAL_VALID',method='CAKE_NATIVE',batches=100,requests=10000,W_chain_links=99,
             compute_z=sum(x['compute_z'] for x in committed),solve_calls=sum(x['solve_calls'] for x in committed),history_append_passes=100,layer_history_updates=500,
             actual_final_state=prior,final_evaluation='B100/seen-full.json',no_persistent_checkpoint=True,exact_restart_available=False,
             seconds=time.monotonic()-started,manifest_members=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(output.rglob('*')) if p.is_file()],
             nonfinite=0,scientific_promotion=False))
    except BaseException as e:
        save(output/'failure.json',dict(status='TYPED_FAILURE',stage=stage,error=repr(e),traceback=traceback.format_exc(),
             prior_committed_batches=len(committed),prior_committed_requests=len(seen),final_denominator=0,
             seconds=time.monotonic()-started,exact_restart_available=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True)
    run(p.parse_args().lock)
