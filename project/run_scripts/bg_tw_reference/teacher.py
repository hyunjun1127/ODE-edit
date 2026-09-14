"""Pinned W0 full-vocabulary FP32 teacher192. No editor/solver/history import."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from .control import identity, save, sha, verify_dispatch


def validate_tokens(tokens, roles, bos):
    import numpy as np
    ids = tokens['input_ids']
    assert ids.shape == (768,257) and ids.dtype == np.int64
    assert ids.min() >= 0 and ids.max() < 128256
    assert np.all(ids[:,0] == bos)
    assert np.array_equal(tokens['score_input_indices'], np.arange(129,257))
    assert np.array_equal(tokens['score_logits_indices'], np.arange(128,256))
    assert list(tokens['split_roles']) == ['S64']*64+['Dev128']*128+['Reserve320']*320+['Report256']*256
    assert len(set(map(str,tokens['source_row_ids']))) == 768
    for role,start,end in [('S64',0,64),('Dev128',64,192),('Reserve320',192,512),('Report256',512,768)]:
        assert roles['roles'][role]['indices'] == list(range(start,end))
        assert roles['roles'][role]['source_row_ids'] == list(map(str,tokens['source_row_ids'][start:end]))
    return ids


def verify_preparation_lock(lock):
    """Require actual complete builder seal and exact source/model/tokenizer binding."""
    reference = Path(lock['reference_root'])
    known = {m['path']:m for m in lock['members']}
    assert len(known) == len(lock['members']), 'DUPLICATE_LOCK_PATH'
    required = [lock['dispatch']['path'], __file__, str(reference/'reference-tokens.npz'),
        str(reference/'splits.json'),str(reference/'build-status.json'),str(reference/'source-manifest.json')]
    snapshot = Path(lock['snapshot'])
    required += [str(snapshot/x) for x in ['config.json','tokenizer.json','tokenizer_config.json',
        'special_tokens_map.json','model.safetensors.index.json']]
    index = json.loads((snapshot/'model.safetensors.index.json').read_text())
    required += [str(snapshot/x) for x in sorted(set(index['weight_map'].values()))]
    assert all(p in known for p in required), 'INCOMPLETE_PREPARATION_MEMBER_SET'
    for m in known.values():
        assert Path(m['path']).stat().st_size == m['bytes'] and sha(m['path']) == m['sha256'], ('INPUT_DRIFT',m['path'])
    built = json.loads((reference/'build-status.json').read_text())
    assert built['status'] == 'TEXT_TOKEN_SEALED_TEACHER_PENDING'
    assert built['formal_text_set_ready'] and built['exact_token_set_ready']
    for m in built['members']:
        assert Path(m['path']).stat().st_size == m['bytes'] and sha(m['path']) == m['sha256']
    manifest = json.loads((reference/'source-manifest.json').read_text())
    assert Path(manifest['tokenizer']['path']).resolve() == snapshot.resolve()
    for m in manifest['tokenizer']['members']:
        # Source builder consumed the same immutable tokenizer bytes as teacher.
        assert known[m['path']]['sha256'] == m['sha256']
    assert sha(lock['dispatch']['path']) == lock['dispatch']['sha256']
    return built


def teacher_kl(logp0, logpw):
    """Vocabulary sum, then position mean, then document mean. No clamp."""
    assert logp0.shape == logpw.shape
    return (logp0.exp()*(logp0-logpw)).sum(-1).mean(-1).mean()


def run(lock_path, output):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from scripts.fixed_counterfact import load_prefix
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    lock = json.loads(Path(lock_path).read_text())
    start = time.monotonic()
    stage = 'VERIFY_LOCK'
    try:
        verify_dispatch(lock['dispatch']['path'])
        assert lock['phase'] == 'W0_TEACHER192' and lock['scientific_chains_submitted'] == 0
        assert lock['model_revision'] == '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
        assert os.environ.get('SLURMD_NODENAME') == 'server4'
        assert torch.__version__ == lock['torch'] and transformers.__version__ == lock['transformers']
        verify_preparation_lock(lock)
        # Governing fixed10k policy, no requests enter teacher or sampling.
        assert len(load_prefix(lock['dataset_root'],1000)) == 1000
        torch.set_num_threads(8)
        transformers.set_seed(lock['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        reference = Path(lock['reference_root'])
        tokens = np.load(reference/'reference-tokens.npz',allow_pickle=False)
        roles = json.loads((reference/'splits.json').read_text())
        tok = AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        tok.pad_token_id = tok.eos_token_id
        ids = validate_tokens(tokens,roles,tok.bos_token_id)
        assert tok.padding_side == 'right'
        stage = 'W0_LOAD'
        load_start = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,
            torch_dtype=torch.float32, low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        assert model.config.vocab_size == 128256
        assert all(p.dtype == torch.float32 for p in model.parameters())
        for p in model.parameters(): p.requires_grad_(False)
        versions = [(p,p.data_ptr(),p._version) for p in model.parameters()]
        load_seconds = time.monotonic()-load_start
        save(root/'loaded.json', dict(lock_sha256=sha(lock_path), model_revision=lock['model_revision'],
            GPU=torch.cuda.get_device_name(),initial_gate='NOT_RUN',edit_calls=0,
            selected_state='PRETRAINED_W0_NO_CHECKPOINT_RESTORE'))
        cache_members, documents = [], []
        forward_seconds = io_seconds = 0.
        first_logp = None
        for role,begin,end in [('S64',0,64),('Dev128',64,192)]:
            role_dir=root/'teacher'/role;role_dir.mkdir(parents=True,exist_ok=True)
            for offset in range(begin,end,8):
                stage = f'{role}_DOC{offset}'
                path=role_dir/f'logp0-{(offset-begin)//8:03d}.npy'
                if path.exists():raise FileExistsError(path)
                cache=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(8,128,128256))
                for j in range(8):
                    index=offset+j;t=time.monotonic()
                    input_ids=torch.tensor(ids[index:index+1],dtype=torch.long,device='cuda')
                    with torch.inference_mode():
                        logits=model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),use_cache=False).logits
                        assert logits.shape == (1,257,128256)
                        logp=torch.log_softmax(logits[:,128:256,:].float(),dim=-1)
                        assert torch.isfinite(logp).all()
                        normalizer=float(torch.logsumexp(logp,dim=-1).abs().max())
                        assert normalizer <= lock['normalizer_tolerance'], 'TEACHER_NORMALIZATION'
                        nll=float(-logp.gather(-1,input_ids[:,129:257,None]).mean())
                        self_kl=float(teacher_kl(logp,logp))
                        assert self_kl == 0.
                        cpu=logp[0].cpu().numpy()
                        if index == 0:first_logp=cpu.copy()
                    torch.cuda.synchronize();forward_seconds+=time.monotonic()-t
                    t=time.monotonic();cache[j]=cpu;io_seconds+=time.monotonic()-t
                    documents.append(dict(index=index,role=role,source_row_id=str(tokens['source_row_ids'][index]),
                        natural_token_nll=nll,scored_positions=128,normalizer_max_abs=normalizer,self_kl=self_kl))
                    del logits,logp,cpu,input_ids
                t=time.monotonic();cache.flush();del cache
                # Read-only CPU mmap bridge verifies actual finite serialized payload.
                read=np.load(path,mmap_mode='r',allow_pickle=False)
                assert read.shape == (8,128,128256) and read.dtype == np.float32
                for j in range(8): assert np.isfinite(read[j]).all()
                if offset == 0:assert np.array_equal(read[0],first_logp)
                del read
                cache_members.append(identity(path));io_seconds+=time.monotonic()-t
                print('TEACHER_SHARD_SEALED',role,offset+8,flush=True)
        assert len(documents)==192 and len(cache_members)==24
        assert all(p.data_ptr()==ptr and p._version==v for p,ptr,v in versions)
        assert not model.training and all(p.grad is None for p in model.parameters())
        total=time.monotonic()-start
        result=dict(status='TEACHER192_READY',phase='PREPARATION_NOT_G0_PASS',lock=identity(lock_path),
            reference_tokens=identity(reference/'reference-tokens.npz'),splits=identity(reference/'splits.json'),
            model_revision=lock['model_revision'],model_weight_dtype='float32',cache_dtype='float32',
            vocab=128256,microbatch=1,shape_per_document=[128,128256],cache_shards=cache_members,
            tensor_payload_bytes=192*128*128256*4,actual_cache_bytes=sum(x['bytes'] for x in cache_members),
            documents=documents,scored_position_count=192*128,edit_calls=0,history_append=0,
            Reserve320_teacher='DEFERRED',Report256_teacher='DEFERRED',
            kernel=dict(torch=torch.__version__,transformers=transformers.__version__,
                attention='eager',use_cache=False,tf32_matmul=False,tf32_cudnn=True,device=torch.cuda.get_device_name()),
            self_kl_comparison='SAME_LOGP_EXACT_ZERO; independent W0 reload/forward parity NOT_TESTED',
            model_nonmutation='ALL_PARAMETER_POINTER_VERSION_NO_GRAD',
            seconds=dict(total=total,load=load_seconds,forward_and_D2H=forward_seconds,cache_io_verify=io_seconds),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return save(root/'teacher-manifest.json',result)
    except BaseException as e:
        save(root/'failure.json',dict(status='PREPARATION_FAILURE',stage=stage,error=repr(e),
            traceback=traceback.format_exc(),seconds=time.monotonic()-start,scientific_submissions=0))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lock',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();print(json.dumps(run(a.lock,a.output)))
