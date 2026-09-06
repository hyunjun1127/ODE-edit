"""Seal the already printed native context cache; no generation or model replay.

Only our exact chain log prefix is read. Raw context strings stay in local state,
never in Git/report tables. This complements the real selected-weight/M snapshot.
"""
import argparse,ast,hashlib,json,os
from pathlib import Path
from .reporting import read,sha,jwrite


def seal(chain,log,output):
    chain,log,output=map(Path,(chain,log,output))
    runtime=read(chain/'runtime.lock.json');prefix=b'Cached context templates ';offset=0;found=None
    with log.open('rb') as f:
        for line in f:
            i=line.find(prefix)
            if i>=0:
                if not line.endswith(b'\n'):raise RuntimeError('CONTEXT_LOG_RECORD_INCOMPLETE')
                value=ast.literal_eval(line[i+len(prefix):].decode('utf-8').strip())
                digest=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
                if digest!=runtime['contexts_sha256']:raise RuntimeError('NATIVE_CONTEXT_CACHE_IDENTITY')
                if not isinstance(value,list) or not all(isinstance(g,list) and all(isinstance(s,str) for s in g) for g in value):
                    raise RuntimeError('NATIVE_CONTEXT_CACHE_SCHEMA')
                found=(value,digest,offset,offset+len(line),hashlib.sha256(line).hexdigest());break
            offset+=len(line)
            if offset>1024*1024:raise RuntimeError('NATIVE_CONTEXT_PREFIX_NOT_FOUND')
    if found is None:raise RuntimeError('NATIVE_CONTEXT_CACHE_NOT_RECORDED')
    value,digest,start,end,line_sha=found;output.mkdir(parents=True,mode=0o700)
    path=output/'native-context-cache-local-only.json'
    with os.fdopen(os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600),'w') as f:
        json.dump(value,f,ensure_ascii=False);f.write('\n')
    receipt=dict(contexts_sha256=digest,raw_local_path=str(path.absolute()),file_sha256=sha(path),bytes=path.stat().st_size,
        runtime_lock=str((chain/'runtime.lock.json').absolute()),runtime_lock_sha256=sha(chain/'runtime.lock.json'),
        source_log=str(log.absolute()),source_byte_start=start,source_byte_end=end,source_line_sha256=line_sha,
        origin='EXISTING_NATIVE_STDOUT_RECORD_EXACT_RUNTIME_CACHE_HASH',additional_generation_count=0,
        model_replay_count=0,context_publication_to_git=False,existing_checkpoint_mutation_count=0)
    jwrite(output/'context-recovery-receipt.json',receipt);os.chmod(output/'context-recovery-receipt.json',0o600)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('chain',type=Path);p.add_argument('log',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(seal(a.chain,a.log,a.output),sort_keys=True))
