import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

TASK='gpt2-xl-wikipedia-alpha-projector'
NONCE='USER-GH-SH1-GPT2XL-WIKIPEDIA-ALPHA-PREP-20261007-R1'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/gpt2-xl-stats-projector/20261007-v1')
MODEL=Path('/mnt/raid5/janghj/.cache/huggingface/hub/git-checkouts/openai-community--gpt2-xl')
EASY=Path('/mnt/raid5/janghj/EasyEdit')
OLD=Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-alpha-jv-diagnosis-sweep-v1/local/state/easyedit-gpt-stats-v1')
STOCK=Path('/mnt/raid5/janghj/.codex/worktrees/easyeditsh1-gpt-stats-stock3488-v1')
LAYERS=(13,14,15,16,17)
WT=Path(__file__).resolve().parents[3]
PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
TRACKING='/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env'

def require(ok,code):
    if not ok:raise RuntimeError(code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def member(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p),bytes=s.st_size,sha256=sha(p),inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns)

def verify(row,full=False):
    p=Path(row['path']);s=p.stat()
    require(s.st_size==row['bytes'],'ASSET_SIZE:'+str(p))
    if full:require(sha(p)==row['sha256'],'ASSET_SHA:'+str(p))
    else:
        for key,val in [('inode',s.st_ino),('device',s.st_dev),('mtime_ns',s.st_mtime_ns)]:
            if key in row:require(row[key]==val,'ASSET_STAT:'+str(p))

def write(path,obj):
    """Create-once atomic complete publication, no replace of existing artifact."""
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    require(p.is_relative_to(LOCAL) or p.is_relative_to(WT),'OUTPUT_SCOPE')
    require(not any(x.is_symlink() for x in [p,*p.parents]),'OUTPUT_SYMLINK')
    fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'-',dir=p.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(tmp,p)
    finally:os.unlink(tmp) # only our mkstemp staging file, never a source artifact

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def stats_path(layer):
    return EASY/f'examples/data/stats/gpt2-xl/wikipedia_stats/transformer.h.{layer}.mlp.c_proj_float32_mom2_100000.npz'
