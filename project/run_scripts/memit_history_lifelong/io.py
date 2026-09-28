"""Small scalar-only JSON, atomic create-once; tensor persistence is absent."""
import hashlib,json,os,tempfile
from pathlib import Path
from project.run_scripts.blue_alphaedit_sequential_comparison.integrity import file_sha,tensor_sha,signature,content,restore,digest

def scalar(x):
    import numpy as np
    if isinstance(x,np.generic):return x.item()
    raise TypeError('Unsupported JSON object: '+type(x).__name__)

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False,default=scalar,separators=(',',':')).encode()+b'\n'
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(tmp,path) # fails if published; never overwrite prior artifacts
    finally:os.unlink(tmp)
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
