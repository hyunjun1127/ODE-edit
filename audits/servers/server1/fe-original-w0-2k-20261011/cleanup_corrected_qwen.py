from pathlib import Path
import stat
from official.runners.server1.common import read,member
from official.experiments.prepare import write_new
O=Path(__file__).parent
root=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/server1/preparation-r1')
cpath=root/'configs/qwen25.json';c=read(cpath);assert c['method']=='MEMIT_FE_HISTORY'
out=Path(c['output']);assert out==root/'runs/qwen25'
done=read(out/'COMPLETE.json');assert done['identity']['config_sha256']==c['config_sha256']
p=read(out/'checkpoint/latest.json');f=out/'checkpoint'/p['file'];m=member(f);assert m['sha256']==p['sha256']
s=f.lstat();assert stat.S_ISREG(s.st_mode) and not f.is_symlink()
write_new(O/'corrected-qwen-delete-manifest.json',dict(config=member(cpath),terminal=member(out/'COMPLETE.json'),payload=m,inode=s.st_ino,bytes=s.st_size))
assert f.stat().st_ino==s.st_ino;f.unlink();assert not f.exists()
write_new(O/'corrected-qwen-deleted.json',dict(path=str(f),bytes=m['bytes'],deleted=True,backup=False))
print('CORRECTED_QWEN_DELETED',m['bytes'])
