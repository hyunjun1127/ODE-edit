"""One full destination SHA and CPU shape/dtype check per locked new member."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);plan=json.loads((root/'transfer-allowlist.json').read_text());pending=list(plan['items']);checked=[];started=time.monotonic()
 out=root/'transfer-verified-members.jsonl'
 assert not out.exists()
 with out.open('x') as receipt:
  while pending:
   progressed=False
   for item in list(pending):
    dest=Path(item['destination'])
    if not dest.exists():continue
    before=dest.stat();assert before.st_size==item['size'],str(dest)
    h=hashlib.sha256()
    with dest.open('rb') as f:
     for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    assert h.hexdigest()==item['sha256'],str(dest)
    if item['shape']:
     arr=np.load(dest,mmap_mode='r',allow_pickle=False);assert list(arr.shape)==item['shape'] and str(arr.dtype)==item['dtype'];del arr
    after=dest.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    row=dict(relative=item['relative'],size=after.st_size,sha256=h.hexdigest(),device=after.st_dev,inode=after.st_ino,mtime_ns=after.st_mtime_ns,status='PASS_FULL_SHA_SIZE_CPU_SHAPE_DTYPE')
    receipt.write(json.dumps(row)+'\n');receipt.flush();checked.append(row);pending.remove(item);progressed=True
    if len(checked)%128==0:print('verified',len(checked),'bytes',sum(x['size'] for x in checked),flush=True)
   if not progressed:time.sleep(10)
 verified=dict(status='PASS',members=len(checked),bytes=sum(x['size'] for x in checked),seconds=time.monotonic()-started,missing=0,mismatch=0,free_bytes=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize,validation='each_new_destination_once;source_seal_reuse;runtime_no_rehash')
 (root/'transfer-verified.json').write_text(json.dumps(verified,indent=2)+'\n');print(json.dumps(verified),flush=True)
if __name__=='__main__':main()
