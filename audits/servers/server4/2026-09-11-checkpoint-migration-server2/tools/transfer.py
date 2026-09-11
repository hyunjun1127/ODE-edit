"""Sole SH4 writer, exact files-from private staging, no source removal."""
from inventory import *
import subprocess,argparse
DEST='/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('bundle',choices=['new177','jvp1k']);a=ap.parse_args()
 phases=[('payload','payload-files.txt'),('closure','companion-files.txt'),('supplement','supplement-files.txt')] if a.bundle=='new177' else [('payload','jvp-payload-files.txt'),('closure','jvp-companion-files.txt')]
 result=[]
 for target,allow in phases:
  dest=DEST+a.bundle+'-v1.partial/'+target+'/'
  cmd=['rsync','-a','--no-owner','--no-group','--ignore-existing','--partial-dir=.rsync-partial','--stats','--relative','--files-from='+str(C/allow),'-e','ssh -o BatchMode=yes -o ConnectTimeout=15',str(R)+'/', 'rke-server2:'+dest]
  start=time.time();print('TRANSFER_START',a.bundle,target,flush=True)
  p=subprocess.run(cmd,text=True,capture_output=True)
  row=dict(command=cmd,returncode=p.returncode,seconds=time.time()-start,stdout=p.stdout,stderr=p.stderr,allowlist=member(C/allow),exception='user-approved S4-to-S2 archive only; generic broadcast would target unrelated hosts/path and overwrite; exact no-overwrite staged transfer instead',source_remove=False,destination_seal_owner='SH2')
  save(a.bundle+'-'+target+'-transfer.json',row);result.append(row)
  assert p.returncode==0,p.stderr
  print('TRANSFER_DONE',a.bundle,target,round(row['seconds'],1),flush=True)
 save(a.bundle+'-transfer-complete.json',dict(status='RSYNC_COMPLETED_NOT_DESTINATION_VERIFIED',phases=result))
if __name__=='__main__':main()
