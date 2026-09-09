"""SH4 sole-writer allowlisted rsync; neither delete nor source removal nor overwrite."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEST='/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (ROOT/name).open('x') as f:json.dump(obj,f,sort_keys=True,indent=2)
def remote(mode,expected=''):
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','rke-server2','python3 - '+mode+(' --expected '+expected if expected else '')]
    r=subprocess.run(cmd,input=(ROOT/'remote_seal.py').read_text(),text=True,capture_output=True,check=True)
    save('remote-'+mode+'.json',dict(command=cmd,stdout=r.stdout,stderr=r.stderr,remote_script_sha256=sha(ROOT/'remote_seal.py')))
    print(r.stdout,flush=True)
def main(phase):
    started=time.time()
    if phase=='source':remote('init')
    args=['rsync','-a','--no-owner','--no-group','--ignore-existing','--partial-dir=.rsync-partial','--stats','-e','ssh -o BatchMode=yes -o ConnectTimeout=15']
    if phase=='source':args += [str(ROOT/'source-package')+'/', 'rke-server2:'+DEST+'/source.partial/']
    else:args += ['--relative','--files-from='+str(ROOT/'payload-files.txt'),'/data/janghj/ODE-edit/','rke-server2:'+DEST+'/payload.partial/']
    r=subprocess.run(args,text=True,capture_output=True)
    save('rsync-'+phase+'.json',dict(command=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.time()-started,exception='generic helper preserves source-relative destination and permits overwrites; user-directed custom imports and no-overwrite staged seal require direct allowlisted rsync',source_removal=False,delete=False))
    assert r.returncode==0,r.stderr
    remote(phase,sha(ROOT/'source-package'/('source-seal.json' if phase=='source' else 'checkpoint-manifest.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['source','payload']);main(p.parse_args().phase)
