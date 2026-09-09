"""One-shot terminal raw inventory; publish identities, never raw payloads."""
import argparse
import json
import stat
from pathlib import Path
from .import_assets import sha
from .records import save,digest

def inventory(roots):
    members=[]
    for root in sorted(map(Path,roots)):
        if root.is_symlink() or not root.is_dir():raise ValueError('INVALID_RAW_ROOT')
        if not (root/'terminal.json').is_file():raise ValueError('TERMINAL_RECEIPT_REQUIRED')
        for path in sorted(root.rglob('*')):
            st=path.lstat()
            if stat.S_ISDIR(st.st_mode):continue
            if not stat.S_ISREG(st.st_mode):raise ValueError(f'NONREGULAR_RAW_MEMBER {path}')
            before=(st.st_ino,st.st_size,st.st_mtime_ns);checksum=sha(path);after=path.stat()
            if before!=(after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('MUTATING_TERMINAL_MEMBER')
            row=dict(root=str(root),path=str(path.relative_to(root)),bytes=st.st_size,mode=oct(stat.S_IMODE(st.st_mode)),sha256=checksum)
            if path.suffix=='.json':
                data=json.loads(path.read_text())
                row['rows']=len(data['rows']) if isinstance(data,dict) and isinstance(data.get('rows'),list) else None
                row['status']=data.get('status') if isinstance(data,dict) else None
            members.append(row)
    return members

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--stage',choices=['A','B','C'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();registry=json.loads(a.registry.read_text());roots=[]
    for config in registry.values():
        roots += [config['native'],*config.get('direct',[])] if a.stage=='A' else ([config[a.stage]] if a.stage in config else [])
    members=inventory(roots)
    save(a.output,dict(stage=a.stage,members=members,members_root=digest(members),member_count=len(members),
         bytes=sum(r['bytes'] for r in members),raw_payload_git=0,terminal_immutable_check=True,registry_sha=sha(a.registry)))

if __name__=='__main__':main()
