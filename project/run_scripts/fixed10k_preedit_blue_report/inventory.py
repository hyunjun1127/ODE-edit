"""Publish only raw-free path/byte/hash closure, never raw content."""
import argparse
from pathlib import Path
from .reduce import read, require, sha, write_csv


def emit(out):
    v=read(out/'independent-validation.json')
    for key in ('execution_lock','terminal'):
        require(sha(v[key]['path'])==v[key]['sha256'],'INPUT_BINDING_'+key)
    lock=read(v['execution_lock']['path']);terminal=read(v['terminal']['path'])
    root=Path(v['terminal']['path']).parent
    source=[dict(path=m['path'],bytes=m['bytes'],sha256=m['sha256'],
                 original_server4_path=m.get('source_path',''),verification='RECALL_INDEPENDENT_FULL_SHA_PASS') for m in lock['members']]
    raw=[dict(path=str(root/m['path']),relative_path=m['path'],bytes=m['bytes'],sha256=m['sha256'],
              verification='RECALL_INDEPENDENT_FULL_SHA_PASS') for m in terminal['manifest_members']]
    raw.append(dict(path=str(root/'terminal.json'),relative_path='terminal.json',bytes=v['terminal']['bytes'],sha256=v['terminal']['sha256'],verification='RECALL_ROOT_SHA_BINDING'))
    write_csv(out/'preedit-source-member-inventory.csv',source)
    write_csv(out/'preedit-output-member-inventory.csv',raw)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);emit(p.parse_args().out)
