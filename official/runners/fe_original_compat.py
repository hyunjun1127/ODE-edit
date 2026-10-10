"""Exact import/container-only edits to the pinned author clone, idempotent."""
import argparse,subprocess
from pathlib import Path
from official.runners.fe_original import AUTHOR
CHANGES={
 'z_methods/__init__.py': [('from .compute_z_mlp import compute_z as compute_z_mlp',"def compute_z_mlp(*args, **kwargs):\n    raise ImportError('Pinned author tree does not provide compute_z_mlp; unused by firstforward')")],
 'precompute_z.py': [('from load import load_data\n',''),('def precompute_z(cfg: DictConfig) -> None:\n','def precompute_z(cfg: DictConfig) -> None:\n    from load import load_data\n'),
 ('                if cur_out[0].shape[0] == 1:\n                    cur_out[0][0, lookup_idx, :] = z_to_inject','                hidden = cur_out[0] if isinstance(cur_out, (tuple, list)) else cur_out\n                if hidden.shape[0] == 1:\n                    hidden[0, lookup_idx, :] = z_to_inject'),
 ('                    cur_out[0][lookup_idx, 0, :] = z_to_inject','                    hidden[lookup_idx, 0, :] = z_to_inject')]
}
def expected(root):
    assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==AUTHOR
    result={}
    for path,changes in CHANGES.items():
        raw=subprocess.check_output(['git','-C',str(root),'show',f'{AUTHOR}:{path}'],text=True)
        for old,new in changes:
            assert raw.count(old)==1,(path,old);raw=raw.replace(old,new)
        result[path]=raw
    return result
def check(root):
    for path,value in expected(root).items():assert (root/path).read_text()==value,path
    changed=subprocess.check_output(['git','-C',str(root),'diff','--name-only'],text=True).splitlines()
    assert set(changed)==set(CHANGES)
    return True
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();check(a.root);print('EXACT_PACKAGING_CONTAINER_DIFF_PASS')
