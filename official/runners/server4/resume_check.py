"""Compare actual continuous B3 and independently cold B2 -> resumed B3 receipts."""
import argparse
from pathlib import Path
from official.experiments.prepare import read, write_new


def compare(continuous, resumed):
    a,b=(read(Path(p)/'batch-03-commit.json') for p in (continuous,resumed))
    x,y=(read(Path(p)/'W3-factual.json') for p in (continuous,resumed))
    if a['state']!=b['state']:raise ValueError('NATIVE_RESUME_WEIGHT_HISTORY_CONTEXT_MISMATCH')
    if a['rng_sha256']!=b['rng_sha256']:raise ValueError('NATIVE_RESUME_RNG_MISMATCH')
    if x!=y:raise ValueError('NATIVE_RESUME_FACTUAL_MISMATCH')
    return dict(status='ACTUAL_B3_RECEIPTS_EXACT',weight_history_context=True,
        factual=True,paths=[str(continuous),str(resumed)],
        note='Only valid for actual native receipts; this comparator does not execute a model.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('continuous','resumed','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();write_new(a.output,compare(a.continuous,a.resumed))
