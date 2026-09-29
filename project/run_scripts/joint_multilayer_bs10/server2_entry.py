"""Explicit S2 platform binding before importing frozen scientific modules.

No native, solver, scoring or snapshot formula changes. The original S4 modules
remain byte-exact; this process-local binding never edits shared environments.
"""
import argparse
import sys
from pathlib import Path
from . import common

ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1')
NONCE='ODEEDIT-GH-SH4-SH2-JOINT-BS1-MIGRATION-20260929-R1'
DEPS=Path('/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1/deps-transformers-4.44.2')
NATIVE=ROOT/'inputs/server4-handoff/native-imports'
MODEL=Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots')/common.REVISION
DATA=Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')
PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'

def bind():
    for name in ('ROOT','NONCE','DEPS','NATIVE','MODEL','DATA','PYTHON'):
        setattr(common,name,globals()[name])
    sys.path.insert(0,str(DEPS))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','collect','audit'))
    p.add_argument('--lock');p.add_argument('--index',type=int);p.add_argument('--repo');p.add_argument('--output');p.add_argument('--configuration')
    a=p.parse_args();bind()
    if a.action=='run':
        from .runner import run
        run(a.lock,a.index)
    elif a.action=='collect':
        from .reduce import collect
        collect(a.lock)
    else:
        from .audit import audit
        audit(a.repo,a.output,a.configuration)

if __name__=='__main__':main()
