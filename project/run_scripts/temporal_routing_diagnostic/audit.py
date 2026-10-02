"""Bounded owner CPU audit. Does not claim independent-agent/model validation."""
import argparse
import ast
import io
from pathlib import Path
import subprocess
import unittest
from .common import *


def run(repo,out):
    repo=Path(repo);out=Path(out);ns=repo/'project/run_scripts/temporal_routing_diagnostic'
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('project.run_scripts.temporal_routing_diagnostic.test_core')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    require(result.wasSuccessful(),'CPU_TESTS')
    files=[]
    for p in sorted(ns.glob('*.py')):
        ast.parse(p.read_text());files.append(record(p))
    for f in ('runner.py','native.py','observations.py','reduce.py'):
        tree=ast.parse((ns/f).read_text())
        require(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('system','Popen','run','check_output') for n in ast.walk(tree)),'NO_RUNTIME_SUBMIT_OR_SHELL')
    require('cache_template=None' in (ns/'native.py').read_text() and 'blue=False,L2=10' in (ns/'native.py').read_text(),'NATIVE_CONSTANTS')
    require('step in (50,100)' in (ns/'runner.py').read_text(),'SNAPSHOT_STEPS')
    receipt=dict(status='PASS_CPU_ONLY',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        source_files=files,owner_audit=True,independent_agent_red=False,actual_GPU_validation='NOT_TESTED',
        source_numeric_native='EXACT_OFFICIAL_CALL_NO_AST_OR_FORMULA_REWRITE',input_raw_model_preserved=True,
        static_checks=['no runtime submit/repair/shell','native L2=10/blue=false/targetcacheOFF','only50/100 weight snapshots'],
        limitations=['toy algebra and source checks do not establish Llama parity','GPU checks execute inside each scientific branch, no extra fitting'])
    with (out/'cpu-test-log.txt').open('x') as f:f.write(stream.getvalue())
    print(save(out/'owner-cpu-audit.json',receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();Path(a.output).mkdir(parents=True,exist_ok=True);run(a.repo,a.output)
