"""Owner CPU regression and import closure receipt, no scheduler/model load."""
import argparse
import ast
import contextlib
import importlib.util
import io
import os
import sys
import time
import unittest
from .common import *

def audit(repo,out,configuration):
    repo=Path(repo).resolve();out=Path(out);start=time.monotonic();sys.path.insert(0,str(DEPS))
    suite=unittest.defaultTestLoader.loadTestsFromName('project.run_scripts.joint_multilayer_bs10.test_core')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    config=read(configuration);validate_execution(config);sys.path.insert(0,str(NATIVE));old=os.getcwd()
    try:
        os.chdir(NATIVE)
        p=config['official_native']['path'];spec=importlib.util.spec_from_file_location('AlphaEdit.joint_import_audit',p)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
        from .runtime import Trace
        hp=AlphaEditHyperParams(**config['hparams']);tr=Trace(module)
        require(hp.layers==list(LAYERS) and hp.L2==10 and hp.blue is False,'NATIVE_HPARAMS')
        native=dict(path=module.__file__,sha256=sha(module.__file__),trace_loss_line=tr.loss_line,
           dependencies={k:str(getattr(v,'__file__','')) for k,v in sys.modules.items() if k.startswith(('AlphaEdit','rome','util'))})
    finally:os.chdir(old)
    from transformers import AutoTokenizer
    from .observations import catalog
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
    token_checks=[]
    for cp in config['checkpoints']:
        data,ids,panels,current,native_rows,neighborhood=catalog(config,tok,cp,config['checkpoints'][cp]['metadata']['contexts'])
        allrows=panels+sum(current.values(),[])+sum(native_rows.values(),[])+sum(neighborhood.values(),[])
        require(len(ids)==100 and len(panels)==288 and len(current)==100 and len(native_rows)==100,'BS1_TOKEN_COUNTS')
        require(all(len(x['positions'])==len(x['target_ids']) and max(x['positions'])<len(x['input_ids']) for x in allrows),'TF_POSITION_SHIFT')
        require(set(neighborhood)=={ids[n-1] for n in MILESTONES if n},'BS1_NEIGHBORHOOD_MILESTONES')
        token_checks.append(dict(checkpoint=cp,rows=len(allrows),token_catalog_sha=digest(allrows),offered100_sha=digest(ids),
            max_input_length=max(len(x['input_ids']) for x in allrows),answer_positions=sum(len(x['positions']) for x in allrows)))
    save(out/'token-preflight.json',dict(configuration=record(configuration),checks=token_checks,model_loaded=False,actual_gpu=False))
    sources=[record(p) for p in sorted((repo/'project/run_scripts/joint_multilayer_bs10').glob('*.py'))]
    for p in (repo/'project/run_scripts/joint_multilayer_bs10').glob('*.py'):ast.parse(p.read_text())
    rec=dict(status='PASS_CPU_ONLY' if result.wasSuccessful() else 'CPU_FAILED',tests=result.testsRun,
       errors=len(result.errors),failures=len(result.failures),seconds=time.monotonic()-start,source=sources,native=native,
       independent_red_agent=False,actual_gpu='NOT_RUN',source_freeze='NOT_YET',
       user_override=OVERRIDE_NONCE,configuration=record(configuration),token_checks=token_checks,
       prior_development_findings=['CPU fixture supplied no raw rows; fixture corrected, production guard unchanged',
           'standalone import audit lacked native globals.yml cwd; runtime already sets approved native cwd',
           'launcher design path missing diagnostic suffix; fixed before any source freeze or submission'])
    save(out/'owner-cpu-audit.json',rec)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'cpu-tests.txt').open('x') as f:f.write(stream.getvalue())
    require(result.wasSuccessful(),'CPU_REGRESSION');print(rec['status'],result.testsRun)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--configuration',required=True)
    a=p.parse_args();audit(a.repo,a.output,a.configuration)
