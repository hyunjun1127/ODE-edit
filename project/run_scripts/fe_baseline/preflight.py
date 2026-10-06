"""Owner focused CPU evidence, explicitly not an actual model PASS."""
import ast
import argparse
import io
import time
import unittest
from . import *
def main():
    import torch
    torch.set_num_threads(6);start=time.monotonic();buffer=io.StringIO()
    for p in (ROOT/'project/run_scripts/fe_baseline').glob('*.py'):ast.parse(p.read_text())
    result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('project.run_scripts.fe_baseline.test_fe'))
    p=argparse.ArgumentParser();p.add_argument('--receipt-name',default='cpu');name=p.parse_args().receipt_name
    require('/' not in name and name.startswith('cpu'),'CPU_RECEIPT_NAME')
    out=LOCAL/name;out.mkdir(exist_ok=False);(out/'tests.txt').write_text(buffer.getvalue())
    receipt=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),seconds=time.monotonic()-start,
        source=[member(p) for p in sorted((ROOT/'project/run_scripts/fe_baseline').glob('*.py'))],owner_review=True,independent_reviewer=0,
        actual_model='NOT_RUN',model_load=0,new_GPU=0,checkpoint_saved=False)
    write(out/'receipt.json',receipt);print(buffer.getvalue());require(receipt['passed'],'CPU_TEST_FAILURE')
if __name__=='__main__':main()
