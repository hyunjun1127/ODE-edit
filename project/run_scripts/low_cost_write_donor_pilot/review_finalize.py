"""Finalize code-generated plots/report, reproducibility receipt and package."""
import argparse,csv,json,platform
from pathlib import Path
import matplotlib,numpy
from .review_provenance import sha,write
from .review_plots import run as plots
from .review_report import run as report
from .review_package import seal
def run(root,attempt,state):
 root=Path(root);attempt=Path(attempt)
 # The first report assembly also exports audited scalar layer actions.
 report(root,attempt,state)
 first=plots(root);second=plots(root)
 assert [x['sha256'] for x in first]==[x['sha256'] for x in second]
 write(root/'reproduction-receipt.json',dict(status='BYTE_STABLE_PNG_REGENERATION_PASS',png_count=len(first),output_sha256={x['path']:x['sha256'] for x in first},python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=numpy.__version__,backend='Agg',seed=20260913,DPI=160,model_forward=0,review_tests=12,reused_preGPU_CPU_tests=12,original_GPU_initial_tests='Stored job receipt reviewed retrospectively, not previously observed by agent',code_only_PNG=True))
 names=['execution.lock.json','execution-source.tar','panel-lock.json','config4.json','config8.json','P0-cpu-receipt.json']
 write(root/'input-manifest.json',dict(attempt=str(attempt),members=[dict(path=str(attempt/n),bytes=(attempt/n).stat().st_size,sha256=sha(attempt/n)) for n in names],raw_output_inventory=dict(path='raw-member-inventory.csv',sha256=sha(root/'raw-member-inventory.csv')),state_CPU_receipt=dict(path=str(state),sha256=sha(state)),new_raw_transfer=0,raw_broadcast='NO_BROADCAST_NOT_REQUIRED'))
 report(root,attempt,state)
 return seal(root)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--attempt',required=True);p.add_argument('--state',required=True);a=p.parse_args();print(json.dumps(run(a.root,a.attempt,a.state)))
