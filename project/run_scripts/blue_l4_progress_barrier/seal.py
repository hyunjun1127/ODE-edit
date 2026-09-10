"""Create-once exact source execution closure, without raw Git additions."""
import json
from pathlib import Path
import subprocess
from .transfer import DEST,sha
from project.run_scripts.single_layer_cumulative_risk.records import save

REPO=Path(__file__).resolve().parents[3]
def create():
    status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True)
    assert not status,'DIRTY_SOURCE'
    paths=subprocess.check_output(['git','ls-files','project/run_scripts','scripts/fixed_counterfact.py'],cwd=REPO,text=True).splitlines()
    members=[dict(path=p,sha256=sha(REPO/p)) for p in paths if p.endswith(('.py','.sbatch'))]
    locks={name:sha(DEST/name) for name in ('input.lock.json','science.lock.json','resource.lock.json','transfer-receipt.json')}
    save(DEST/'execution.lock.json',dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=REPO,text=True).strip(),source_root=str(REPO),
        members=members,locks=locks,dependency_runtime_member_parity='1735/1735 .py/.so exact against server1 deps-py312',
        dependency_nonruntime_exclusions='41 type-stub/unrelated compiled-kernel source files absent locally; not imported by eager Llama',
        scientific_promotion=False))

def verify():
    lock=json.loads((DEST/'execution.lock.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==lock['source_head']
    for m in lock['members']:assert sha(REPO/m['path'])==m['sha256'],m['path']
    for name,h in lock['locks'].items():assert sha(DEST/name)==h,name
    return lock

if __name__=='__main__':create()
