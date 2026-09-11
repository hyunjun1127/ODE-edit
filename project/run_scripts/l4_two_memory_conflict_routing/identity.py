"""Create-once receipts and explicitly read-only legacy asset locations."""
from pathlib import Path
from project.run_scripts.single_layer_cumulative_risk.records import canonical, digest, save, tensor_save, tensor_sha, Ledger
from project.run_scripts.single_layer_cumulative_risk.import_assets import sha

REPO = Path(__file__).resolve().parents[3]
ABC = Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1')
ROOT = REPO/'local/l4-two-memory-conflict-routing/20260911-v2'
INSTRUCTION = 'ODEEDIT-S06-L4-TWO-MEMORY-CONFLICT-ROUTING-SH1-V2'
PREPARED = {e:ABC/'A'/e/r/'prepared.pt' for e,r in [('Early','native-r4'),('Middle','native-r1'),('Late','native-r4')]}
WEIGHT = 'model.layers.4.mlp.down_proj.weight'

def member(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError(f'NONREGULAR_MEMBER: {p}')
    s = p.stat()
    return dict(path=str(p),sha256=sha(p),bytes=s.st_size,mode=oct(s.st_mode & 0o777))

def m0():
    import subprocess
    files = [Path('/mnt/raid5/janghj/.codex/attachments/6c0efb88-0cfa-45ab-8c82-9b49db565a41/pasted-text.txt'),
             REPO/'plans/global/2026-09-11-l4-two-memory-conflict-routing-barrier-design.md',
             REPO/'messages/head/2026-09-11-l4-two-memory-routing-sh1-v2.md',REPO/'PROTOCOL.md']
    for name in ('binding','runtime','evaluation','panels','objective','microbatch','algebra','records'):
        files.append(REPO/f'project/run_scripts/single_layer_cumulative_risk/{name}.py')
    for name in ('AlphaEdit_main','compute_ks','compute_z'):
        files.append(ABC/f'imports/blue-source/AlphaEdit/{name}.py')
    files += [ABC/'imports/config.json',REPO/'experiment-reports/servers/server2/blue-l4-progress-barrier-2026-09-11-v1/diagnostic-report-ko.md']
    receipt = dict(instruction_id=INSTRUCTION,status='FULL_READ_M0',members=[member(p) for p in files],
        base_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        base_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=REPO,text=True).strip(),
        worktree=str(REPO),prepared_presence=[dict(path=str(p),bytes=p.stat().st_size) for p in PREPARED.values()],
        prepared_tensor_validation='PENDING_ACTUAL_LOAD',project_gpu_cap=2,memory_mib=182272,gpu_hour_cap=None,
        new_path_count=16,model_loads=0,gpu_actions=0,slurm_submissions=0,scientific_promotion=False)
    receipt['identity']=digest(receipt)
    save(ROOT/'control/m0.json',receipt)
    print(str(ROOT/'control/m0.json'),sha(ROOT/'control/m0.json'),flush=True)

if __name__=='__main__':m0()
