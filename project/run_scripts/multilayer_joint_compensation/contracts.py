"""Source identities and task-local policy; no model or legacy imports at import."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from project.run_scripts.single_layer_cumulative_risk.records import canonical,digest,save,tensor_save,tensor_sha,Ledger

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'local/multilayer-joint-compensation/20260911-v1'
ABC=Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1')
DATA=Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')
MODEL=Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')
P_RAW=Path('/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt')
NAMESPACE='ODEEDIT-MULTILAYER-JOINT-COMPENSATION-20260911-V1'
ENTRIES={'Early':(10,1000,'native-r4'),'Middle':(50,5000,'native-r1'),'Late':(90,9000,'native-r4')}
WEIGHTS={l:f'model.layers.{l}.mlp.down_proj.weight' for l in (4,5,6,7,8)}
DESIGN='plans/global/2026-09-11-multilayer-joint-edit-and-compensation-design.md'
ATTACHMENT=Path('/mnt/raid5/janghj/.codex/attachments/cb28c4e5-2f55-4a68-ab49-2691d5de5559/pasted-text.txt')

@dataclasses.dataclass(frozen=True)
class Science:
    version: str='multilayer-joint-compensation-ab-v1'
    current_tau: float=.1
    past_tau: float=.1
    native_risk_floor: float=1e-3
    native_ridge: float=1.
    metric_tau: float=.01
    current_gamma: float=1.
    balance_lambda: float=.1
    adam_updates: int=25
    adam_lr: float=.1
    adam_betas: tuple=(.9,.999)
    adam_epsilon: float=1e-8
    bf_nodes: int=4
    horizon: float=1.
    pcg_relative_target: float=1e-4
    pcg_maxiter_per_rhs: int=20
    risk_order: tuple=('Base','Past')
    slack_rho: tuple=(2.,1.)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def member(path):
    p=Path(path);s=p.lstat()
    if not stat.S_ISREG(s.st_mode):raise ValueError(f'NONREGULAR_INPUT: {p}')
    return dict(path=str(p.absolute()),bytes=s.st_size,mode=oct(stat.S_IMODE(s.st_mode)),sha256=sha(p))

def hf_member(path):
    """Bind a normal HF snapshot link AND its regular in-repository blob."""
    p=Path(path).absolute(); snapshot=MODEL.absolute()
    p.relative_to(snapshot)
    s=p.lstat()
    if stat.S_ISREG(s.st_mode):return member(p)
    if not stat.S_ISLNK(s.st_mode):raise ValueError(f'INVALID_HF_MEMBER: {p}')
    resolved=p.resolve(strict=True)
    resolved.relative_to(snapshot.parent.parent/'blobs')
    result=member(resolved)
    return dict(path=str(p),type='HF_SNAPSHOT_SYMLINK',link_target=os.readlink(p),
      blob=result,sha256=result['sha256'],bytes=result['bytes'])

def m0():
    names=[DESIGN,'messages/head/2026-09-11-multilayer-joint-edit-a-sh1.md',
      'messages/head/2026-09-11-multilayer-joint-compensation-common.md',
      'transfers/approvals/2026-09-11-multilayer-joint-compensation-input-sharing.md',
      'plans/global/2026-09-11-multilayer-joint-edit-and-compensation-design-math-check.json',
      'experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md',
      'experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/independent-review-ko.md',
      'PROTOCOL.md','servers/connection-inventory.md']
    paths=[ATTACHMENT]+[REPO/n for n in names]
    paths += [ABC/'imports/blue-source/AlphaEdit'/n for n in ('AlphaEdit_main.py','compute_z.py','compute_ks.py')]
    paths += [REPO/'project/run_scripts/single_layer_cumulative_risk'/n for n in ('binding.py','objective.py','microbatch.py','records.py','import_assets.py')]
    paths += [REPO/'project/run_scripts/l4_two_memory_conflict_routing'/n for n in ('native.py','geometry.py','controller_repair.py','runtime.py','evaluation.py')]
    expected={str(ATTACHMENT):('9094300705dca8f07b73dc9a6ba116c8016f660081f3dc9171a57544ece66400',8940,114),
      str(REPO/DESIGN):('a8f32937ddd2fc4f87cb3a2cfb9ef95de67d2fd39ce877142fa7b80a1884bf05',35133,280)}
    members=[]
    for path in paths:
        item=member(path);item['newline_count']=path.read_bytes().count(b'\n')
        if str(path) in expected:
            assert (item['sha256'],item['bytes'],item['newline_count'])==expected[str(path)]
        members.append(item)
    prepared=[ABC/'A'/e/attempt/'prepared.pt' for e,(_,_,attempt) in ENTRIES.items()]
    receipt=dict(status='FULL_READ_M0',base_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      base_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),
      worktree=str(REPO),members=members,members_root=digest(members),science=dataclasses.asdict(Science()),
      prepared_inventory=[dict(path=str(p),exists=p.is_file(),bytes=p.stat().st_size if p.exists() else None,
          tensor_validation='PENDING_ACTUAL_LOAD') for p in prepared],
      source_owner='SH1',rsync_owner='SH2',cross_server_VERIFIED=False,
      project_cap=2,memory_mib=182272,gpu_hour_cap=None,new_model_actions=0,new_GPU_actions=0,new_submissions=0,
      scientific_promotion=False)
    out=ROOT/'common/control/m0.json';save(out,receipt)
    print(json.dumps(dict(path=str(out),sha256=sha(out),status=receipt['status'])),flush=True)

if __name__=='__main__':m0()
