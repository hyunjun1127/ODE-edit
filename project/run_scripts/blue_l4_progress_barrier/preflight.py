"""Minimal input/source/dependency locks; CPU only, no model evaluation."""
import json
import subprocess
from pathlib import Path
import torch
from scripts.fixed_counterfact import load_prefix,verify
from project.run_scripts.single_layer_cumulative_risk import binding
from project.run_scripts.single_layer_cumulative_risk.records import save,tensor_sha,digest
from .transfer import DEST,sha

DEPS=Path('/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1/deps-transformers-4.44.2')
REPO=Path(__file__).resolve().parents[3]
DESIGN='plans/global/2026-09-11-blue-l4-only-progress-preserving-barrier-experiment-design.md'

def main():
    torch.set_num_threads(8)
    root=DEST/'imports';old=json.loads((root/'input.lock.json').read_text())
    binding.ROOT=root
    assert sha(REPO/DESIGN)=='30be08eccc4b74b34acfeb9e3deb5cbffc9e6c4a02768e6f69fddf5593d99a8e'
    data=verify(binding.DATA);records=load_prefix(binding.DATA,10000)
    assert sha(binding.PROJECTOR)==old['projector']['stack_sha'],'P_SOURCE_MISMATCH'
    assert sha(binding.COVARIANCE)==old['covariance']['sha256'],'C0_SOURCE_MISMATCH'
    models=[]
    for member in old['model_members']:
        p=Path(member['path']);assert p.stat().st_size==member['bytes']
        if len(member['sha256'])==64:assert sha(p)==member['sha256']
        models.append(member)
    entries={}
    for name,revision in [('Early','r4'),('Middle','r1'),('Late','r4')]:
        native=root/f'A/{name}/native-{revision}'
        receipt=json.loads((native/'prepared-receipt.json').read_text())
        assert sha(native/'prepared.pt')==receipt['sha256']
        prep=torch.load(native/'prepared.pt',weights_only=True,map_location='cpu',mmap=True)
        cp,targets,current=binding.load_entry(name,records)
        assert torch.equal(prep['We'],cp['weights'][binding.WEIGHT])
        assert torch.equal(prep['M'],cp['cache_c'])
        assert prep['contexts']==cp['metadata']['contexts'] and prep['entry']==name
        for key in ('We','WN','W0','Ub','K','M'):
            assert prep[key].dtype==torch.float32 and torch.isfinite(prep[key]).all(),key
        assert prep['WN'].shape==(4096,14336) and prep['Ub'].shape==(14336,100)
        panel=old['entries'][name]['panels']
        for i,h in panel['record_hashes'].items():assert digest(records[int(i)])==h
        for endpoint in ('W0','ENTRY','N'):
            obs=json.loads((native/f'{endpoint}-full.json').read_text())
            assert obs['panel_identity']==digest(panel) and len(obs['rows'])==3900
        entries[name]=dict(prepared=str(native/'prepared.pt'),prepared_sha=receipt['sha256'],
                 WN_sha=tensor_sha(prep['WN']),We_sha=tensor_sha(prep['We']),W0_sha=tensor_sha(prep['W0']),
                 Ub_sha=tensor_sha(prep['Ub']),M_sha=tensor_sha(prep['M']),K_sha=tensor_sha(prep['K']),
                 panel_sha=digest(panel),baseline_full_files={s:sha(native/f'{s}-full.json') for s in ('W0','ENTRY','N')})
        del prep,cp,targets
    dependencies=[]
    for pkg in ('transformers','tokenizers'):
        for p in sorted((DEPS/pkg).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                dependencies.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    assert dependencies
    save(DEST/'input.lock.json',dict(status='CPU_INPUT_IDENTITY_PASS',design_sha=sha(REPO/DESIGN),
         source_parent='285d464361c82d033cbaf88c177fd6c5af8f83df',native_input_lock_sha=sha(root/'input.lock.json'),
         transfer_receipt_sha=sha(DEST/'transfer-receipt.json'),entries=entries,dataset=data,model_members=models,
         P=old['projector'],C0=old['covariance'],dependencies=dependencies,
         dependency_binding=str(DEPS),new_native_calls=0,cross_hardware_numerical_parity='NOT_CLAIMED'))
    save(DEST/'science.lock.json',dict(arms=['H','R','EP','EP-Free','EP-J4','EP-N16'],trajectories=12,logical_steps=104,
         N=8,T=1,h=.125,N16_h=.0625,d_ref_h=.125,kappa=2,epsilon_factor=.1,rank_tolerance=1e-10,
         essence_kl=.0625,action_penalty=.1,nominal_initial_native_norm_fraction=.01,precision='FULL_FP32',
         geometry='FP64',compute_z=0,native_writer=0,history_append=0,renormalize=0,line_search=0,scientific_promotion=False))
    save(DEST/'resource.lock.json',dict(server='server2',cap=2,gpu_per_process=1,cpu=8,mem_mib=60416,gpu_hour_cap=None,
         estimate_gpu_hours=[2.5,4],estimate_not_limit=True,other_jobs_mutation=0))
    print('CPU_INPUT_IDENTITY_PASS',sha(DEST/'input.lock.json'),flush=True)

if __name__=='__main__':main()
