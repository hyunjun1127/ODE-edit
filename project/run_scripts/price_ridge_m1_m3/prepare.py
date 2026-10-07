"""CPU-only task binding from existing immutable attempts and saved W0 rows."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from .profile import modified_profile
from .run import TASK, NONCE

LOCAL=Path('/data/janghj/ODE-edit/local')
PARENTS={
    'LLAMA_REPRO': ('llama3',LOCAL/'jlz-price-cap-base-repair-2k/logging-repair-20261007','LLAMA_CAP075'),
    'GPTJ_M1': ('gptj',LOCAL/'jlz-price-gptj-2k/checkpoint-repair-20261007','MEMIT_CAP075'),
}


def prepare(out,phase0):
    from project.run_scripts.jlz_interference_l1.cap_common import cell_config as llama_cell,member,verify,digest,require
    from project.run_scripts.jlz_price_gptj.common import cell_config as gptj_cell
    from project.run_scripts.jlz_price_gptj.inputs import augment
    evidence=json.loads(Path(phase0).read_text())
    bymodel={x['model']:x for x in evidence['models']}
    previous_gen=json.loads((LOCAL/'gpt2xl-alpha-l13-2k/attempt-logging-repair/config.json').read_text())['generation']
    cells={};inputs=[member(phase0)]
    for cell,(model,old,oldcell) in PARENTS.items():
        cfg=json.loads((old/'config.json').read_text())
        oldlock=json.loads((old/'execution.lock.json').read_text())
        require(member(old/'config.json')['sha256']==oldlock['config_sha256'],'PARENT_CONFIG_BINDING')
        c=(llama_cell if model=='llama3' else gptj_cell)(cfg,oldcell)
        if model=='gptj':
            ready=json.loads((old/'inputs/ready.json').read_text())
            augment(c,ready);c['native_ready']=member(old/'inputs/ready.json');inputs.append(c['native_ready'])
        profile=modified_profile(c['arm_profiles'][oldcell],cell,bymodel[model])
        c.update(task_id=TASK,instruction_id=NONCE,cell=cell,writer='memit',model_alias=model,
            arm_profiles={cell:profile},phase0_status='PASS',w0_policy='REUSE_ONLY_NO_FORWARD',generate_contexts=False,
            parent_attempt=str(old),parent_cell=oldcell,parent_source=oldlock['source_commit'],
            run_instance=dict(attempt='saved-evidence-cold-20261008'),model_revision=Path(c['model']).name)
        c['resources'].update(project_cap=2,task_cap=2,cpu=8,gpu=1,host_mib=59392,hard_host_mib=60416)
        memory=copy.deepcopy(c['resources']['memory_plans']['LLAMA' if model=='llama3' else 'MEMIT'])
        memory['generation_reference_extra_host_GiB']=2.5
        memory['host_peak_with_generation_GiB']=memory['host_peak_GiB']+2.5
        require(memory['host_peak_with_generation_GiB']<58,'RESOURCE_BLOCKED_GENERATION_HOST')
        c['task_memory_plan']=memory
        # Independent full generation outputs remain local. Conservative 8 GiB
        # additional allowance per run; no W0 output allowance is consumed.
        c['generation_reserve_bytes']=8*1024**3
        c['generation']=copy.deepcopy(previous_gen)
        c['generation']['W0_producer']='DISABLED_USER'
        for key in ('READY','reference_manifest'):inputs.append(c['generation'][key])
        for key in ('native_input_alignment','native_full_input_binding','observer_identity'):inputs.append(c[key])
        inputs += [member(old/'config.json'),member(old/'execution.lock.json'),member(c['contexts'])]
        reuse=c.get('W0_reuse',{})
        require(reuse.get('status')=='QUALIFIED_EXACT_REUSE','SAVED_W0_REQUIRED_NO_NEW_EVALUATION')
        inputs += [reuse['runtime'],reuse['summary']]+reuse['chunks']
        # Existing independent raw reducer and state/token identity, CPU only.
        w0=__import__('project.run_scripts.'+('jlz_interference_l1.cap_w0' if model=='llama3' else 'jlz_price_gptj.w0'),fromlist=['install'])
        checkdir=out.parent/'preflight-W0-reference'/cell
        w0.install(checkdir,c,c['cold_W0_H0'],reuse)
        if model=='llama3':
            c['reference_B1_commit']=member(old/oldcell/'batch-01/writer/commit.json')
            c['reference_W20_commit']=member(old/oldcell/'batch-20/commit.json')
            inputs += [c['reference_B1_commit'],c['reference_W20_commit']]
        for row in c['assets']:
            st=Path(row['path']).stat()
            require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_STAT_CHANGED')
        for path in c['stats'].values():require(Path(path).is_file(),'C0_MISSING')
        cells[cell]=c
    # The old bound is conservative for six RPN trajectories; add only two
    # currently admissible generation trajectories plus an error reserve.
    reserve=max(c['storage']['reserve_bytes'] for c in cells.values())+sum(c['generation_reserve_bytes'] for c in cells.values())
    free=shutil.disk_usage(out.parent).free
    require(free>=reserve,'RESOURCE_BLOCKED_STORAGE')
    for row in inputs:verify(row)
    data=dict(task_id=TASK,instruction_id=NONCE,cells=cells,input_members=inputs,
        phase0=member(phase0),storage=dict(free_bytes=free,reserve_bytes=reserve,
        measured=False,shared_reservation=False),blocked_cells={
            'GPT2XL_M1_M2':'SERVER4_C0_L14_L15_L16_L17_AND_NATIVE_INPUT_BINDING_NOT_AVAILABLE',
            'GPT2XL_M1_M3':'NOT_RECORDED_SAVED_B1_K',
            'GPT2XL_M1_M2_M3':'NOT_RECORDED_SAVED_B1_K'},
        later_phases='WAITING_FINAL_PROFILE; held baselines preserved')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2)
    return dict(cells=list(cells),config=member(out),storage=data['storage'],blocked_cells=data['blocked_cells'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--phase0',type=Path,required=True)
    args=p.parse_args();print(json.dumps(prepare(args.out,args.phase0),indent=2))
