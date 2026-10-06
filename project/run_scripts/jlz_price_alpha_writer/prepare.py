"""Bind exact existing model/native assets and new Alpha configuration, CPU only."""
import argparse,copy,json,os,shutil
from pathlib import Path
from .common import *
from .storage import storage_plan
from .predecessor import PREVIOUS,SOURCE
from .geometry import projector

PROJECTORS={
 'LLAMA':('/data/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt',
          '6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec',4110419877,14336),
 'QWEN':('/data/janghj/EasyEdit/examples/null_space_project_Qwen2.5-7B-Instruct.pt',
         'd3a9687d196f7ee06739253813917a632542717ce1166c0433aa7eec70c5c8ff',7177504758,18944)}

def prepare(out,preflight):
    out=Path(out).resolve();require(out.is_relative_to(LOCAL),'LOCAL_OUTPUT_SCOPE')
    previous_lock=json.loads((PREVIOUS/'execution.lock.json').read_text())
    require(previous_lock['source_commit']==SOURCE and sha(PREVIOUS/'config.json')==previous_lock['config_sha256'],'BASE_EXECUTION_SOURCE')
    c=copy.deepcopy(json.loads((PREVIOUS/'config.json').read_text()))
    # Every imported predecessor Python byte stays exact, not a new numerical PASS.
    unchanged=[]
    for row in previous_lock['source_members']:
        rel=Path(row['path']).relative_to(PREVIOUS/'source')
        if rel.suffix=='.py':
            require(sha(ROOT/rel)==row['sha256'],'READONLY_BASE:'+str(rel));unchanged.append(str(rel))
    for row in c['runtime']['source_members']+c['native_reference']+c['dependency_sources']:verify(row)
    for row in c['assets']:
        st=Path(row['path']).stat()
        require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'EXISTING_ASSET_FRESH_STAT')
    manifest=ROOT/DESIGN/'dispatch-package-manifest.json'
    require(sha(manifest)=='147ea5d9ae3432a623bcbed8181486399133424b1e47da6dd86de41ef7c3581b','CANONICAL_MANIFEST_SHA')
    require(sha(ROOT/ENVELOPE)=='95ada04d7cf1e6cc19a3fc9be10fd10fa2381705c83c7a681e52e0b7cf2ead80','ENVELOPE_SHA')
    authority=[member(manifest),member(ROOT/ENVELOPE),member(ROOT/'plans/global/jlz-price-alpha-writer-2k/execution.json')]
    for row in json.loads(manifest.read_text())['files']:
        p=ROOT/row['path'];require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'CANONICAL_MEMBER_SHA_SIZE')
        authority.append(member(p))
    storage=storage_plan();free=shutil.disk_usage(LOCAL).free
    previous_reserve=c['storage']['reserve_bytes']
    require(free>=storage['reserve_bytes']+previous_reserve,'RESOURCE_BLOCKED_COMBINED_STORAGE')
    memory={};bindings={}
    for name in MODELS:
        mc=c['models'][name]
        for field in ('native_input_alignment','native_full_input_binding','observer_identity'):verify(mc[field])
        p,expected,size,d=PROJECTORS[name];binding=member(p)
        require(binding['bytes']==size and binding['sha256']==expected,'PROJECTOR_CURRENT_SHA_SIZE')
        N=projector(p)
        require(tuple(N.shape)==(5,d,d) and str(N.dtype)=='torch.float32','PROJECTOR_MODEL_SHAPE')
        binding.update(physical_layers=[4,5,6,7,8],shape=list(N.shape),dtype=str(N.dtype),cutoff=.02,
            current_asset=True,requested_path=p,layer_index='physical_layer-4')
        del N;projector.cache_clear()
        mc['projector']=binding;mc['assets'].append(binding);c['assets'].append(binding);bindings[name]=binding
        for arm,profile in mc['profiles'].items():
            profile.update(writer='alphaedit',lambda_alpha=1.,projector_cutoff=.02,
                projector_sha256=expected,lambda_C=0.,diagnostic_C0_scale=1.,
                factor_backend='A0_I_plus_NH_LU',response='Q=solve(I+N(H+KKt),NK)',
                historical_MEMIT_lambda_C=15000.,alpha_geometry_FP64=True)
        # Readonly matching W0 data may exist when the predecessor frontier ends.
        mc['predecessor_W0_roots']=[str(PREVIOUS/(name+'_'+arm)) for arm in ARMS]
        mc['predecessor_observation_config']=member(PREVIOUS/'config.json')
        mp=copy.deepcopy(mc['memory_plan']);hp=mp['host_parts_GiB'];gp=mp['GPU_parts_GiB'];G=1024**3
        hp.pop('raw_A');hp['projector_mmap_resident_max']=5*d*d*4/G;hp['C0_FP32_all_layers']=5*d*d*4/G
        gp['factor_transients']=4*d*d*8/G
        mp.update(host_peak_GiB=max(sum(hp.values()),mp['host_peak_GiB']),GPU_peak_GiB=sum(gp.values()),
            alpha_A0_CPU_copy=False,Alpha_nonsymmetric_LU=True,projector_full_residency_counted=True,
            C0_diagnostic_only=True,measured=False,SDK_sidecar_included=True)
        require(mp['host_peak_GiB']<58 and mp['GPU_peak_GiB']<95,'RESOURCE_BLOCKED_ALPHA_MEMORY_ESTIMATE')
        mc['memory_plan']=mp;memory[name]=mp
    c.update(task_id=TASK,instruction_id=NONCE,writer='alphaedit',attempt=str(LOCAL/'attempt'),
        run_instance=dict(date='2026-10-07',attempt='attempt'),cpu_preflight=member(preflight),
        authority_members=authority,tracking=dict(env_file=str(LOCAL/'tracking.env')),storage=storage,
        readonly_closure=dict(source=SOURCE,unchanged_python=unchanged,shared_modules_changed=[]),
        alpha_binding=dict(projectors=bindings,lambda_alpha=1.,C0_diagnostic_scale=1.,
            operator_residual_relative=1e-8,LOO_relative=1e-6,LOO_zero_absolute=1e-8,
            LOO_normalization='original_unprojected_key_norm',no_tolerance_changes=True))
    c['resources'].update(memory_plans=memory,reserve_bytes=storage['reserve_bytes'],free_bytes=free,
        free_inodes=os.statvfs(LOCAL).f_favail,existing_MEMIT_output_reserve_bytes=previous_reserve,
        combined_output_reserve_bytes=storage['reserve_bytes']+previous_reserve,
        wall_request_basis='20x25 candidate horizon, larger Alpha LU plus native observer; conservative48h ceiling, measured ETA unavailable')
    write(out,c)
    return dict(config=str(out),cells=list(CELLS),memory=memory,combined_reserve_GiB=(storage['reserve_bytes']+previous_reserve)/1024**3,
        actual_B1='NOT_OBSERVED',no_model_load=True,no_new_numerical_test=True)

def finalize(config,preflight,out):
    path=Path(config).resolve();require(path.is_relative_to(LOCAL),'LOCAL_CONFIG_SCOPE')
    c=json.loads(path.read_text());p=json.loads(Path(preflight).read_text())
    require(c['task_id']==p['task']==TASK and p['passed'] and p['numeric_tests']==0,'FINAL_STATIC_BINDING')
    for row in p['source']:verify(row)
    for row in c['assets']:
        s=Path(row['path']).stat();require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'FINAL_ASSET_STAT')
    output=Path(out).resolve();require(output.is_relative_to(LOCAL),'FINAL_CONFIG_SCOPE')
    c['cpu_preflight']=member(preflight);write(output,c)
    return dict(config=str(output),sha256=sha(output),actual_B1='NOT_OBSERVED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--preflight',required=True)
    p.add_argument('--finalize-config')
    a=p.parse_args();print(json.dumps(finalize(a.finalize_config,a.preflight,a.out) if a.finalize_config else prepare(a.out,a.preflight)))
