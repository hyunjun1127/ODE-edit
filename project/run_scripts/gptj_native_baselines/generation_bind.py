"""Bind received immutable SH1 evaluator/reference, CPU only, no regeneration."""
import argparse
import json
import time
import resource
import subprocess
from pathlib import Path
from .generation_common import *
from .generation_plan import ready
from project.run_scripts.experiment_generation_eval.assets import load_assets
SHARED_SOURCE='83535c6a47c552cc4e5c6385f3a587d752820150'
PACKAGE='project/run_scripts/experiment_generation_eval'
PACKAGE_TREE='6b9ed049ddaf0cfd431d036125ea8aa7be3724d1'
REFERENCE='75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6'
MANIFEST_SHA='6d9a713ab7eaa10871f277e10a3974e0bd5b140c265be80052c279f258876ca8'
def bind():
    authority()
    out=LOCAL/'preparation-r2'
    require(not out.exists(), 'FINAL_BIND_CREATE_ONCE')
    start=time.monotonic()
    c=read(LOCAL/'preparation-r1/config-provisional.json')
    reference=LOCAL/'inputs/reference-r1'
    received=read(reference/'receipt.json')
    require(received['status']=='READY_VERIFIED_RECEIVER'
            and received['reference_identity']==REFERENCE,'REFERENCE_RECEIVE')
    manifest=reference/'reference-ready-r1/manifest.json'
    require(sha(manifest)==MANIFEST_SHA,'MANIFEST_EXACT_BYTES')
    paths={r['relative'].split('/')[-1]:r['path'] for r in received['files'] if r['relative'].startswith('download-r1/')}
    entry=dict(generation_assets=member(manifest),asset_paths=paths)
    assets=load_assets(entry)
    records=read(c['stream'])[:2000]
    coverage=assets.coverage(records)['summary']
    require(coverage['planned_count']==coverage['reference_available_count']
            ==coverage['generation_prompts_available_count']==2000
            and not coverage['missing_reference_count'] and not coverage['missing_generation_prompts_count'],
            'EXACT_REFERENCE_COVERAGE')
    require(assets.sha==REFERENCE,'LOADED_REFERENCE_IDENTITY')
    tree=subprocess.check_output(['git','rev-parse','HEAD:'+PACKAGE],cwd=ROOT,text=True).strip()
    require(tree==PACKAGE_TREE,'SH1_IMMUTABLE_PACKAGE_TREE')
    model_identity=digest(dict(model=c['model'],revision=c['model_revision'],
        model_assets=c['model_assets'],runtime=c['runtime'],scorer=c['observer_identity'],
        seed=c['seed'],precision='FP32/eager/TF32off/autocastoff'))
    c['generation'].update(common_source_status='READY_BOUND',reference_status='READY_VERIFIED',
        source_sha=SHARED_SOURCE,package_tree=PACKAGE_TREE,reference_assets_sha256=REFERENCE,
        generation_assets=entry['generation_assets'],asset_paths=paths,
        model_identity=model_identity,shared_source_members=[small_member(p) for p in sorted((ROOT/PACKAGE).rglob('*.py'))],
        reference_receive=small_member(reference/'receipt.json'),
        reference_manifest=small_member(manifest),reference_READY=small_member(reference/'reference-ready-r1/READY.json'),
        W0_state_identity=dict(model_identity=model_identity,model_revision=c['model_revision'],
            selected_physical_W=c['cold_W'],actual_model_edits=0,model_state='cold_W0',
            no_history_identity_shared=True),
        generator_route='SH1 full-prefix unpadded row/noKV-cache/nativeEOS actual shared API',
        scoring_versions=assets.manifest['versions'],coverage=coverage,
        reference_loader_no_refit=True,
        cache_scope='Single cold W0 generation observation owner; no edited state/history resume')
    ready(c)
    write(out/'config.json',c)
    receipt=dict(status='READY_BOUND_CPU_SOURCE_REFERENCE_NOT_GPU_PASS',source=SHARED_SOURCE,
        package_tree=tree,reference_identity=REFERENCE,
        config=small_member(out/'config.json'),coverage=coverage,
        reference_receive=received,seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        model_loads=0,GPU=0,new_native_fit=0,reference_regeneration=0,checkpoint_save=False,
        public_scoring_versions=assets.manifest['versions'],
        prior_provisional=small_member(LOCAL/'preparation-r1/config-provisional.json'))
    write(out/'binding.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('reference_receive',)}))
    return c
if __name__=='__main__': bind()

