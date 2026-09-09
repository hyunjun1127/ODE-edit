"""Bounded CPU-only publication pipeline for an already completed stage."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .import_assets import sha
from .records import save,digest

PREFIX='project.run_scripts.single_layer_cumulative_risk.'
def call(module,*arguments):
    subprocess.run([sys.executable,'-m',PREFIX+module,*map(str,arguments)],check=True)

def verify(package):
    manifest=json.loads((package/'package-manifest.json').read_text())
    for member in manifest['members']:
        p=package/member['path']
        if p.is_symlink() or not p.is_file():raise ValueError('INVALID_PACKAGE_MEMBER')
        if p.stat().st_size!=member['bytes'] or sha(p)!=member['sha256']:raise ValueError('PACKAGE_REHASH_MISMATCH')
    if digest(manifest['members'])!=manifest['members_root']:raise ValueError('MEMBERS_ROOT_MISMATCH')
    receipt=json.loads((package/'rooted-receipt.json').read_text());identity=receipt.pop('identity')
    if identity!=digest(receipt) or receipt['manifest_sha']!=sha(package/'package-manifest.json'):raise ValueError('ROOTED_RECEIPT_MISMATCH')
    return dict(members=len(manifest['members']),members_root=manifest['members_root'],receipt_identity=identity,
                report_sha=receipt['report_sha'],manifest_sha=receipt['manifest_sha'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['A','B','C'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--covariance-repair',type=Path,action='append',default=[])
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:print(json.dumps(verify(a.output)));return
    if a.output.exists() or a.evidence.exists():raise ValueError('CREATE_ONCE_PUBLICATION_NAMESPACE_REQUIRED')
    call('completeness','--registry',a.registry,'--stage',a.stage,'--output',a.evidence)
    completion=a.evidence/'completion.json';completed=json.loads(completion.read_text())
    if completed['status']!='COMPLETE':raise ValueError(f"INCOMPLETE_STAGE see {completion}")
    call('analysis','--root',a.root,'--registry',a.registry,'--stage',a.stage,'--output',a.output)
    repairs=[arg for path in a.covariance_repair for arg in ['--covariance-repair',path]]
    call('auxiliary_analysis','--registry',a.registry,'--stage',a.stage,'--output',a.output/'auxiliary',*repairs)
    call('metadata_analysis','--request-table',a.output/'request-metrics.csv.gz','--output',a.output/'metadata')
    if a.stage=='A':call('baseline_reuse_audit','--registry',a.registry,'--output',a.output/'baseline-reuse')
    if a.stage=='B':call('direction_analysis','--registry',a.registry,'--output',a.output/'directions')
    call('plotting','--input',a.output/'paired-summary.csv','--trajectory',a.output/'trajectory.csv','--output',a.output/'figures')
    call('mechanism_plotting','--summary',a.output/'paired-summary.csv','--trajectory',a.output/'trajectory.csv',
         '--structures',a.output/'auxiliary/structural-risk.csv','--output',a.output/'mechanism-figures')
    call('discussion','--package',a.output,'--completion',completion)
    # Raw endpoint payloads remain local. This artifact contains identities only.
    call('artifact_inventory','--registry',a.registry,'--stage',a.stage,'--output',a.output/'raw-artifact-inventory.json')
    for name in ['completion.json','requirements-evidence.csv']:
        with (a.output/name).open('xb') as f:f.write((a.evidence/name).read_bytes())
    save(a.output/'publication-inputs.json',dict(registry_path=str(a.registry),registry_sha=sha(a.registry),
         input_lock_path=str(a.root/'input.lock.json'),input_lock_sha=sha(a.root/'input.lock.json'),
         control_override_path=str(a.root/'control-override.json'),control_override_sha=sha(a.root/'control-override.json'),
         raw_payload_git=0,model_action=0,performance_gates=0,stage=a.stage))
    call('report','--package',a.output,'--completion',completion)
    print(json.dumps(verify(a.output)))

if __name__=='__main__':main()
