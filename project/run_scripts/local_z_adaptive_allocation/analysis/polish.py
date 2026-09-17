"""One-off figure layout correction and an apply_patch payload for metadata.

Prior generated publication bytes are preserved in local review staging. No
scientific raw access, metric changes, runtime imports, or scheduler calls.
"""
import json
from pathlib import Path
import subprocess
from .bootstrap import REVIEW,identity
from .publish import REPORT,copy
from .finalize import AUDIT
from .plot import plots

def main():
    w=Path.cwd();p=w/REPORT;a=w/AUDIT
    backup=REVIEW/'publication-before-layout-polish-v1';backup.mkdir(exist_ok=False)
    paths=[p/'analysis-manifest.json',p/'rooted-receipt.json',p/'evidence-reuse-manifest.json',a/'postrun-checks.json',p/'figures/final-seven-arm.png']
    for f in paths:copy(f,backup/f.name)
    fresh=REVIEW/'figures-polished-v1';repeated=REVIEW/'figures-polished-repeat-v1';plots(p,fresh);plots(p,repeated)
    for f in fresh.glob('*.png'):
        assert identity(f)['sha256']==identity(repeated/f.name)['sha256']
        # Generated scientific figure replacement, never an old scientific input.
        (p/'figures'/f.name).write_bytes(f.read_bytes())
    changes={}
    def revised(f,x):
        old=f.read_text();new=json.dumps(x,ensure_ascii=False,indent=2)+'\n';changes[f]=(old,new)
    def predicted(f):
        import hashlib
        data=changes[f][1].encode() if f in changes else f.read_bytes()
        return dict(path=str(f),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    def load(f):return json.loads(f.read_text())
    reuse=load(p/'evidence-reuse-manifest.json')
    for m in reuse['members']:
        if m['publication'].startswith('figures/'):
            m['prior_layout_source']=m['source'];m['source']=identity(fresh/Path(m['publication']).name)
    reuse['layout_only_repair']=dict(reason='suptitle y1.03 clipped; y.99, no data change',backup=str(backup))
    revised(p/'evidence-reuse-manifest.json',reuse)
    post=load(a/'postrun-checks.json')
    post['PNG']=[dict(file=f.name,bytes=f.stat().st_size,sha256=identity(f)['sha256'],byte_reproduced=True) for f in sorted((p/'figures').glob('*.png'))]
    post['figure_visual_check']='Final bars and selection heatmap inspected; title clipping repaired without metric change'
    revised(a/'postrun-checks.json',post)
    manifest=load(p/'analysis-manifest.json')
    manifest['analysis_source_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    manifest['analysis_source_tree']=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    manifest['source']=[identity(f) for f in sorted((w/'project/run_scripts/local_z_adaptive_allocation/analysis').glob('*')) if f.is_file()]
    manifest['reused_evidence']=predicted(p/'evidence-reuse-manifest.json');manifest['postrun']=predicted(a/'postrun-checks.json')
    for m in manifest['artifacts']:
        f=p/m['path'];current=predicted(f);m.update(bytes=current['bytes'],sha256=current['sha256'])
    revised(p/'analysis-manifest.json',manifest)
    receipt=load(p/'rooted-receipt.json');receipt['manifest']=predicted(p/'analysis-manifest.json');receipt['postrun']=predicted(a/'postrun-checks.json');receipt['analysis']=manifest['analysis_source_head']
    revised(p/'rooted-receipt.json',receipt)
    audit=a/'audit-report-ko.md';old=audit.read_text();new=old.replace(load(p/'rooted-receipt.json')['manifest']['sha256'],receipt['manifest']['sha256'])
    changes[audit]=(old,new)
    print('*** Begin Patch')
    for f,(old,new) in changes.items():
        print('*** Update File: '+str(f));print('@@')
        for line in old.splitlines():print('-'+line)
        for line in new.splitlines():print('+'+line)
    print('*** End Patch')

if __name__=='__main__':main()
