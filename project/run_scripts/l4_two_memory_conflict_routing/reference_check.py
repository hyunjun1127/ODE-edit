"""Bind reused N/W0/We observations to the existing Git-sealed ABC manifest."""
import json
from .identity import REPO,ROOT,PREPARED,sha,save

def main():
    published=REPO/'experiment-reports/servers/server1/single-layer-cumulative-risk-abc-2026-09-10-v1/A/analysis-manifest.json'
    auxiliary=published.parent/'auxiliary/auxiliary-manifest.json'
    expected={}
    for source in (published,auxiliary):
        for r in json.loads(source.read_text())['inputs']:
            if r['path'] in expected and expected[r['path']]['sha256']!=r['sha256']:raise ValueError('PUBLISHED_IDENTITY_CONFLICT')
            expected[r['path']]=r
    locked={r['path']:r for r in json.loads((ROOT/'control/input.lock.json').read_text())['assets']}
    checked=[]
    for entry,prepared in PREPARED.items():
        if str(prepared) in expected:
            if expected[str(prepared)]['sha256']!=locked[str(prepared)]['sha256']:raise ValueError('PUBLISHED_PREPARED_MISMATCH')
        for name in ('N-full.json','ENTRY-full.json','W0-full.json','N-generation.json'):
            path=prepared.parent/name;row=expected[str(path)]
            if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('PUBLISHED_OBSERVATION_MISMATCH')
            checked.append(dict(entry=entry,**row))
    output=ROOT/'control/published-reference-check.json'
    save(output,dict(status='PUBLISHED_REFERENCE_HASH_PASS',published_manifest=str(published),published_sha=sha(published),
        auxiliary_manifest=str(auxiliary),auxiliary_sha=sha(auxiliary),
        checked=checked,large_prepared_hashes_reused_from_initial_verified_lock=True,model_loads=0,GPU_actions=0))
    print(str(output),sha(output),flush=True)

if __name__=='__main__':main()
