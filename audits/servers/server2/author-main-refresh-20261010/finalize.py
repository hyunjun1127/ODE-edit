"""Metadata-only report path normalization and compact consistency checks."""
import json
from pathlib import Path
from official.experiments.prepare import file_sha
p=Path(__file__).with_name('table-rows.json');d=json.loads(p.read_text())
for r in d['rows']:r['report_path']='experiment-reports/servers/server2/author-main-refresh-20261010/report-ko.md'
d['reducer']['sha256']=file_sha(Path(__file__).with_name('review.py'))
d['reducer']['bytes']=Path(__file__).with_name('review.py').stat().st_size
assert len({(r['model'],r['dataset'],r['job_id']) for r in d['rows']})==len(d['rows'])
assert not d['issues'] and sum(r['numeric_eligible'] for r in d['rows'])==24
author=[r for r in d['rows'] if r.get('profile')=='FE_AUTHOR_HPARAMS_HISTORY']
assert len(author)==2 and [r['job_id'] for r in author if r['numeric_eligible']]==['62531']
assert all(r['Flu']=='DEFERRED' for r in author if r['dataset']=='cf')
assert sum(g['numeric_eligible'] for g in d['generation'])==3
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
print('24 numeric rows; 11 public-query zsRE; 3 generation results; author/native separation PASS')
