"""Read-only actual W0 scalar replay. No SDK, model or historical upload."""
from pathlib import Path
import json
from official.experiments.prepare import read,digest,write_new
from official.runners.server2 import run,cf_display_repair as repair
from official.tracking import schema
old=read(repair.OLD/'manifest.json')
new=dict(old,code_commit=repair.control.command(['git','rev-parse','HEAD']),
    source_members={k:run.file_sha(Path('official')/k) for k in old['source_members']})
compat=repair.compatibility(old,new)
raw=read(repair.OLD/'W0_CF/factual/W0.json');before=digest(raw)
payload=run.evaluate_payload(raw,'W0',0)
cfg=run.tracking_config(new,run.configuration('MEMIT','cf'),'chain',Path('/CPU_ONLY'))
schema.metrics(payload,scientific=True,config_values=cfg)
legacy={k:v for k,v in payload.items() if not k.endswith(('Efficacy_AlphaEdit_display','Generalization_AlphaEdit_display','Specificity_AlphaEdit_display'))}
try:
 schema.metrics(legacy,scientific=True,config_values=cfg);old_failure=None
except ValueError as error: old_failure=str(error)
assert old_failure and 'OFFICIAL_DISPLAY_SCORE_MISMATCH' in old_failure
assert digest(raw)==before
result=dict(status='ACTUAL_RAW_CPU_CALLER_SCHEMA_PASS',source_relabel=False,
 raw_sha256=run.file_sha(repair.OLD/'W0_CF/factual/W0.json'),compatibility=compat,
 requests=len(raw['cases']),legacy_error=old_failure,
 official_metrics={k:v for k,v in payload.items() if k.startswith('official/')},
 model_loads=0,forwards=0,SDK_online=False,historical_upload=False)
write_new(repair.control.OUTPUT/'cf-display-reconcile-r1/actual-raw-check.json',result)
print(json.dumps(result))
