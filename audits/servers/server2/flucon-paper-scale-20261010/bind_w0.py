"""Cross-owner W0 table composition, not cross-hardware generation parity."""
import csv
import json
from pathlib import Path
from official.experiments.prepare import read, file_sha, write_new
from official.evaluation.generation.observer import _record_identity
from paper_scale import OUT, REPO, ROOT, require, member, verified

data=read(OUT/'paper-table-ready.json')
sh3path=REPO/'audits/servers/server3/flucon-paper-scale-20261010/qwen-w0-provenance.json'
sh3=read(sh3path)
assets=read(ROOT/'asset-preflight.json')
snapshot=assets['assets']['model_snapshot']
def flatten(value):
    if isinstance(value,dict):
        if 'observed_sha256' in value and 'path' in value:
            yield value
        for child in value.values():
            yield from flatten(child)
    elif isinstance(value,list):
        for child in value:
            yield from flatten(child)
byname={Path(r['path']).name:r for r in flatten(snapshot)}
matches=[]
for item in sh3['tokenizer_config_assets_prior_locked']+sh3['model_payload_prior_locked']:
    name=Path(item['requested_path']).name
    local=byname[name]
    require(local['verification']=='SHA256_VERIFIED' and local['observed_sha256']==item['sha256'] and local['bytes']==item['bytes'],'ASSET_'+name)
    matches.append(dict(filename=name,bytes=item['bytes'],sha256=item['sha256'],source='BOTH_OWN_LOCKED_FULL_SHA_RECEIPTS'))
require(len(matches)==10,'TEN_ASSET_MEMBERS')
w0=data['W0_generation']; generation=verified(w0['generation']['raw'])
stream=verified(w0['stream'])
require(sh3['identity']['stream_sha256']==w0['stream']['sha256'] and sh3['ordered_case_ids_sha256']==w0['ordered_sample_sha256'],'COHORT')
require(sh3['cold_state_verified'] and sh3['requests']==2000,'SH3_COLD')
require(Path(sh3['identity']['model_snapshot']).name==w0['generation_conditions']['model_identity'].split('@')[1],'REVISION')
require(generation['identity']['endpoint']=='W0','SH2_COLD_ENDPOINT')
for row,record in zip(generation['rows'],stream):
    raw=verified(row['provenance']['raw_member'])
    require(raw['identity']['record_identity']==_record_identity(record,record['occurrence_index']),'GENERATION_RECORD')
    require(raw['identity']['runtime']==w0['generation']['runtime_sha256'],'GENERATION_RUNTIME')
    require(raw['metrics']==row['metrics'],'ROW_METRICS')
conditions=w0['generation_conditions']
require(conditions['reference_assets_sha256']==assets['asset_identity']['generation_reference_identity_sha256'],'REFERENCES')
require(conditions['eval_seed']==20261007 and conditions['profile']=='cf-cake-native-casebatch-kv-total100-globalrng-v1','GENERATION_PROFILE')
require(conditions['max_total_tokens']==100 and conditions['top_k']==5 and conditions['EOS_stop'] is False,'GENERATION_CONDITIONS')
binding=dict(status='COMPATIBLE_FOR_SPLIT_PROVENANCE_W0_GENERATION_TWO_CELLS',
    SH3_provenance=member(sh3path),SH2_asset_receipt=member(ROOT/'asset-preflight.json'),
    matched_members=matches,ordered_stream_sha256=w0['stream']['sha256'],
    model_revision=assets['model_identity']['revision'],requests=2000,prompts=20000,
    local_observation_members_fullSHA_and_record_identity_verified=2000,
    SH2_generation_conditions=conditions,reference_files=assets['asset_identity']['generation_reference_files_sha256'],
    SH3_factual=dict(job_id='61813',source=sh3['identity']['source'],values_unchanged=True),
    SH2_generation=dict(job_id='61898',source=w0['source_commit'],raw=w0['generation']['raw'],
        Flu=w0['generation']['Flu'],Con=w0['generation']['Con']),
    limitations=['SH3 generation NOT_MEASURED: seed20261002 is not a generation seed',
        'SH2 seed20261007 generation conditions and provenance are separate, not relabeled as SH3',
        'Same base model/cohort table composition only, no cross-hardware or cross-runtime bitwise reproduction claim',
        'Model payload hashes reused from actual locked full-SHA receipts; no model load or new weight hash campaign'],
    GPU=0,forwards=0,raw_mutations=0,WB_mutations=0)
write_new(OUT/'w0-compatibility-final.json',binding)
w0['selected_table_generation_status']=binding['status']
data['compatibility']=member(OUT/'w0-compatibility-final.json')
data['binding_reducer']=member(__file__)
write_new(OUT/'table-rows-final.json',data)
with (OUT/'table-rows-final.csv').open('x',newline='') as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['model','dataset','method','job_id','status','Flu_raw','Flu_paper_x100','Con_raw','Con_paper_x100'])
    for r in data['rows']+[w0]:
        a,b=r['generation']['Flu'],r['generation']['Con']
        writer.writerow([r['model'],r['dataset'],r['method'],r['job_id'],r.get('status',r.get('selected_table_generation_status')),
            a['raw_value'] if isinstance(a,dict) else a,a['paper_display_x100'] if isinstance(a,dict) else a,
            b['raw_value'] if isinstance(b,dict) else b,b['paper_display_x100'] if isinstance(b,dict) else b])
print(json.dumps(dict(status=binding['status'],asset_members=len(matches),observation_members=2000)))
