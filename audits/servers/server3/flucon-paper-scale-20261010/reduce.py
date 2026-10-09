"""Read-only saved W0 reducer and paper-unit rendering; no model imports."""
import collections,csv,datetime,hashlib,json,math
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
REPORT=ROOT/'experiment-reports/servers/server3/flucon-paper-scale-20261010'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d): (OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def display(v):return None if v is None else str((Decimal(str(v))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
assert display('6.252105796227186')=='625.21' and display('0.2591242773267912')=='25.91'
assert display('0.00005')=='0.01' and display(None) is None and display(0)=='0.00'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
global_row=read(ROOT/'audits/global/w0-main-table-20261009/results.json')['observations']['qwen25-cf']
for x in global_row['inventory']:
 p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
base=Path('/data/janghj/ODE-edit/local/qwen-price-q3-eot-2k-20261009/execution-r1')
initial=read(base/'final/initial.json');w0=read(base/'final/W0-result.json')
assert initial['cold']==w0['state']
counts=collections.Counter();correct=collections.Counter();ids=collections.defaultdict(list); seen=set()
for p in sorted((base/'final/W0').glob('chunk-*.json')):
 d=read(p)
 for r in d['rows']:
  key=(r['case_id'],r['kind'],r['prompt_index']);assert key not in seen;seen.add(key)
  a,b=r['true_nll'],r['new_nll'];assert math.isfinite(a) and math.isfinite(b)
  k=r['kind'];ok=(a<b) if k=='N' else (b<a)
  counts[k]+=1;correct[k]+=ok;ids[(r['case_id'],k)].append(int(ok))
assert dict(counts)=={'R':2000,'P':4000,'N':20000}
metrics={}
for k,name in [('R','Eff'),('P','Gen'),('N','Loc')]:
 assert correct[k]==w0['summary'][k]['numerator']
 values=[sum(v)/len(v) for (case,kind),v in ids.items() if kind==k];assert len(values)==2000
 metrics[name]=100*math.fsum(values)/2000
metrics['Score']=3/sum(1/metrics[k] for k in ['Eff','Gen','Loc'])
prep=read(base.parent/'preparation-v1/preparation.json');cell=read(prep['parent_config']['path'])['cells'][prep['parent_cell']]
assets=[{k:x[k] for k in ['path','bytes','sha256'] if k in x} for x in cell['assets'] if any(y in Path(x['path']).name for y in ['tokenizer','vocab','merges','special_tokens','config.json'])]
prov={'observed_at':now,'server':'server3','job_id':'61813','job_name':'qwen-price-q3-eot-2k','endpoint':'W0','identity':global_row['identity'],'ordered_case_ids_sha256':prep['ordered_ids_sha256'],'requests':2000,'cold_state_verified':True,'cold_state':initial['cold'],'initial_rng_sha256':initial['RNG'],'runtime':{k:v for k,v in cell['runtime'].items() if isinstance(v,(str,int,float,bool))},'seed':cell['seed'],'precision':'FP32/eager/TF32off','tokenizer':'AutoTokenizer local snapshot; pad=eos/right at load; factual observer has own masked batching','tokenizer_config_assets_prior_locked':assets,'model_payload_prior_locked':[{k:x[k] for k in ['path','bytes','sha256'] if k in x} for x in cell['assets'] if Path(x['path']).suffix in ['.safetensors','.bin']],'generation':{'status':'NOT_MEASURED_ON_SH3','generation_seed':None,'generation_protocol':None,'reference_assets_sha256':None,'compatibility_with_SH2':'SH2_MUST_BIND_ITS_GENERATION_SOURCE_PROTOCOL_REFERENCE_TOKENIZER_AND_COHORT; no SH3 generation parity claim'},'factual':metrics,'denominators':dict(counts),'numerators':dict(correct),'inventory':global_row['inventory'],'preservation':'raw/WB/checkpoint unchanged; no new forward'}
save('qwen-w0-provenance.json',prov)
old=read(ROOT/'audits/servers/server3/main-table-refresh-20261009/table-rows.json')
rows=[{'model':'qwen25','dataset':'CF','endpoint':'W0','job_id':'61813','job_name':'qwen-price-q3-eot-2k','main_reference_eligible':True,'factual':metrics,'source_commit':initial['source'],'config_sha256':initial['profile_sha'],'cohort_sha256':prep['baseline_stream']['sha256'],'requests':2000,'Flu':{'status':'DEFERRED','raw_unit':'entropy_bits','raw_value':None,'paper_display_x100':None,'display_unit':'entropy_bits_x100'},'Con':{'status':'DEFERRED','raw_unit':'tfidf_cosine','raw_value':None,'paper_display_x100':None,'display_unit':'cosine_x100'}}]
for x in old['rows']:
 entry={k:x.get(k) for k in ['model','method','job_id','job_name','source_commit','config_sha256','main_table_eligible']}
 root=Path('/data/janghj/ODE-edit/local')/('qwen-price-q3-eot-2k-20261009' if x['job_id']=='61813' else 'llama-price-final-2k-20261009')/'execution-r1/final'
 entry.update(result_present=(root/'result.json').exists(),terminal_present=(root/'terminal.json').exists(),scope='selected settings: no automatic main promotion',Flu_status='DEFERRED',Con_status='DEFERRED')
 if entry['result_present']:
  d=read(root/'result.json');entry.update(status=d.get('status'),batches=d.get('batches'),result_sha256=sha(root/'result.json'))
 rows.append(entry)
save('table-rows.json',{'nonce':'USER-GH-FLUCON-PAPER-SCALE-TABLE-REFRESH-20261010-R1','observed_at':now,'rows':rows,'own_new_measured_FluCon_cells':0,'own_zsre':'No completed own CP per existing bounded inventory; no replica evaluation','SH2_supplied_unverified_reference_only':{'Flu':{'raw_value':'6.252105796227186','paper_display_x100':display('6.252105796227186')},'Con':{'raw_value':'0.2591242773267912','paper_display_x100':display('0.2591242773267912')},'status':'SH2_COMPATIBILITY_PENDING_NOT_SH3_MEASURED'},'CPU_checks':'47 existing evidence files SHA-size; 26000 rows finite/unique; request-macro/count/cold-state; Decimal half-up tests','reducer_sha256':sha(__file__),'new_GPU_jobs_forward':0,'reviewer':'owner CPU audit, no independent reviewer','broadcast':'NO_BROADCAST_NOT_REQUIRED'})
with (REPORT/'table-rows.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['server','job','endpoint','metric','status','raw_unit','raw_value','paper_display_x100','display_unit'])
 for k in ['Flu','Con']:
  d=rows[0][k];w.writerow(['server3','61813','W0',k,d['status'],d['raw_unit'],'','',d['display_unit']])
print(json.dumps({'W0':metrics,'counts':dict(counts),'verified_files':len(global_row['inventory']),'rows':len(seen),'own_FluCon':0}))
