"""Validate frozen panel membership only; never score audit or load a model."""
import argparse,json,hashlib
from pathlib import Path
from .review_provenance import write,sha
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def run(attempt,out):
 p=Path(attempt);lock=json.loads((p/'execution.lock.json').read_text());panels=json.loads((p/'panel-lock.json').read_text())
 records=json.loads((Path(lock['dataset_root'])/'counterfact.json').read_text())
 assert sha(Path(lock['dataset_root'])/'counterfact.json')=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
 for group,count in [('Current',100),('Historical',128),('Audit',128)]:
  rows=panels[group]['rows'];assert len(rows)==count and len({r['case_id'] for r in rows})==count
  for r in rows:assert records[r['ordinal']]['case_id']==r['case_id'] and digest(records[r['ordinal']])==r['record_sha256']
  assert digest(rows)==panels[group]['identity']
 dev=panels['Current']['rows']+panels['Historical']['rows'];audit=panels['Audit']['rows']
 assert not {r['case_id'] for r in dev}&{r['case_id'] for r in audit}
 def keys(rows):return {(records[r['ordinal']]['requested_rewrite']['subject'],records[r['ordinal']]['requested_rewrite'].get('relation_id')) for r in rows}
 assert not keys(dev)&keys(audit)
 m=panels['MMLU']['groups'];assert len(m['development']['indices'])==32 and len(m['audit']['indices'])==68
 assert set(m['development']['indices']).isdisjoint(m['audit']['indices']) and set(m['development']['indices']+m['audit']['indices'])==set(range(100))
 for a in ['W0','ENTRY','N4','S875','S75','FULL8','RES8','REFIT4']:
  d=json.loads((p/'output'/a/'evaluation.json').read_text())
  assert [r['row_sha256'] for r in d['mmlu']['rows']]==m['development']['row_hashes']
  assert not set(r['row_sha256'] for r in d['mmlu']['rows'])&set(m['audit']['row_hashes'])
  for name in ['current','historical']:
   for metric,ratio in [('RS',1),('PS',2),('NS',10)]:
    actual=[r['case_id'] for r in d[name]['metrics'][metric]['rows']]
    expected=[r['case_id'] for r in panels[name.title()]['rows'] for _ in range(ratio)]
    assert actual==expected
 write(Path(out)/'panel-verification.json',dict(status='CPU_IDENTITY_DISJOINTNESS_PASS',current=100,historical=128,audit_sealed_not_evaluated=128,audit_neighbors_not_evaluated=1280,mmlu_development=32,mmlu_audit_not_evaluated=68,wiki_sequences=128,wiki_predicted_tokens=24999,panel_sha256=sha(p/'panel-lock.json'),audit_data_metadata_reviewed=True,audit_performance_read=False,audit_global_blind=False,unknown_semantic_overlap='NOT_EXCLUDED',current_historical_audit_identity=[panels[x]['identity'] for x in ['Current','Historical','Audit']],audit_selection_details={k:v for k,v in panels['Audit'].items() if k!='rows'},source_runtime_audit_access='NO_SCORING_BRANCH',future_N='NOT_EXECUTED',claim_decision='PENDING_GH_REVIEW'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.attempt,a.out)
