"""Read-only CPU check of the unchanged native input contract.

No prompt generation, P/N inspection, tokenizer/model calls, source edits, or
output-file writes. This checks sealed source data/text constructions, not a
new execution's tokenizer/model equivalence.
"""
from pathlib import Path
import argparse,hashlib,json,math

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--manifest',type=Path,default=Path(__file__).with_name('exact-native-inputs.json'))
 parser.add_argument('--dataset',type=Path)
 parser.add_argument('--contexts',type=Path)
 args=parser.parse_args();m=json.loads(args.manifest.read_text())
 dp=args.dataset or Path(m['dataset']['local_path']);cp=args.contexts or Path(m['contexts']['local_path'])
 db=dp.read_bytes();cb=cp.read_bytes()
 assert hashlib.sha256(db).hexdigest()==m['dataset']['sha256'],'DATASET_BYTES_CHANGED'
 assert hashlib.sha256(cb).hexdigest()==m['contexts']['sha256'],'CONTEXT_BYTES_CHANGED'
 allrows=json.loads(db);contexts=json.loads(cb)
 assert contexts==m['contexts']['groups'] and list(map(len,contexts))==[1,5] and contexts[0]==['{}'],'CONTEXT_GROUPS_CHANGED'
 assert digest([r['case_id'] for r in allrows])==m['full_fixed10k_ordered_case_ids_sha256'],'FULL_ORDER_CHANGED'
 lo,hi=m['scope']['dataset_slice'];records=allrows[lo:hi]
 assert len(records)==m['scope']['requests']==2000
 ids=[r['case_id'] for r in records]
 assert digest(ids)==m['ordered_case_ids_sha256'],'FIRST2K_ORDER_CHANGED'
 assert digest([{'case_id':r['case_id'],'requested_rewrite':r['requested_rewrite']} for r in records])==m['source_requested_rewrite_sha256'],'SOURCE_PROMPT_SUBJECT_TARGET_CHANGED'
 flat=[x for group in contexts for x in group];assert len(flat)==6
 assert all(x.count('{}')==1 for x in flat)
 keyweights=[1/(len(contexts)*len(group)) for group in contexts for _ in group]
 assert keyweights==m['weights']['key_context_mean_of_group_means']==[.5,.1,.1,.1,.1,.1]
 assert m['weights']['rewrite_context_NLL']==[1/6]*6
 rows=[]
 for r in records:
  rw=r['requested_rewrite'];assert rw['prompt'].count('{}')==1
  assert rw['target_new']['str'] and rw['target_true']['str']
  request_rows=[]
  for i,ctx in enumerate(flat):
   template=ctx.format(rw['prompt']);assert template.count('{}')==1
   request_rows.append([r['case_id'],'rewrite',i,template.format(rw['subject']),rw['target_new']['str']])
  request_rows.append([r['case_id'],'kl',0,'{} is a'.format(rw['subject']),None])
  assert len(request_rows)==7
  rows.extend(request_rows)
 assert digest(rows)==m['target_free_rewrite_and_KL_rows_sha256'],'NATIVE_ROWS_CHANGED'
 assert len(rows)==14000 and sum(r[1]=='rewrite' for r in rows)==12000
 print(json.dumps({'status':'PASS_UNCHANGED_NATIVE_SOURCE_INPUTS','requests':2000,'rewrite_rows':12000,'native_KL_rows':2000,'per_request_rows':[6,1],'context_sha256':m['contexts']['sha256'],'first2k_order_sha256':m['ordered_case_ids_sha256'],'source_requested_rewrite_sha256':m['source_requested_rewrite_sha256'],'key_weights':keyweights,'semantic_curriculum':'WITHDRAWN','writes':0,'tokenizer_model_GPU_calls':0,'tokenization_runtime_claim':'NOT_TESTED_BY_THIS_CPU_INPUT_CHECK'},indent=2))

if __name__=='__main__':main()
