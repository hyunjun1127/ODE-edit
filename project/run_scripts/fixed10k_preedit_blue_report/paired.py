"""Bounded SSH read-only reduction of six explicitly approved sealed final files.

Only local W0 scalar identities/NLLs/bits travel in memory; no remote file writes,
prompts, tensor transfer, scheduler access, model imports or evaluations.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path
from .reduce import read, save, write_csv, sha, require, member

REMOTE = r'''
import sys,json,hashlib,math,statistics
from pathlib import Path
x=json.load(sys.stdin);result=[];cohorts=[];evidence=[]
def check(ok,why):
 if not ok: raise ValueError(why)
def hashfile(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def aggregate(rows,arm,metric,scope):
 counts={k:0 for k in ['retained','lost','gained','both_failed']}
 for w,r in rows:
  k='retained' if w[3] and r['success'] else 'lost' if w[3] else 'gained' if r['success'] else 'both_failed'
  counts[k]+=1
 n=len(rows);before=counts['retained']+counts['lost'];after=counts['retained']+counts['gained']
 out=dict(arm=arm,metric=metric,scope=scope,denominator=n,before_num=before,after_num=after,
          delta_pp=100*(after-before)/n,**counts,
          W0_success_den=before,W0_success_to_failure_num=counts['lost'],
          W0_failure_den=n-before,W0_failure_to_success_num=counts['gained'])
 for side,idx in [('new',4),('true',5)]:
  ds=[r[side+'_nll']-w[idx] for w,r in rows]
  out.update({side+'_NLL_paired_delta_mean':statistics.fmean(ds),side+'_NLL_paired_delta_median':statistics.median(ds),
              side+'_NLL_decreased':sum(d<0 for d in ds),side+'_NLL_increased':sum(d>0 for d in ds),side+'_NLL_equal':sum(d==0 for d in ds)})
 return out
for m in x['members']:
 p=Path(m['path']);check(p.is_file() and not p.is_symlink(),'TYPE')
 check(p.stat().st_size==int(m['bytes']) and hashfile(p)==m['sha256'],'SEALED_INPUT_SHA')
 raw=json.loads(p.read_text());check(raw['requests']==10000,'REQUEST_COUNT')
 for metric,mult in [('RS',1),('PS',2),('NS',10)]:
  wrows=x['w0'][metric];rows=raw['metrics'][metric]['rows'];check(len(rows)==len(wrows)==10000*mult,'DEN')
  seen=set()
  for w,r in zip(wrows,rows):
   check((w[0],w[1],w[2])==(r['case_id'],r['prompt_index'],r['identity']),'EXACT_PAIR_ID_ORDER')
   check(r['identity'] not in seen,'DUPLICATE');seen.add(r['identity'])
   check(math.isfinite(r['new_nll']) and math.isfinite(r['true_nll']),'NONFINITE')
   expected=r['true_nll']<r['new_nll'] if metric=='NS' else r['new_nll']<r['true_nll']
   check(expected==r['success'],'PREFERENCE_DEFINITION')
  pairs=list(zip(wrows,rows));a=aggregate(pairs,m['arm'],metric,'FINAL_W100_FULL10000')
  check(a['after_num']==raw['metrics'][metric]['numerator'],'SOURCE_REDUCER')
  result.append(a)
  for b in range(100):cohorts.append(dict(cohort=b+1,**aggregate(pairs[b*100*mult:(b+1)*100*mult],m['arm'],metric,'FINAL_W100_FIXED_B100_COHORT')))
 check(p.stat().st_size==int(m['bytes']) and hashfile(p)==m['sha256'],'SOURCE_CHANGED_DURING_READ')
 evidence.append(m)
print(json.dumps(dict(status='PASS',exact_id_order_denominator=True,members=evidence,rows=result,cohorts=cohorts,
 no_remote_file_write=True,no_raw_prompt_or_tensor_transfer=True,new_model_GPU_evaluator_Slurm_actions=0),allow_nan=False))
'''


def run(reduction, v2, out):
    require(not out.exists(), 'CREATE_ONCE '+str(out))
    rows=list(csv.DictReader((v2/'raw-member-inventory.csv').open()))
    members=[{k:r[k] for k in ('arm','path','bytes','sha256')} for r in rows if r['path'].endswith('/B100/seen-full.json')]
    require(len(members)==6 and len({m['arm'] for m in members})==6,'EXACT_SIX_MEMBERS')
    w=read(reduction/'w0-scalar-rows.local.json')
    data=dict(members=members,w0={metric:[[r[k] for k in ('case_id','prompt_index','identity','success','new_nll','true_nll')] for r in rs] for metric,rs in w.items()})
    payload=json.dumps(data,separators=(',',':'))
    import shlex
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rke-server4','python3 -c '+shlex.quote(REMOTE)]
    result=subprocess.run(command,input=payload,text=True,capture_output=True,timeout=240)
    require(result.returncode==0,'REMOTE_READ_ONLY_REDUCER '+result.stderr)
    parsed=json.loads(result.stdout);require(parsed['status']=='PASS','PAIRING')
    out.mkdir(parents=True)
    write_csv(out/'w0-final-paired-transitions.csv',parsed.pop('rows'))
    write_csv(out/'w0-final-paired-cohorts.csv',parsed.pop('cohorts'))
    parsed.update(instruction_id='ODEEDIT-S06-FIXED10K-PREEDIT-BLUE-LIFELONG-REPORT-INTEGRATION-SH2-V1',
                  authority='SH4 SEALED FINAL SIX READY; identity receipt SHA1915ab402459cc8eeeb05cbb8473c2fed5f9048873e4f700e6274f8344d79eab',
                  scalar_in_memory_request_bytes=len(payload.encode()),aggregate_response_bytes=len(result.stdout.encode()),
                  raw_replication_bytes=0,raw_automatic_broadcast_exception='approved immutable remote read-only CPU aggregate; six 399737324-byte raw files remain on server4; no prompts or weights transferred',
                  w0_scalar_input=member(reduction/'w0-scalar-rows.local.json'),remote_code_sha256=__import__('hashlib').sha256(REMOTE.encode()).hexdigest())
    save(out/'paired-validation.json',parsed)
    print(json.dumps(parsed,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reduction',type=Path,required=True);p.add_argument('--v2',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.reduction,a.v2,a.out)
