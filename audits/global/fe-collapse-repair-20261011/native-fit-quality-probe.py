import json,re,statistics
from pathlib import Path
base=Path('/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011')
old=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/qwen-server1-move-r1/registration-r1/logs/qwen25-cf-fe-original-63217.out')
current=base/'native-r2/qwen25-63385.out'
num=r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)'
def collect(path,native):
 rows=[];batch=1;case=None;last=None;counter=0
 for line in path.read_text(errors='replace').splitlines():
  m=re.match(r'NATIVE_FE_BATCH_START (\d+)',line)
  if m:batch=int(m[1]);counter=0
  m=re.search(r'avg prob of \[.*\] ('+num+r')',line)
  if m:last=float(m[1])
  if line.startswith('Init norm '):
   rows.append(dict(batch=batch if native else len(rows)//100+1,prob=last));counter+=1
 return rows
result=dict(scope='Final native z-fit probability over contexts, not CF efficacy/accuracy',reference_log=str(old),current_log=str(current),rows={})
for arm,path,native in [('w0_fixed',old,False),('batch_entry',current,True)]:
 rows=collect(path,native);result['rows'][arm]={}
 for batch in sorted({r['batch'] for r in rows}):
  probs=[r['prob'] for r in rows if r['batch']==batch];result['rows'][arm][str(batch)]=dict(count=len(probs),mean=statistics.mean(probs),median=statistics.median(probs),below_point5=sum(p<.5 for p in probs),below_point01=sum(p<.01 for p in probs))
print(json.dumps(result,indent=2));(base/'native-r2/qwen-fit-quality.json').write_text(json.dumps(result,indent=2)+'\n')
