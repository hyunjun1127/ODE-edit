"""Pure planning and occurrence identity; exact canonical cells are consumed."""
import csv,hashlib,json
from pathlib import Path
ROOT='5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def past_panel(rows,n):
 if n==0:return []
 groups=[[] for _ in range(4)]
 for i,r in enumerate(rows[:n]):
  fields=dict(namespace='memit-hj-v2-past400',anchor_requests=n,ordered_root=ROOT,occurrence_ordinal=i,request_hash=digest(r['requested_rewrite']))
  groups[4*i//n].append((digest(fields),i))
 return sorted(i for g in groups for _,i in sorted(g)[:100])
def sample_batch(rows,start):return sorted(range(start,start+len(rows)),key=lambda i:(digest(dict(namespace='memit-hj-v2-refresh5',ordinal=i,request_hash=digest(rows[i-start]['requested_rewrite']),ordered_root=ROOT)),i))[:5]
def cells(path):
 with Path(path).open() as f:x=list(csv.DictReader(f))
 assert len(x)==28 and sum(r['family']=='main' for r in x)==8
 return x
