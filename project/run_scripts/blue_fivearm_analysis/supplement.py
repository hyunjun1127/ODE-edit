"""Endpoint-vs-online arithmetic and prompt transition supplements."""
import argparse
from .common import *
from .details import transitions
from .metrics import TAGS

def run(repo):
 out=repo/OUT_REL;current=csvread(out/'current_batch.csv');final={r['arm']:r for r in csvread(out/'final_metrics.csv')};online=[];pairs=[]
 for arm in ARMS:
  rs=[r for r in current if r['arm']==arm];assert len(rs)==10
  z=dict(arm=arm,scope='ONLINE_OWN_BATCH_POOL_DIFFERENT_W_PER_COHORT',request_count=1000)
  for tag in TAGS:
   n=sum(int(r[tag+'_num']) for r in rs);d=sum(int(r[tag+'_den']) for r in rs)
   z.update({tag+'_num':n,tag+'_den':d,tag+'_rate':n/d,tag+'_final_num':int(final[arm][tag+'_num']),tag+'_final_minus_online_pp':100*(float(final[arm][tag+'_rate'])-n/d)})
  online.append(z)
 for arm,root in BLUE_ROOTS.items():
  full=read(root/'B10/seen-full.json')
  for tag in TAGS:
   own=[r for b in range(1,11) for r in read(root/f'B{b:02d}/current.json')['metrics'][tag]['rows']]
   pairs.append(transitions(own,full['metrics'][tag]['rows'],arm,'ONLINE_OWN_BATCH',tag,10))
 csvwrite(out/'online-final-comparison.csv',online);csvwrite(out/'atwrite-final-prompt-transitions.csv',pairs)
 # All three BLUE W0 publications share prompt inventory and exact metrics; shared selected weight hashes also match.
 originals=read(BLUE_ROOTS['BLUE']/'runtime.json')['W0']['weights']
 for arm in ['BLUE_L4_ONLY','BLUE_L8_ONLY']:
  for k,v in read(BLUE_ROOTS[arm]/'runtime.json')['W0']['weights'].items():assert v['sha256']==originals[k]['sha256']
 save(out/'supplement-receipt.json',dict(shared_BLUE_W0_selected_weights_exact=True,online_comparison_arms=6,paired_transitions=9,scope='NO_NEW_EVALUATION'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);run(p.parse_args().repo)
