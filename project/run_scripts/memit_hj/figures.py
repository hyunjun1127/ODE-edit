"""CPU-only scientific tables and figures from already sealed scalar raw."""
import csv,json
from pathlib import Path
def make(output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(output);dest=out/'reduced';layers=[];z=[]
 for p in sorted((out/'cells').glob('*/C*/writer.json')):
  r=json.loads(p.read_text());cell=p.parent.parent.name;n=int(p.parent.name[1:])
  for a in r['layers']:
   layers.append(dict(cell=cell,cursor=n,layer=a['layer'],q=a.get('q'),residual_norm=a.get('residual_norm'),
    capacity_trace=a.get('trace'),energy=a.get('energy'),anchor_A_energy=a.get('anchor_A_energy'),
    D_norm=a.get('D_norm'),observed_norm=a.get('observed_norm'),alpha=a.get('alpha'),cosine=a.get('cosine'),error=a.get('error'),
    ideal_to_FP32_relative=a.get('ideal_to_FP32_update_relative',a.get('ideal_to_FP32_relative'))))
  for item in r.get('z',r.get('shadow_divisor',{}).get('z',[])):
   z.append(dict(cell=cell,cursor=n,case_id=item['case_id'],status=item.get('status','NATIVE'),calls=item.get('calls',item.get('native_loss_calls')),
    final_loss=item.get('value',(item.get('loss_trace') or [{}])[-1].get('total')),normalized_PG=item.get('normalized_PG')))
 for name,rows in [('geometry.csv',layers),('z-states.csv',z)]:
  if rows:
   with (dest/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True)
 for root in sorted((out/'cells').glob('main_*')):
  xs=[];ys={k:[] for k in ['RS','PS','NS']}
  for p in sorted(root.glob('C*/all-seen.json')):
   r=json.loads(p.read_text())
   if set(r['metrics'])!=set(ys):continue
   xs.append(int(p.parent.name[1:]))
   for k in ys:ys[k].append(100*r['metrics'][k]['rate'])
  if xs:
   for ax,k in zip(axes,ys):ax.plot(xs,ys[k],label=root.name.removeprefix('main_'));ax.set_title(k);ax.set_xlabel('Committed occurrences');ax.grid(alpha=.2)
 axes[0].set_ylabel('NLL preference (%)');axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'main-preference.png',dpi=180);plt.close(fig)
 if layers:
  fig,ax=plt.subplots(figsize=(7,4))
  for cell in sorted({x['cell'] for x in layers if x['cell'].startswith('writer_0_')}):
   rs=[x for x in layers if x['cell']==cell and x['layer']==8 and x['q'] is not None]
   if rs:ax.plot([x['cursor'] for x in rs],[x['q'] for x in rs],label=cell.removeprefix('writer_0_'))
  ax.set_xlabel('W0 diagnostic continuation occurrences');ax.set_ylabel('L8 residual norm / entry residual norm');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'W0-L8-residual.png',dpi=180);plt.close(fig)
