"""Static scientific figures from verified measurements; no performance forecast."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P=Path(__file__).resolve().parent
D=json.loads((P/'layer-ledger-evidence.json').read_text())
def read(name):return list(csv.DictReader((P/'report-snapshot'/name).open()))
arms=['N4','F48','G48','C4','C48','C45678']
colors=dict(zip(arms,['#454545','#9a9a9a','#008c79','#518bca','#d27b20','#a343a3']))

fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
ax=axes[0,0]
s=[s for s in D['selections'] if s['arm']=='C45678']
mat=np.array([x['gates'] for x in s]).T
im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis',aspect='auto')
for i in range(5):
 for j in range(10):ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]<.65 else 'black')
ax.set(xticks=range(10),xticklabels=range(1,11),yticks=range(5),yticklabels=[f'L{x}' for x in range(4,9)],xlabel='Batch (100 edits each)',title='C45678: selected native-endpoint gates')
fig.colorbar(im,ax=ax,shrink=.65,label='Gate (not contribution share)')

ax=axes[0,1]
fits=[x for x in D['fits'] if x['arm']=='C45678']
nonzero=[sum(x['layer']==l and x['adam_updates']>0 for x in fits) for l in range(4,9)]
zero=[sum(x['layer']==l and x['adam_updates']==0 for x in fits) for l in range(4,9)]
ax.bar(range(5),nonzero,label='At least one Adam update',color='#a343a3')
ax.bar(range(5),zero,bottom=nonzero,label='Zero Adam updates in whole B100 fit',color='#cfcfcf')
for i,(a,b) in enumerate(zip(nonzero,zero)):ax.text(i,a+b+1,str(a+b),ha='center')
ax.set(xticks=range(5),xticklabels=[f'L{x}' for x in range(4,9)],ylabel='Native B100 fits (incl. rejected / incomplete)',title='C45678: L7-L8 consume 175 of 321 suffix fits',ylim=(0,107))
ax.legend(fontsize=8,loc='upper left')

ax=axes[1,0]
generic=read('generic.csv')
for a in arms:
 rr=[r for r in generic if r['arm']==a and r['role']=='S64_SELECTED_ONLINE']
 ax.plot([0]+[100*int(x['batch']) for x in rr],[0]+[float(x['D']) for x in rr],marker='o',markersize=3,label=a,color=colors[a])
ax.set(xlabel='Cumulative edits',ylabel='KL from fixed W0 (S64)',title='All six chains accumulate output drift')
ax.legend(fontsize=8,ncol=3);ax.grid(alpha=.2)

ax=axes[1,1]
perf=read('performance.csv')
for a in arms:
 vals=[float(next(r for r in perf if r['arm']==a and r['view']==v and r['metric']=='NS')['percent']) for v in ['W5_FIRST500','W10_FIRST500']]
 ax.plot([500,1000],vals,'o-',label=a,color=colors[a])
 offset={'N4':-.12,'C4':.12,'G48':-.10,'C45678':.13}.get(a,0)
 ax.annotate(f'{a} {vals[1]:.2f}',(1000,vals[1]),(1040,vals[1]+offset),fontsize=8,va='center',color=colors[a],arrowprops=dict(arrowstyle='-',color=colors[a],lw=.6))
ax.set(xticks=[500,1000],xlim=(450,1220),ylim=(78,87),xlabel='Cumulative edits (same first 500 cases)',ylabel='Neighborhood success (%)',title='Same-cohort locality keeps declining after edit 500')
ax.grid(alpha=.2)
fig.savefig(P/'allocation-and-horizon.png',dpi=170)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for ax,a in zip(axes,['C48','C45678']):
 for b in range(1,11):
  cs=[c for c in D['candidates'] if c['arm']==a and c['batch']==b]
  base=cs[0]['scores']['base_kl'];best=base;xx=[];yy=[]
  for c in cs:
   if c['feasible']:best=min(best,c['scores']['base_kl'])
   xx.append(c['counts']['suffix_fits']);yy.append(100*(base-best)/base)
  terminal=next(s for s in D['selections'] if s['arm']==a and s['batch']==b)
  xx.append(terminal['counts']['suffix_fits']);yy.append(100*(base-best)/base)
  ax.step(xx,yy,where='post',label=f'B{b}')
 ax.set(xlabel='Cumulative suffix fits (baseline L4 excluded)',ylabel='Best feasible S64 KL reduction vs own N4 (%)',title=a)
 ax.grid(alpha=.2)
axes[1].legend(fontsize=7,ncol=2)
fig.savefig(P/'measured-search-frontier.png',dpi=170)
