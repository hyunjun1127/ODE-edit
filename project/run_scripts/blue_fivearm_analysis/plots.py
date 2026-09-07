"""Deterministic headless plots from sealed raw-free tables only."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

COLORS=['#0072B2','#009E73','#E69F00','#CC79A7','#D55E00','#666666']
STYLE={'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160,'figure.dpi':160,'axes.grid':False}

def generate(tables,destination):
 tables=Path(tables);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update(STYLE);np.random.seed(0);ledger=[]
 def emit(fig,name,inputs,caption):
  fig.tight_layout();p=destination/(name+'.png');fig.savefig(p,metadata={'Software':'blue_fivearm_analysis deterministic Agg'});plt.close(fig)
  ledger.append(dict(path=p.name,sha256=sha(p),caption=caption,source_sha256=sha(Path(__file__)),
   inputs=[dict(path=n,sha256=sha(tables/n)) for n in inputs],command=f'python -m project.run_scripts.blue_fivearm_analysis.plots --tables {tables} --output {destination}',
   backend='Agg',DPI=160,style=STYLE,seed=0,missing='No imputation; missing points omitted; lines connect recorded checkpoints only',arm_order=ARMS,colors=COLORS,figure_size=list(fig.get_size_inches())))
 final={r['arm']:r for r in csvread(tables/'final_metrics.csv')}
 fig,axs=plt.subplots(1,3,figsize=(13,4))
 for ax,t in zip(axs,['RS','PS','NS']):
  ax.bar(np.arange(6),[100*float(final[a][t+'_rate']) for a in ARMS],color=COLORS);ax.set_xticks(range(6),ARMS,rotation=40,ha='right');ax.set_ylim(0,100);ax.set_title(t);ax.set_ylabel('Prompt success (%)')
 emit(fig,'final_full1000_performance',['final_metrics.csv'],'Each arm final W10; RS1000/PS2000/NS10000 prompts. O_NATIVE is reference, not a fifth experimental arm. Five experimental arms + one reference.')
 for tab,name,tags in [('allseen_rewrite.csv','allseen_rewrite_trajectory',['RS']),('current_batch.csv','current_batch_trajectory',['RS','PS','NS']),('seen_prefix.csv','checkpoint_seen_prefix',['RS','PS','NS'])]:
  rr=csvread(tables/tab);fig,axs=plt.subplots(1,len(tags),figsize=(6 if len(tags)==1 else 13,4),squeeze=False)
  for ax,t in zip(axs.flat,tags):
   for arm,color in zip(ARMS,COLORS):
    r=sorted([x for x in rr if x['arm']==arm],key=lambda x:int(x['batch']))
    ax.plot([int(x['batch']) for x in r],[100*float(x[t+'_rate']) for x in r],marker='o',label=arm,color=color,lw=1.3)
   ax.set_ylim(0,101);ax.set_xlim(.8,10.2);ax.set_xlabel('Committed batch');ax.set_ylabel('Prompt success (%)');ax.set_title(t)
  axs.flat[0].legend(fontsize=7)
  emit(fig,name,[tab], 'Current:100/200/1000 prompts per batch; all-seen rewrite:100k requests at Wk; checkpoint full:only k=1,5,10, denominators100k/200k/1000k. No unrecorded PS/NS interpolation.')
 rr=csvread(tables/'retention_cohort.csv');fig,axs=plt.subplots(2,3,figsize=(12,7))
 for ax,arm in zip(axs.flat,ARMS):
  m=np.full((10,10),np.nan)
  for r in rr:
   if r['arm']==arm:m[int(r['cohort'])-1,int(r['batch'])-1]=100*int(r['current_success'])/int(r['canonical_denominator'])
  im=ax.imshow(m,vmin=0,vmax=100,cmap='viridis',origin='lower',extent=[.5,10.5,.5,10.5]);ax.set_title(arm);ax.set_xlabel('Evaluation W batch');ax.set_ylabel('Edit cohort')
  fig.colorbar(im,ax=ax,label='RS (%)',shrink=.7)
 emit(fig,'rewrite_retention_heatmap',['retention_cohort.csv'],'Every observed cohort cell n=100 requests; triangular future-cohort cells missing, not zero. Diagonal=current; off-diagonal=historical retention.')
 fig,axs=plt.subplots(1,3,figsize=(13,4))
 for ax,cat,side in zip(axs,['rewrite','rephrase','locality'],['new','new','true']):
  for field,marker in [('median','o'),('p90','^'),('max','x')]:
   ys=[float(final[a][cat+'_target_'+side+'_nll_'+field]) for a in ARMS]
   ax.plot(range(6),ys,marker=marker,label=field)
  ax.set_xticks(range(6),ARMS,rotation=40,ha='right');ax.set_ylim(bottom=0);ax.set_title(cat+' target-'+side);ax.set_ylabel('NLL (nat / target token)');ax.legend()
 emit(fig,'final_nll_tails',['final_metrics.csv'],'Prompt-level NLL median/p90/max at final W10. n=1000 rewrite,2000 rephrase,10000 locality per arm; not request-cluster quantiles.')
 fig,axs=plt.subplots(1,3,figsize=(13,4))
 for ax,cat in zip(axs,['rewrite','rephrase','locality']):
  for field,marker in [('mean','o'),('median','s'),('p90','^')]:
   ys=[]
   for a in ARMS:
    try:ys.append(float(final[a][cat+'_margin_'+field]))
    except (ValueError,KeyError):ys.append(np.nan)
   ax.plot(range(6),ys,marker=marker,label=field)
  ax.set_xticks(range(6),ARMS,rotation=40,ha='right');ax.set_title(cat);ax.set_ylabel('Success-oriented NLL margin (nat/token)');ax.legend()
 emit(fig,'final_margin_distributions',['final_metrics.csv'],'Margin positive means canonical preference success. O/JVP prompt-pair quantiles unavailable, omitted; marginal quantiles never subtracted.')
 rr=csvread(tables/'layer_action.csv');fig,axs=plt.subplots(2,3,figsize=(12,7))
 fig.suptitle('Layer-wise Update Magnitude')
 for ax,arm,color in zip(axs.flat,ARMS,COLORS):
  yy=[]
  for layer in range(4,9):
   v=[float(r['batch_net_norm']) for r in rr if r['arm']==arm and int(r['layer'])==layer]
   yy.append(float(np.mean(v)) if v else 0.)
  ax.bar(range(4,9),yy,color=color);ax.set_title(arm);ax.set_xticks(range(4,9));ax.set_ylabel('Mean batch-net Frobenius norm');ax.set_xlabel('Physical layer')
 emit(fig,'layer_wise_update_magnitude',['layer_action.csv'],'Actual batch-entry to endpoint update norm, mean over10 sequential batches, each batch joint100 requests. Known noneditable layers are zero support, not missing-value imputation. No equal-allocation reference line.')
 rr={r['arm']:r for r in csvread(tables/'compute.csv')};fig,axs=plt.subplots(1,2,figsize=(12,4))
 for ax,field,label in zip(axs,['process_seconds','target_seconds'],['Total process wall time','Native target computation wall time']):
  ax.bar(range(6),[float(rr[a][field])/60 for a in ARMS],color=COLORS);ax.set_xticks(range(6),ARMS,rotation=40,ha='right');ax.set_ylabel('Minutes / 1000-edit chain');ax.set_title(label)
 emit(fig,'compute_wall_time',['compute.csv'],'Recorded process/target wall time; O/JVP Server2 A6000, others Server4 RTX PRO6000. Includes different instrumentation/evaluation overhead; not controlled speedup or actual utilization.')
 return ledger

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tables',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=generate(a.tables,a.output);save(a.output/'plot-manifest.json',result)
