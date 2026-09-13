"""Deterministic PNG plots from public, independently reduced CSV only."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ARMS=['N4','S875','S75','FULL8','RES8','REFIT4']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def run(root):
 root=Path(root);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
 records=[]
 def save(fig,name,inputs,caption):
  fig.tight_layout();p=root/name;fig.savefig(p,metadata={'Software':'seq10-reviewed-code-v1'});plt.close(fig)
  records.append(dict(path=name,sha256=sha(p),source_sha256=sha(__file__),inputs={x:sha(root/x) for x in inputs},command=f'{sys.executable} -m project.run_scripts.low_cost_write_donor_pilot.seq_review_plots --root {root.resolve()}',cwd=str(Path.cwd()),caption=caption))
 bm=rows(root/'batchmetrics.csv');fm=rows(root/'finalpopulationmetrics.csv');nll=rows(root/'nll-distributions.csv');gen=rows(root/'general-panels.csv')
 fig,axs=plt.subplots(1,3,figsize=(13,4))
 for ax,m in zip(axs,['RS','PS','NS']):
  for pop,marker in [('fullseen','o'),('entry_old','s'),('suffix','^'),('current','x')]:
   d={r['arm']:r for r in fm if r['metric']==m and r['population']==pop and r['group']=='ALL'}
   ax.plot(range(6),[100*float(d[a]['rate']) for a in ARMS],marker+'-',label=pop)
  ax.set_xticks(range(6),ARMS,rotation=35);ax.set_title('W60 '+m);ax.set_ylabel('Prompt success (%)');ax.legend(fontsize=7)
 save(fig,'final-populations.png',['finalpopulationmetrics.csv'],'W60 full6000/old5000/suffix1000/current100 are distinct populations; overlapping rows are not independent evaluations.')
 fig,axs=plt.subplots(2,3,figsize=(13,7))
 for i,pop in enumerate(['current','historical']):
  for j,m in enumerate(['RS','PS','NS']):
   ax=axs[i,j]
   for arm in ARMS:
    rr=[r for r in bm if r['arm']==arm and r['population']==pop and r['metric']==m and r['group']=='ALL'];ax.plot([int(r['batch']) for r in rr],[100*float(r['rate']) for r in rr],'o-',label=arm)
   ax.set_title(pop+' '+m);ax.set_xlabel('Global batch');ax.set_ylabel('Success (%)');ax.legend(fontsize=6)
 save(fig,'batch-trajectories.png',['batchmetrics.csv'],'Current changes cohort each batch; fixed Historical128 does not. Neither current pooling nor Historical128 is all-seen retention.')
 co=rows(root/'cohort-final.csv');fig,axs=plt.subplots(3,1,figsize=(15,8))
 for ax,m in zip(axs,['RS','PS','NS']):
  vals=[[100*float(next(r['rate'] for r in co if r['arm']==a and r['metric']==m and int(r['cohort_batch'])==b)) for b in range(1,61)] for a in ARMS]
  im=ax.imshow(vals,aspect='auto',vmin=0,vmax=100,cmap='viridis');ax.set_yticks(range(6),ARMS);ax.set_xticks([0,9,19,29,39,49,59],[1,10,20,30,40,50,60]);ax.axvline(49.5,color='white',linestyle=':');ax.set_title('W60 '+m+' by original cohort');fig.colorbar(im,ax=ax,label='%')
 save(fig,'cohort-retention.png',['cohort-final.csv'],'Actual W60 cohort outcomes. Vertical separator is historical entry5000/new suffix1000. Not a full checkpoint-by-cohort matrix.')
 fig,axs=plt.subplots(2,3,figsize=(13,7))
 for i,pop in enumerate(['current','fullseen']):
  for j,m in enumerate(['RS','PS','NS']):
   ax=axs[i,j];field='true_nll' if m=='NS' else 'new_nll'
   d={r['arm']:r for r in nll if r['batch']=='60' and r['population']==pop and r['metric']==m and r['group']=='ALL' and r['aggregation_unit']=='PROMPT' and r['field']==field}
   for stat,style in [('median','o-'),('p90','s--'),('p99','^:')]:ax.plot(range(6),[float(d[a][stat]) for a in ARMS],style,label=stat)
   ax.set_xticks(range(6),ARMS,rotation=35);ax.set_title(pop+' '+m+' desired NLL');ax.set_ylabel('nats / target token');ax.legend()
 save(fig,'nll-tails.png',['nll-distributions.csv'],'Median/p90/p99 are prompt distributions, not confidence intervals; length normalized target NLL.')
 fig,axs=plt.subplots(1,2,figsize=(12,4))
 for a in ARMS:
  rr=[r for r in gen if r['arm']==a];xx=[int(r['batch']) for r in rr]
  axs[0].plot(xx,[float(r['wiki_nll']) for r in rr],'o-',label=a);axs[1].plot(xx,[int(r['mmlu_correct']) for r in rr],'o-',label=a)
 axs[0].set_title('Wiki128 sequence-mean NLL');axs[1].set_title('MMLU development32 correct count')
 for ax in axs:ax.set_xlabel('Global batch');ax.legend(fontsize=7)
 save(fig,'general-panels.png',['general-panels.csv'],'Fixed Wiki128/24999 next-token positions and MMLUdev32; no audit68, no generation F1, no full-benchmark claim.')
 cc={r['arm']:r for r in rows(root/'compute-summary.csv')};fig,axs=plt.subplots(1,2,figsize=(12,4))
 for ax,fields in [(axs[0],['first_fit_seconds','second_fit_seconds','finalization_seconds','materialization_seconds']),(axs[1],['policy_instrumented_online_seconds','evaluation_seconds'])]:
  bottom=np.zeros(6)
  for k in fields:
   v=np.array([float(cc[a][k]) for a in ARMS]);ax.bar(range(6),v,bottom=bottom,label=k);bottom+=v
  ax.set_xticks(range(6),ARMS,rotation=35);ax.set_ylabel('Recorded seconds / ten batches');ax.legend(fontsize=6)
 axs[0].set_title('Instrumented policy components');axs[1].set_title('Policy and evaluation; other overhead excluded')
 save(fig,'compute-components.png',['compute-summary.csv'],'Recorded synchronous policy stages include instrumentation. Unseparated I/O/restore/setup is not reassigned to pure writer.')
 la=root/'layer-action.csv'
 if la.exists():
  rr=rows(la);fig,axs=plt.subplots(1,3,figsize=(14,4))
  for ax,field in zip(axs,['incremental_frobenius','cumulative_path_frobenius_sum','checkpoint_net_from_W50']):
   for a in ARMS:
    for layer,style in [('4','-'),('8','--')]:
     dd=[r for r in rr if r['arm']==a and r['layer']==layer and r[field] not in ('NOT_CHECKPOINT_BATCH','NOT_RECORDED')]
     if dd:ax.plot([int(r['batch']) for r in dd],[float(r[field]) for r in dd],style,marker='.',label=a+' L'+layer)
   ax.set_title(field);ax.set_xlabel('Global batch');ax.set_ylabel('Frobenius norm');ax.legend(fontsize=5)
  fig.suptitle('Layer-wise Update Magnitude')
  save(fig,'layer-update-magnitude.png',['layer-action.csv'],'Actual stored increments, sum of batch increment norms, and actual checkpoint net from W50 are distinct. Net sampled only at B51/55/60; no uniform-allocation reference.')
 (root/'plot-receipt.json').write_text(json.dumps(records,indent=2)+'\n');return records
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
