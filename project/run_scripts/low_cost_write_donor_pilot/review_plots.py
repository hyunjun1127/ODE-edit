"""Deterministic headless plots from reviewed CSV only; no raw/model access."""
import argparse,csv,hashlib,json,sys,shlex
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ARMS=['N4','S875','S75','FULL8','RES8','REFIT4']
COLORS=['#4c566a','#88c0d0','#5e81ac','#a3be8c','#d08770','#b48ead']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def run(root):
 root=Path(root);np.random.seed(20260913)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':120,'savefig.dpi':160,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'lowcost46451'})
 sources=['first-core-table.csv','nll-distributions.csv','paired-transitions.csv','layer-action.csv','compute-ledger.csv']
 tables={x:rows(root/x) for x in sources};manifest=[]
 def save(fig,name,inputs,caption):
  fig.tight_layout();path=root/name;fig.savefig(path,metadata={'Software':'lowcost-review-plots-v1'});plt.close(fig)
  command=shlex.join([sys.executable,'-m','project.run_scripts.low_cost_write_donor_pilot.review_plots','--root',str(root.resolve())])
  manifest.append(dict(path=name,sha256=sha(path),source_sha256=sha(__file__),inputs={x:sha(root/x) for x in inputs},command=command,cwd=str(Path.cwd()),caption=caption,missing='No imputation; unavailable rows omitted with report disclosure',dpi=160))
 c={r['state']:r for r in tables['first-core-table.csv']};x=np.arange(6)
 fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True)
 for i,p in enumerate(['current','historical']):
  for j,m in enumerate(['RS','PS','NS']):
   ax=axs[i,j];ax.bar(x,[100*float(c[a][p+'_'+m+'_rate']) for a in ARMS],color=COLORS);ax.set_ylim(0,105);ax.set_title(p.title()+' '+m);ax.set_ylabel('Prompt success (%)');ax.set_xticks(x,ARMS,rotation=30)
 save(fig,'core-performance.png',['first-core-table.csv'],'Six frozen endpoints. Current R100/P200/N1000; Historical R128/P256/N1280. Strict NLL preference; ties fail. No pooled current/history denominator.')
 fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True)
 for i,p in enumerate(['current','historical']):
  for j,m in enumerate(['RS','PS','NS']):
   field='true_nll' if m=='NS' else 'new_nll';dd={r['state']:r for r in tables['nll-distributions.csv'] if r['panel']==p and r['metric']==m and r['field']==field and r['population']=='ALL' and r['aggregation_unit']=='PROMPT'}
   for stat,style in [('median','o-'),('p90','s--'),('p99','^:')]:axs[i,j].plot(x,[float(dd[a][stat]) for a in ARMS],style,label=stat)
   axs[i,j].set_ylim(bottom=0);axs[i,j].set_title(p.title()+' '+m+' desired-target NLL');axs[i,j].set_ylabel('nats / target token');axs[i,j].set_xticks(x,ARMS,rotation=30);axs[i,j].legend()
 save(fig,'nll-tails.png',['nll-distributions.csv'],'Prompt-level desired-target NLL, length-normalized. Current100/200/1000 and Historical128/256/1280. Median/p90/p99 are different summaries, not uncertainty intervals. Arms categorical, connected lines only visual guide.')
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for ax,p in zip(axs,['current','historical']):
  dd={r['after']:r for r in tables['paired-transitions.csv'] if r['before']=='N4' and r['panel']==p and r['metric']=='NS' and r['population']=='ALL'}
  aa=ARMS[1:];xx=np.arange(5)
  ax.bar(xx-.18,[int(dd[a]['lost']) for a in aa],.36,label='N4 success lost',color='#bf616a');ax.bar(xx+.18,[int(dd[a]['gained']) for a in aa],.36,label='N4 failure gained',color='#a3be8c');ax.set_xticks(xx,aa,rotation=30);ax.set_title(p.title()+' paired NS');ax.set_ylabel('Prompt count');ax.legend()
 save(fig,'paired-locality.png',['paired-transitions.csv'],'Identity-matched N4→candidate loss/gain counts. Current N1000 and Historical N1280, within100/128 request clusters. Opposing transitions not netted away.')
 fig,ax=plt.subplots(figsize=(9,4));la=tables['layer-action.csv']
 for layer,off,col in [(4,-.18,'#5e81ac'),(8,.18,'#d08770')]:
  dd={r['arm']:float(r['norm']) for r in la if int(r['layer'])==layer};ax.bar(x+off,[dd[a] for a in ARMS],.36,label='L'+str(layer),color=col)
 ax.set_xticks(x,ARMS);ax.set_ylabel('Frobenius norm of stored endpoint − entry');ax.set_title('Layer-wise Update Magnitude');ax.legend()
 save(fig,'layer-update-magnitude.png',['layer-action.csv'],'CPU FP64 norm of actual stored FP32 endpoint minus common We, one batch endpoint per arm/layer. REFIT4 net action is not summed path work. No equal-allocation reference.')
 fig,axs=plt.subplots(1,2,figsize=(11,4));cc={r['arm']:r for r in tables['compute-ledger.csv']}
 axs[0].bar(x,[float(cc[a]['policy_instrumented_online_seconds']) for a in ARMS],color=COLORS);axs[0].set_ylabel('Instrumented seconds / policy B100');axs[0].set_title('Same-host recorded policy cost')
 axs[1].bar(x,[float(cc[a]['policy_M8_setup_seconds']) for a in ARMS],color=COLORS);axs[1].set_ylabel('One-time M8 setup seconds');axs[1].set_title('Additional historical input preparation')
 for ax in axs:ax.set_xticks(x,ARMS,rotation=30)
 save(fig,'compute.png',['compute-ledger.csv'],'Policy accounting includes shared native fit per policy; study total counts native once. M8 setup shared once by FULL8/RES8 in study, not two actual executions. Diagnostics included; pure writer cost NOT_SEPARATED.')
 (root/'plot-receipt.json').write_text(json.dumps(manifest,indent=2)+'\n')
 return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
