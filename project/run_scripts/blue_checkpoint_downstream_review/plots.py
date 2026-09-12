"""Deterministic figures from CPU-verified aggregate CSV, never model output synthesis."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import TASKS

def render(csv_path,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(Path(csv_path).open()))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.grid':True,'grid.alpha':.22})
    for family in ('MEMIT','AlphaEdit'):
        for branch in ('generation','alternative'):
            fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
            for task,ax in zip(TASKS,axes.flat):
                w=next(r for r in rows if r['state']=='W0' and r['task']==task and r['branch']==branch)
                ax.axhline(float(w['weighted_f1'])*100,color='black',ls='--',label='W0 (shared once)')
                for variant,color in [('BLUE','#0072B2'),('L4_ONLY','#009E73'),('L8_ONLY','#D55E00')]:
                    rr=sorted([r for r in rows if r['family']==family and r['variant']==variant and r['task']==task and r['branch']==branch],key=lambda r:int(r['edits']))
                    ax.plot([int(r['edits']) for r in rr],[float(r['weighted_f1'])*100 for r in rr],'-o',ms=3,color=color,label=variant)
                ax.set(title=task.upper()+(' (corrected)' if task=='rte' else ''),xlabel='Edits',ylabel='Weighted F1 (%)',ylim=(-2,102),xticks=[0,2000,4000,6000,8000,10000])
                ax.tick_params(axis='x',labelrotation=30)
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(f'{family} | {branch} | source-reference 100 items/task; lines connect measured checkpoints')
            fig.savefig(destination/f'{family}-{branch}.png',dpi=150,metadata={'Software':'blue_checkpoint_downstream_review/plots.py'})
            plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.csv,a.output)
