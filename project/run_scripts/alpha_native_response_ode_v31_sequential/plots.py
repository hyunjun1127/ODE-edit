"""Deterministic PNGs from exact aggregate CSVs; no replay or visualization API."""
import argparse,csv
from pathlib import Path


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=Path(report)
    def read(name):
        with (report/name).open(newline='') as f:return list(csv.DictReader(f))
    current=read('current_batch_metrics.csv');cohorts=read('retention_cohort_metrics.csv')
    layers=read('physical_layer_batch_metrics.csv')
    aliases=sorted({r['alias'] for r in current});colors={'O_NATIVE':'#555555','JV_NATIVE':'#0066aa','L8_ONLY_NATIVE':'#cc6600'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.grid':True,'grid.alpha':.2,
        'figure.dpi':120,'savefig.dpi':160,'svg.hashsalt':'alpha-jv-sequential-routing-v1'})
    outputs=[]
    for alias in aliases:
        fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        for arm in ('O_NATIVE','JV_NATIVE','L8_ONLY_NATIVE'):
            rows=sorted((r for r in current if r['alias']==alias and r['arm']==arm),key=lambda r:int(r['batch']))
            if not rows:continue
            xs=[int(r['batch']) for r in rows]
            for ax,key in zip(axes[0],('RS','PS','NS')):
                ax.plot(xs,[float(r[key+'_num'])/float(r[key+'_den']) for r in rows],'.-',label=arm,color=colors[arm]);ax.set_title('Current B100 '+key);ax.set_ylim(0,1.02)
            old=sorted((r for r in cohorts if r['alias']==alias and r['arm']==arm and int(r['cohort'])==1),key=lambda r:int(r['batch']))
            axes[1,0].plot([int(r['batch']) for r in old],[int(r['current_success'])/int(r['canonical_denominator']) for r in old],'.-',color=colors[arm],label=arm)
            lr=[r for r in layers if r['alias']==alias and r['arm']==arm];by={}
            for r in lr:by.setdefault(int(r['batch']),[]).append(r)
            for ax,which in [(axes[1,1],'L8'),(axes[1,2],'other')]:
                yy=[]
                for k in sorted(by):
                    rr=by[k];total=sum(float(r['endpoint_DeltaW_squared']) for r in rr)
                    last=sum(float(r['endpoint_DeltaW_squared']) for r in rr if int(r['layer'])==8)
                    yy.append(last/total if which=='L8' and total else (float('nan') if which=='L8' else total-last))
                ax.plot(sorted(by),yy,'.-',color=colors[arm],label=arm)
        axes[1,0].set_title('B1 cohort RS at current Wk (den=100)');axes[1,0].set_ylim(0,1.02)
        axes[1,1].set_title('Actual batch endpoint L8 energy share');axes[1,1].set_ylim(0,1.02)
        axes[1,2].set_title('Actual L4-7 endpoint energy (absolute)')
        for ax in axes.flat:ax.set_xlabel('Committed batch k');ax.set_xticks(range(1,11))
        axes[0,0].legend(fontsize=7);fig.suptitle(alias+' | online current, retained cohort, physical writes are distinct')
        path=report/(alias+'-trajectory-physical.png')
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,metadata={'Software':'alpha-jv-sequential-routing deterministic report v1'});plt.close(fig);outputs.append(str(path))
    return outputs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('report',type=Path);a=p.parse_args();print(plot(a.report))
