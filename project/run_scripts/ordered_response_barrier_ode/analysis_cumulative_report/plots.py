"""Deterministic Agg plots from sealed CPU tables; no image-generation tools."""
import argparse,hashlib,io,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..sequential_plots import panels,weight_figure,STYLE,COLORS,LABELS
from .common import *

def lines(frame,field,title,ylabel,*,rate=False,tail=None):
    fig,axes=panels(title,ylabel,ylim=(0,102) if rate else None)
    for cell,ax in zip(CELLS,axes.flat):
        for arm in ARMS:
            g=frame[(frame.cell==cell)&(frame.arm==arm)].sort_values('batch')
            ax.plot(g.batch,g[field]*(100 if rate else 1),marker='o',ms=2,color=COLORS[arm],label=arm)
            if tail:ax.plot(g.batch,g[tail],ls='--',color=COLORS[arm],alpha=.7)
        ax.set_xticks(range(1,11));ax.set_xlim(.8,10.2);ax.set_xlabel('Committed B100 batch');ax.legend(fontsize=7,ncol=3)
    return fig

def final_plot(frame):
    fig,axes=panels('Final W_B10 on ALL 1,000 edited requests','Canonical success (%)',ylim=(0,102))
    for cell,ax in zip(CELLS,axes.flat):
        g=frame[frame.cell==cell].set_index('arm').reindex(ARMS)
        for j,(metric,color) in enumerate(zip(('RS','PS','NS'),('#0072B2','#009E73','#D55E00'))):
            ax.bar(np.arange(5)+(j-1)*.24,100*g[metric+'_rate'],width=.22,color=color,label=metric)
        ax.set_xticks(range(5),ARMS);ax.legend(fontsize=8)
    return fig

def retention_plot(frame):
    fig,axes=plt.subplots(5,4,figsize=(14,16),layout='constrained');fig.suptitle('Rewrite retention: each cohort at each committed W_B')
    for ai,arm in enumerate(ARMS):
        for ci,cell in enumerate(CELLS):
            ax=axes[ai,ci];g=frame[(frame.cell==cell)&(frame.arm==arm)]
            mat=g.pivot(index='cohort',columns='batch',values='now_success').reindex(index=range(1,11),columns=range(1,11))
            ax.imshow(np.ma.masked_invalid(mat.to_numpy()),vmin=0,vmax=100,cmap='viridis',origin='lower',aspect='auto')
            ax.set_title(f'{cell} / {arm}');ax.set_xticks([0,4,9],[1,5,10]);ax.set_yticks([0,4,9],[1,5,10]);ax.set_xlabel('W_B checkpoint');ax.set_ylabel('Edit cohort B (100 requests)')
    return fig

def age_plot(frame):
    fig,axes=plt.subplots(5,4,figsize=(14,15),sharex=True,sharey=True,layout='constrained');fig.suptitle('All-seen NS by relative edit-age stratum')
    for ai,arm in enumerate(ARMS):
        for ci,cell in enumerate(CELLS):
            ax=axes[ai,ci]
            for name,ls in [('early','-'),('middle','--'),('recent',':')]:
                g=frame[(frame.cell==cell)&(frame.arm==arm)&(frame.age==name)].sort_values('batch')
                ax.plot(g.batch,100*g.NS_rate,ls=ls,label=name)
            ax.set_title(f'{cell}/{arm}');ax.set_ylim(0,102);ax.set_xlim(1,10);ax.set_ylabel('NS (%)');ax.set_xlabel('W_B');ax.legend(fontsize=6)
    return fig

def build(perf,mech,comparison,out):
    perf,mech,comparison,out=map(Path,(perf,mech,comparison,out));require(not out.exists(),'create-once plot directory');out.mkdir(parents=True)
    plt.rcParams.update(STYLE);np.random.seed(20260907)
    sources={};figures=[]
    def load(directory,name):
        p=directory/name;sources[str(p)]=dict(path=str(p),sha256=sha256_file(p),bytes=p.stat().st_size)
        return pd.read_csv(p)
    def save(make,name,caption):
        images=[];axes=[]
        for repeat in range(2):
            fig=make();b=io.BytesIO();fig.savefig(b,format='png',metadata={'Software':'ODE-edit ORBODE cumulative analysis v1'},dpi=160)
            images.append(b.getvalue());axes=[dict(x=list(ax.get_xlim()),y=list(ax.get_ylim()),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel()) for ax in fig.axes]
            size=fig.get_size_inches().tolist();plt.close(fig)
        require(images[0]==images[1],'plot byte instability '+name)
        with (out/name).open('xb') as f:f.write(images[0])
        figures.append(dict(path=name,sha256=hashlib.sha256(images[0]).hexdigest(),bytes=len(images[0]),caption=caption,byte_stable_rerender=True,size_inches=size,dpi=160,axes=axes))
    final=load(perf,'final-20-arm.csv');cum=load(perf,'cumulative-core.csv');current=load(perf,'current-B100.csv');dist=load(perf,'category-distributions.csv')
    save(lambda:final_plot(final),'final-full1000.png','20 primary chains; each final W10 RS1000, PS2000, NS10000 prompts. No duplicate final evaluation.')
    for metric in ('RS','PS','NS'):
        save(lambda m=metric:lines(cum,m+'_rate',f'CHECKPOINT W_B on all seen requests: {m}',m+' (%)',rate=True),f'cumulative-{metric}.png','Each W_B on first100B requests; RS100B, PS200B, NS1000B. Ten actual observations, missing not imputed.')
        save(lambda m=metric:lines(current,m+'_rate',f'Current B100 only (NOT cumulative): {m}',m+' (%)',rate=True),f'current-{metric}.png','Each newly edited B100 only; RS100/PS200/NS1000 prompt denominator.')
    for category in ('rewrite','rephrase','locality'):
        for target in ('new','true'):
            d=dist[(dist.stage=='CHECKPOINT_W_ON_ALL_SEEN_REQUESTS')&(dist.category==category)&(dist.target==target)&(dist.unit=='request_cluster')]
            save(lambda d=d,c=category,t=target:lines(d,'nll_median',f'All-seen {c} target-{t} NLL','Request-cluster NLL; solid median / dashed p90',tail='nll_p90'),f'cumulative-{category}-{target}-nll.png','Request-cluster mean within each request; distribution over100B requests; nat/token. Dashed p90 distinct from median.')
        d=dist[(dist.stage=='CHECKPOINT_W_ON_ALL_SEEN_REQUESTS')&(dist.category==category)&(dist.target=='new')&(dist.unit=='request_cluster')]
        save(lambda d=d,c=category:lines(d,'margin_true_minus_new_median',f'All-seen {c}: signed true-minus-new NLL','Request-cluster margin; solid median / dashed p90',tail='margin_true_minus_new_p90'),f'cumulative-{category}-margin.png','True-minus-new: positive favors rewrite/rephrase new; negative favors locality true. Same100B request clusters.')
    retention=load(perf,'retention-matrix.csv');save(lambda:retention_plot(retention),'rewrite-retention-20arms.png','Each heatmap cell counts success /100 requests in that cohort. Future unedited cohorts masked, never zero-imputed.')
    age=load(perf,'age-rates.csv');save(lambda:age_plot(age),'age-strata-NS.png','Within each seen-prefix: early20%, middle60%, recent20%; neighborhood prompts10 per request; relative cohorts change over checkpoints.')
    action=load(mech,'layer-actual-action.csv');w=action.rename(columns={'batch_net_update_magnitude':'update_magnitude'})
    save(lambda:weight_figure(w),'layer-wise-update-magnitude.png','Actual stored-weight batch net Frobenius magnitude; ten B100 batch summaries per layer/arm, not100 independent update samples. No equal-share reference line.')
    for field,title in [('batch_net_update_magnitude','Stored-weight batch net'),('net_from_original_W0_magnitude','Stored-weight W0-to-W_B net')]:
        g=action.groupby(['cell','arm','batch'],sort=False)[field].apply(lambda v:float(np.linalg.norm(v))).reset_index()
        save(lambda g=g,f=field,t=title:lines(g,f,t,'Frobenius norm (model-specific scale)'),field+'.png','Five disjoint edited weight blocks combined by squared sum; path work is a different object.')
    history=load(mech,'historical-checkpoint-residual.csv')
    for relation in ('CURRENT','HISTORICAL'):
        g=history[history.relationship==relation].groupby(['cell','arm','batch'],sort=False).q_after_mean.mean().reset_index()
        save(lambda g=g,r=relation:lines(g,'q_after_mean',r+' original fixed-z residual','Mean q after commit'),relation.lower()+'-fixed-z-residual.png','Each cohort100; mean across eligible cohorts. Historical rows have no new command; absent B1 historical cohort omitted.')
    command=load(mech,'current-command-realization.csv')
    for field in ('rho_command_median','tau_command_median','E_over_origin_mean','M_over_origin_mean'):
        g=command.groupby(['cell','arm','batch'],sort=False)[field].mean().reset_index()
        save(lambda g=g,f=field:lines(g,f,'Current command realization: '+f,'Visit-summary mean (not pooled request quantile)'),field+'.png','Per-visit request summary followed by visit mean; source CSV has complete per-visit n/quantiles. No-command/Official missing M remains missing.')
    nodes=load(mech,'node-mechanism.csv');g=nodes.groupby(['cell','arm','batch'],sort=False).coefficient_u.mean().reset_index()
    save(lambda:lines(g,'coefficient_u','Response velocity multiplier','Mean u across20 visits'),'response-multiplier.png','Dynamic arms only,20 layer visits/B; Official has no u. QCL/NQFIX fixed1 remain displayed.')
    g=nodes.groupby(['cell','arm','batch'],sort=False).discretization_defect.sum().reset_index()
    save(lambda:lines(g,'discretization_defect','Actual positive potential increments','Sum finite-step defect'),'finite-step-defect.png','Continuous response condition does not certify finite steps; positive defects retained, no outcome filtering.')
    comp=load(comparison,'mechanism-performance-checkpoints.csv')
    def scatter():
        fig,axes=panels('Fixed-target residual vs cumulative NS (descriptive)','All-seen NS (%)',ylim=(0,102))
        for cell,ax in zip(CELLS,axes.flat):
            for arm in ARMS:
                g=comp[(comp.cell==cell)&(comp.arm==arm)];ax.scatter(g.q_after_mean,100*g.NS_rate,s=18,color=COLORS[arm],label=arm)
            ax.set_xlabel('Mean original-target residual q');ax.legend(fontsize=7)
        return fig
    save(scatter,'mechanism-cumulative-association.png','Ten repeated checkpoints/arm, shared histories; association only, no independent trial or causal inference.')
    def layer_heat(frame,value,title):
        fig,axes=plt.subplots(5,4,figsize=(14,14),layout='constrained');fig.suptitle(title)
        for ai,arm in enumerate(ARMS):
            for ci,cell in enumerate(CELLS):
                ax=axes[ai,ci];g=frame[(frame.cell==cell)&(frame.arm==arm)]
                mat=g.groupby(['layer','batch'])[value].mean().unstack().reindex(index=range(4,9),columns=range(1,11))
                im=ax.imshow(np.ma.masked_invalid(mat.to_numpy()),origin='lower',aspect='auto',cmap='viridis')
                ax.set_title(f'{cell}/{arm}');ax.set_yticks(range(5),range(4,9));ax.set_xticks([0,4,9],[1,5,10]);ax.set_xlabel('B100 commit');ax.set_ylabel('Layer');fig.colorbar(im,ax=ax,shrink=.6)
        return fig
    lv=load(mech,'layer-visit-cohort-response.csv.gz')
    for relation in ('CURRENT','HISTORICAL'):
        g=lv[lv.relationship==relation]
        save(lambda g=g,r=relation:layer_heat(g,'normalized_potential_reduction_mean',r+' cohort: layer progress toward original target'),f'layer-{relation.lower()}-progress-heatmap.png','Mean normalized potential reduction across eligible100-request cohorts/visits; per-panel scales shown. Historical has no new command; future/absent cohorts masked.')
    for field in ('batch_magnitude_share','dynamic_workload_sum_h_u_batch_increment'):
        save(lambda f=field:layer_heat(action,f,f'Layer × checkpoint: {f}'),field+'-heatmap.png','One actual batch-level layer update/share or nominal workload per checkpoint. No request-replicated independent update denominator; Official u workload absent.')
    for metric in ('rho_command_mean','tau_command_mean'):
        save(lambda m=metric:layer_heat(command,m,'Current command: '+m),metric+'-heatmap.png','Visit mean from100 current requests, then mean over layer visits; no historical command interpretation, missing not imputed.')
    code=Path(__file__);m=dict(status='PLOTS_BYTE_STABLE',figures=figures,inputs=list(sources.values()),source=dict(path=str(code),sha256=sha256_file(code)),reused_style_source=member(Path(__file__).parents[1]/'sequential_plots.py'),
        seed=20260907,backend=matplotlib.get_backend(),matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,style=STYLE,cells=CELLS,arms=ARMS,colors=COLORS,
        command=f'{sys.executable} -m project.run_scripts.ordered_response_barrier_ode.analysis_cumulative_report.plots --performance {perf} --mechanism {mech} --comparisons {comparison} --output <NEW_OUTPUT_DIR>',missing='OMIT_OR_MASK_NOT_IMPUTE')
    write_json_once(out/'plot-manifest.json',m);print('BYTE_STABLE_PLOTS',len(figures),flush=True);return m

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('performance','mechanism','comparisons','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();build(a.performance,a.mechanism,a.comparisons,a.output)
