"""Publish compact CPU review tables/figures; never mutates scientific output."""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from .review_b1 import ARMS,dump
from .review_completed_b1 import read,member,csv_write


def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def plot(package):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    package=Path(package);trials=[r for r in rows(package/'candidate-arithmetic.csv') if r['arm']=='EN_KL_Q']
    cost=rows(package/'standalone-accounting.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    x=[int(r['trial']) for r in trials];y=[float(r['loss']) for r in trials]
    axes[0].plot(x,y,'o-',color='#315f9a');axes[0].scatter([4],[y[3]],s=90,color='#278456',zorder=5)
    native=next(float(r['KL']) for r in rows(package/'KL.csv') if r['arm']=='N4' and r['panel']=='R512')
    axes[0].axhline(native,color='#8f3b39',linestyle='--',label='Native KL')
    axes[0].set(xticks=x,xlabel='Actual trial (1, 1/2, 1/4, 1/8)',ylabel='R512 document-mean KL',title='EN-KL-Q: first accepted trial = 4')
    axes[0].legend();axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    names=['N4','EN-KL-Q','DEC-LINE','DEC-MODES-CUM'];base=[float(r['shared_native_seconds']) for r in cost]
    extra=[float(r['correction_accounted_seconds']) for r in cost]
    axes[1].barh(names,base,label='Shared native (charged in full)',color='#929eab')
    axes[1].barh(names,extra,left=base,label='Required correction blocks',color='#315f9a')
    axes[1].set(xlabel='Seconds (boundary accounting; not isolated runs)',title='B1 standalone accounting, excluding observer')
    axes[1].legend(fontsize=8,loc='lower right');axes[1].invert_yaxis()
    fig.savefig(package/'trials-and-cost.png',dpi=150,metadata={'Software':'SLMF CPU review'});plt.close(fig)
    modes=rows(package/'writer-modes.csv');components=rows(package/'postselection-components.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    scatter=axes[0].scatter([float(r['sigma']) for r in modes],[float(r['raw_map_mode_weight_norm_squared']) for r in modes],
        c=[float(r['target_loading_squared']) for r in modes],cmap='viridis',s=22)
    axes[0].set(xlabel='Native ideal-metric singular value',ylabel='Raw-map mode weight norm squared',title='100 writer modes (postselection algebra)')
    fig.colorbar(scatter,ax=axes[0],label='Target loading squared')
    short=['Native','Random m0','Random m99','Remove m0','Remove m99']
    axes[1].plot(short,[int(r['reference_mismatches']) for r in components],'o-',label='Reference token mismatches (32 docs)')
    axes[1].plot(short,[int(r['N_count']) for r in components],'s-',label='Neighborhood success (32 prompts)')
    axes[1].set(title='Bounded postselection panel, not policy arms',ylabel='Count');axes[1].tick_params(axis='x',rotation=22)
    axes[1].legend(fontsize=8)
    fig.savefig(package/'writer-and-components.png',dpi=150,metadata={'Software':'SLMF CPU review'});plt.close(fig)


def run(local,output,package):
    local,out,dest=map(Path,(local,output,package));dest.mkdir(parents=True,exist_ok=False)
    first=local/'first-reduction';full=local/'full-reduction-r2';sup=local/'supplement-r3'
    chosen=[(first/'first-table.csv','first-table.csv'),(first/'independent-reducer.json','first-independent-reducer.json')]
    chosen += [(p,p.name) for p in sorted(full.glob('*')) if p.is_file() and p.name!='analysis-manifest.json']
    chosen += [(p,p.name) for p in sorted(sup.glob('*')) if p.is_file()]
    chosen += [(local/n,n) for n in ('tensor-inventory.json','local-solver.json','provenance.json')]
    for src,name in chosen:
        if (dest/name).exists():raise ValueError('DUPLICATE_PUBLICATION_MEMBER')
        shutil.copyfile(src,dest/name)
        assert member(src)['sha256']==member(dest/name)['sha256']
    compute=read(full/'compute.json');times=compute['batch']['timing'];cost=[]
    for key,value in times.items():cost.append(dict(scope='B1_batch',component=key,value=value,unit='seconds',relation='NONOVERLAPPING_TOP_BLOCK; contained in B1 total'))
    for key in ('program_seconds','peak_GPU_allocated','peak_GPU_reserved','peak_host_KiB'):
        cost.append(dict(scope='terminal',component=key,value=compute['terminal'][key],unit='seconds' if key=='program_seconds' else ('KiB' if key=='peak_host_KiB' else 'bytes'),relation='NOT_ADDITIVE_TO_BLOCKS'))
    cost.extend([dict(scope='allocation',component='51058_parent',value=3629,unit='GPU_sec',relation='step/extern excluded'),
                 dict(scope='prior_lineage',component='failed_T0_parents',value=1113,unit='GPU_sec',relation='prior sealed receipt; not new scheduler queries')])
    native=read(out/'B1/native/native-binding.json')['receipt']
    for k,v in native.items():
        if isinstance(v,(int,float)) and not isinstance(v,bool):cost.append(dict(scope='native_receipt',component=k,value=v,unit='seconds' if 'seconds' in k else 'count',relation='NESTED_NATIVE; do not add to native total'))
    for scope,path in [('decision_reference',out/'B1/reference-native.json'),('empty_history',out/'B1/history-native.json')]:
        for k,v in read(path)['work'].items():
            if isinstance(v,(int,float)) and not isinstance(v,bool):cost.append(dict(scope=scope,component=k,value=v,unit='bytes' if 'bytes' in k else ('seconds' if 'seconds' in k else 'count'),relation='NESTED; no double addition'))
    observer=read(out/'B1/observers/current/OBSERVERS_COMPLETE.json');mech=read(out/'B1/mechanism/COMPLETE.json')
    cost += [dict(scope='postseal',component='canonical_reference_observers',value=observer['seconds'],unit='seconds',relation='includes canonical plus reference/KL'),
             dict(scope='postseal',component='mechanism_inclusive',value=mech['seconds'],unit='seconds',relation='valid upper timer; invalid writer timer excluded')]
    csv_write(dest/'compute.csv',cost)
    standalone=[]
    for arm in ARMS:
        extra=times[arm+'_controller_inclusive'];parts=[arm+'_controller_inclusive']
        if arm!='N4':extra+=times['geometry_current_prefix'];parts+=['geometry_current_prefix']
        if arm.startswith('DEC_'):
            extra+=times['decision_reference_derivative']+times['projection_functional_basis'];parts+=['decision_reference_derivative','projection_functional_basis']
        standalone.append(dict(arm=arm,shared_native_seconds=times['native'],correction_accounted_seconds=extra,
            accounted_edit_seconds=times['native']+extra,correction_to_native_ratio=extra/times['native'],
            included='+'.join(parts),boundary='no observer/setup/commit; unseparated glue excluded; LINE projection+functional block not separated',
            independent_run_wall=False,S3_median_cost_gate='NOT_APPLICABLE_B1_ONLY'))
    csv_write(dest/'standalone-accounting.csv',standalone)
    generation=[]
    for arm in ARMS:
        d=read(out/'B1/observers/current'/f'{arm}.json');g=d['generation']
        generation.append(dict(arm=arm,requests=len(g),target_prefix_match=sum(r['target_prefix_match'] for r in g),
            censored=d['generation_censored'],actual_original_EOS=sum(r['stopped_on_original_eos'] for r in g),
            max_new_tokens_reached=sum(r['reached_max_new_tokens'] for r in g),
            reused=bool(d.get('same_endpoint_reuse'))))
    csv_write(dest/'greedy32.csv',generation)
    plot(dest)
    dump(dest/'publication-copy-manifest.json',dict(copies=[dict(source=member(s),published=n) for s,n in chosen],
        raw_payload_copied=False,model_or_GPU_calls=0,plots='repository matplotlib code only'))
    return dict(files=len(list(dest.iterdir())),standalone=standalone)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--local');p.add_argument('--output');p.add_argument('--package',required=True);p.add_argument('--plots-only',action='store_true')
    a=p.parse_args()
    if a.plots_only:plot(a.package)
    else:print(run(a.local,a.output,a.package))
