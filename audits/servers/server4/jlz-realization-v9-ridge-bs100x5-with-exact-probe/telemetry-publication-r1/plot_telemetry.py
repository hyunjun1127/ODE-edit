"""게시 CSV만 사용하는 정적 과학 그림. 모델/원 tensor/GPU 호출 없음."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(root, name):
    with (root/(name+'.csv')).open() as f:
        return list(csv.DictReader(f))


def select(data, **values):
    return [x for x in data if all(x[k] == str(v) for k,v in values.items())]


def mean(data, name):
    return np.mean([float(x[name]) for x in data])


def draw(root, out):
    assert not out.exists(), '그림 경로 create-once'
    out.mkdir(parents=True)
    plt.rcParams.update({'font.size':10, 'axes.grid':True, 'grid.alpha':.2,
                         'savefig.dpi':160, 'font.family':'DejaVu Sans'})
    colors={'A':'#1769aa','B':'#d46a1f'}
    data=rows(root,'terminal-layer-realization')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    layers=np.arange(4,9)
    for arm in ('A','B'):
        v=select(data,arm=arm,batch=5)
        axes[0].plot(layers,[float(x['gamma_mean']) for x in v],'-o',color=colors[arm],label=arm+' directional gamma')
        axes[0].plot(layers,[float(x['M_diagonal_mean']) for x in v],'--x',color=colors[arm],label=arm+' mean diag(M)')
        for realized,offset,alpha in ((False,-.21, .4),(True,-.07,1.)) if arm=='A' else ((False,.07,.4),(True,.21,1.)):
            field='realized_layer_share_mean' if realized else 'planned_layer_share_mean'
            axes[1].bar(layers+offset,[100*float(x[field]) for x in v],width=.14,alpha=alpha,
                        color=colors[arm],label=arm+(' realized' if realized else ' planned'))
    axes[0].set(title='W5: request-mean directional gamma and diag(M)',xlabel='Layer',ylabel='Ratio',xticks=layers,ylim=(.35,.7))
    axes[1].set(title='W5: mean per-request normalized-rho layer share',xlabel='Layer',ylabel='Share (%)',xticks=layers,ylim=(0,25))
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(out/'realization-and-allocation.png');plt.close(fig)

    loss=rows(root,'candidate-objectives');clamp=rows(root,'clamp-trajectory')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fields={'native_nll_mean':'Native NLL','native_kl_weighted_mean':'0.0625 native KL',
            'native_norm_weighted_mean':'Native norm','policy_weighted_mean':'0.1 allocation'}
    for col,arm in enumerate(('A','B')):
        for field,label in fields.items():
            y=[mean(select(loss,arm=arm,candidate=k),field) for k in range(1,26)]
            axes[0,col].plot(range(1,26),y,label=label)
        pulse=[mean(select(loss,arm=arm,candidate=k),'pulse_weighted_mean') for k in (5,10,15,20)]
        axes[0,col].plot((5,10,15,20),pulse,'ko',markersize=4,label='Weighted pulse')
        axes[0,col].set(title=arm+': objective components (mean of 5 batches)',xlabel='Candidate (pre-update)',ylabel='Weighted request-mean loss')
        axes[0,col].set_yscale('symlog',linthresh=1e-4)
        axes[0,col].set_ylim(-1e-4,20)
        axes[0,col].legend(fontsize=8)
        values=[mean(select(loss,arm=arm,candidate=k),'total_q_gradient_F') for k in range(1,25)]
        axes[1,0].plot(range(1,25),values,color=colors[arm],label=arm+' total q gradient')
        maxrho=[max(float(x['post_rho_max']) for x in select(clamp,arm=arm,update=k)) for k in range(1,25)]
        axes[1,1].plot(range(1,25),maxrho,color=colors[arm],label=arm+' maximum post-update rho')
    axes[1,0].set(title='Only total gradients were recorded',xlabel='Candidate 1-24',ylabel='Frobenius norm (5-batch mean)',yscale='log')
    axes[1,0].legend(fontsize=8)
    axes[1,1].axhline(.75,color='black',linestyle='--',label='Native clamp radius')
    total=sum(int(x['groups']) for x in clamp);clipped=sum(int(x['clipped']) for x in clamp)
    axes[1,1].set(title=f'Clipped request-layer updates: {clipped}/{total}',xlabel='Completed update',ylabel='rho = ||D|| / anchor',ylim=(0,.8))
    axes[1,1].legend(fontsize=8)
    fig.savefig(out/'objectives-and-clamp.png');plt.close(fig)

    decomp=rows(root,'terminal-context-decomposition')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for ax,arm in zip(axes,('A','B')):
        v=select(decomp,arm=arm,batch=5,scope='all_native')
        for off,field,label in ((-.24,'inherited_gap_mean','Inherited path term'),
                                (0,'mean_key_error_mean','Mean-key realization term'),
                                (.24,'context_key_difference_mean','Context-key term')):
            ax.bar(layers+off,[float(x[field]) for x in v],width=.24,label=label)
        ax.set(title=arm+': W5 terminal full-context decomposition',xticks=layers,xlabel='Layer',ylabel='Mean vector norm (600 context rows)')
        ax.legend(fontsize=8)
    fig.suptitle('Unstacked norms are not additive contributions; signed cross terms are in CSV',fontsize=11)
    fig.savefig(out/'terminal-decomposition.png');plt.close(fig)
    manifest=dict(source=str(Path(__file__).resolve()),matplotlib=matplotlib.__version__,numpy=np.__version__,
                  note='고정 CSV 재생성. 목적값은 A/B 동일 symlog 축(1e-4 이내 선형)으로 0 및 작은 음수도 그대로 표시; 목적식 smoothing이 아님',files=[])
    for p in sorted(out.glob('*.png')):
        data=p.read_bytes();manifest['files'].append(dict(path=p.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    with (out/'manifest.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();draw(a.data,a.out)
