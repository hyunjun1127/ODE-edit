"""Measured-data-only figures; deterministic raster output, no image generation."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reducer import REPORT,ARMS,sha,writejson
def rows(n):return list(csv.DictReader((REPORT/n).open()))
def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':120,'savefig.dpi':150})
    colors=['#444444','#999999','#008877','#5588cc','#cb7722','#aa4499'];inputs={}
    f=rows('first-final-table.csv');fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,tag in zip(axes,['RS','PS','NS']):
        y=[float(x[tag+'_percent']) for x in f];ax.bar(ARMS,y,color=colors)
        ax.set_title('Actual W10 '+tag);ax.set_ylabel('Percent');ax.tick_params(axis='x',rotation=35);ax.set_ylim((98.5,100.4) if tag=='RS' else (93,99) if tag=='PS' else (77,86))
        for i,v in enumerate(y):ax.text(i,v+.06,f'{v:g}',ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.2)
    fig.savefig(REPORT/'final-rpn.png',metadata={'Software':'SLZV2 CPU review'});plt.close(fig)
    gate=rows('selected-gates.csv');fig,axes=plt.subplots(2,3,figsize=(13,6),layout='constrained')
    for ax,a in zip(axes.flat,ARMS):
        m=np.full((5,10),np.nan)
        for r in gate:
            if r['arm']==a:m[int(r['layer'])-4,int(r['batch'])-1]=float(r['gate'])
        im=ax.imshow(m,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_title(a);ax.set_xticks(range(10),range(1,11));ax.set_yticks(range(5),range(4,9));ax.set_xlabel('Batch');ax.set_ylabel('Physical layer')
        for i,j in zip(*np.where(np.isfinite(m))):ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if m[i,j]<.65 else 'black')
    fig.colorbar(im,ax=list(axes.flat),shrink=.7,label='Selected gate (blank = outside arm)');fig.savefig(REPORT/'selected-gates.png',metadata={'Software':'SLZV2 CPU review'});plt.close(fig)
    cs=rows('candidate.csv');fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,a in zip(axes,['C4','C48','C45678']):
        for flag,color,label in [('False','#bbbbbb','Infeasible'),('True','#228899','Feasible')]:
            d=[r for r in cs if r['arm']==a and r['feasible']==flag];ax.scatter([float(r['delta_E']) for r in d],[float(r['delta_B']) for r in d],s=13,alpha=.6,c=color,label=label)
        d=[r for r in cs if r['arm']==a and r['selected']=='True'];ax.scatter([float(r['delta_E']) for r in d],[float(r['delta_B']) for r in d],s=45,c='#aa3333',marker='x',label='Selected')
        ax.axvline(.0001,color='k',ls='--',lw=.7);ax.axhline(0,color='k',lw=.5);ax.set_title(a+' measured candidates');ax.set_xlabel('E(candidate) - E(own N4)');ax.set_ylabel('B(candidate) - B(own N4)');ax.set_xscale('symlog',linthresh=.0001);ax.legend(fontsize=8)
    fig.savefig(REPORT/'candidate-quality.png',metadata={'Software':'SLZV2 CPU review'});plt.close(fig)
    writejson(REPORT/'figure-manifest.json',{'generator':str(__file__),'inputs':{n:sha(REPORT/n) for n in ['first-final-table.csv','selected-gates.csv','candidate.csv']},'outputs':{p.name:sha(p) for p in REPORT.glob('*.png')},'unmeasured_interpolation':False})
if __name__=='__main__':main()
