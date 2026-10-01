"""Deterministic standard-library SVG plots from independently reduced CSV."""
import argparse
import csv
from html import escape
from pathlib import Path

def chart(path,title,series,ylabel,fixed=None):
    values=[y for _,_,points in series for _,y in points]
    hi=fixed or max([1e-12,*values])*1.05;lo=0. if fixed else min([0.,*values])*1.05
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360" viewBox="0 0 900 360">',
         '<rect width="900" height="360" fill="white"/>',
         f'<text x="70" y="25" font-family="sans-serif" font-size="18">{escape(title)}</text>',
         f'<text x="70" y="48" font-family="sans-serif" font-size="12">{escape(ylabel)}</text>']
    for i in range(5):
        value=lo+(hi-lo)*i/4;y=300-230*i/4
        svg += [f'<path d="M70 {y} H720" stroke="#ddd"/>',f'<text x="8" y="{y+4}" font-family="sans-serif" font-size="11">{value:.4g}</text>']
    for x in (0,5,10,15,20):
        svg += [f'<text x="{70+32.5*x}" y="320" font-family="sans-serif" font-size="12">{x}</text>']
    svg += ['<text x="330" y="347" font-family="sans-serif" font-size="13">offered batch (BS100)</text>']
    for i,(label,color,points) in enumerate(series):
        coordinates=' '.join(f'{70+32.5*x:.4f},{300-230*(y-lo)/(hi-lo):.4f}' for x,y in points)
        if coordinates:svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x,y in points:svg.append(f'<circle cx="{70+32.5*x:.4f}" cy="{300-230*(y-lo)/(hi-lo):.4f}" r="3" fill="{color}"/>')
        svg.append(f'<text x="740" y="{75+22*i}" fill="{color}" font-family="sans-serif" font-size="12">{escape(label)}</text>')
    if not values:svg.append('<text x="230" y="180" font-family="sans-serif">NO COMPLETED MEASUREMENTS</text>')
    svg.append('</svg>');Path(path).write_text('\n'.join(svg)+'\n')

def generate(root):
    root=Path(root)
    with (root/'endpoint-metrics.csv').open() as f:rows=list(csv.DictReader(f))
    series=[]
    for arm,colors in [('A',('#1b6ca8','#6794b5')),('B',('#c55a16','#dea277'))]:
        for kind,color in zip(('R','N'),colors):
            points=[(int(r['batch']),float(r['preference_rate'])) for r in rows if r['stage']=='main' and r['method']=='JLZ_'+arm and r['kind']==kind and r['panel']=='allseen_RP_current_or_milestone_N' and (kind=='R' or int(r['batch']) in (5,10,20))]
            series.append((arm+' '+kind+' allseen',color,points))
    chart(root/'quality.svg','Observed preference (not free generation)',series,'strict NLL inequality; ties=failure',1.)
    with (root/'fit-reference-losses.csv').open() as f:rows=list(csv.DictReader(f))
    series=[]
    for arm,color in [('A','#1b6ca8'),('B','#c55a16')]:
        points=[(int(r['batch']),float(r['general_mean'])) for r in rows if r['stage']=='main' and r['arm']==arm and r.get('general_mean')]
        series.append((arm+' general KL',color,points))
    chart(root/'reference-kl.svg','Training reference KL (not official observer)',series,'current || W0; token mean then prompt mean')
    return ['quality.svg','reference-kl.svg']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);generate(p.parse_args().root)
