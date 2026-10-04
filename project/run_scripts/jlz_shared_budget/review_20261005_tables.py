"""Historical endpoint comparison. Reuses stored counts; never evaluates a model."""
import argparse,csv,json,math
from pathlib import Path
from review_20261005 import check,save,table,sha,TASK

def score(values):
    check(len(values)==3 and all(math.isfinite(v) and 0<=v<=1 for v in values),'SCORE_RANGE')
    return 0. if min(values)==0 else 3/math.fsum(1/v for v in values)

def compare(root,out):
    s=json.loads((out/'summary.json').read_text());rows=[];inventory=[]
    def load(path,expected=None):
        p=root/path;h=sha(p)
        if expected:check(h==expected,'BASELINE_SHA')
        inventory.append(dict(path=path,bytes=p.stat().st_size,sha256=h));return list(csv.DictReader(p.open()))
    def add(method,b,k,num,den,strict,correct,tokens,macro,new,true,path,scope):
        check(k in ('R','P','N') and den==b*100*dict(R=1,P=2,N=10)[k],'BASELINE_DENOMINATOR')
        check(0<=num<=den and 0<=strict<=den and 0<=correct<=tokens,'BASELINE_COUNTS')
        rows.append(dict(method=method,batch=b,edits=b*100,family=k,numerator=num,denominator=den,preference=num/den,
            strict_numerator=strict,strict=strict/den,token_correct=correct,token_denominator=tokens,token_micro=correct/tokens,
            prompt_macro=macro,new_nll=new,true_nll=true,desired_nll=true if k=='N' else new,source=path,comparison_scope=scope))
    for r in s['metrics']:
        if r['endpoint']:
            add('v12-MEMIT',r['endpoint'],r['family'],r['success'],r['n'],r['strict_num'],r['correct'],r['tokens'],r['prompt_macro'],r['new_nll'],r['true_nll'],'summary.json','REVIEWED_CURRENT_RUN')
    p='experiment-reports/servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1/all-seen-metrics.csv'
    for r in load(p):
        b=int(r['batch'])
        if b not in (5,10,15,20):continue
        add('MEMIT-H',b,r['metric'][0],int(r['numerator']),int(r['denominator']),int(r['tf_strict_numerator']),int(r['tf_token_correct']),int(r['tf_token_count']),float(r['tf_prompt_macro']),float(r['new_nll']),float(r['true_nll']),p,'HISTORICAL_REFERENCE')
    mapping={'BASE_MEMIT':'MEMIT','MEMIT_ORIGINAL':'MEMIT-BLUE','BASE_ALPHAEDIT':'AlphaEdit','AlphaEdit_ORIGINAL':'AlphaEdit-BLUE'}
    base='experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/'
    for name,h in [('MEMIT','5f405e36e3bd0b4785fd3d2b9466fea4ea7af51db1370bdef608c206cf3dae0e'),('AlphaEdit','19ddace1472b89b23acb43ffed0450924c7342c12fd3fdedf277ebca5cd9ab07')]:
        p=base+name+'-cumulative-metrics.csv'
        for r in load(p,h):
            b=int(r['batch'])
            if b not in (5,10,15,20) or r['arm'] not in mapping:continue
            check(r['scope']=='CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS','BASELINE_SCOPE')
            k=r['metric'][0];d='true' if k=='N' else 'new';den=int(r['denominator']);check(int(r[d+'_strict_den'])==den,'TF_DEN')
            add(mapping[r['arm']],b,k,int(r['numerator']),den,int(r[d+'_strict_num']),int(r[d+'_token_correct']),int(r[d+'_token_den']),None,float(r['new_nll_prompt_mean']),float(r['true_nll_prompt_mean']),p,'HISTORICAL_REFERENCE')
    p='experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2/seen-prefix.csv'
    for r in load(p,'584012115d6c4f5bc537765631cb3ac35e866f2ba985a99e02c8d2c747e1a051'):
        b=int(r['batch'])
        if b not in (5,10,15,20) or r['population']!='ACTUAL_FULL_SEEN':continue
        check(int(r['desired_strict_d'])==int(r['denominator']),'CAKE_TF_DEN')
        add('CAKE',b,r['metric'][0],int(r['numerator']),int(r['denominator']),int(r['desired_strict_n']),int(r['desired_token_correct']),int(r['desired_token_d']),None,float(r['new_nll_mean']),float(r['true_nll_mean']),p,'HISTORICAL_REFERENCE')
    check(len({(r['method'],r['batch'],r['family']) for r in rows})==len(rows),'DUPLICATE_BASELINE')
    order=['v12-MEMIT','MEMIT-H','MEMIT','AlphaEdit','AlphaEdit-BLUE','CAKE','MEMIT-BLUE'];wide=[];missing=[]
    for b in (5,10,15,20):
        for method in order:
            group={r['family']:r for r in rows if r['method']==method and r['batch']==b}
            if not group:continue
            if set(group)!=set('RPN'):
                missing.append(dict(method=method,batch=b,missing_families=sorted(set('RPN')-set(group))))
                continue
            r=dict(method=method,batch=b,edits=b*100,comparison_scope=group['R']['comparison_scope'])
            for k in 'RPN':
                r[k+'S_percent']=100*group[k]['preference'];r[k+'_strict_percent']=100*group[k]['strict'];r[k+'_token_micro_percent']=100*group[k]['token_micro']
            r['score_harmonic_percent']=100*score([group[k]['preference'] for k in 'RPN']);wide.append(r)
    check(sum(r['batch']==20 for r in wide)==7,'W20_METHOD_COVERAGE')
    for r in wide:
        target=next(v for v in wide if v['batch']==r['batch'] and v['method']=='v12-MEMIT')
        for k in ('RS_percent','PS_percent','NS_percent','score_harmonic_percent'):r['v12_minus_row_'+k.replace('_percent','_pp')]=target[k]-r[k]
    table(out/'baseline-metrics-long.csv',rows);table(out/'baseline-comparison.csv',wide)
    save(out/'baseline-manifest.json',dict(files=inventory,source_type='existing tracked completed CPU review tables',new_model_calls=0,
        endpoints='each own W5/10/15/20 evaluated on first500/1000/1500/2000; no finalW100 substitution',
        score='100 * 3 / (1/RS + 1/PS + 1/NS), rates from integer counts, zero convention 0; equal family weights, no rounded-input calculation',
        missing_endpoint_families=missing,cross_run_paired_ids='NOT_AVAILABLE; arithmetic aggregate difference only',missing_prompt_macro='NOT_RECORDED, never substituted with token_micro'))
    print(json.dumps([r for r in wide if r['batch']==20],ensure_ascii=False))
    return wide

def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((out/'baseline-comparison.csv').open()));w=[r for r in rows if r['batch']=='20']
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for k,color in zip(('RS','PS','NS'),('#2166ac','#1b9e77','#d95f02')):
        rr=[r for r in rows if r['method']=='v12-MEMIT'];axs[0].plot([int(r['edits']) for r in rr],[float(r[k+'_percent']) for r in rr],'o-',label=k,color=color)
    axs[0].set(title='v12-MEMIT: growing all-seen population',xlabel='Edits',ylabel='Preference success (%)',ylim=(50,102));axs[0].legend();axs[0].grid(alpha=.2)
    axs[1].barh([r['method'] for r in w],[float(r['score_harmonic_percent']) for r in w],color=['#2166ac']+['#999999']*6)
    axs[1].invert_yaxis();axs[1].set(title='W20 first2000: harmonic score',xlabel='Score (%)',xlim=(0,100))
    for i,r in enumerate(w):axs[1].text(float(r['score_harmonic_percent'])+.4,i,f"{float(r['score_harmonic_percent']):.2f}",va='center',fontsize=9)
    fig.suptitle('Stored-result CPU review | baselines are historical, not matched reruns',fontsize=11)
    fig.savefig(out/'metrics-and-baselines.png',dpi=180);fig.savefig(out/'metrics-and-baselines.pdf');plt.close(fig)
    s=json.loads((out/'summary.json').read_text());pairs=[r for r in s['paired'] if r['comparison']=='first500_W5_to_W20' and r['metric']=='preference']
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for off,key,label in [(-.18,'before','W5'),(.18,'after','W20')]:axs[0].bar([i+off for i in range(3)],[100*r[key]/r['n'] for r in pairs],width=.36,label=label)
    axs[0].set_xticks(range(3),['RS','PS','NS']);axs[0].set(title='Fixed first500 cohort (same requests)',ylabel='Preference success (%)',ylim=(0,105));axs[0].legend()
    ls=sorted(s['layers']);x=range(len(ls))
    axs[1].bar([i-.18 for i in x],[100*s['layers'][l]['terminal_plan_share']['mean'] for l in ls],width=.36,label='Terminal plan share')
    axs[1].bar([i+.18 for i in x],[100*s['layers'][l]['applied_share']['mean'] for l in ls],width=.36,label='Direct realized share')
    axs[1].set_xticks(x,['L'+l for l in ls]);axs[1].set(title='Mean request shares; distinct quantities',ylabel='Share (%)');axs[1].legend(fontsize=8)
    fig.savefig(out/'retention-and-shares.png',dpi=180);fig.savefig(out/'retention-and-shares.pdf');plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path.cwd());ap.add_argument('--out',type=Path,required=True);ap.add_argument('--plots',action='store_true');a=ap.parse_args();compare(a.root,a.out)
    if a.plots:plots(a.out)
