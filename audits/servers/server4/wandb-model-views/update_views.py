"""User-authorized model-specific saved views; no run histories or jobs modified."""
import copy
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
import wandb
from inspect_views import ROOT,ENTITY,PROJECT,QUERY

SOURCE='nw-y9vet5iwjzr-v'
IDS={'LLAMA':'nw-pricefirst2kllama-v','QWEN':'nw-pricefirst2kqwen-v','GPTJ':'nw-pricefirst2kgptj-v'}
LABELS={'LLAMA':'Llama-3-8B-Instruct','QWEN':'Qwen2.5-7B-Instruct','GPTJ':'GPT-J-6B'}
BASELINES=['historical-'+x+'-first2k' for x in ('memit','memith','memitblue','alphaedit','alphaeditblue','cake','v12memit','v13md','v13cd')]
LLAMA_OLD=BASELINES+['historical-w0-42673-cohort-matched','price-59768-metrics']
MUTATION='''mutation($id:ID,$e:String!,$p:String!,$n:String!,$d:String!,$s:String!){upsertView(input:{id:$id,entityName:$e,projectName:$p,type:"project-view",name:$n,displayName:$d,spec:$s,createdUsing:WANDB_SDK}){view{id name} inserted}}'''

def leaf(section,name,op,value,connector=None):
    d=dict(key=dict(section=section,name=name),op=op,value=value,disabled=False)
    if connector:d['connector']=connector
    return d

def model_filter(model):
    arms=[model+'_'+writer+arm for writer in ('','AE_','MEMIT_','ALPHA_','ALPHAEDIT_')
          for arm in ('CAP075','CAP100','FREE100')]
    filters=[leaf('config','model_family','=',model),leaf('config','model_profile','=',model,'OR'),
             leaf('config','arm','IN',arms,'OR')]
    if model=='LLAMA':filters.append(leaf('run','name','IN',LLAMA_OLD,'OR'))
    return dict(filterFormat='filterV2',filters=filters)

def url(name):
    return 'https://forge.coreweave.com/wandb/'+ENTITY+'/'+quote(PROJECT,safe='')+'?nw='+name[3:-2]

def view_spec(base,model):
    spec=copy.deepcopy(base);sec=spec['section'];sec['name']=LABELS[model]
    rs=sec['runSets'][0];rs['filters']=model_filter(model);rs['search']={'query':''}
    rs['name']=LABELS[model]+' — ours / baselines';rs['grouping']=[]
    rs['runFeed'].update(columnVisible={x:True for x in ('run:displayName','run:state','config:model_family.value',
        'config:arm.value','config:method.value','config:task_id.value')},pageSize=50,onlyShowSelected=False)
    rs['selections']={'root':1,'bounds':[],'tree':[]}
    sections=sec['panelBankConfig']['sections']
    note=(f'# {LABELS[model]} — Ours and Baselines / First 2k\n'
        '모델이 확인된 run만 표시합니다. x축 0–2000 edits; current100과 all-seen은 분리합니다. '
        '미완료·미측정·없는 baseline은 0으로 채우지 않습니다. W0는 동일 모델의 reference만 사용합니다. '
        '같은 모델이어도 runtime/seed/history 차이는 historical comparison으로 남습니다.\n\n'
        '새 run은 model_family/model_profile 또는 정확한 model-prefixed arm 필터로 자동 포함됩니다. '
        '기존 봉인 logger의 eval/*는 current와 milestone all-seen이 섞인 legacy namespace이므로 '
        '아래 별도 실시간 패널에만 표시합니다. baseline % 곡선과 무단 결합하지 않습니다.\n\n')
    note+=('등록된 historical baselines와 W0는 Llama입니다.' if model=='LLAMA' else
           '현재 이 모델로 확인된 imported baseline/W0는 NOT_AVAILABLE. 다른 모델 baseline을 대체 표시하지 않습니다.')
    sections[0]['panels'][0]['config']['value']=note
    # Existing model-scoped comparison sections and all missing-value semantics survive.
    for s in sections:
        for panel in s.get('panels',[]):
            if panel['viewType']=='Run History Line Plot':
                panel['config']['xAxisMax']=2000.;panel['config']['limit']=100
    template=next(p for s in sections for p in s.get('panels',[]) if p['viewType']=='Run History Line Plot')
    live=copy.deepcopy(sections[1]);live['name']='Live legacy logger — endpoint denominators differ (not baseline overlay)';live['panels']=[]
    for metric in ('RS','PS','NS','harmonic'):
        p=copy.deepcopy(template);p['__id__']='live-'+model+'-'+metric
        p['config'].update(chartTitle='Live '+metric+' — inspect R/P/N denominator',metrics=['eval/'+metric],
            yAxisTitle='Fraction (0–1), mixed endpoint scope',yAxisMin=0.,yAxisMax=1.)
        live['panels'].append(p)
    for kind in 'RPN':
        p=copy.deepcopy(template);p['__id__']='denominator-'+model+'-'+kind
        p['config'].update(chartTitle='Live '+kind+' denominator',metrics=['eval/'+kind+'_denominator'],
            yAxisTitle='Observed requests/prompts',yAxisMin=0.)
        p['config'].pop('yAxisMax',None);live['panels'].append(p)
    sections.append(live)
    return spec

def main():
    api=wandb.Api(timeout=30)
    before=json.loads((ROOT/'views-before.json').read_text())
    current=api._service_api.execute_graphql(QUERY,variables={'e':ENTITY,'p':PROJECT})
    old={e['node']['name']:e['node'] for e in before['project']['allViews']['edges']}
    now={e['node']['name']:e['node'] for e in current['project']['allViews']['edges']}
    assert now[SOURCE]['spec']==old[SOURCE]['spec'],'Concurrent view edit: stop'
    base=json.loads(old[SOURCE]['spec']);receipts=[]
    for model,name in IDS.items():
        spec=view_spec(base,model)
        response=api._service_api.execute_graphql(MUTATION,variables=dict(id=now.get(name,{}).get('id'),
            e=ENTITY,p=PROJECT,n=name,d=LABELS[model]+' - Ours and Baselines - First 2k',
            s=json.dumps(spec,ensure_ascii=False)))
        receipts.append(dict(model=model,name=name,url=url(name),response=response,spec=spec))
    # Original comparison becomes a model navigation view. Preserve the backed-up
    # legacy Llama comparison below the index; no cross-model overlay is implied.
    index=view_spec(base,'LLAMA')
    panel=index['section']['panelBankConfig']['sections'][0]['panels'][0]
    panel['config']['value']='# Ours and Baselines — First 2k / 모델별 보기\n\n'+ '\n'.join(
        f'- [{LABELS[m]}]({url(n)})' for m,n in IDS.items())+'\n\n아래 기존 비교 곡선은 Llama 전용입니다. Qwen/GPT-J는 위의 모델별 Saved View에서 확인하세요. 없는 baseline은 NOT_AVAILABLE이며 기존 full-history view는 변경하지 않았습니다.'
    response=api._service_api.execute_graphql(MUTATION,variables=dict(id=now[SOURCE]['id'],e=ENTITY,p=PROJECT,
        n=SOURCE,d='Ours and Baselines - First 2k Comparison',s=json.dumps(index,ensure_ascii=False)))
    after=api._service_api.execute_graphql(QUERY,variables={'e':ENTITY,'p':PROJECT})
    nodes={e['node']['name']:e['node'] for e in after['project']['allViews']['edges']}
    for r in receipts:assert json.loads(nodes[r['name']]['spec'])==r['spec']
    assert json.loads(nodes[SOURCE]['spec'])==index
    untouched=[name for name in old if name!=SOURCE]
    assert all(nodes[n]['spec']==old[n]['spec'] for n in untouched)
    for r in receipts:
        r['spec_sha256']=hashlib.sha256(json.dumps(r.pop('spec'),sort_keys=True).encode()).hexdigest()
    result=dict(status='REMOTE_SAVED_VIEWS_READBACK_VERIFIED',views=receipts,index_url=url(SOURCE),
        untouched_views=untouched,run_history_writes=0,job_writes=0,new_science_runs=0,
        existing_run_configs_changed=0,model_assignment='historical source identities + exact arm fields; unknown excluded',
        new_metrics_schema='Existing common helper lacks split endpoint namespace; legacy live metrics kept separate, never wrong baseline overlay',
        helper_owner_request='SH1 direct app message unavailable; shared helper not modified')
    with (ROOT/'views-after.json').open('x') as f:json.dump(after,f,ensure_ascii=False,indent=2)
    with (ROOT/'update-receipt.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
