"""USER authorized zsRE-only saved views. No run/history/config mutation."""
import copy
import hashlib
import json
from pathlib import Path
import wandb

ENTITY='wkdguswns2256'
PROJECT='layer allocation'
OUT=Path('/mnt/raid5/janghj/ODE-edit/local/zsre-wandb-20261009/views-r1')
QUERY='''query($e:String!,$p:String!){project(name:$p,entityName:$e){allViews(viewType:"project-view"){edges{node{id name displayName spec}}}}}'''
MUTATION='''mutation($id:ID,$e:String!,$p:String!,$n:String!,$d:String!,$s:String!){upsertView(input:{id:$id,entityName:$e,projectName:$p,type:"project-view",name:$n,displayName:$d,spec:$s,createdUsing:WANDB_SDK}){view{id name} inserted}}'''
MODELS={'llama3':'Llama3-8B-Instruct','qwen25':'Qwen2.5-7B-Instruct','gptj':'GPT-J-6B'}
FIELDS=('Efficacy','Generalization','Specificity','Specificity_loc_ans','Score','requests')


def url(name):
    return 'https://forge.coreweave.com/wandb/'+ENTITY+'/layer%20allocation?nw='+name[3:-2]


def write(path,value):
    with path.open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2)


def spec_for(base,model):
    spec=copy.deepcopy(base);sec=spec['section']
    name='zsRE — '+MODELS.get(model,'Model navigation')
    sec['name']=name;sec['customRunColors']={};sec['customRunNames']={}
    rs=copy.deepcopy(sec['runSets'][0]);sec['runSets']=[rs]
    filters=[dict(key=dict(section='config',name='dataset'),op='=',value='zsre',disabled=False),
        dict(key=dict(section='config',name='metric_schema'),op='=',value='official-baselines-scalar-v1',disabled=False,connector='AND')]
    if model:
        filters.append(dict(key=dict(section='config',name='model'),op='=',value=model,disabled=False,connector='AND'))
    else:
        filters.append(dict(key=dict(section='config',name='model'),op='IN',value=list(MODELS),disabled=False,connector='AND'))
    rs.update(name=name,filters=dict(filterFormat='filterV2',filters=filters),
        search={'query':''},grouping=[],selections={'root':1,'bounds':[],'tree':[]},expandedRowAddresses=[])
    rs['runFeed'].update(columnVisible={k:True for k in ('run:displayName','run:state',
        'config:dataset.value','config:model.value','config:writer.value','config:job_id.value')},onlyShowSelected=False)
    sections=sec['panelBankConfig']['sections']
    template=next(p for s in sections for p in s.get('panels',[]) if p['viewType']=='Run History Line Plot')
    section_template=copy.deepcopy(sections[0]);section_template['panels']=[]
    note='# '+name+'\n\n'+'\n'.join('['+label+']('+url('nw-zsre'+m+'-v')+')' for m,label in MODELS.items())
    note+='\n\nzsRE teacher-forced target accuracy/request-macro. Specificity = same-model W0 token prediction agreement; Specificity_loc_ans = separate labeled locality-answer accuracy. Percent 0–100, requests is a count. CF NLL preference and CF Flu/Con are excluded. Missing means unmeasured, never zero. Current cohort and measured all-seen are separate; W0 appears only at edits=0. No historical backfill; saved-view creation is not scientific evidence.'
    section_template.update(name='zsRE definitions and model pages',panels=[dict(__id__='zsre-note',
        layout=dict(x=0,y=0,w=12,h=5),viewType='Markdown Panel',config=dict(value=note))])
    new=[section_template]
    # Index is navigation only, avoiding cross-model overlays.
    if model:
        for endpoint in ('W0_first2000','current/pre','current/post','all_seen/post'):
            section=copy.deepcopy(section_template);section.update(name=endpoint,panels=[])
            for i,field in enumerate(FIELDS):
                panel=copy.deepcopy(template);panel['__id__']='zsre-'+model+'-'+endpoint.replace('/','-')+'-'+field
                panel['layout']=dict(x=(i%2)*6,y=(i//2)*6,w=6,h=6)
                panel['config']=dict(chartTitle=endpoint+' — '+field,xAxis='edits',
                    metrics=['zsre/'+endpoint+'/'+field],xAxisMin=0,xAxisMax=2000,
                    xAxisTitle='Actual edits (cold W0 = 0)',yAxisTitle='Requests' if field=='requests' else 'Percent',
                    yAxisMin=0,yAxisMax=2000 if field=='requests' else 100,
                    ignoreOutliers=False,smoothingWeight=0,smoothingType='none',limit=100,
                    legendPosition='south',legendTemplate='${run:displayName}',aggregate=False,
                    plotType='line',overrideSeriesTitles={},overrideMarks={})
                section['panels'].append(panel)
            new.append(section)
    sec['panelBankConfig']['sections']=new
    sec['panelBankSectionConfig']={}
    return spec


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    api=wandb.Api(timeout=30)
    before=api._service_api.execute_graphql(QUERY,variables={'e':ENTITY,'p':PROJECT})
    write(OUT/'before.json',before)
    nodes={e['node']['name']:e['node'] for e in before['project']['allViews']['edges']}
    base=json.loads(nodes['nw-pricefirst2kgptj-v']['spec']);published={}
    for model in (None,*MODELS):
        name='nw-zsre'+(model or 'index')+'-v'
        if name in nodes: raise ValueError('EXISTING_ZSRE_VIEW_REQUIRES_RECONCILE')
        spec=spec_for(base,model)
        response=api._service_api.execute_graphql(MUTATION,variables=dict(id=None,e=ENTITY,p=PROJECT,n=name,
            d='zsRE — '+MODELS.get(model,'Model navigation'),s=json.dumps(spec,ensure_ascii=False)))
        published[name]=dict(url=url(name),model=model,spec=spec,response=response)
        write(OUT/(name+'.json'),published[name])
    after=api._service_api.execute_graphql(QUERY,variables={'e':ENTITY,'p':PROJECT})
    write(OUT/'after.json',after)
    observed={e['node']['name']:e['node'] for e in after['project']['allViews']['edges']}
    for name,value in published.items():
        assert json.loads(observed[name]['spec'])==value['spec']
        value['spec_sha256']=hashlib.sha256(json.dumps(value.pop('spec'),sort_keys=True).encode()).hexdigest()
    assert all(observed[n]['spec']==v['spec'] for n,v in nodes.items()),'CONCURRENT_OTHER_VIEW_CHANGE'
    receipt=dict(status='SAVED_VIEWS_REMOTE_SPEC_READBACK_VERIFIED',views=published,
        existing_views_unchanged=list(nodes),browser_rendering='NOT_OBSERVED',
        run_history_writes=0,run_config_writes=0,new_science_runs=0,
        empty_charts_are_not_missing_value_zero=True)
    write(OUT/'receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__': main()
