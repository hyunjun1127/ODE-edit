"""Read-only saved-view/run metadata inventory; credentials never serialized."""
import json
from pathlib import Path
import wandb

ROOT=Path('/data/janghj/ODE-edit/local/wandb-model-views')
ENTITY='wkdguswns2256'
PROJECT='layer allocation'
QUERY='''query($e:String!,$p:String!){project(name:$p,entityName:$e){allViews(viewType:"project-view"){edges{node{id name displayName spec}}}}}'''

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    api=wandb.Api(timeout=30)
    response=api._service_api.execute_graphql(QUERY,variables={'e':ENTITY,'p':PROJECT})
    backup=ROOT/'views-before.json'
    with backup.open('x') as f:json.dump(response,f,ensure_ascii=False,indent=2)
    for edge in response['project']['allViews']['edges']:
        node=edge['node'];spec=json.loads(node['spec'])
        print(json.dumps(dict(name=node['displayName'],internal=node['name'],spec=spec),ensure_ascii=False))
    rows=[]
    keys=('model','model_name','model_family','model_profile','task_id','arm','method','writer','benchmark','edits','requests','dataset','cohort','batch_size')
    for run in api.runs(ENTITY+'/'+PROJECT,per_page=100):
        rows.append(dict(id=run.id,name=run.name,tags=run.tags,
            config={k:v for k,v in run.config.items() if k in keys},
            config_key_names=sorted(run.config),summary_key_names=sorted(run.summary.keys())))
    with (ROOT/'runs-before.json').open('x') as f:json.dump(rows,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(runs=rows),ensure_ascii=False))

if __name__=='__main__':main()
