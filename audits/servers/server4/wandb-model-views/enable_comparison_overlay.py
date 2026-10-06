"""Promote endpoint-aligned companion metrics into existing comparison panels."""
import json
import hashlib
import wandb
from inspect_views import ROOT,ENTITY,PROJECT,QUERY
from update_views import IDS,SOURCE,LABELS,MUTATION,url

def main():
    api=wandb.Api(timeout=30)
    before=api._service_api.execute_graphql(QUERY,variables={'e':ENTITY,'p':PROJECT})
    with (ROOT/'comparison-views-before.json').open('x') as f:json.dump(before,f)
    nodes={e['node']['name']:e['node'] for e in before['project']['allViews']['edges']}
    changed={}
    for model,name in list(IDS.items())+[('LLAMA',SOURCE)]:
        node=nodes[name];spec=json.loads(node['spec']);bank=spec['section']['panelBankConfig']
        bank['sections']=[s for s in bank['sections'] if not s['name'].startswith('Live legacy logger')]
        note=(f'# {LABELS[model]} — First 2k: Baselines / PRICE-59768 / New runs\n\n'
            '현재 실행의 **[comparison]** run은 원본 job의 확정 batch 관측을 CPU로 재집계한 실시간 비교용 기록입니다. '
            '추가 모델 평가와 실험 코드 변경은 없습니다. 원본 run은 학습 loss/시스템 추적을 계속합니다.\n\n'
            '**Current100 post**: 각 batch의 새 요청 R100/P200/N1000. '
            '**All-seen post**: W5/W10/W15/W20의 누적 실제 평가. '
            '동일 패널의 모든 run은 동일 metric 이름·% 단위·edits 축을 사용합니다. '
            'RS/PS/NS 조화평균도 해당 endpoint의 같은 분모에서 계산합니다. '
            '아직 도달하지 않은 milestone과 없는 baseline은 결측이며 current 평균으로 대체하지 않습니다.\n\n'
            '주의: 같은 그래프/평가 정의이지 서로 다른 seed/runtime/history까지 동일한 실험 환경이라는 뜻은 아닙니다. '
            '기존 baseline은 historical comparison입니다.\n\n')
        if model!='LLAMA':note+='이 모델의 imported baseline과 PRICE-59768은 없습니다. Llama 결과를 혼합하지 않습니다.\n'
        if name==SOURCE:note='# Ours and Baselines - First 2k Comparison\n\n'+'\n'.join(
            f'- [{LABELS[m]}]({url(n)})' for m,n in IDS.items())+'\n\n아래는 Llama 전용 비교입니다.\n\n'+note
        bank['sections'][0]['panels'][0]['config']['value']=note
        api._service_api.execute_graphql(MUTATION,variables=dict(id=node['id'],e=ENTITY,p=PROJECT,
            n=name,d=node['displayName'],s=json.dumps(spec,ensure_ascii=False)))
        changed[name]=spec
    after=api._service_api.execute_graphql(QUERY,variables={'e':ENTITY,'p':PROJECT})
    actual={e['node']['name']:e['node'] for e in after['project']['allViews']['edges']}
    for name,spec in changed.items():assert json.loads(actual[name]['spec'])==spec
    for name,node in nodes.items():
        if name not in changed:assert actual[name]['spec']==node['spec']
    receipt=dict(status='COMPARISON_OVERLAY_REMOTE_READBACK_VERIFIED',
        views={n:dict(url=url(n),sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()) for n,s in changed.items()},
        metric_namespace=['current/pre','current/post','all_seen/post'],success_unit='percent',
        original_training_runs_modified=False,no_new_model_evaluation=True)
    with open('audits/servers/server4/wandb-model-views/comparison-overlay-receipt.json','x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))

if __name__=='__main__':main()
