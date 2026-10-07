"""Rebuild all published tables offline from the compact immutable snapshots."""
import collections
import csv
import datetime
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REPORT=ROOT/'experiment-reports/global/2026-10-08-ours-baseline-results'


def write_csv(name, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (REPORT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def pct(n,d): return str((Decimal(n)*100/Decimal(d)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))


def main():
    captures=[json.loads(p.read_text()) for p in sorted(REPORT.glob('server*-snapshot.json'))]
    cells=[r for s in captures for r in s['cells']]
    assert len(cells)==len(json.loads((HERE/'registry.json').read_text()))==54
    metrics=[];milestones=[];inventory=[];cost=[];prices=[];fits=[];generation=[];sources=[]
    for index,r in enumerate(cells):
        ident={k:r[k] for k in ['server','model','role','series','task','arm']}
        ident['cell_id']=index+1
        terminal=r.get('terminal',{})
        last=max([s['batch'] for s in r['summaries'] if s['phase']=='post'],default=0)
        row=dict(ident,root_exists=r['root_exists'],commits_observed=len(r['batches']),
            last_post_batch=last,terminal_status=terminal.get('status','NOT_RECORDED'),
            job=terminal.get('job',''),root=r['root'],generation_endpoints=len(r['generation']),
            partial_generation_observation_files=r.get('partial_generation_observation_files',0))
        inventory.append(row)
        for s in r['summaries']:
            for kind,m in s['metrics'].items():
                v=dict(ident,batch=s['batch'],phase=s['phase'],scope=s['scope'],
                    state_edits=0 if s['phase']=='W0' else 100*(s['batch']-1) if s['phase']=='pre' else 100*s['batch'],
                    requests=s['requests'],family=kind,**m,
                    success_pct=100*m['numerator']/m['denominator'],source_path=s['source_path'],source_sha256=s['source_sha256'])
                metrics.append(v)
        for b in [5,10,15,20]:
            found=[s for s in r['summaries'] if s['batch']==b and s['phase']=='post' and s['scope']=='all_seen']
            assert len(found)<=1
            v=dict(ident,edits=b*100,availability='MEASURED' if found else 'NOT_PRODUCED')
            if found:
                for k,m in found[0]['metrics'].items():
                    v[k+'S_pct']=pct(m['numerator'],m['denominator'])
                    v[k+'_numerator']=m['numerator'];v[k+'_denominator']=m['denominator']
                v['source_path']=found[0]['source_path'];v['source_sha256']=found[0]['source_sha256']
            milestones.append(v)
        for b in r['batches']:
            cost.append(dict(ident,**{k:v if not isinstance(v,(dict,list)) else json.dumps(v,sort_keys=True) for k,v in b.items()}))
        for p in r['prices']:
            for layer,stats in p['effective_pi'].items():
                prices.append(dict(ident,batch=p['batch'],layer=layer,**stats,
                    cheapest_count_ties_included=p['cheapest_layer_count_ties_included'].get(layer,0),
                    beta_base_mean=p['beta_base']['mean'],beta_max_mean=p['beta_max']['mean']))
        for f in r['fits']:
            fits.append(dict(ident,**{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in f.items()}))
        for g in r['generation']:
            generation.append(dict(ident,**{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in g.items()}))
        for f in r['files']: sources.append(dict(cell_id=index+1,**f))
    for name,rows in [('metrics.csv',metrics),('milestones.csv',milestones),('inventory.csv',inventory),
                      ('batch-cost-provenance.csv',cost),('price-distributions.csv',prices),
                      ('fit-terminal-expansion.csv',fits),('source-manifest.csv',sources)]:write_csv(name,rows)
    if generation:write_csv('generation-metrics.csv',generation)
    measured=sum(r['availability']=='MEASURED' for r in milestones)
    ours_measured=sum(r['availability']=='MEASURED' and r['role']=='ours' for r in milestones)
    end=max(s['finished_utc'] for s in captures)
    when=datetime.datetime.fromisoformat(end).astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    text=[ '# OURS와 baseline 중간 결과 2026년 10월 8일', '',
      f'{when}까지 서버별로 읽은 시점의 snapshot이다. 현재 세 모델 PRICE 18개 arm, 취소된 Qwen 6개 arm, 이전 baseline 12개 arm, 생성 평가를 추가한 baseline 18개 arm의 총54개 실행 구성을 구분한다. 실행 중인 결과의 한정 관측이며 이후 자동 갱신하지 않는다.', '',
      f'0.5K·1K·1.5K·2K 누적 평가 {measured}개 지점(OURS {ours_measured}, baseline {measured-ours_measured})이 확보됐다. 표의 값은 RS / PS / NS (%)이며, 미산출을 0으로 대체하지 않았다. 각 endpoint의 분모는 R=N, P=2N, N=10N이다. 같은 이름의 기존 baseline과 generation 재실험은 별도 실행이며 값을 이어붙이지 않는다.', '',
      '## OURS 누적 성능', '', '| 모델 | Arm | 0.5K | 1K | 1.5K | 2K |', '|---|---|---|---|---|---|' ]
    def table_row(r):
        vals=[]
        for b in [500,1000,1500,2000]:
            m=next(x for x in milestones if x['cell_id']==r['cell_id'] and x['edits']==b)
            vals.append(' / '.join(m[k+'S_pct'] for k in ['R','P','N']) if m['availability']=='MEASURED' else '미산출')
        return '| '+r['model']+' | '+r['arm']+' | '+' | '.join(vals)+' |'
    for r in inventory:
        if r['series']=='price':text.append(table_row(r))
    text+=['','## 이전 baseline 누적 성능','','아래는 생성 평가 추가 전 실행이다. 저장된 R/P/N 성능을 게시하며 fluency·consistency 측정으로 표시하지 않는다.','','| 모델 | Method | 0.5K | 1K | 1.5K | 2K |','|---|---|---|---|---|---|']
    for r in inventory:
        if r['series']=='prior_no_generation':text.append(table_row(r))
    text+=['','## 생성 평가 재실험과 보존 결과','',
      '생성 평가 재실험의 편집 후 누적 지표와 완료된 generation endpoint는 이 snapshot에서 미산출이다. GPT2-XL의 개별 생성 관측 파일은 작성 중이며, 그 수만 inventory에 보존했다. 전체 cohort 점수로 평균 내거나 완료된 endpoint로 취급하지 않는다. W0에서 확보된 R/P/N은 metrics.csv에 별도 scope로 남겼다.', '',
      '취소된 Qwen의 보존된 B1 및 W0 결과도 metrics.csv에 포함하며, 새 실행 또는 재개를 뜻하지 않는다. GPT-J OURS의 terminal TECHNICAL_BLOCKED와 완료 전 결과를 inventory에서 구분한다. terminal이 없는 항목은 NOT_RECORDED이며 이를 RUNNING으로 추정하지 않는다.', '',
      '## 파일과 검산 범위','',
      '- [전체 관측 지표](metrics.csv): W0, 각 batch current/pre·current/post, 실제 milestone all_seen/post. NLL·TF token/prompt/strict 집계도 포함한다.',
      '- [누적 지표와 분모](milestones.csv), [실행 상태 및 미산출](inventory.csv).',
      '- [Batch 비용 및 source/config](batch-cost-provenance.csv), [층별 가격 분포와 최저가 층](price-distributions.csv), [종료 상태 및 예산 확장](fit-terminal-expansion.csv).',
      '- [원자료 경로·SHA256](source-manifest.csv), 서버별 snapshot JSON. 원 prompt·생성문·토큰열·event stream·checkpoint는 각 서버 local에 보존한다.', '',
      '이 게시 검산은 endpoint/cohort 크기, R/P/N 분모, numerator/rate 산술, finite 값, 원 observer의 no_mutation 표시 및 읽은 파일 SHA를 확인했다. 기존 summary를 사용한 게시 검산이며 모든 raw row의 독립 재평가나 method 정합 인증은 아니다. 비용은 원 receipt의 의미를 보존하며 중첩 시간을 합산하지 않는다. 가격 최저층 집계는 동률을 모두 포함하므로 요청수보다 합이 클 수 있다.', '',
      '모델별 effective hparams/source와 이전 baseline의 writer·history 정책 차이는 원 실행 보고서에 따른다. 이 표만으로 완전히 통제된 인과 비교를 주장하지 않는다. 기존 Llama 10k baseline 게시 자료는 [통합 리뷰](../../servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/diagnostic-report-ko.md), MEMIT-history는 [완료 리뷰](../../servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1/report-ko.md)에 보존되어 있다.', '',
      '오프라인 표 재생성: `python3 audits/global/2026-10-08-ours-baseline-results/build_report.py`. 원 실험 코드·job·대형 raw를 변경하지 않았다. 수집 스크립트는 별도 read-only snapshot용이며 재실행 시 이미 저장한 서버 snapshot을 보존한다.', '' ]
    (REPORT/'report-ko.md').write_text('\n'.join(text))
    checks=dict(cells=len(cells),metric_rows=len(metrics),milestone_rows=len(milestones),
        measured_milestones=measured,ours_measured_milestones=ours_measured,baseline_measured_milestones=measured-ours_measured,
        price_layer_rows=len(prices),fit_rows=len(fits),batch_receipts=len(cost),source_receipts=len(sources),
        generation_endpoint_rows=len(generation),snapshot_finished_kst=when,
        raw_rows_independently_reduced=False,new_model_forwards=0,new_jobs=0,large_raw_exported=False)
    (HERE/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    files=[]
    for parent in [REPORT,HERE]:
        for p in sorted(parent.iterdir()):
            if p.is_file() and p.name!='artifact-manifest.json':
                raw=p.read_bytes();files.append(dict(path=str(p.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    (HERE/'artifact-manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
