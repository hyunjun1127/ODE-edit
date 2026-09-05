"""Canonical factual Korean report/receipt from validated sequential tables.

No raw mutation, model, torch, scheduler, or evaluator execution. Large scalar
tables remain compressed. Generated package is create-once.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
from .sequential_analysis import ARMS,CELLS,SOURCE,NR,require,stats,write_csv
from .sequential_plots import render
from .round0_analysis_contracts import canonical_hash,member,sha256_file,write_json_once,verify_canonical_identity

REPORT='orbode-sequential-b100x10-fourcell-factual-ko.md'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows])+'\n'


def fmt(x):
    if x is None or pd.isna(x):return NR
    return f'{float(x):.6g}'


def rate(r,k):
    return f'{int(r[k+"_num"])}/{int(r[k+"_den"])} ({100*r[k+"_rate"]:.2f}%)'


def metric_table(frame):
    return table(['cell','arm','RS','PS','PS strict','NS','rewrite acc','rephrase acc'],
        [[r.cell,r.arm,*[rate(r,k) for k in ('RS','PS','PS_strict','NS','rewrite_acc','rephrase_acc')]] for _,r in frame.iterrows()])


def compile_report(t:Path,package:Path,audit:dict,source:list,plots:list) -> str:
    read=lambda n:pd.read_csv(t/n)
    p=read('performance-summary.csv');online=p[p.scope=='ONLINE_AT_EDIT_TIME_10_BATCHES'];b10=p[p.scope=='FINAL_W10_CURRENT_B10_ONLY']
    b=read('batch-performance.csv');post=b[b.stage=='IMMEDIATE_POST'];entry=b[b.stage=='ENTRY_CURRENT_COHORT']
    m=read('batch-mechanism-compute.csv');n=read('aggregate-distributions.csv');d=read('batch-distributions.csv')
    pr=read('cell-provenance-compute.csv');layer=read('layer-updates.csv');node=read('node-telemetry.csv.gz')
    cumulative=read('online-prefix-not-cumulative-W.csv');paired=read('paired-versus-official.csv');der=read('derived-prefix.csv')
    lines=['# ORBODE B100×10 sequential — 4-cell factual report (Server4)',
       '', '## 1. 핵심 결과와 평가 범위', '',
       '이 실험은 **arm 내부 B1→B10 cumulative W/cache sequential**이다. '
       '각 cell에서 O/QCL/NQFIX/ORBFH/JAC가 동일 W0·cold method state로 각각 시작하고, '
       'arm 안에서는 앞 batch의 물리적 W와 applicable AlphaEdit cache를 다음 batch로 전달했다. '
       '모든 요청은 해당 batch entry state에서 z*를 한 번 계산했다. 네 arm을 선별하거나 나쁜 결과를 분모에서 제외하지 않았다.',
       '', '다음 첫 표는 **서로 다른 W1…W10에서 각각 자기 batch를 편집한 직후 측정한 1,000건 합계**다. '
       '최종 W10으로 전체 1,000건을 재평가한 값이 아니다. 후자의 평가와 복원 가능한 W checkpoint는 저장되지 않았다. '
       '따라서 최종 전체 유지율·진짜 누적 seen-prefix 성능·forgetting은 **NOT_RECORDED_NO_WEIGHT_CHECKPOINT**다. '
       '사용자 지시(보고서 먼저)에 따라 이번 분석에서 GPU 재실행·edit replay·값 추정은 하지 않았다.',
       '', metric_table(online),
       '분모: cell×arm당 request=1,000, RS=1,000 rewrite prompts, PS=2,000 rephrase prompts, NS=10,000 neighborhood prompts. '
       '전체 20개 arm×1,000=20,000 request-arm 관측이며 독립된 20,000개 데이터가 아니다. 공통 unique requests는 1,000개다. '
       'LM=Llama/MEMIT, LA=Llama/AlphaEdit, QM=Qwen/MEMIT, QA=Qwen/AlphaEdit.',
       '', '### 1.1 O 대비 산술 차이 (편집 직후 10-batch 합계)', '']
    comparisons=[]
    for c in CELLS:
        o=online[(online.cell==c)&(online.arm=='O')].iloc[0]
        for _,r in online[(online.cell==c)&(online.arm!='O')].iterrows():
            comparisons.append([c,r.arm,*[f'{100*(r[k+"_rate"]-o[k+"_rate"]):+.2f}' for k in ('RS','PS','NS')],
                fmt(r.rewrite_target_new_nll_mean-o.rewrite_target_new_nll_mean),
                fmt(r.rephrase_target_new_nll_mean_mean-o.rephrase_target_new_nll_mean_mean)])
    lines += [table(['cell','arm−O','ΔRS pp','ΔPS pp','ΔNS pp','Δrewrite new NLL mean','Δrephrase new cluster-NLL mean'],comparisons),
       '주요 관측: ORBFH−O의 NS 차이는 QA +6.80pp, LA +0.14pp, QM −2.97pp, LM −0.66pp다. '
       '동시에 PS 차이는 QA −1.45pp, LA −2.95pp, QM +0.35pp, LM −7.65pp다. '
       '따라서 네 cell에서 동일한 성능 방향은 관측되지 않았다. QA NQFIX의 RS/PS/NS는 '
       '71.8/64.25/57.28%로 O의 99.2/96.95/70.57%보다 모두 낮았다. '
       '이 수치들은 관측된 정책 간 산술 비교이며 단독 barrier의 인과 효과를 분리하지 않는다.',
       '이 비교는 같은 요청·순서의 **전체 sequential 정책 차이**다. B2부터는 각 arm의 앞선 W/cache가 달라져 '
       'z*/keys도 그 arm의 entry state에서 계산된다. 동일 z*·동일 중간 W에서 controller 하나만 바꾼 paired ablation으로 해석하지 않는다.',
       '', '## 2. 지표·arm 읽는 법', '',
       table(['지표','정의 / 집계 단위','방향·주의'],[
         ['RS','각 rewrite prompt에서 mean NLL(new) < mean NLL(true)','높을수록 new 선호. tie=failure; 1/request'],
         ['PS','각 rephrase prompt에서 mean NLL(new) < mean NLL(true)','높을수록 new 선호. 2/request를 개별 비교'],
         ['PS strict','한 request의 두 rephrase 모두 PS 성공','1/request, PS prompt 평균과 다름'],
         ['NS','각 neighborhood prompt에서 mean NLL(true) < mean NLL(new)','높을수록 true 선호. tie=failure; 10/request'],
         ['rewrite/rephrase acc','해당 new target의 모든 teacher-forced token이 top-1 정답인 prompt 비율','token 평균 accuracy 아님; free generation 아님'],
         ['PP-token','batch entry 대비 locality teacher-forced predicted token 일치수/비교 token수','NS가 아니며 W0 대비 전체 retention도 아님'],
         ['NLL','target token당 평균 negative log probability','낮을수록 해당 target 확률이 큼; token count로 이미 길이 정규화'],
         ['margin','NLL(true)−NLL(new)','양수면 new 선호. locality에서는 음수가 true 선호'],
         ['prompt / request-cluster','각 prompt 분포 / 먼저 request 안 prompt NLL 평균 후 request 분포','rephrase2, locality10을 혼합하지 않음'],
         ['mean/median/IQR/p90/max','산술평균/50%/25–75%/90%/최댓값; linear quantile','분모는 표의 n. batch 평균들의 median을 전체 median으로 쓰지 않음'],
         ['q','||z*−z_terminal||₂ / ||z*−z_entry||₂ (active request)','낮으면 entry residual 대비 더 근접; O의 이 항목은 미기록'],
         ['V','active request에 대해 mean(q²)/2','heldout NLL이 아닌 target activation potential'],
         ['u / h','response gain / Euler h=.25; 적용 coefficient=h·u','O는 stock one-pass; dynamic은 N4×5 layers=20 visits'],
         ['update magnitude','batch entry→endpoint의 layer별 ||ΔW||F','네 모델 raw norm pooling 금지. magnitude share와 squared-norm share 별도'],
         ['HORIZON_SEMANTIC_MISS','controller semantic predicate가 고정 horizon 안 all-request strict-hit하지 못함','technical failure 아님; RS/PS/NS 성공과 동일하지 않음'],
       ]),
       table(['arm','accepted source semantics'],[
         ['O','Official family-native one-pass / remaining-layer allocation'],
         ['QCL','current remaining-layer quota, unit response gain, per-visit refresh'],
         ['NQFIX','current full residual (no quota), unit response gain, per-visit refresh'],
         ['ORBFH','current full residual, current ordered-response gain, per-visit refresh; fixed horizon'],
         ['JAC','current full residual/response gain, per-sweep frozen Jacobi refresh'],
         ['ORBHit','ORBFH-derived materialized prefix; separate arm=0, primary denominator=0'],
       ]),
       '', '## 3. Source / jobs / 정확한 분모와 무결성', '',
       table(['cell','canonical job','source HEAD','source tree','runtime wall h','peak allocated GiB'],[
          [r.cell,r.job,r.source_head,r.source_tree,fmt(r.cell_wall_seconds/3600),fmt(r.peak_allocated_GiB)] for _,r in pr.iterrows()]),
       f"4/4 cell, 20/20 arm, 200/200 batch terminal-valid. 20,000/20,000 primary request endpoints; "
       f"W commit→next entry {audit['W_links']}/180, method-state link {audit['cache_links']}/180; "
       f"compute_z={audit['fixed_z_compute']}, batch 내부 recompute=0, final W0/method restore=4/4. "
       f"{audit['raw_members']} linked raw members와 {audit['deep_identity_count']} schema-bearing canonical identities를 다시 계산했다. "
       '실패·nonfinite·duplicate·imputation은 canonical scientific rows에서 0이다.',
       'AlphaEdit 각 arm: cache width 0→100→…→1000, 성공 batch당 append100; inner append/cache mutation0. '
       'MEMIT: covariance는 static computation cache, request-history append0. 상세 20개 chain은 `tables/chain-integrity.csv`.',
       'Model parameters는 FP32; autocast/quantization/BF16/FP16 cast0. '
       '**stock MEMIT의 ephemeral linear solve는 기존 승인된 FP64이며 model/storage FP32와 분리한다.** '
       '실행 source 두 lineage의 차이는 AlphaEdit cache_c_new lifecycle binding과 distinct launcher/run token의 기술 수정이다. '
       'source·hparam·HF/attention/dtype/numerical lock·EasyEdit identity 전체는 `tables/source-and-runtime-provenance.json`.',
       f"공통 stream root `{audit['stream_root']}`; 1,000개 request order root `{audit['order_root']}`. "
       '샘플 추가·교체·순서 변경·arm pruning0. 같은 seed의 단일 stream에 대한 기술적으로 유효한 관측이지 다중-seed 검증이 아니다.',
       '', '## 4. Sequential 진행과 누적 분석 — 가능한 것과 없는 것', '',
       table(['평가 타입','저장 여부','답하는 질문'],[
         ['IMMEDIATE_POST','200×B100, 20,000 request-arm rows','W_b가 방금 편집한 B_b를 얼마나 잘 반영했나'],
         ['ENTRY_CURRENT_COHORT','같은 200개 batch의 entry 평가','앞선 batch가 진행된 상태에서 새 B_b 편집 전 성능은 어떤가'],
         ['ONLINE_AT_EDIT_TIME_PREFIX','저장 행의 정확한 누계, 아래 200 rows','각 요청을 편집하던 시점의 성공률을 합치면 얼마인가'],
         ['CHECKPOINT_Wb_ON_ALL_SEEN','NOT_RECORDED_NO_WEIGHT_CHECKPOINT','W_b가 B1…B_b 모두를 현재 얼마나 유지하나: 판정 불가'],
         ['FINAL_W10_FULL1000','NOT_RECORDED_NO_WEIGHT_CHECKPOINT','최종 전체 1,000건 retention: 판정 불가'],
         ['W10_CURRENT_B10','20×B100만 있음','최종 상태의 마지막 100건 성능; 전체 유지율 아님'],
       ]),
       '실제 runtime은 매 batch 현재 100개 요청만 evaluator에 bind하고 JSON scalar receipt만 publish한다. '
       '완료 후 W0로 복구하며 checkpoint/delta tensor를 디스크에 저장하지 않는다. commit SHA는 byte identity의 증거이지 '
       'W를 재구성할 수 있는 state가 아니다. `cumulative-availability.csv`에 20 arms×10 시점의 부재를 명시했다. '
       '편집 직후 누계를 진짜 누적 유지율로 대체하지 않는다. 재실행 승인은 이번 보고서 작업에 포함하지 않았다.',
       '', '### 4.1 모든 batch의 편집 직후 성능과 entry→post 변화', '']
    for c in CELLS:
        rows=[]
        for a in ARMS:
            for _,r in post[(post.cell==c)&(post.arm==a)].sort_values('batch').iterrows():
                en=entry[(entry.cell==c)&(entry.arm==a)&(entry.batch==r.batch)].iloc[0]
                rows.append([a,int(r.batch),rate(r,'RS'),rate(r,'PS'),rate(r,'NS'),
                    *[f'{100*(r[k+"_rate"]-en[k+"_rate"]):+.1f}' for k in ('RS','PS','NS')]])
        lines += [f'#### {c} — 50 batch endpoints',table(['arm','B','RS','PS','NS','ΔRS entry pp','ΔPS entry pp','ΔNS entry pp'],rows)]
        text=[]
        for a in ARMS:
            g=post[(post.cell==c)&(post.arm==a)].sort_values('batch')
            low=g.loc[g.RS_rate.idxmin()];first,last=g.iloc[0],g.iloc[-1]
            text.append(f"{a}: RS B1 {100*first.RS_rate:.1f}%→B10 {100*last.RS_rate:.1f}%; "
                        f"최저 B{int(low.batch)} {100*low.RS_rate:.1f}%, PS {100*first.PS_rate:.1f}→{100*last.PS_rate:.1f}%, "
                        f"NS {100*first.NS_rate:.1f}→{100*last.NS_rate:.1f}%.")
        lines += [' '.join(text)+' 각 batch의 요청 자체가 다르므로 이 변화만으로 과거 요청의 forgetting을 측정했다고 할 수 없다.','']
    lines += ['### 4.2 편집 직후 누계 (명시적으로 W_t 누적 재평가가 아님)',
        'B1…B_b 각각의 자기 편집 직후 평가를 합친 값. N=100b; PS denominator=2N, NS=10N. '
        '아래 누계는 early requests를 W_b에서 다시 평가하지 않는다. 낮아지거나 높아지는 값 모두 관측 누계이며 retention 판정은 유보한다.','']
    for c in CELLS:
        g=cumulative[cumulative.cell==c]
        lines += [f'#### {c} — online prefix',table(['arm','through B','N','RS','PS','NS'],[
            [r.arm,int(r.through_batch),int(r.request_n),rate(r,'RS'),rate(r,'PS'),rate(r,'NS')] for _,r in g.iterrows()])]
    lines += ['### 4.3 W10의 마지막 B10 100건 (전체 1,000건 아님)',metric_table(b10),
        '', '## 5. Rewrite / rephrase / neighborhood NLL와 margin 분포',
        '각 소표는 **request-cluster** 기준 mean/median/p90/max이다. rewrite는1 prompt, rephrase는 request 내2 prompts 평균, '
        'neighborhood는10 prompts 평균 후 1,000 request 분포를 계산했다. primary PS/NS는 이 평균 NLL 비교가 아니라 개별 prompt pair 비교다. '
        '별도 prompt-level 전체분포와 모든 B100 분포는 `aggregate-distributions.csv`, `batch-distributions.csv`에 완전 수록. '
        'target-new/true 모두 제공하며 margin=true−new이다.','']
    for c in CELLS:
        g=n[(n.cell==c)&(n.stage=='IMMEDIATE_POST')&(n.scope=='ALL_BATCH_ENDPOINTS')&(n.unit=='request_cluster')]
        rows=[]
        for _,r in g.iterrows():
            rows.append([r.arm,r.category,int(r.nll_new_n),
                '/'.join(fmt(r['nll_new_'+k]) for k in ('mean','median','p90','max')),
                '/'.join(fmt(r['nll_true_'+k]) for k in ('mean','median','p90','max')),
                '/'.join(fmt(r['margin_true_minus_new_'+k]) for k in ('mean','median','p90','max'))])
        lines += [f'### {c}',table(['arm','category','cluster n','new NLL mean/median/p90/max','true NLL mean/median/p90/max','margin mean/median/p90/max'],rows)]
    lines += ['### 5.1 同一 request의 arm−O paired 차이',
        'request hash와 batch index로 1:1 join한 1,000개 paired delta. min/max outcome에 의한 case selection0. '
        '같은 batch 100개는 joint write를 공유하고 batch가 sequentially dependent하므로 독립 표본 p-value/확증 CI를 부여하지 않았다. '
        'positive/zero/negative counts와 모든 분포는 `paired-versus-official.csv`.','',
        table(['cell','arm','metric','n','Δmean','Δmedian','Δp90','Δmax','positive/zero/negative'],[
           [r.cell,r.arm,r.metric,int(r.n),*[fmt(r[k]) for k in ('mean','median','p90','max')],f'{r.positive}/{r.zero}/{r.negative}']
           for _,r in paired[paired.metric.isin(['rewrite_target_new_nll','rephrase_target_new_nll_mean','canonical_ns_rate'])].iterrows()]),
        '', '## 6. Target trajectory / response / semantic horizon',
        '각 dynamic batch는20 visits를 모두 실행한다. semantic first-hit 이후에도 fixed horizon을 완결하며 ORBHit는 별도 derived prefix다. '
        'HORIZON_SEMANTIC_MISS를 technical failure로 제외하지 않았다. semantic all-request strict는 별도 native-context 조건이어서 '
        'rewrite PS/RS와 같지 않다. raw에는 no-op/response gain0도 관측값으로 보존한다.','']
    mechrows=[]
    for c in CELLS:
        for a in ARMS:
            g=m[(m.cell==c)&(m.arm==a)];ss=node[(node.cell==c)&(node.arm==a)]
            mechrows.append([c,a,len(g),int((g.endpoint_status=='HORIZON_SEMANTIC_MISS').sum()),
                int(g.semantic_request_strict_count.sum()),len(ss),int((ss.coefficient_u==0).sum()) if len(ss) else NR,
                fmt(g.q_terminal_mean.mean()),fmt(g.q_terminal_max.max()),fmt(g.terminal_net_frobenius.mean())])
    lines += [table(['cell','arm','batch n','horizon miss','semantic strict /1000','node n','u=0 nodes','mean batch q-mean','worst q','mean ||ΔW||F'],mechrows),
       'O의 q는 해당 telemetry schema에 없어 NOT_RECORDED; O의 horizon miss=0은 동적 horizon 판정 PASS를 뜻하지 않는다. '
       '`node-telemetry.csv.gz`는 각 visit의 residual/potential/JVP response/gain/rank/actual-vs-predicted reduction과 '
       'request-vector 통계(n,mean,median,IQR,p90,max)를 기록한다. 원본 per-request vectors는 rooted raw journal에 있으며 Git에 복사하지 않았다.',
       '', '### 6.1 Layer-wise Update Magnitude',
       '실제 각 batch entry→endpoint의 layer별 Frobenius update 크기. 아래는 10개 batch의 관측 평균이며 '
       '최종 W10−W0 크기가 아니다. 순차 ΔW의 norm을 더하면 최종 net norm이 된다고 가정하지 않는다. '
       'magnitude share=||ΔW_l||/Σ||ΔW_j||, squared share=||ΔW_l||²/Σ||ΔW_j||²를 별도 CSV column으로 유지한다.',
       table(['cell','arm','L4','L5','L6','L7','L8'],[
          [c,a,*[fmt(layer[(layer.cell==c)&(layer.arm==a)&(layer.layer==l)].update_magnitude.mean()) for l in range(4,9)]] for c in CELLS for a in ARMS]),
       '', '### 6.2 ORBHit derived prefix',
       table(['cell','derived batches','first-hit observed','status counts'],[
          [c,len(der[der.cell==c]),int((der[der.cell==c].status=='FIRST_HIT').sum()),str(der[der.cell==c].status.value_counts().to_dict())] for c in CELLS]),
       'ORBHit는 ORBFH에서 파생된 평가이므로 independent arm처럼 primary20개 분모에 추가하지 않는다. '
       'horizon miss이면 source가 제공한 derived status 그대로 남긴다. 상세 실제 평가분모/성능은 `derived-prefix.csv`.',
       '', '## 7. 계산량과 wall time',
       '아래 endpoint call wall은 method execution 및 그 endpoint evaluator를 포함하고, batch-entry의 shared compute_z와 pre-evaluation은 밖에 있다. '
       '따라서 순수 solver/edit-core 시간으로 부르지 않는다. cell total에는 model load/모든 target 계산/entry+endpoint 평가/receipt 등이 포함된다. '
       'setup/evaluator-only 분리 시간, 전체 backward 횟수, CPU RSS는 NOT_RECORDED. JVP는 해당 scoped ledger 값이다. '
       'O의 adapter key/solve counter0은 stock 내부 호출을 계측하지 않는 scope의 0이지 stock solve가 없었다는 뜻이 아니다.',
       table(['cell','arm','endpoint call sum s','O 대비×','forward calls','JVP calls','factor builds','physical endpoint copies','temporary observation copies'],[
         [c,a,fmt((g:=m[(m.cell==c)&(m.arm==a)]).wall_seconds.sum()),
          fmt(g.wall_seconds.sum()/m[(m.cell==c)&(m.arm=='O')].wall_seconds.sum()),
          int(g.model_forward_invocation_count.sum()),int(g.jvp_jvp_call_count.sum()),int(g.factor_build_count.sum()),
          int(g.physical_write_count.sum()),NR] for c in CELLS for a in ARMS]),
       '임시 observation copy의 exact per-batch 값은 raw endpoint에 남아 있으며 위 요약표에서는 미집계다. 내부 counter는 `batch-mechanism-compute.csv`의 계측 scope를 확인한다. '
       '주어진 field가 없는 값은0으로 보충하지 않았다. model/storage FULL-FP32와 stock MEMIT ephemeral FP64의 경계는 §3과 동일하다.',
       '', '## 8. 기술 제외·출처 보존',
       table(['lineage','상태/원인','처리/과학 분모'],[
         ['37151_1 (child37155)','FAILED 00:39:59: B2 Alpha cache pointer/content boundary','B1 1개 publish되었으나 이 failed lineage의 final denominator0; 원본 immutable'],
         ['37151_3','CANCELLED_BEFORE_START, elapsed0','잘못된 cache binding 예방. denominator0'],
         ['37214_1 / 37214_3','TECH-R2 canonical completed','cache_c_new=true lifecycle repair와 run token 수정. source science/stream/tolerance unchanged'],
       ]),
       '모든 logs/result/실패 receipt는 local raw root에 보존했다. bytes/SHA는 `external-inputs.json`에 결속했다. '
       'MEMIT canonical job37151_0/2는 변경·재실행하지 않았다. 기존 raw/result bytes를 수정하거나 실패 분모를 성공과 합치지 않았다.',
       '', '## 9. 한계·판정',
       'FACT: 4 cell×5 arms×10 batch의 순차 실행 및 기술적 무결성은 확인됐다. canonical CounterFact RS/PS/NS는 prompt NLL pair로 '
       '다시 계산했으며 stored numerator/denominator 및 bit hash와 일치한다. 위 표의 성능 차이는 그대로 보존한다.',
       'SCOPE: observed immediate batch endpoints와 native activation trajectory만 확인했다. '
       '현재 저장 상태로 cumulative W_t all-seen retention, final W10 full1000 성능, edit-age forgetting, generation fluency, '
       'general benchmark, output-distribution KL은 확인할 수 없다. Missing 값을0으로 두지 않는다. '
       '서로 다른 batch 성능 추세를 같은 요청 longitudinal forgetting으로 부르지 않는다.',
       'DECISION: 보고서·코드 통합만 수행한다. 성능이 나쁘거나 horizon miss라는 이유의 사후 선택·재튜닝·재실행은0. '
       'ODE necessity, numerical convergence, barrier causal benefit 또는 보편적 모델 주장은 하지 않는다. scientific_promotion=false. '
       '진짜 누적 성능을 요구하는 후속 실행은 W_t에서 seen-prefix 평가 또는 recoverable checkpoint를 저장하는 별도 사용자 승인 범위다.',
       '', '## 10. 재현 가능한 그림',
       '모든 PNG는 저장소 Python/Agg 코드로만 생성했다. 동일 명령 재생성 SHA를 비교했다. '
       'seed20260906, DPI160, 11×7 inch, DejaVu Sans, cell order LM/LA/QM/QA, arm order O/QCL/NQFIX/ORBFH/JAC, '
       'missing interpolation/imputation0. rate 축0–102%, NLL·norm 축0부터 actual data autoscale. '
       'plot source/input/output SHA와 command는 `plot-reproduction.json`.', '']
    captions={
       'immediate-rs-trajectory.png':'각 B100에서 RS100 prompts; 서로 다른 cohort의 post-W_b 관측.',
       'immediate-ps-trajectory.png':'각 B100에서 PS200 prompts; request strict와 분리.',
       'immediate-ns-trajectory.png':'각 B100에서 NS1000 neighborhood prompt pairs; PP-token 아님.',
       'online-prefix-not-cumulative-W.png':'각 시점 N=100b, 각 요청의 편집 직후 score 누계. W_t 누적 retention이 아님.',
       'immediate-rewrite-nll.png':'batch당 request100; 실선median, 점선p90; 단일 rewrite prompt NLL.',
       'immediate-rephrase-nll.png':'batch당 request100; 먼저 두 rephrase NLL 평균. 실선median, 점선p90.',
       'layer-wise-update-magnitude.png':'각 bar는 10 sequential batches의 실제 layer update 크기 평균. 균등선/ideal line 없음.',
       'dynamic-terminal-residual.png':'dynamic arm당 batch100 requests의 terminal q 평균. O missing은 그리지 않음.',
       'endpoint-call-compute-ratio.png':'cell별 동일10 batch의 endpoint-call wall sum/O sum; evaluator 포함, compute_z 제외.',
    }
    for name in plots:lines += [f'![{name}](figures/{name})',captions[name],'']
    lines += ['## 11. Artifact inventory / reproduction',
        '제안/계약: GH-mediated authoritative local seals (`authoritative-documents.json`)을 authority로 사용했다. '
        '원래 repository proposal path의 부재와 전송 FULL_READ identity를 보존했으며 문서 사본을 Git에 추가하지 않았다.',
        'Raw root: `/data/janghj/ODE-edit/local/state/orbode-sequential-b100x10-server4-v1`. '
        'raw 로그·weight/checkpoint·model/cache·tensor·dataset·credential은 Git 포함0. 보고서에는 hash와 curated scalar 통계만 포함한다.',
        table(['table','rows','SHA256'],[[f'[{p.name}](tables/{p.name})',audit['table_rows'].get(p.name,'JSON'),sha256_file(p)]
             for p in sorted(t.iterdir()) if p.is_file()]),
        '추가 `cumulative-availability.csv`, `paired-batch-rate-deltas.csv`, `technical-exclusions.csv`는 completeness/누적 한계/기술 lineage를 고정한다. '
        '모든 package file은 `analysis-manifest.json`→`rooted-analysis-receipt.json`에 결속된다.',
        '분석 재현: README.md의 aggregate→package 명령. 실행 source를 바꾸거나 model을 로드하지 않는다. '
        '검증 테스트 및 package rehash 결과는 `verification.json`.','']
    return '\n\n'.join(lines)


def package(tables:Path,out:Path,raw:Path,repo:Path,test_receipt:Path):
    require(not out.exists(),'package create-once')
    audit=json.loads((tables/'analysis-audit.json').read_text());verify_canonical_identity(audit)
    require(audit['cells']==4 and audit['batches']==200,'canonical denominator')
    # Input tables are sealed at aggregation, and rehashed again below.
    out.mkdir(parents=True);(out/'tables').mkdir();(out/'figures').mkdir()
    for path in sorted(tables.iterdir()):
        if path.is_file():shutil.copyfile(path,out/'tables'/path.name)
    availability=pd.DataFrame([{'cell':c,'arm':a,'batch':b,'seen_requests':100*b,
        'cumulative_Wt_all_seen_evaluation':'NOT_RECORDED','recoverable_W_checkpoint':'NOT_RECORDED',
        'stored_endpoint_cohort_requests':100,'replay_or_imputation':0} for c in CELLS for a in ARMS for b in range(1,11)])
    write_csv(out/'tables/cumulative-availability.csv',availability)
    perf=pd.read_csv(tables/'batch-performance.csv');perf=perf[perf.stage=='IMMEDIATE_POST']
    delta=[]
    for c in CELLS:
        o=perf[(perf.cell==c)&(perf.arm=='O')]
        for a in ARMS[1:]:
            pair=perf[(perf.cell==c)&(perf.arm==a)].merge(o,on='batch',suffixes=('_arm','_O'),validate='one_to_one')
            for _,r in pair.iterrows():
                delta.append({'cell':c,'arm':a,'batch':int(r.batch),**{k+'_delta_pp':100*(r[k+'_rate_arm']-r[k+'_rate_O']) for k in ('RS','PS','NS','PS_strict')}})
    write_csv(out/'tables/paired-batch-rate-deltas.csv',pd.DataFrame(delta))
    repair_path=raw/'tech-r2/technical-repair-receipt.json';repair=json.loads(repair_path.read_text())
    fail=repair['failed_lineage'];require(sha256_file(Path(fail['failure_boundary_path']))==fail['failure_boundary_sha256'],'technical failure binding')
    write_csv(out/'tables/technical-exclusions.csv',pd.DataFrame([
        {'job':'37151_1','status':'FAILED','reason':'PURE_TECHNICAL_ALPHA_CACHE_BINDING_B2','published_batches':1,'canonical_denominator':0},
        {'job':'37151_3','status':'CANCELLED_BEFORE_START','reason':'PREVENT_SAME_TECHNICAL_DEFECT','published_batches':0,'canonical_denominator':0}]))
    ext=[]
    for path in sorted(raw.rglob('*')):
        if path.is_file() and '/results/' not in str(path):ext.append(member(path,kind='external_operational_evidence'))
    ext.append(member(Path(fail['failure_boundary_path']),kind='excluded_failure'))
    write_json_once(out/'external-inputs.json',{'members':ext,'members_root':canonical_hash(ext),'technical_repair':repair})
    docs=Path('/data/janghj/ODE-edit/local/state/orbode-sequential-b100x10-v1/authoritative-docs')
    expected={
      '2026-09-03-ordered-response-barrier-ode-edit-proposal.md':'03a6fc61258e7643fc281fab8ab0d3ca700bb80f09aaa3eb34591ce5827087be',
      '2026-09-03-ordered-response-barrier-ode-edit-gh-fast-main-prompt.md':'8548d016fda343f8b8917bcbe43647b58ad99ab6f796ece2fe4ef661faa4c431',
      '2026-09-03-ordered-response-barrier-ode-edit-fast-main-table.md':'e134ac708c556482b912d7103e12a12347a78958943e7fa3d3da0a473908aa40'}
    docmembers=[]
    for name,sha in expected.items():
        path=docs/name;require(sha256_file(path)==sha,'authority document identity');docmembers.append(member(path,kind='GH_authoritative_transfer'))
    write_json_once(out/'authoritative-documents.json',docmembers)
    sources=[member(p,relative_to=repo,kind='reusable_source') for p in sorted((repo/'project/run_scripts/ordered_response_barrier_ode').rglob('*'))
             if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in str(p)]
    write_json_once(out/'analysis-source-manifest.json',{'members':sources,'members_root':canonical_hash(sources),
        'HEAD':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'tree':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip()})
    plots=render(out/'tables',out/'figures')
    with tempfile.TemporaryDirectory(prefix='orbode-plot-reproduction-') as tmp:
        render(out/'tables',Path(tmp))
        require(all(sha256_file(out/'figures'/n)==sha256_file(Path(tmp)/n) for n in plots),'plot byte reproduction')
    cmd='python -m project.run_scripts.ordered_response_barrier_ode.sequential_plots --tables PACKAGE/tables --output NEW_EMPTY_FIGURE_DIR'
    write_json_once(out/'plot-reproduction.json',{'command':cmd,'backend':'Agg','seed':20260906,'DPI':160,'figure_inches':[11,7],
        'cell_order':list(CELLS),'arm_order':list(ARMS),'missing_policy':'NO_IMPUTATION_OR_INTERPOLATION',
        'source_sha256':sha256_file(repo/'project/run_scripts/ordered_response_barrier_ode/sequential_plots.py'),
        'input_members':[member(out/'tables'/n,relative_to=out) for n in ['batch-performance.csv','online-prefix-not-cumulative-W.csv','batch-distributions.csv','layer-updates.csv','batch-mechanism-compute.csv']],
        'figures':[member(out/'figures'/n,relative_to=out) for n in plots],'byte_stable_reproduction':True})
    report=compile_report(out/'tables',out,audit,sources,plots)
    with (out/REPORT).open('x') as f:f.write(report)
    readme=(f'# ORBODE sequential canonical report\n\n[한국어 보고서]({REPORT})\n\n'
       'Scope: arm-local B100×10 cumulative W/cache sequential. Reports are online/immediate; final full-history evaluation was not recorded.\n\n'
       'Analysis-only reproduction (use the pinned CPU environment with NumPy/Pandas/Matplotlib):\n\n```bash\n'
       'python -m project.run_scripts.ordered_response_barrier_ode.sequential_analysis --raw-root RAW_ROOT --output NEW_ANALYSIS_DIR\n'
       'python -m project.run_scripts.ordered_response_barrier_ode.sequential_report --tables NEW_ANALYSIS_DIR --raw-root RAW_ROOT --output NEW_PACKAGE --test-receipt TEST_RECEIPT\n'
       'python -m project.run_scripts.ordered_response_barrier_ode.sequential_package_verify NEW_PACKAGE\n```\n\n'
       'Static figures: regenerate from sealed CSV using plot-reproduction.json. No model/GPU or evaluator invocation.\n')
    with (out/'README.md').open('x') as f:f.write(readme)
    test=json.loads(test_receipt.read_text())
    write_json_once(out/'verification.json',{'focused_tests':test,'raw_audit':audit,'plots_byte_stable':True,
        'package_rehash':'RUN_SEQUENTIAL_PACKAGE_VERIFY','GPU':0,'model_load':0,'Slurm_submit':0,'raw_mutation':0})
    members=[member(p,relative_to=out,kind='canonical_report_member') for p in sorted(out.rglob('*')) if p.is_file()]
    manifest={'schema':'orbode.server4.sequential-analysis-manifest.v1','members':members,'members_root':canonical_hash(members),
       'raw_members_root':audit['raw_member_root'],'execution_sources':sorted(set(SOURCE.values())),
       'report':REPORT,'report_sha256':sha256_file(out/REPORT),'scientific_promotion':False,
       'evaluation_scope':'IMMEDIATE_AND_ONLINE_PREFIX_NOT_FINAL_W_FULL_HISTORY'}
    manifest['identity_sha256']=canonical_hash(manifest);write_json_once(out/'analysis-manifest.json',manifest)
    receipt={'schema':'orbode.server4.rooted-sequential-analysis-receipt.v1','status':'REPORT_COMPLETE_CUMULATIVE_RETENTION_NOT_RECORDED',
       'report':REPORT,'report_sha256':sha256_file(out/REPORT),'manifest_sha256':sha256_file(out/'analysis-manifest.json'),
       'manifest_identity':manifest['identity_sha256'],'members_root':manifest['members_root'],'raw_member_root':audit['raw_member_root'],
       'cells':4,'arms':20,'batches':200,'request_endpoints':20000,'scientific_promotion':False,'new_experiment_action':0}
    receipt['identity_sha256']=canonical_hash(receipt);write_json_once(out/'rooted-analysis-receipt.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tables',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--raw-root',required=True,type=Path);p.add_argument('--test-receipt',required=True,type=Path)
    args=p.parse_args();print(json.dumps(package(args.tables,args.output,args.raw_root,Path.cwd(),args.test_receipt),ensure_ascii=False))
