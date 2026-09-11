"""Continuous Korean diagnostic review, generated from verified tables."""
from .common import *
from .plots import ordered,variant
import json
from .common import csvwrite as create_csv

def csvwrite(path,rows):
    if path.exists():
        keys=list(dict.fromkeys(k for r in rows for k in r))
        assert csvread(path)==[{k:str(r.get(k,'')) for k in keys} for r in rows]
    else:create_csv(path,rows)

def pct(n,d):return f'{int(float(n))}/{int(float(d))} ({100*float(n)/float(d):.3f}%)'
def f(v):
    try:return f'{float(v):.4f}'
    except (ValueError,TypeError):return str(v) if v else 'NOT_RECORDED'
def main():
    names=['final-summary','final-distributions','cumulative-metrics','current-metrics','prompt-transitions','layer-summary','compute-summary','chain-integrity','configured-policy-comparison','compatibility','source-config-compatibility','final-fixed-cohorts','paired-final-transitions','paired-locality-nll-changes','technical-exclusions','overwrite-strata','mechanism-performance-associations','outlier-ledger']
    t={n:csvread(OUT/(n+'.csv')) for n in names};summary=t['final-summary'];fs={r['arm']:r for r in summary}
    sel=lambda name,a,**kw:[r for r in t[name] if r.get('arm')==a and all(str(r.get(k))==str(v) for k,v in kw.items())]
    def metrics(a,b,table_name='cumulative-metrics'):
        rr=sel(table_name,a,batch=b)
        return {r['metric']:r for r in rr}
    text='# Server4 Llama fixed10k lifelong 상세 통합 리뷰 — native · BLUE · L4–L8\n\n'
    text+='instruction_id: '+INSTRUCTION+'\n\n2026-09-11. 기존 BLUE6+W0 v3의 후속 **독립 보고서**이며 기존 bytes를 수정하지 않았다. 사용자 “자세히 리뷰” 및 단일-layer/BLUE/원본 양계열 비교 지시에 따라 FACT와 가능한 설명을 분리한다. 모든 수치는 **최종 실제 W100 전체10,000 재평가**가 우선이다. current100 또는 online 합계로 대체하지 않았다. 과학 promotion=false, 새 GPU/model/forward/evaluator/edit/Slurm 변경0.\n\n'
    text+='## 1. 최종 성능 — 두 계열 각각의 주표\n\n14 scheduler COMPLETED0, 신규8 raw/selected CP CPU 검산 및 기존6 봉인 검증 재사용을 구분했다. 14 chains×10,000=140,000 arm-request 관측, 고유 요청10,000; W0는 SH2의 공통1회이며 두 표에 표시해도 두 실험으로 계수하지 않는다.\n\n'
    for method in ['MEMIT','AlphaEdit']:text+='### '+method+'\n\n'+family_table(summary,method)+'\n'
    text+='![family final](figures/final-family-rspsns.png)\n\n분모는 각 arm RS10,000 / PS20,000 / NS100,000이다. 같은 sample이지만 **native와 BLUE는 layers·target 정책·regularization이 달라** method-only 인과효과를 주장하지 않는다. W0 NS89.212%보다 모든 편집 arm NS가 낮다. W0 RS7.910%는 편집 전 새 target 선호율이므로 pretrained 전체 능력 저하가 아니다.\n\n'
    text+='### 핵심 관측과 해석 경계\n\n'
    for method in ['MEMIT','AlphaEdit']:
        single=[a for a in ordered(method) if variant(a).startswith('L')];blue=next(a for a in ordered(method) if variant(a)=='BLUE');base=next(a for a in ordered(method) if variant(a)=='Native')
        best={tag:max(single,key=lambda a:float(fs[a][tag+'_rate'])) for tag in MULT}
        text+=f"- **FACT — {method}:** 단일-layer 최고 RS는 {label(best['RS'])} {100*float(fs[best['RS']]['RS_rate']):.3f}%, 최고 PS는 {label(best['PS'])} {100*float(fs[best['PS']]['PS_rate']):.3f}%, 최고 NS는 {label(best['NS'])} {100*float(fs[best['NS']]['NS_rate']):.3f}%다. 하나의 layer가 모든 지표를 동시에 대표한다고 가정하지 않았다.\n"
        for a in [base,blue,best['RS']]:
            q=sel('prompt-transitions',a,batch=100,metric='RS',comparison='AT_WRITE_TO_CHECKPOINT')[0]
            text+=f"- **FACT — {label(a)}:** at-write RS {q['before_num']}/10000 → final {q['after_num']}/10000. 이후 lost {q['lost']} / at-write 성공 {q['conditional_loss_den']}, recovery {q['gained']}; 최초 실패와 후속 forgetting을 분리했다.\n"
    text+='- **설명 가능 범위:** current batch에서는 쓰기 성공했으나 final에서 실패한 항목은 이 순서의 후속 editing에 따른 유지 손실로 기술할 수 있다. 다만 parameter 위치·local target·정규화·history의 개별 원인 효과는 분리하지 못했다. BLUE가 native보다 높다고 그 이유가 layer 수 감소 하나라고 결론내리지 않는다.\n- **DECISION:** 보고 및 코드/표 공개만 수행한다. 후속 실험·promotion·중단 ORBODE 감사 재개는 하지 않는다.\n\n'
    text+='## 2. 지표 읽는 법과 분모\n\n'
    glossary=[dict(지표='RS',정의='rewrite 1/request: new 평균-token NLL < true 평균-token NLL',단위='prompt 10000; 높을수록 새 edit 목표 선호'),dict(지표='PS',정의='rephrase 2/request 각각 new NLL < true NLL; request평균 비교 아님',단위='prompt 20000; 높을수록 선호 일반화'),dict(지표='NS',정의='neighborhood 10/request 각각 true NLL < new NLL',단위='prompt 100000; 높을수록 기존 true 선호'),dict(지표='NLL',정의='teacher-forced target token −log p 평균; new와true 각각',단위='nats/token; 해당 target에는 낮을수록 높은확률'),dict(지표='margin',정의='true NLL − new NLL (NS에서도 부호 그대로)',단위='nats/token; RS/PS 양수성공, NS 음수성공'),dict(지표='TF strict / token accuracy',정의='모든 target token top1 일치 / 정확 token 합÷실제 token수',단위='secondary; 자유 생성 정확도·primary선호와 다름'),dict(지표='request pair-strict',정의='한 request의 해당 category 모든 prompt 선호 성공',단위='request 10000; TF strict와 다름'),dict(지표='mean/median/IQR/p90/p99/max',정의='산술평균/중앙값/q75−q25/상위분위/최댓값',단위='prompt와 request-cluster 평균 후 분포를 별도column'),dict(지표='Layer-wise Update Magnitude',정의='||W_l after batch − W_l entry||_F',단위='stored-weight 실제 batch-net Frobenius norm; native action 아님'),dict(지표='share / path',정의='layer magnitude÷같은batch selected합 / 100 batch-net norms의 합',단위='비중 / 경로길이 proxy; W100−W0 net 또는 원소별누적벡터와 다름'),dict(지표='lost / gained',정의='동일 prompt identity 성공→실패 / 실패→성공',단위='conditional loss 분모는 이전 성공; 전체전이합은 원분모')]
    text+=table(glossary,['지표','정의','단위'])+'\n모든 부등식은 strict `<`, tie는 failure. RS와 NS는 요구 선호 방향이 반대다. NS가 높아져도 같은 문항을 유지했다는 뜻이 아니므로 전이를 별도로 센다. 통계는 단일 고정 order 기술통계이며 100k neighborhood prompt를 독립 실험 반복처럼 취급하지 않는다. p-value/확증적 유의성 주장0.\n\n'
    text+='## 3. 실행 inventory · availability · source/config\n\n'
    rows=[]
    for a in ARMS:
        ci=sel('chain-integrity',a)[0];r=fs[a]
        rows.append(dict(arm=label(a),job=r['job'],scheduler='COMPLETED 0:0',batches=ci['batches'],requests=ci['requests'],checkpoints=ci['checkpoints'],검증='기존v3/72CP transfer검증 재사용' if a in OLDARMS else '이번 full file SHA+CPU selected/target/state 검사'))
    text+=table(rows,['arm','job','scheduler','batches','requests','checkpoints','검증'])
    text+='\n현재 core14에 미완료/결측 arm은 없지만 과거 superseded/미실행은 별도 제외표에 남겼다. Scheduler만으로 이 판단을 내리지 않았다. CPU reload는 GPU continuation replay가 아니다. source/edit equation correctness 전체를 file hash 하나로 증명한다는 뜻도 아니다.\n\n'
    text+=table([dict(arm=label(r['arm']),**{k:r[k] for k in ['blue','layers','L2','mom2_update_weight','v_weight_decay','v_lr','v_num_grad_steps','v_loss_layer','target_policy','divisor']}) for r in t['configured-policy-comparison']],['arm','blue','layers','L2','mom2_update_weight','v_weight_decay','v_lr','v_num_grad_steps','v_loss_layer','target_policy','divisor'])
    text+='\nBase는 BLUE repository 원본 `blue=False` entrypoint이며 JVP controller가 아니다. BLUE는 current-W의 각 선택 layer에서 native z를 다시 계산하고 전체 residual로 solve한다. Single-layer는 native layer 자체가 target/readout이다. `v_loss_layer=31`은 loss/readout head 경로 설정으로 activation target layer를 L31로 교체했다는 뜻이 아니다.\n\nMEMIT은 `(λC + KKᵀ)⁻¹K`를 FP64 solve 후 residual과 곱하고 FP32 weight에 materialize한다. Base는 남은 layer 수5/4/3/2/1로 residual을 나누고 BLUE는 divisor1. MEMIT 내부 temporary apply→entry restore→returned deltas materialize를 wrapper의 실제 endpoint에서 평가한 증거를 확인했다. AlphaEdit은 `P(KKᵀ+M)+L2 I` 선형계와 `PKRᵀ`를 사용하며 끝에 선택 layer의 실제 key로 M을 append한다. Blue L2=1, native L2=10 차이를 제거하지 않았다.\n\n'
    text+='### 호환성 / 불변성\n\n'
    text+='모든14 모델은 Llama3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32 parameter, eager, torch2.9.1+cu128/transformers4.44.2, TF32matmulFalse/cudnnTrue다. Native MEMIT solve FP64 예외는 원본정책이며 전부FP32 연산이라고 쓰지 않는다. selectedW0는 five-layer native reference와 key별 SHA 일치; 미선택 전체parameter는 runtime pointer/version guard로, full byte parity는 미측정이다. evaluator wrapper는 explicit left padding과 source-default position 경로, microbatch16을 유지한다. 원문 source/config/archive, tokenizer property, context hash는 `source-config-compatibility.csv`, `configured-policy-comparison.csv`, `compatibility.csv`에 각각 결속한다.\n\n'
    text+=table([dict(arm=label(r['arm']),source_HEAD=r['source_head'],archive_SHA=r['source_archive_sha256'],runtime_SHA=r['runtime_sha256']) for r in t['source-config-compatibility']],['arm','source_HEAD','archive_SHA','runtime_SHA'])
    text+='\n동일 fixed10k root5b013569…5729 및 dataset SHA3d7f5e31…10ae1를 검증했다. BLUE historical full-dataset selector와 native dedicated subset 파일의 **파일SHA는 다르지만 실제10000 record/order/targets는 동일**하며 원본fullCounterFact SHA와 subset SHA를 억지 동일시하지 않는다. contexts의 동일성/차이는 table의 hash로 판단한다. 단일-layer도 target 위치가 달라 layer-only 원인분리 실험이 아니다.\n\n'
    text+='Alpha projector physical L4..L8→indices0..4→local index를 새L567 및 native 실제 CPU tensor SHA로 검증했다(`new-projector-binding.csv`). Native는5weights+5M, BLUE(L4+L8)는2weights+2M, singleton은1weight+1M. MEMIT cache는 empty sentinel/static covariance이며 Alpha history가 아니다. 기존 L4 observer metadata 오표기는 이전v3 evidence대로 보존하고 physical P0 binding과 구분한다. 14×99 W-links, 각Alpha99 history-links, coldreset1/chain과 terminalW0restore assertion을 확인/재사용했다. 12CP/chain 실제 selected tensors와 contexts/RNG를 CPU검사했지만 모델전체를 매번저장한 checkpoint 또는 GPU복원성공이라고 하지 않는다.\n\n'
    text+='## 4. 핵심 비교 I — 각 단일 layer ↔ BLUE ↔ native\n\n![single layer comparison](figures/single-layer-vs-blue-native.png)\n\n'
    for method in ['MEMIT','AlphaEdit']:
        text+='### '+method+'\n\n';blue=next(a for a in ordered(method) if variant(a)=='BLUE');native=next(a for a in ordered(method) if variant(a)=='Native');rows=[]
        for a in ordered(method):
            row=dict(arm=label(a))
            for tag in MULT:
                row[tag+'_%']=100*float(fs[a][tag+'_rate']);row[tag+'_ΔBLUE_pp']=100*(float(fs[a][tag+'_rate'])-float(fs[blue][tag+'_rate']));row[tag+'_Δnative_pp']=100*(float(fs[a][tag+'_rate'])-float(fs[native][tag+'_rate']))
            rows.append(row)
        csvwrite(OUT/(method+'-final-blue-native-deltas.csv'),rows)
        text+=table(rows,list(rows[0]))+'\n같은10000/20000/100000 prompt 분모, Δ는 행arm−reference. 이 표는 각자 다른W/target에서 나온 end-to-end 차이다.\n\n'
        l4=next(a for a in ordered(method) if variant(a)=='L4');l8=next(a for a in ordered(method) if variant(a)=='L8')
        text+=f"L4→L8 final 차이: RS {100*(float(fs[l8]['RS_rate'])-float(fs[l4]['RS_rate'])):+.3f}pp, PS {100*(float(fs[l8]['PS_rate'])-float(fs[l4]['PS_rate'])):+.3f}pp, NS {100*(float(fs[l8]['NS_rate'])-float(fs[l4]['NS_rate'])):+.3f}pp. 중간L5/L6/L7을 포함했으므로 양끝만 보고 단조 변화를 가정하지 않는다. 각 지표 최고 layer가 같지 않으면 trade-off로 남긴다.\n\n"
    text+='### 원본 AlphaEdit ↔ 원본 MEMIT 및 같은 variant의 두 family\n\n'
    rows=[]
    for v in ['Native','BLUE','L4','L5','L6','L7','L8']:
        a=next(a for a in ordered('MEMIT') if variant(a)==v);b=next(a for a in ordered('AlphaEdit') if variant(a)==v);row=dict(variant=v)
        for tag in MULT:row['MEMIT_'+tag]=pct(fs[a][tag+'_numerator'],fs[a][tag+'_denominator']);row['Alpha_'+tag]=pct(fs[b][tag+'_numerator'],fs[b][tag+'_denominator']);row[tag+'_Alpha_minus_MEMIT_pp']=100*(float(fs[b][tag+'_rate'])-float(fs[a][tag+'_rate']))
        rows.append(row)
    csvwrite(OUT/'cross-family-final.csv',rows);text+=table(rows,list(rows[0]))+'\nHistory-projected AlphaEdit과 covariance-weighted MEMIT의 equation·regularization이 다르다. 같은 variant 또는 같은sample만으로 그 차이를 history 하나의 효과로 해석하지 않는다.\n\n'
    text+='## 5. 전체 checkpoint cumulative seen-prefix · current · online\n\n![cumulative](figures/cumulative-rspsns.png)\n\n![current vs seen](figures/current-vs-cumulative.png)\n\n12개 기록checkpoint에서 Wk에 first100k requests 전체를 평가했다. full PS/NS는 그12시점에서만 존재하며 임의 중간값을 만들지 않았다. Native는 `seen-rewrite.json`이100batch 모두 있어 별도 `native-all-batches-seen-rewrite.csv`와 cohort표에 반영했다. BLUE full noncheckpoint 평가가 있는 것처럼 표시하지 않는다. 같은current100은 checkpoint full의 마지막100으로 재사용되므로 중복분모로 합산하지 않는다.\n\n'
    for method in ['MEMIT','AlphaEdit']:
        text+='### '+method+' — 7arm ×12 실제 checkpoint\n\n'
        for a in ordered(method):
            text+='#### '+label(a)+'\n\n';rows=[]
            for b in SCHEDULE:
                m=metrics(a,b);curr=metrics(a,b,'current-metrics');x=dict(edits=b*100)
                for tag in MULT:x[tag]=pct(m[tag]['numerator'],m[tag]['denominator']);x['current_'+tag]=pct(curr[tag]['numerator'],curr[tag]['denominator'])
                x['RS_new_NLL_mean']=f(m['RS']['new_nll_prompt_mean']);x['PS_new_NLL_p90']=f(m['PS']['new_nll_prompt_p90']);x['NS_margin_mean']=f(m['NS']['margin_prompt_mean']);rows.append(x)
            text+=table(rows,list(rows[0]))+'\n'
            for tag in MULT:
                rr=[metrics(a,b)[tag] for b in SCHEDULE];rate=[float(x['rate']) for x in rr];d=[100*(y-x) for x,y in zip(rate,rate[1:])];i=int(np.argmin(d));j=int(np.argmax(d));start=metrics(a,10)[tag];last=metrics(a,100)[tag]
                text+=f"{tag}: 1k→10k {100*float(start['rate']):.3f}%→{100*float(last['rate']):.3f}% ({100*(float(last['rate'])-float(start['rate'])):+.3f}pp). 기록점 간 최소 Δ는 {SCHEDULE[i]*100}→{SCHEDULE[i+1]*100}: {d[i]:+.3f}pp, 최대 Δ는 {SCHEDULE[j]*100}→{SCHEDULE[j+1]*100}: {d[j]:+.3f}pp. "
            text+='이 변화에는 seen-prefix 구성이 커지는 효과가 포함된다. 같은 옛cohort의 손실/회복은 아래 전이와 heatmap에서 별도 확인한다.\n\n'
            for tag in MULT:
                start=metrics(a,10)[tag];end=metrics(a,100)[tag]
                dm=float(end['new_nll_prompt_median'])-float(start['new_nll_prompt_median']);dt=float(end['new_nll_prompt_p90'])-float(start['new_nll_prompt_p90'])
                text+=f"{tag} target-new NLL 1k→10k median {float(start['new_nll_prompt_median']):.4f}→{float(end['new_nll_prompt_median']):.4f}, p90 {float(start['new_nll_prompt_p90']):.4f}→{float(end['new_nll_prompt_p90']):.4f} nats/token. "
                if dm*dt<0:text+='중앙부와tail 변화 방향이 반대이므로 하나의악화/개선으로축약하지않는다. '
            text+='NS의new NLL상승은 competing새target약화방향이므로 RS/PS의new NLL악화와같이읽지않는다.\n\n'
    text+='## 6. At-write 실패와 후속 forgetting/recovery · age\n\n![cohort](figures/cohort-retention-heatmap.png)\n\n'
    for method in ['MEMIT','AlphaEdit']:
        text+='### '+method+'\n\n';rows=[]
        for a in ordered(method):
            for tag in MULT:
                q=sel('prompt-transitions',a,batch=100,metric=tag,comparison='AT_WRITE_TO_CHECKPOINT')[0]
                rows.append(dict(arm=label(a),metric=tag,online=q['before_num'],final=q['after_num'],den=q['denominator'],initial_failure=int(q['denominator'])-int(q['before_num']),retained=q['retained'],lost=q['lost'],recovery=q['gained'],both_failed=q['both_failed'],conditional_loss=pct(q['lost'],q['conditional_loss_den'])))
        text+=table(rows,list(rows[0]))+'\nNS at-write 전이는 각 문항이 속한 batch endpoint를 시작점으로 쓴 것으로 W0 NS preservation과 다르다. initial failure 중 회복되지 않은 것은 both_failed에 남는다.\n\n'
        rows=[]
        for a in ordered(method):
            for stratum in ['first100','first500','first1000','early','middle','recent','last1000']:
                rr=sel('final-fixed-cohorts',a,stratum=stratum);x=dict(arm=label(a),cohort=stratum)
                for r in rr:x[r['metric']]=pct(r['numerator'],r['denominator'])
                rows.append(x)
        text+=table(rows,list(rows[0]))+'\nfirst100/500/1000은 중첩 참조이며 합산하지 않는다. early first20%, middle60%, recent last20%는 W100의 고정10000내 분할이다. 체크포인트별 relative bins는 `age-strata-metrics.csv`에 완전수록했다.\n\n'
    text+='동일 subject/relation에 이후 다른 target을 쓰는 annotation도 원래sample에서 유지했다. `overwrite-strata.csv`는 의도된 overwrite 가능성/같은target/무후속을 구분하되 실제 실패원인으로 확정하거나 요청을 제외하지 않는다. `cohort-retention.csv`는 각기록state×고정cohort×metric 전체와 lost/recovery 분모를 제공한다.\n\n'
    text+='## 7. Locality 총점과 같은 문항 loss/recovery · NLL 분해\n\nW0→기존6 final 전이는 SH2 봉인v3의18rows를 재사용한다(`w0-final-paired-transitions.csv`). 신규8 W0 pair는 **NOT_AVAILABLE_LOCAL_W0_PROMPT_ROWS**: 원격W0 raw 탐색/추가전송을 하지 않아 총점만으로 전이를 역산하지 않았다. 새8의 W0대비 점수 차이는 계산 가능하지만 문항전이와 구분한다.\n\n이번14 final 사이에는 case_id/prompt_index/prompt+target identity를 직접대조했으므로 family별21쌍×3metric=126 paired rows를 계산했다. Native와 BLUE를 포함한 모든쌍은 `paired-final-transitions.csv`; 아래는 NS다.\n\n'
    for method in ['MEMIT','AlphaEdit']:
        rows=[]
        for r in t['paired-final-transitions']:
            if r['method']==method and r['metric']=='NS':rows.append(dict(before=label(r['arm_before']),after=label(r['arm_after']),n=r['denominator'],retained=r['retained'],loss=r['lost'],recovery=r['gained'],both_failed=r['both_failed'],delta_pp=100*float(r['paired_rate_delta'])))
        text+='### '+method+' NS 동일prompt 전이\n\n'+table(rows,list(rows[0]))+'\n'
    text+='`paired-locality-nll-changes.csv`에서 전체/loss/recovery 각각에 true NLL 변화(기존 사실 약화와 연관) 및 competing-new NLL 변화(새target 강화와 연관)를 별도로 저장한다. 방향의 관측이지 직접 causal mechanism 증명이 아니다. 차이는 after−before이고 new NLL 하락은 competing new의 확률상승이다.\n\n'
    rows=[]
    for r in t['paired-locality-nll-changes']:
        if r['transition']=='all' and ('BASE_' in r['arm_after'] or 'ORIGINAL' in r['arm_before']):
            rows.append(dict(before=label(r['arm_before']),after=label(r['arm_after']),n=r['denominator'],new_delta_mean=f(r.get('new_nll_delta_mean')),true_delta_mean=f(r.get('true_nll_delta_mean')),new_increased=r.get('new_nll_increased'),true_increased=r.get('true_nll_increased')))
    text+=table(rows,list(rows[0]))+'\n'
    text+='## 8. NLL 분포·tail·strict/token secondary\n\n![NLL](figures/nll-median-tail.png)\n\n'
    for method in ['MEMIT','AlphaEdit']:
        text+='### '+method+' final prompt-level 분포\n\n';rows=[]
        for a in ordered(method):
            for r in sel('final-distributions',a):
                x=dict(arm=label(a),metric=r['metric'],n=r['denominator'])
                for side in ['new_nll','true_nll','margin']:x[side+'_mean_med_p90_p99_max']=' / '.join(f(r.get(side+'_prompt_'+st)) for st in ['mean','median','p90','p99','max'])
                x['new_TF_strict']=pct(r['new_strict_num'],r['new_strict_den']);x['new_token_accuracy']=pct(r['new_token_correct'],r['new_token_den']);rows.append(x)
        text+=table(rows,list(rows[0]))+'\n모든 target-true strict/token 및 request-cluster NLL/margin 분포는 `final-distributions.csv`에 있음. Median과p99가 다른 방향이면 평균하나로 tail을 감추지 않는다. 구v3의 중간CP p99는 당시 aggregate 미기록으로 빈칸/NOT_RECORDED, 신규8 CP와14final p99는 원시pair에서 계산했다. p99가 p90대신 기존표에 몰래 들어가지 않는다.\n\n'
    text+='## 9. 실제 layer별 업데이트 · state · 비용\n\n![update](figures/layer-wise-update-magnitude.png)\n\n'
    for method in ['MEMIT','AlphaEdit']:
        rows=[dict(arm=label(r['arm']),layer=r['layer'],mean=f(r['update_norm_mean']),median=f(r['update_norm_median']),p90=f(r['update_norm_p90']),max=f(r['update_norm_max']),mean_share=f(r['magnitude_share_mean']),sum_batch_net_lengths=f(r['sum_batch_net_lengths'])) for r in t['layer-summary'] if family(r['arm'])==method]
        text+='### '+method+' — 각 selected layer 100 batch 관측\n\n'+table(rows,list(rows[0]))+'\n'
    text+='BLUE두layer의 실제write는 각weight에 기록되며 singleton은지정layer만이다. 비selected는pointer/version guard까지이다. 합산 batch-net lengths는 실제 batch endpoint 간 길이 합이고 intra-solve 경로작업량 또는 W100−W0 net과 같지 않다. checkpoint간 net차는 `checkpoint-tensors.csv`에 있다. Native covariance/history action 또는 공통 fixed-z post-write realization은 **NOT_RECORDED**이므로 Frobenius에 native 명칭을 붙이지 않는다. BLUE layer별 z가 달라 JVP fixed-z potential과 직접 pooling하지 않는다.\n\n'
    text+='업데이트 magnitude/path와 current 성능100점 또는 cumulative12점의 Spearman은 `mechanism-performance-associations.csv`에 n·부호·계수로 보존한다. 시계열상관과 누적구성변화가 있으므로 기전인과 또는 독립반복 유의성을 주장하지 않는다. 표본수가 같은10000이라고 checkpoint별 관측을 독립request로 과대계수하지 않는다.\n\n![cost](figures/cost-breakdown.png)\n\n'
    for method in ['MEMIT','AlphaEdit']:
        rows=[]
        for r in t['compute-summary']:
            if family(r['arm'])!=method:continue
            x=dict(arm=label(r['arm']),GPUh=f(r['gpu_hours']),wall_h=f(float(r['wall_seconds'])/3600))
            for key in ['edit_seconds','target_seconds','key_seconds','solve_seconds','evaluation_seconds']:x[key+'_h']=f(float(r[key])/3600)
            x['checkpoint_GiB']=f(float(r['checkpoint_bytes'])/2**30);x['raw_GiB']=f(float(r['raw_bytes'])/2**30);x['peak_GiB']=f(float(r['peak_gpu_bytes'])/2**30);rows.append(x)
        text+='### '+method+'\n\n'+table(rows,list(rows[0]))+'\n'
    text+='Native는 every-batch seen-rewrite를 추가 수행하므로 evaluator 부하가BLUE와 완전히 같지 않다. writer/edit시간과evaluation을 분리해 읽는다. edit에는target/key/solve가포함되어 합산시중복금지. snapshot/history별 순수wall/forward/backward/JVP카운트는없는경우NOT_RECORDED. Slurm GPUh는할당시간이며직접utilization측정이아니다. W0는Server2 A6000, 편집은Server4 PRO6000이므로crosshardware속도효과주장0.\n\n'
    text+='## 10. 이전 실험 인덱스 · technical exclusions\n\n'
    history=[('BLUE 1k + JVP/JVP-L8/O','experiment-reports/servers/server4/blue-alphaedit-fivearm-sequential1000-review-2026-09-07-v1/factual-report-ko.md','38940/38997/38988; same1kprefix, final1k와final10k합산0'),('JVP L8 takeover','experiment-reports/servers/server4/alpha-jv-l8-only-sequential1000-review-2026-09-07-v1/factual-report-ko.md','38433_4/5; Qwen은Llama14core에포함0'),('JVP/O primary publication','experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1','sealed publication only; remote raw/再監査0'),('ORBODE cumulative','experiment-reports/servers/server4/orbode-cumulative-b100x10-exhaustive-2026-09-07-v1/','37649; raw사용자삭제, STOPPED_USER_REVIEW correctness 감사 재개0'),('BLUE6+W0 v3',str(OLD.relative_to(REPO))+'/factual-report-ko.md','본보고서의재사용source; v3원문불변')]
    inventory=[]
    for name,p,note in history:
        q=REPO/p;inventory.append(dict(experiment=name,path=p,sha256=sha(q) if q.is_file() else 'DIRECTORY_LINK_ONLY',status='SEALED_PUBLICATION_REUSED_NO_RAW_REAUDIT',boundary=note))
    for a in ARMS:inventory.append(dict(experiment=label(a),path=str(root(a)),job=fs[a]['job'],status='TERMINAL_VALID_EVIDENCE_VERIFIED',boundary='core fixed10k; raw local only'))
    csvwrite(OUT/'experiment-inventory.csv',inventory)
    text+=table(inventory,['experiment','job','path','status','boundary'])+'\n1k matched-prefix 과거결과는 `prior-1k-comparison.csv`의 기존봉인비교를참조하며새원시재감사0. native10k 완성으로v3 NOT_AVAILABLE을이번버전에서만채웠다. 다른오래된Server4결과와SH1/SH2진단은원publication링크범위이며범용재감사하지않는다.\n\n'
    text+=table(t['technical-exclusions'],['job','status','canonical_denominator','committed_prefix_requests','allocated_seconds','evidence'])+'\n39183_0의600prefix/B7미commit은새canonical분모0, 기존receipt2766GPU초를재사용했다. smoke39172와old pending취소를PASS로승격하지않았다. 42656preedit이관취소와SH2실측42673을분리한다.\n\n'
    text+='## 11. 근거 한계 · 미측정 · FACT/INFERENCE/DECISION\n\n'
    limits=[('FACT','14 terminal/file/metric, new96CP CPU 검산+old72CP 재사용, 140000 final observations / unique10000, source equations/native mappings 확인'),('INFERENCE','같은항목의atwrite→final손실은이순서에서유지손실로기술. 다만적용layer/target/L2/history/환경의개별원인은분리되지않음'),('NOT_MEASURED','full model nonselected byte parity; GPU checkpoint continuation; raw W0→new8 prompt transitions; common postwrite activation realization/native action; purehistory/snapshot시간'),('MISSING_POLICY','새 평가/보간/nearby CP 치환/성공기반제외0. finite poor 결과도전체표유지'),('DECISION','analysis/report ownscope mainpush만,scientific_promotion=false,후속run없음, 완료후STOP')]
    text+=table([dict(type=a,boundary=b) for a,b in limits],['type','boundary'])+'\n단일“전부정상”은사용하지않는다. runtime assertions와독립file/CPU검증,scientific explanation,미실행검증을구분한다. 낮은 성능은무결성실패가아니다.\n\n'
    text+='## 12. 완전 산출물·재현\n\n'
    for p in sorted(OUT.glob('*.csv')):
        text+=f"- `{p.name}` — {len(csvread(p))} rows, SHA256 `{sha(p)}`.\n"
    text+='\n코드: `project/run_scripts/server4_experiments_review/`. 실행순서 first→audit 및 aggregate→supplement→plots→report→package. Python `/data/janghj/EasyEdit/.venv/bin/python`, CPU threads1/2. 신규CP/hash만audit; 기존6은sealed재사용이다. 최초scheduler 관측을 재실행하지 않으며 `initial.json`에 단발 결과가 남아 있다. raw file 경로/size/SHA는new-raw-member-inventory와oldpublication원래inventory로결속한다. 원raw/weights/cache/log/prompts는Git0, broadcast는analysis-only/local-only보존사유로생략(새대형전송0).\n\n'
    pr=read(OUT/'figures/plot-receipt.json')
    for r in pr['figures']:text+=f"- PNG `{r['path']}` SHA `{r['sha256']}`: {r['caption']}\n"
    text+='\n재현명령: `OPENBLAS_NUM_THREADS=1 /data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.server4_experiments_review.plots --out <새빈경로>`. source/input/outputSHA는 `figures/plot-receipt.json`, 독립재실행 byte비교는 `plot-reproduction.json`. 모든PNG는코드생성, imagegen/visualization/manualedit0. 보고서/CSV의missing은미기록으로표시하고plot에서는mask; 보간값을표에생성하지않는다. package검산은analysis-manifest.json/rooted-receipt.json에있다.\n'
    text=text.replace('//W_l after batch − W_l entry//_F','‖W_l after batch − W_l entry‖_F').replace('first100k requests','첫 100×k개 requests')
    import sys
    assert not (OUT/'analysis-manifest.json').exists(), 'SEALED_REPORT_NO_REGEN'
    with (OUT/'diagnostic-report-ko.md').open('w' if '--refresh-unsealed' in sys.argv else 'x') as out:out.write(text)
    print('REPORT',sha(OUT/'diagnostic-report-ko.md'),flush=True)

if __name__=='__main__':main()
