"""Korean factual report built from complete, independently reduced tables."""
from collections import defaultdict
import json
import numpy as np
from .common import *

def main(out):
    names=['final-summary','cumulative-metrics','current-metrics','age-strata-metrics','cohort-retention','prompt-transitions','paired-final-transitions','layer-updates','batch-cost','chain-integrity','source-config-compatibility','checkpoint-tensors','prior-1k-comparison','technical-exclusions','overwrite-strata','mechanism-performance-associations']
    data={n:csvread(out/(n+'.csv')) for n in names}
    def sel(name,arm,tag=None):return [r for r in data[name] if r['arm']==arm and (tag is None or r.get('metric')==tag)]
    def rate(r):return f"{r['numerator']}/{r['denominator']} ({100*float(r['rate']):.3f}%)"
    def stat4(r,prefix):return '/'.join(f'{float(r[prefix+k]):.5g}' for k in ['mean','median','p90','max'])
    audit=read(out/'full-audit-receipt.json');agg=read(out/'aggregation-receipt.json')
    txt=['# Llama3 BLUE 원본·L4-only·L8-only lifelong 상세 리뷰 — B100×100\n',
         '본 보고서는 지정된 MEMIT_BLUE/AlphaEdit_BLUE 6개 chain의 **실제 최종 W100에서 전체 10,000 요청을 재평가한 결과**를 대표값으로 사용한다. Current-B100 합이나 online-at-write 누계가 아니다. 신규 model/GPU/evaluator/replay/Slurm 실행0, L5/L6/L7 접근0, stopped ORBODE 감사 재개0. scientific_promotion=false.\n',
         '## 1. Executive factual findings\n']
    headline=[]
    for r in data['final-summary']:
        z={k:r[k] for k in ['arm','job','status','batches','requests']}
        for tag in MULT:z[tag]=f"{r[tag+'_numerator']}/{r[tag+'_denominator']} ({100*float(r[tag+'_rate']):.3f}%)"
        z['rewrite_TF_exact']=f"{r['RS_new_strict_num']}/{r['RS_new_strict_den']}";z['rephrase_TF_exact']=f"{r['PS_new_strict_num']}/{r['PS_new_strict_den']}"
        headline.append(z)
    txt+=[table(headline,list(headline[0])),
          '집계 단위는 RS rewrite prompt, PS rephrase prompt, NS neighborhood prompt다. 분모는 각 arm 10,000 / 20,000 / 100,000으로 raw inventory에서 검산했다. TF exact는 모든 target token의 teacher-forced top-1 일치 여부로 primary NLL preference와 다르다.\n',
          'MEMIT의 최종 RS는 original 71.83%, L4-only 77.22%, L8-only 81.77%다. PS/NS는 같은 순서로 67.65/53.287%, 74.83/57.848%, 72.30/48.425%다. AlphaEdit은 original 98.88/95.775/63.726%, L4-only 99.39/95.68/65.348%, L8-only 93.96/77.78/54.703%다. 어느 한 수치의 개선을 전체 지표의 우위로 일반화하지 않는다.\n',
          'AlphaEdit의 기존1k BLUE/full/L4/L8 결과와 이번 lifelong t=1000은 세 metric의 **전체 저장 NLL-pair row가 각각 exact 일치**했다. 이는 확인한 첫1k 결과에 한정되며 전체 GPU 연산 또는 다른 환경의 보편적 재현성 주장이 아니다.\n',
          f"검증 coverage: raw {audit['raw_members']:,} members / {audit['raw_bytes']:,} bytes, 600 committed batches, 72 selected-W/method-state checkpoint, 594 batch W links. Checkpoint all-seen request-state 관측 {agg['cumulative_request_state_rows']:,}, unique requests10,000, final arm-request 관측60,000. Final은 cumulative10000 member 재사용이며 별도분모로 합산하지 않는다.\n",
          '![Final performance](figures/final-full10000.png)\n',
          '## 2. Metric glossary / 읽는 방법\n',
          '|지표|정의·집계·방향|\n|---|---|\n'
          '|RS|rewrite prompt에서 target-new의 평균 target-token NLL < target-true NLL. 1 prompt/request. 높을수록 해당 요청의 새 target 선호가 많다.|\n'
          '|PS|각 rephrase prompt의 new NLL < true NLL. 2 prompts/request의 prompt별 판정; 두 prompt NLL을 먼저 평균한 판정 아님.|\n'
          '|NS|각 neighborhood prompt의 true NLL < new NLL. 10 prompts/request. target-true top-1 보존 정확도와 다르다.|\n'
          '|Tie|strict <만 성공. 두 NLL이 같으면 실패. tie 횟수는 metric CSV에 보존.|\n'
          '|NLL|target token의 −log p를 token 길이로 평균, 단위 nats/token. 낮으면 해당 target의 예측확률이 높다. new/true를 분리하며 true NLL 저하가 편집 성공을 의미하지 않는다.|\n'
          '|Margin|모든 category에서 true NLL−new NLL. 양수는 new 선호(RS/PS 성공), 음수는 true 선호(NS 성공). NS도 임의 부호 반전하지 않았다.|\n'
          '|TF exact / strict|teacher forcing하에서 모든 target token이 top-1로 맞으면 해당 prompt true. primary success가 아님. `new_strict_num/den`, `true_strict_num/den`.|\n'
          '|Token accuracy|teacher-forced correct token count / actual target-token count. prompt exact와 집계단위가 다름.|\n'
          '|Request pair-strict|한 request의 모든 해당 category prompt가 canonical preference 성공. PS는2개, NS는10개. Primary prompt PS/NS와 구분.|\n'
          '|Prompt vs request-cluster|prompt-level은 각 prompt NLL; request-cluster는 request 내 prompt NLL부터 평균하고 request별 분포를 계산. 모든 분포의 n, mean, median, q25/q75(IQR), p90, max는 CSV에 있다.|\n'
          '|Current B100|편집 직후 W_k에서 그 batch100개만 평가. 누적 seen-prefix 성능과 다름.|\n'
          '|All-seen cumulative|고정된 W_k에서 first100*k 요청 전체를 평가. k={1,5,10,20,...,100}만 저장. 연결선은 미기록시점 추정치가 아님.|\n'
          '|Online-at-write|각 request가 처음 쓰인 서로 다른 W의 결과를 합친 것. 최종 W 성능 아님.|\n'
          '|Loss/recovery|동일 prompt identity의 before 성공→after 실패 / before 실패→after 성공. conditional loss의 분모는 before 성공 수. 평가 사이 중간 경로는 모름.|\n'
          '|Layer-wise Update Magnitude|실제 batch-entry 대비 endpoint의 ||ΔW_l||_F. 단위 weight norm. Share=해당 norm/선택 layer norm 합. squared norm과 다름.|\n'
          '|Path/net|Σbatch ||ΔW_l,b||는 batch-net 길이 합이지 native 내부 trajectory 길이나 ||W100−W0||와 같지 않다. checkpoint interval net은 실제 저장된 두 checkpoint 차이.|\n'
          '|Native z/residual|BLUE는 선택 layer마다 current W에서 z를 재최적화. log의 z error는 해당 layer write 전 ||z−h|| 평균(반올림 scalar). post-write realization ratio/rho/tau나 JVP fixed-L8 potential로 부를 수 없다.|\n'
          '|History/state|AlphaEdit은 선택 layer별 dense cache M 누적. MEMIT은 static covariance computation cache만 있고 임의 lifelong history는 없다.|\n',
          '현재6chain의 W0 전체10k 성능은 NOT_RECORDED다. 이전1k W0 점수를 이번10k의 W0 reference로 대입하지 않는다. Fixed sentinel 별도패널도 기록되지 않았다.\n',
          '## 3. Provenance / 설정 / 실행 의미\n']
    prov=[]
    for r in data['source-config-compatibility']:
        hp=json.loads(r['hparams']);z={k:r[k] for k in ['arm','job','source_archive_sha256','config_sha256','model_revision']}
        z.update(layers=str(hp['layers']),z_steps=hp['v_num_grad_steps'],z_lr=hp['v_lr'],z_decay=hp['v_weight_decay'],loss_layer=hp['v_loss_layer'],clamp=hp['clamp_norm_factor'],native_regularizer=hp.get('L2',hp.get('mom2_update_weight')))
        prov.append(z)
    txt+=[table(prov,list(prov[0])),
          'BLUE source HEAD311b076a92e4ed0f14f5c8b4909732da781bc5f7/tree f3c933c31cba2fe979c5c34546a99a72e6beb763. ODE helper1075540b45c29269e690ac63aae44758d8d63174/tree172b6b9b5c0de4e3aa9e94a05920edaa84a2b323. 실제 실행은 별도 local lifelong modules+archive이며 tracked BLUE 또는 예전 L4/L8 local hook을 변경하지 않았다. 동일 method내 layers 외 config 차이0을 원본 JSON과 대조했다. Source/member/path 전체는 source-member-inventory.csv, source-config-compatibility.csv, source-findings.csv에 있다.\n',
          '모든 run은 Llama3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, NVIDIA RTX PRO6000 Blackwell Server Edition, torch2.9.1+cu128, transformers4.44.2, eager attention. Model parameters FP32 inventory8,030,261,248 elements/291 tensors, autocast=False, TF32 matmul=False, cudnn=True. MEMIT native solve FP64와 AlphaEdit native FP32 정책을 구분한다. 저장 FP32 및 runtime assertion을 확인했으며 GPU 내부 모든 연산을 새로 추적하지 않았다. Writer/evaluator의 tokenizer.padding_side 속성은 **right**, pad=eos128009지만 실제 evaluator는 tensor를 **수동 left padding**하고 attention_mask/target offset을 구성한다. position_ids는 명시 인자로 전달하지 않아 native model 기본 경로를 쓴다. Writer add_bos_token=False, evaluator 해당 속성 NOT_EXPOSED. 이번 감사에서 새 GPU padding/parity 검사는 실행하지 않았다. Dataset의 prompt/target string→실제 scalar row identity를 672개 current/full 평가 파일에 대해 독립 대조했다.\n',
          '|method|계산·state 의미|\n|---|---|\n'
          '|MEMIT_BLUE|각 layer에서 current-W native z, K, residual R를 계산. B=(λ C+KKᵀ)^−1K, ΔW=R Bᵀ(물리 shape [4096,14336]). BLUE는 full residual(divisor1). execute에서 일시 적용→entry 복원 후 apply가 ΔW를 실제 materialize; runner는 반환된 actual endpoint를 평가·commit.|\n'
          '|AlphaEdit_BLUE|각 layer마다 native z와 R 재계산. physical P로 projected closed-form solve [P(KKᵀ+M)+L2 I]X=PKRᵀ, transpose orientation 적용. terminal selected-layer key로 M append1 pass/batch. Pstack L4..L8에서 original[0,4], L4[0], L8[4] 추출을 CPU tensor hash로 확인.|\n'
          '|Single layer|layers만[4] 또는[8]; z target/optimization layer도 해당 layer로 바뀐다. 반복 JV/Euler/controller/substep0. 단순히 같은 target에서 support만 바꾼 비교가 아님.|\n',
          '표본은 unique10,000, sample root `'+SAMPLE_ROOT+'`. 첫1,000 root `'+PREFIX_ROOT+'`를 보존했고 이후9,000은 metadata SHA-order selection이다. 125개 repeated subject-relation group은 교체·삭제하지 않았다. 상충 target의 후속 요청 여부는 overwrite-strata.csv에 따로 표시하되 관측된 망각의 원인으로 확정하지 않는다.\n',
          '### 3.1 상태/저장 검산\n',table(data['chain-integrity'],['arm','batches','requests','checkpoints','W_links','history_links','compute_z','solve_calls','history_append_passes','context_sha256']),
          '선택 W와 Alpha history의 entry→commit→next-entry99 links/cell. Original compute_z20,000/solve200, single10,000/100. Alpha append100, MEMIT0. Chain cold reset1, runtime terminal W0/cache byte·pointer restore assertion true. **nonselected weight는 pointer/version 보존을 기록했으나 전체 byte 해시는 기록하지 않았다.** 최종 version restore는 주장하지 않으며 evaluator 전후 version exact와 혼동하지 않는다. Hash/shape PASS를 방법의 과학적 정상성 또는 효과 증명으로 사용하지 않는다.\n',
          '12 checkpoints는 original에서 L4+L8 두 weight, single에서 한 weight를 실제 저장했다. Alpha는 모든 selected M tensor, MEMIT은 empty sentinel+static NPZ identity. Context/RNG(Python/NumPy/torch/CUDA)/base/config/sample binding 존재, CPU weights_only reload/hash/finite 확인. Full base model 중복저장0. GPU continuation replay0이며 비선택 base tensor의 immutable 참조가 필요하다. 구체 key/shape/SHA는 checkpoint-tensors.csv의96 layer-checkpoint행에 있다.\n',
          '## 4. Checkpoint actual-W all-seen 누적 성능\n',
          '![Cumulative](figures/cumulative-seen-prefix.png)\n']
    for arm in ARMS:
        txt.append('### 4.'+str(ARMS.index(arm)+1)+' '+arm+'\n')
        rr=sel('cumulative-metrics',arm);rows=[]
        for b in SCHEDULE:
            q={'batch':b,'seen_requests':b*100}
            for tag in MULT:q[tag]=rate(next(r for r in rr if int(r['batch'])==b and r['metric']==tag))
            rows.append(q)
        txt.append(table(rows,list(rows[0])))
        for tag in MULT:
            a=[r for r in rr if r['metric']==tag];d=[(float(y['rate'])-float(x['rate']),x,y) for x,y in zip(a,a[1:])];worst=min(d,key=lambda z:z[0]);best=max(d,key=lambda z:z[0]);start=next(r for r in a if r['batch']=='10');end=a[-1]
            txt.append(f"{tag}: 1k→10k {100*float(start['rate']):.3f}%→{100*float(end['rate']):.3f}% ({100*(float(end['rate'])-float(start['rate'])):+.3f}pp). 저장된 연속 checkpoint 중 최대 감소는 B{worst[1]['batch']}→B{worst[2]['batch']} {100*worst[0]:+.3f}pp, 최대 변화(증가가 없으면 가장 작은 감소)는 B{best[1]['batch']}→B{best[2]['batch']} {100*best[0]:+.3f}pp. Seen-prefix 분모도 함께 증가하므로 동일 고정 cohort의 인과효과가 아니다.\n")
    txt+=['## 5. Current efficacy와 과거 요청의 loss/recovery 분리\n','![Current versus cumulative](figures/current-vs-cumulative.png)\n','![Cohort](figures/cohort-retention-heatmap.png)\n']
    for arm in ARMS:
        tr=[r for r in sel('prompt-transitions',arm) if r['batch']=='100' and r['comparison']=='AT_WRITE_TO_CHECKPOINT']
        txt+=['### '+arm+'\n',table(tr,['metric','denominator','before_num','retained','lost','gained','both_failed','after_num','conditional_loss_den','conditional_loss_rate'])]
        rs=next(r for r in tr if r['metric']=='RS');cc=sel('current-metrics',arm,'RS');mins=min(cc,key=lambda r:float(r['rate']));fail=next((r for r in cc if int(r['numerator'])<100),None)
        txt.append(f"RS at-write {rs['before_num']}/10000에서 final {rs['after_num']}/10000. 처음 성공했다가 최종 실패한 요청 {rs['lost']}, 처음 실패했으나 최종 회복한 요청 {rs['gained']}. Current RS 최저 batch B{mins['batch']}={rate(mins)}; 첫 current failure 관측은 "+(f"B{fail['batch']}={rate(fail)}" if fail else '없음')+'. 이것은 사전 scientific failure threshold가 아닌 기술적 표산술이다.\n')
    txt+=['전체100 current batch×6arm×3metric=1,800행은 current-metrics.csv; checkpoint×write-cohort×metric10,008행은 cohort-retention.csv. 저장되지 않은 checkpoint 사이의 개별 loss→recovery 경로는 알 수 없으며 100×100 dense retention matrix를 채우지 않았다. 이전 checkpoint 동일 old-prefix 비교는 prompt-transitions.csv에 별도로 있다.\n',
          '### 5.1 Final 같은 prompt의 arm간 변화\n',table(data['paired-final-transitions'],['arm_before','arm_after','metric','denominator','before_num','after_num','retained','lost','gained','paired_rate_delta']),
          'NS 총점이 같거나 비슷해도 retained/lost/gained 항목은 다를 수 있다. 같은 case/prompt지만 각 arm은 서로 다른 W/history/target 경로다. 100,000 neighborhood prompt를 독립 실험 반복으로 취급한 p-value는 계산하지 않았다.\n',
          '### 5.2 후속 동일 subject/relation 요청 존재 여부\n',table(data['overwrite-strata'],['arm','category','denominator','before_num','after_num','lost','gained']),
          '이 분할은 후속 canonical request의 target hash가 같은지/다른지로 정의한다. 의도된 overwrite와 독립 forgetting을 인과적으로 분리했다고 주장하지 않는다. 원래 분모에서 어떤 요청도 제외하지 않았다.\n',
          '## 6. Edit-age별 유지 성능 — 모든 저장 checkpoint\n','![Age RS](figures/age-retention-RS.png)\n','![Age NS](figures/age-retention-NS.png)\n',
          'Early는 당시 seen-prefix 첫20%, middle 다음60%, recent 마지막20%다. 각 checkpoint에서 상대 bin이 달라진다. 동일한 고정 sentinel longitudinal cohort와 다르다. 각 metric 실제 prompt 분모를 아래에 표시한다.\n']
    for arm in ARMS:
        rows=[]
        for b in SCHEDULE:
            for st in ['early','middle','recent']:
                group=[r for r in sel('age-strata-metrics',arm) if int(r['batch'])==b and r['stratum']==st]
                q=dict(batch=b,stratum=st,requests=group[0]['request_denominator'])
                for tag in MULT:q[tag]=rate(next(r for r in group if r['metric']==tag))
                rows.append(q)
        txt+=['### '+arm+'\n',table(rows,list(rows[0]))]
        g=[r for r in sel('age-strata-metrics',arm,'RS') if r['batch']=='100'];a=next(r for r in g if r['stratum']=='early');b=next(r for r in g if r['stratum']=='recent')
        txt.append(f"Final RS early {rate(a)}, recent {rate(b)}, recent−early {100*(float(b['rate'])-float(a['rate'])):.3f}pp. 이는 final state에서 서로 다른 age-cohort 간 산술 차이다.\n")
    txt+=['## 7. NLL·margin 분포와 tail — checkpoint별 상세\n','![New target NLL](figures/cumulative-nll-new.png)\n','![True target NLL](figures/cumulative-nll-true.png)\n',
          '다음 표의 각 분포 셀은 **mean / median / p90 / max**, 단위 nats/token이다. Prompt 집계와 request-cluster(요청내 prompt 평균 후 분포)를 별도 행으로 제공한다. q25/q75 및 모든 age-bin NLL/margin/strict/token 분포648행은 age-strata-metrics.csv, 전체 prompt/request 분포216행은 cumulative-metrics.csv에 완전 수록한다.\n']
    for arm in ARMS:
        txt.append('### '+arm+'\n')
        for tag in MULT:
            rows=[]
            for r in sel('cumulative-metrics',arm,tag):
                for unit in ['prompt','request']:
                    rows.append(dict(batch=r['batch'],unit=unit,n=r['denominator'] if unit=='prompt' else r['request_denominator'],new_NLL=stat4(r,'new_nll_'+unit+'_'),true_NLL=stat4(r,'true_nll_'+unit+'_'),margin=stat4(r,'margin_'+unit+'_'),TF_new=f"{r['new_strict_num']}/{r['new_strict_den']}",token_new=f"{r['new_token_correct']}/{r['new_token_den']}"))
            txt+=['#### '+tag+'\n',table(rows,list(rows[0]))]
            rr=sel('cumulative-metrics',arm,tag);changes=[]
            for a,b in zip(rr,rr[1:]):
                dm=float(b['new_nll_request_median'])-float(a['new_nll_request_median']);dp=float(b['new_nll_request_p90'])-float(a['new_nll_request_p90'])
                if dm*dp<0:changes.append(f"B{a['batch']}→B{b['batch']} medianΔ{dm:+.4g}/p90Δ{dp:+.4g}")
            txt.append('새 target NLL median과 p90 방향이 다른 구간: '+('; '.join(changes) if changes else '저장된 연속 checkpoint에서 없음')+'. Max/outlier identity는 outlier-ledger.csv에 raw prompt 없이 hash로 기록했다.\n')
    txt+=['## 8. Layer별 실제 update·native target·history\n','![Layer updates](figures/layer-wise-update-magnitude.png)\n',
          '막대는 100 sequential batch의 관측 batch-net Frobenius norm 평균이다. Uniform/ideal 배분선은 없으며 native covariance action으로 표기하지 않는다. Original의 L4/L8 current-layer z와 single-layer native z는 target 정의가 달라 raw residual을 공통 realized efficacy로 직접 비교하지 않는다.\n']
    layer_summary=[]
    for arm in ARMS:
        for l in sorted({int(r['layer']) for r in sel('layer-updates',arm)}):
            rr=[r for r in sel('layer-updates',arm) if int(r['layer'])==l]
            z=dict(arm=arm,layer=l,batches=len(rr))
            for field in ['update_norm','magnitude_share','relative_norm','native_prewrite_z_error_printed']:
                for k,v in stats([float(r[field]) for r in rr]).items():z[field+'_'+k]=v
            z['sum_batch_net_lengths']=rr[-1]['sum_batch_net_lengths'];layer_summary.append(z)
    csvwrite(out/'layer-summary.csv',layer_summary)
    txt.append(table(layer_summary,['arm','layer','batches','update_norm_mean','update_norm_median','update_norm_p90','update_norm_max','magnitude_share_mean','sum_batch_net_lengths','native_prewrite_z_error_printed_mean']))
    txt+=['각100 batch별 target norm mean/median/p90/max, prewrite stdout residual scalar, 실제 update/relative/share는 layer-updates.csv 800행. Z tensor600파일의 layer/request 순서·hash·finite를 CPU 검산했다. Post-write activation vector/Y/E/rho/tau/realization ratio는 NOT_RECORDED. stdout residual은 낮은 자릿수로 반올림된 native write 전 scalar일 뿐이며 원인을 판별하는 충분한 관측이 아니다. Key tensor 실물은 없고 hash/shape/count만 기록됐다. Native covariance/history action, 내부 경로 길이 또는 full W0-net action을 Frobenius 합으로 대체하지 않는다.\n',
          '### 8.1 성능과 mechanism의 기술적 연관\n',table([r for r in data['mechanism-performance-associations'] if r['scope']=='CUMULATIVE' and r['metric']=='RS'],['arm','metric','feature','n_checkpoints_or_batches','spearman']),
          'Cumulative n=12 checkpoint/arm, current n=100 batch/arm이며 시간적으로 종속된다. Spearman은 방향/순위의 기술통계다. P-value·causal claim0. Feature는 선택 layer scalar 합이며 shared-z 반사실 비교가 아니다. 전체 RS/PS/NS/current/cumulative 계수는 mechanism-performance-associations.csv.\n',
          '## 9. Compute / 저장 / 측정 누락\n','![Compute](figures/compute-accounting.png)\n']
    compute=[]
    for arm in ARMS:
        rr=sel('batch-cost',arm);fr=next(r for r in data['final-summary'] if r['arm']==arm)
        q=dict(arm=arm,allocated_GPU_hours=float(fr['gpu_hours']),runtime_hours=float(fr['wall_seconds'])/3600,raw_GiB=int(fr['raw_bytes'])/2**30,peak_GPU_GiB=max(float(r['peak_gpu_bytes']) for r in rr)/2**30,checkpoint_GiB=sum(int(r['checkpoint_bytes']) for r in rr)/2**30)
        for key in ['edit_seconds','target_seconds','key_seconds','solve_seconds','evaluation_seconds']:q[key.replace('_seconds','_hours')]=sum(float(r[key]) for r in rr)/3600
        q['runtime_other_hours']=q['runtime_hours']-q['edit_hours']-q['evaluation_hours'];compute.append(q)
    csvwrite(out/'compute-summary.csv',compute)
    txt+=[table(compute,list(compute[0])),
          'Allocated GPUh는 scheduler 경과시간×할당1GPU이며 GPU busy occupancy 샘플은 NOT_RECORDED. Target/key/solve는 native writer 내부 부분합이고 edit에 중복 포함된다. Evaluation은 current 및 checkpoint all-seen을 포함. Runtime other에는 model load/source rehash/checkpoint I/O/restore 등 분리 계측되지 않은 비용이 있다. 별도 snapshot/history/storage stopwatch, forward/backward/JVP 횟수는 NOT_RECORDED. Native 원본 코드에 JVP controller 없음은 source 사실이나 계측 count=0으로 꾸미지 않는다.\n',
          '## 10. 기존1k 결과·기술 제외·availability\n',table(data['prior-1k-comparison'],['arm','metric','prior_1k_num','lifelong_1k_num','denominator','NLL_pair_rows_exact']),
          '원본1k report bytes와 package members를 재해시했으며 이번 t1000과 prompt identity·NLL/strict/token rows를 비교했다. 같은 모델·seed·context·환경 lock도 source compatibility 파일에 있다. 종전 JV/JVP-L8은 이번6chain 분모가 아니며 새로 실행하지 않았다. Qwen도 이번 범위 아님.\n',
          table(data['technical-exclusions'],['job','status','canonical_denominator','committed_prefix_requests','allocated_seconds']),
          '39183_0/child39218는 OLD_SCHEDULE_USER_SUPERSEDED: committed600 requests, B7 uncommitted, allocated2766s(0.768333GPUh). 새39307에 old prefix carry0. 39183_1..5는 미시작 schedule replacement, smoke39172는 사용자 waiver로 SKIPPED_USER_DIRECTED/취소되었으며 PASS가 아니다. 초기 operational receipt를 operational-receipt-inputs.json 및 supplemental-control-receipt.json으로 별도 보존했다. 기존1k의 technical attempts 비용은 본10k canonical denominator와 합치지 않는다.\n',
          '|미측정/제한|처리|\n|---|---|\n|W0 전체10k RS/PS/NS|NOT_RECORDED; 이전1k 점수로 대입0|\n|12checkpoint 밖 all-seen 성능|NOT_RECORDED; current100은 별도보존; 보간0|\n|Fixed sentinel longitudinal|NOT_RECORDED|\n|Nonselected 전체 byte preservation|pointer/version evidence만; full bytes 주장0|\n|GPU continuation restore|CPU checkpoint reload/hash만 확인, replay0|\n|Native action/실현률|관련metric/forward미기록, Frobenius/printed pre-residual로 대체0|\n|원인/보편성|단일 Llama/order/cross-method config 차이; causal/promotion claim0|\n|초기 첫표 job 표기|historical formatter가 underscore를 숫자로 읽어 표시한 오류. final-summary.csv의 exact39283_N는 맞음; 본문표/최종manifest는 문자열 보존. 최초전달본hash는불변.|\n',
          '## 11. 전체 artifact inventory / 재현\n']
    inv=[]
    for p in sorted(out.glob('*.csv')):inv.append(dict(file=p.name,rows=len(csvread(p)),bytes=p.stat().st_size,sha256=sha(p)))
    txt.append(table(inv,['file','rows','bytes','sha256']))
    figs=read(out/'figures/plot-manifest.json')
    for f in figs:txt.append(f"- `{f['path']}` SHA `{f['sha256']}` — {f['caption']}\n")
    txt+=['\n집계 source는 project/run_scripts/blue_lifelong_analysis/, 기존 raw-free common utilities만 reuse하며 runtime import/model load가 없다. 원본 local hook/archive는 이동·수정하지 않았다. Raw tensors/prompts/log/model/cache는 Git0, local path+SHA inventory만 공유한다.\n',
          '```bash\n/data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.blue_lifelong_analysis.plots --out PACKAGE --dest NEW_DIRECTORY\n```\n',
          '전체 재현 단계는 first → audit → aggregate → supplement → plots → report → package validate/seal이며 각 destination create-once다. Python/environment/source/input/output SHA와 focused tests/reproduction은 최종 manifest/receipt에 결속한다.\n',
          '## 12. FACT / INFERENCE / DECISION\n',
          '**FACT:** 6개 run scheduler COMPLETED0, raw600 batches/72checkpoint 검산, final/cumulative NLL preference와 token-secondary를 분리했다. MEMIT original의 at-write RS9999 중2816은 final에 실패, AlphaEdit original은9999 중111이 final 실패다. 이는 저장된 동일 request의 endpoint 비교다.\n',
          '**INFERENCE 제한:** Current 성공과 final loss는 구분되지만, posterior loss의 원인을 history geometry/특정 layer/residual만으로 확정하지 않는다. Layers-only 변경도 native target layer를 함께 바꾸며 method별 regularizer가 다르다.\n',
          '**DECISION:** 요청된6chain CPU exhaustive review 완료, scientific_promotion=false. 새 experiment/rescue/evaluator/GPU0. 본 새 raw-free analysis/report scope만 main 통합하고 GH 보고 후 STOP. L567/중지된 audit는 그대로 둔다.\n']
    with (out/'factual-report-ko.md').open('x') as f:f.write('\n'.join(txt))
    print('REPORT_WRITTEN',sha(out/'factual-report-ko.md'),flush=True)

if __name__=='__main__':main(cli().out)
