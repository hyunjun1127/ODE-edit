"""Build raw-free Korean factual review from independently reduced tables."""
import argparse
import csv
import json
import subprocess
from pathlib import Path
from review import ARMS, read, dump, csvout, sha, arm_ledger


def table(rows, columns):
    def val(x):
        if isinstance(x,float):return f'{x:.9g}'
        return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(val(r.get(c,'NA')) for c in columns)+' |' for r in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.input;dest=a.output;task=out.parents[2];wt=Path.cwd()
    def rows(name):return list(csv.DictReader((dest/(name+'.csv')).open()))
    mainrows=rows('final-eight-arm-table');native=mainrows[0];checks=read(dest/'artifact-checks.json');term=read(out/'terminal.json')
    constants=['RS','PS','NS']; summary=[]
    for r in mainrows:
        summary.append(dict(arm=r['arm'],RS=f"{r['RS_n']}/100",PS=f"{r['PS_n']}/200",NS=f"{r['NS_n']}/1000",
            R_strict=f"{r['rewrite_strict']}/100",RP_strict=f"{r['R_two_P_strict']}/100",RP_joint=f"{r['R_two_P_NLL_joint']}/100",
            S64_KL=float(r['S64_KL']),Dev128_KL=float(r['Dev128_KL']),Dnorm=float(r['correction_norm']),
            dPS_pp=float(r['PS_delta_pp_vs_N4']),dNS_pp=float(r['NS_delta_pp_vs_N4'])))
    kl=[]
    for r in mainrows:
        kl.append(dict(arm=r['arm'],S64_change_pct=100*(float(r['S64_KL'])/float(native['S64_KL'])-1),
            Dev128_change_pct=100*(float(r['Dev128_KL'])/float(native['Dev128_KL'])-1)))
    csvout(dest/'kl-changes.csv',kl)
    random=[]
    for arm in ['RAND+','RAND-']:
        d=read(out/f'diagnostics/{arm}.json');obs=read(out/f'observers/{arm}.json')
        random.append(dict(arm=arm,status=d['status'],RS=obs['metrics']['RS']['numerator'],PS=obs['metrics']['PS']['numerator'],NS=obs['metrics']['NS']['numerator'],
            S64_KL=read(out/f'observers/{arm}-S64-output-KL.json')['loss'],Dev128_KL=read(out/f'observers/{arm}-Dev128.json')['loss'],
            actual_norm=d['actual_norm'],relative_norm_error=d['relative_norm_error'],invariant_pass=d['invariant']['pass'],
            selected_on_loss=d['selection_on_loss'],weight_sha=d['weight_sha'],retained_weight='NO'))
    csvout(dest/'random-diagnostics.csv',random)
    geometry=[]
    for arm in ['CA','EN-S','EN-F']:
        d=read(out/f'geometry/{arm}.json');g=read(out/f'geometry/{arm}-gradient.json')
        geometry.append(dict(space=arm,allowed=d['allowed_dimension'],blocked=d['blocked_dimension'],remaining=d['dimension'],
            chi=g['chi'],projected_fraction=g['remaining_gradient_fraction'],status=d['status']))
    ca=read(out/'geometry/CA-EXACT.json')
    geo_summary=dict(CA_EXACT_status=ca['status'],rank_A=ca['rank_A']['rank'],rank_AK=ca['rank_raw_AK']['rank'],physical_dimension=ca['physical_dimension'],
        EN_F_rank=read(out/'geometry/EN-F.json')['reduced_rank']['rank'],EN_F_condition=read(out/'geometry/EN-F.json')['reduced_rank']['condition_full'])
    dump(dest/'geometry-summary.json',geo_summary)
    events=rows('controller-events');invariants=[r for r in events if r['event']=='actual_invariant_checked']
    trialcheck=[]
    for arm in ARMS:
        l,_,_=arm_ledger(out,arm)
        for t in l.get('trials',[]):
            if t['accepted']:
                assert t['p_actual']<0 and t['loss']<=t['armijo_bound']
                floor=0 if arm=='EN-COV' else 1e-6
                assert t['decrease']>floor
            trialcheck.append(dict(arm=arm,round=t['round'],trial=t['trial'],accepted=t['accepted'],arithmetic='PASS'))
    dump(dest/'accepted-trial-arithmetic.json',trialcheck)
    design_rows=[
        ('scope/T override','server1_single_batch/reuse.py:54','validation_binding','validation-status.json; terminal.json','b001 only; T skipped; no T PASS'),
        ('same native endpoint / cold history','runtime.py:24','Runtime; native','runtime-load.json; native/; endpoint-artifacts.csv','8 metadata W0/WN/context/order equal; history0'),
        ('reuse selection binding','server1_single_batch/reuse.py:16','verify_endpoint','arms/*/reuse.json; imported selection seals','5 optimization endpoints reused; 3 new'),
        ('old/new all-valid-token union','binding.py:12','protected_sequences','protected-provenance.json','1400 sequences; 624 inputs; 10416 positions; no P/N in lock'),
        ('actual FP32 key dedup','runtime.py:152','protected_oracle','geometry/reuse.json; artifact-checks.json','identical prefix plus identical FP32 key required; S4 4596 vs S1 4482'),
        ('P star and rank rule','geometry.py:280','allowed_range; edit_null_space','P-star-basis seal; geometry/*.json','14326 allowed; cutoff unchanged; T not established'),
        ('W0 forward KL teacher','runtime.py:141','reference_oracle','native-objective.json; teacher path-binding lock','fixed stored payload; current WN gradient recomputed on S1'),
        ('EN-COV cumulative objective','runtime.py:179','covariance','arms/EN-COV/events; selection ledger','W-W0, not W-WN; activation drift distinct from output KL'),
        ('FP64 ideal / FP32 actual','optimizer.py:373','optimize trial materialization','trials.csv; endpoint-artifacts.csv','(WN64+Dideal).float; actual endpoint-WN norm rechecked'),
        ('Armijo / budget / fallback','optimizer.py:450','optimize acceptance','trials.csv; controller-counts.csv','actual negative directional product; c1=1e-4; no quality-based endpoint exclusion'),
        ('per-sequence current guard','binding.py:49','quality_ok','controller-events.csv; raw event hashes','new NLL epsilon1e-4; successful strict/canonical IDs retained'),
        ('ideal/actual response and logits','runtime.py:195','invariant','controller-events.csv','ideal1e-10; actual response/leak1e-5; NLL1e-4; logit max1e-3/rms1e-4'),
        ('EN-F4 fresh gradients','optimizer.py:306','optimize rounds','gradient event state SHAs','4 rounds; shared first +3 fresh; 7/24 available trial slots used'),
        ('selection before observer','server1_single_batch/runner.py:207','run observer phase','ALL_SELECTIONS_SEALED.json; observer seals','P/N/Dev after selection; Report256 not opened by this route'),
        ('canonical RS/PS/NS and strict','observer.py:79','strict_summary; observe','raw rows vs final tables','CPU independent identity-based reduce PASS; ties failure'),
        ('observer and final reset','server1_single_batch/runner.py:270','independent_reset','episode-integrity; before/after manifests','runtime W0/M0/RNG reset evidence; no new GPU continuation verification'),
        ('RAND paired diagnostics','server1_single_batch/runner.py:179','RAND_diagnostic','diagnostics/RAND±.json','both signs; norm match; no selection; final RAND tensor not saved'),
    ]
    cross=[dict(requirement=x[0],file=x[1],function=x[2],runtime_evidence=x[3],bounded_finding=x[4]) for x in design_rows]
    csvout(dest/'design-evidence-crosswalk.csv',cross)
    first_read=task.parents[2]/'audits/servers/server1/2026-09-18-enfc-single-batch-m/full-read-preflight.json'
    # Execution WT is resolved directly, not inferred from scientific source output metadata.
    execution_wt=out.parents[5]
    first_read=execution_wt/'audits/servers/server1/2026-09-18-enfc-single-batch-m/full-read-preflight.json'
    source_files=[]
    package=Path('project/run_scripts/single_layer_edit_preserving_correction')
    for f in sorted(package.rglob('*.py')):
        if '/analysis/' not in str(f):
            frozen=subprocess.check_output(['git','show','3f1941b21538d6a7ad0afd756774ad6a0b605750:'+str(f)])
            assert hashlib_sha(frozen)==sha(f)
            source_files.append(dict(path=str(f),sha256=sha(f),execution_bytes_equal=True))
    source=dict(execution_commit='3f1941b21538d6a7ad0afd756774ad6a0b605750',execution_tree='72503847266864bf5e1b0c96d63ba99aa03d846d',
        M_original_commit='87f65ea2abcbe7e77e04367f73a001d63443734b',analysis_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen_runtime_files=source_files,execution_lock=dict(path=str(task/'submission-r1/execution.lock.json'),sha256=sha(task/'submission-r1/execution.lock.json')),
        prior_FULL_READ=dict(path=str(first_read),sha256=sha(first_read)),reuse_seal=dict(path=str(task/'imports/server4-b001-r1/receiver-seal.json'),sha256=sha(task/'imports/server4-b001-r1/receiver-seal.json')))
    dump(dest/'source-evidence-manifest.json',source)
    missing=[
        ('T full validation','SKIPPED_USER_DIRECTED','FD/ULP/full GPU numerical validation NOT_ESTABLISHED'),
        ('cross-hardware equivalence','NOT_TESTED','S4 and S1 actual key byte inventories differ; no same hardware counterfactual'),
        ('cumulative Frobenius(W-W0)','NOT_RECORDED_WITHIN_LOCAL_INPUTS','WN delta norm recorded; full local W0 weight absent; norms cannot be added'),
        ('RAND final tensors','NOT_RETAINED','seed/rule/space/norm/hash/invariant/observer saved; primary final8 preserved'),
        ('all trial tensors','NOT_RETAINED_BY_POLICY','gradient/factors/formula/hash/events retained, not every rejected physical weight'),
        ('full K tensor','NOT_RETAINED_BY_POLICY','token alias SHA and projector factors saved; actual full K numeric payload not saved'),
        ('GPU continuation / full model CPU restore','NOT_RUN_REVIEW_ONLY','stored selected and nonselected hash receipt audit only'),
        ('Report256 / Past / sequential / M10','NOT_IN_SCOPE','Report256 unopened; cold Past empty; one batch only'),
        ('storage I/O / individual arm complete wall','NOT_SEPARATED','callback and streaming counters overlap; cannot sum as disjoint costs'),
    ]
    csvout(dest/'coverage-and-limitations.csv',[dict(item=i,status=s,limitation=l) for i,s,l in missing])
    tails=rows('request-tail')
    selected_tails=[dict(arm=r['arm'],metric=r['metric'],quantity=r['quantity'],mean=float(r['mean']),median=float(r['median']),p95=float(r['p95']),p99=float(r['p99']))
                    for r in tails if r['arm'] in ARMS and r['quantity']=='new_nll']
    retain=[r for r in rows('strict-and-retention') if r['metric']=='JOINT_AND_W0']
    counts=rows('controller-counts')
    mechanism_table=[{k:r[k] for k in ['arm','accepted_rounds','attempted_trial_slots','gradient_rounds','gradient_sweeps','reject_reasons']} for r in counts]
    components=[dict(group='parent allocation',seconds=9147,scope='single parent; batch/extern not added'),
        dict(group='program wall',seconds=term['wall_seconds'],scope='subset of allocation'),
        dict(group='model load',seconds=term['timing']['model_load'],scope='shared'),
        dict(group='canonical+generation observer',seconds=term['observer_work']['observer_seconds'],scope='9 unique states including W0'),
        dict(group='S64 backward',seconds=term['oracle_work']['S64']['backward_seconds'],scope='nested in objective; not additive'),
        dict(group='S64 suffix forward',seconds=term['oracle_work']['S64']['cached_suffix_seconds'],scope='overlaps objective/invariant components'),
        dict(group='protected suffix forward',seconds=term['oracle_work']['protected']['cached_suffix_seconds'],scope='overlaps guards/invariants'),
        dict(group='Dev128 suffix forward',seconds=term['oracle_work']['Dev128']['cached_suffix_seconds'],scope='observer component')]
    csvout(dest/'compute-summary.csv',components)
    text=f'''# ENFC B1 완료 실험 상세 사실 리뷰

## 1. 범위와 완료 판정

job50098 `odeedit_enfc_m_b001_s1`은 단발 accounting에서 COMPLETED, exit0:0이었다. 시작 2026-09-18T16:42:17, 종료19:14:44, parent allocation **9,147 GPU초(2.540833시간)**, 1GPU/8CPU/178G이다. batch/extern을 중복 합산하지 않았다. 프로그램 terminal은 `COMPLETE_WITH_T_SKIPPED`, wall {term['wall_seconds']:.6f}초이며 8 primary endpoint와 2 RAND 관측이 존재한다. 이 리뷰에서는 GPU/model/forward/evaluator/Slurm 변경·재실행·원격 raw 수신을 하지 않았다.

이는 cold O0/b001 ordinal[0,100)의 **unique100 요청** 비교다. 8arm×100=800 arm-request 관측이지 unique800이 아니다. 성공 총점·checkpoint·저장된 guard를 검산했으나 **T=SKIPPED_USER_DIRECTED / full_numerical_validation=NOT_ESTABLISHED**를 유지한다. 전체 ENFC M10/S/R/L 또는 장기 검증 완료가 아니다. 과학적 선택·인과 해석은 GH 소유다.

## 2. 지표와 첫 8arm 비교

RS/PS는 target-new NLL < target-true NLL, NS는 반대로 true < new다. tie는 실패다. NLL은 낮을수록 해당 target의 확률이 높다. desired margin은 R/P에서 true−new, N에서 new−true로 양수가 성공이다. R100/P200/N1000의 exact case/prompt/target/token identity로 쌍을 묶었다. TF strict는 target 모든 token argmax 일치이며 NLL 선호와 다르다. RP joint는 한 요청의 R+2P 세 NLL 비교가 모두 성공한 수/100이다.

{table(summary,['arm','RS','PS','NS','R_strict','RP_strict','RP_joint','S64_KL','Dev128_KL','Dnorm','dPS_pp','dNS_pp'])}

Dnorm은 저장 FP32 endpoint−WN의 FP64 Frobenius norm이다. Native ΔN norm은 7.6116527986668565이며 Dnorm과 구분한다. N4/SCALE/CA는 실제 weight bytes가 같다. 8 의미상 arm을 유지하되 primary distinct endpoint는 **6개**다. SCALE/CA의 canonical·Dev·generation 재사용을 새 평가 계산으로 세지 않았다. N4 ledger의 fallback=true는 accepted_rounds=0인 reference의 구현 표기이지 N4 최적화 실패가 아니다. SCALE/CA는 각8 trial의 current guard 거절 뒤 WN fallback이다.

S64 output KL과 Dev128 output KL의 N4 대비 산술 변화는 다음과 같다. EN-COV 최적화 목적은 output KL이 아니라 W0 기준 cumulative activation drift이므로 목적값을 섞지 않았다.

{table(kl,['arm','S64_change_pct','Dev128_change_pct'])}

![저장 endpoint 관측](endpoint-observations.png)

## 3. 요청 단위 손실·회복, strict 및 NLL 꼬리

독립 reducer가 모든 저장 true/new raw NLL과 strict token을 재집계하여 기존 n/d 및 row별 성공·NLL·margin과 대조했다. 11개 state(8arm+W0+RAND±), 33개 metric 집계가 일치한다. 입력 중복/누락·nonfinite·token 길이 불일치는 검사상 없었다. raw prompt/token은 Git에 포함하지 않았다.

N4→EN-COV: PS **lost1/gained0/unchanged199**, NS **lost0/gained1/unchanged999**. 나머지 primary arm과 RAND±는 R/P/N 모두 lost0/gained0으로 성공 ID도 동일하다. 동일 총점이라는 이유만으로 ID 동일을 가정하지 않고 쌍별 확인했다. R strict100/100, R+2P strict44/100은 모든 arm에서 같다. R+2P NLL joint는 EN-COV95/100, 나머지96/100이다.

{table(retain,['arm','W0_correct_N_retained','W0_correct_N_d','W0_correct_N_lost','W0_incorrect_N_gained'])}

W0 RS5/100, PS20/200, NS886/1000이다. W0-correct NS 조건분모886은 전체 NS1000과 다르며 분모 오류가 아니다. 전체 PS의 TF strict와 token accuracy는 [strict-and-retention.csv](strict-and-retention.csv)에 별도로 있다.

아래는 target-new NLL의 평균/중앙/p95/p99다. true NLL, desired margin, p90/min/max, N4 대비 paired delta의 동일 통계는 [request-tail.csv](request-tail.csv)에 모두 보존한다. quantile은 NumPy linear interpolation이며 tail cutoff에 따른 선택이나 bootstrap을 하지 않았다.

{table(selected_tails,['arm','metric','quantity','mean','median','p95','p99'])}

greedy32는 모든 primary 및 RAND 행에서 rewrite target-prefix100/100, target>32 censor0, EOS 종료0, 길이32/한도도달100이다. 의미적 정확도나 자연스러운 후속 생성의 검증이 아니다. 두 P 동시 TF strict44/100과 생성 prefix100/100을 같은 지표로 부르지 않는다. 실제 고유 endpoint 8개에서 generation800회/25,600 tokens가 실행됐고 SCALE/CA는 중복 실행하지 않았다. 바뀐 case/prompt ID와 양쪽 margin은 [changed-identities.csv](changed-identities.csv)에 raw 문장 없이 공개한다.

## 4. 교정 경로와 수용/거절

{table(mechanism_table,['arm','accepted_rounds','attempted_trial_slots','gradient_rounds','gradient_sweeps','reject_reasons'])}

모든 실제 trial은 [trials.csv](trials.csv)에 round/trial/eta/eta0/actual step norm/ideal norm/nominal prediction/actual directional product/Armijo bound/감소량/거절 사유로 있다. accepted trial의 실제 음의 방향미분·Armijo·분해능 감소 조건을 CPU 산술로 재확인했다. duplicate trial0, nonfinite selected endpoint0. EN-F4는 shared 첫 gradient+후속3 fresh gradient, 네 수용 round이며 최대4×6 중 실제7 trial을 사용했다. EN-COV는 별도 activation gradient1회를 사용했다. S4 재사용 KL-P/EN-S의 최적화 gradient를 S1에서 다시 실행하지 않았다.

![실제 교정과 trial](correction-trials.png)

EN-F 실제 수용 Dnorm0.01654755, EN-F4 최종0.03395447, EN-COV1.11222008이다. ideal/actual norm과 rounding norm은 별도 기록된다. W−W0 cumulative norm은 허용 local inputs에 W0 원 tensor가 없어 NOT_RECORDED로 남겼다. ΔN norm+Dnorm을 cumulative norm으로 더하지 않았다. 실제 EN-COV 목적식이 W−W0라는 source 사실과 그 cumulative norm 미기록은 다른 사항이다.

## 5. geometry·gradient·RAND

{table(geometry,['space','allowed','blocked','remaining','chi','projected_fraction','status'])}

KL-P의 allowed gradient squared norm은0.004539442695615405, EN-F 잔존 비율은0.5956443443024163이다. 이 수치는 S1 WN에서 새로 계산한 gradient 진단이며 S4의 수용 경로를 같은 하드웨어로 재현한 증거가 아니다. CA-EXACT는 rank(A)=100, rank(AK)=100, physical dimension0/REPAIR_SPACE_EMPTY. CA 자체는 제한된 writer-row 공간의 별도 optimization arm이므로 CA-EXACT와 같은 행으로 합치지 않는다. EN-F reduced rank4482, condition 약1.4306e9, ambiguity indices는 기록상 비어 있다. 편의 rank truncation을 추가하지 않았다.

{table(random,['arm','status','RS','PS','NS','S64_KL','Dev128_KL','actual_norm','relative_norm_error','invariant_pass'])}

RAND±는 EN-F actual norm에 맞춘 동일 난수의 두 부호이며 objective로 부호를 선택하지 않았다. 분모 R/P/N=100/200/1000. 두 actual final RAND weight 파일은 보존되지 않았고 seed/space/norm/hash/관측만 보존됐다. 이는 필수 primary8 endpoint 보존과 구분한다. 모든 rejected trial weight도 원 retention 정책상 미보존이다.

## 6. 실제 invariant와 설계 대조

accepted EN-F 계열에서 ideal response relative≤1e-10, actual token normalized response≤1e-5, P leakage≤1e-5, NLL maxdiff≤1e-4, logit max≤1e-3/rms≤1e-4의 원 경계를 검사했다. EN-F의 실제 logit max8.2016e-5, NLL max2.8610e-5이며 exact-zero/bitwise invariant라고 쓰지 않는다. EN-COV의 PS 손실은 보호된 current rewrite/context 경계와 별개의 post-selection paraphrase 관측이다. 허용 공간 존재와 실제 locality 성공은 별도 열로 유지한다.

{table([{k:r[k] for k in ['arm','round','detail_ideal_response_relative','detail_actual_max_token_normalized_response','detail_actual_projection_leakage_relative','detail_max_NLL_difference','detail_logit_max','detail_logit_rms','passed']} for r in invariants],['arm','round','detail_ideal_response_relative','detail_actual_max_token_normalized_response','detail_actual_projection_leakage_relative','detail_max_NLL_difference','detail_logit_max','detail_logit_rms','passed'])}

{table(cross,['requirement','file','function','runtime_evidence','bounded_finding'])}

파일 위치는 실행 source3f1941b2의 `project/run_scripts/single_layer_edit_preserving_correction/` 상대경로이다. 현재 분석 checkout의 frozen runtime bytes가 실행 commit과 동일함을 파일별 SHA로 확인했다. code의 조건 존재, runtime 기록의 통과, 새 CPU 검산의 통과를 서로 대체하지 않았다. 실제 T/FD/full Llama 수치 검증을 새로 수행하지 않았다.

## 7. 플랫폼·입력·재사용·복원

N4/SCALE/CA/KL-P/EN-S 최적화는 S4 Blackwell 보존 endpoint 재사용, EN-F/EN-COV/EN-F4는 S1 A6000 신규 계산이다. 모든 현재 공식 관측은 S1에서 같은 저장 endpoint로 계산하거나 exact 동일 endpoint 관측을 재사용했다. S4 observer는 REFERENCE_ONLY이며 S1 관측으로 덮어쓰지 않았다.

동일 native WN/target/native K/A/P/zeroM/context/order/teacher payload lineage를 유지한다. 단, full-token key는 S1에서 다시 capture하여 bytes가 다르다. protected prefix alias10416개와 입력624개/sequence1400개는 S4/S1 정확히 같고, token-prefix 자체4315개도 같다. 실제 FP32 key 동일성까지 요구한 dedup 결과가 S4 **4596**열, S1 **4482**열이다. 따라서 같은 token inventory라는 근거는 있으나 geometry/captured-key byte parity는 **false**이고 그 수치적 원인·동등성은 NOT_TESTED다. hardware 차이만으로 원인을 확정하지 않는다. affected EN-F geometry와 WN S64 gradient 재계산 사실을 기록했고 native fit·teacher 생성은 새로 하지 않는 경로이다. native_fit_new0/history_appends0은 terminal 및 source route와 일치한다. teacher generation 별도 runtime 카운터는 없으며 seal 재사용+생성 호출 없는 source 근거다.

final8는 현재 regular file/size/SHA 및 `torch.load(weights_only=True,map_location=cpu)`로 [4096,14336] FP32 finite·case100·WN/W0/context/history0·selection ledger/seal·observer binding을 확인했다. 파일8개, primary unique tensor6개다. [endpoint-artifacts.csv](endpoint-artifacts.csv)와 [artifact-manifest.json](artifact-manifest.json)에 경로/모드/크기/해시를 보존한다. before/after nonselected 전체 parameter hash manifest는 서로 같고 final W0/M0/RNG reset은 저장 receipt로 확인했다. 이번 CPU 검토가 full model을 다시 복원하거나 GPU continuation을 검증한 것은 아니다.

## 8. 실측 비용

{table(components,['group','seconds','scope'])}

peak allocated GPU={term['peak_gpu_allocated']/2**30:.6f}GiB, reserved={term['peak_gpu_reserved']/2**30:.6f}GiB, host peak={term['peak_host_KiB']/2**20:.6f}GiB. 실험 프로그램 wall과 parent allocation 차이는 약{9147-term['wall_seconds']:.3f}초다. CPU geometry·I/O 동안도 GPU allocation 비용에 포함된다. 최종 작은 projection/solve만 전체 계산량이라고 하지 않는다.

canonical observer는 실제9 states(W0 포함), 1494 microbatches, true/new sequence rows23400, forward calls27094(생성 포함), model_input_tokens418064를 기록했다. 이는 primary arm-pair10400와 계산 단위가 다르다. S64 backward document256=4×64, suffix1152, Dev suffix1024; prefix/teacher streaming/guard/invariant 및 arm별 objective timing은 [compute.csv](compute.csv)의 원 counter 단위로 공개한다. callback timing과 oracle timing은 포함관계가 있어 중복 합산 금지. exact arm별 총 wall 및 저장 I/O 별도시간은 NOT_SEPARATED다.

과거 S4 M50050 두 cell 합9768초, 이전 T/M3231초, metadata failure2301초는 이전 lineage 비용으로만 기재한다. B1/B2로 나눌 증거 없이 반분하지 않으며 새 S1 9147초에 재합산하지 않는다. S4 기존5arm 최적화 counter는 S4_REUSED, S1 신규3arm은 S1_NEW로 분리했다.

## 9. 검산·재현·자료 한계

새 reducer 8개 focused CPU test PASS. 실제 raw independently reduced counts/NLL/joint/retention 일치, final8 CPU tensor/selection SHA 검산 PASS. 별도 독립 red subagent는 사용하지 않았고 SH1이 실행 source 대조 및 별도 reducer/postcheck를 수행했다. 이는 독립 연구자 검증이나 T_PASS가 아니다. 원 raw 및 frozen source 수정0, 신규 GPU/Slurm0, raw broadcast=NO_BROADCAST_NOT_REQUIRED.

{table([dict(item=i,status=s,limitation=l) for i,s,l in missing],['item','status','limitation'])}

한 cold batch에서 S64/Dev 변화와 canonical/strict 관측을 제공할 뿐 장기효과·일반화·통계적 동등성·최종 정책 우열을 판정하지 않는다. EN-COV의 PS1개 손실과 NS1개 회복을 모두 유지한다. EN-F/EN-F4의 S64 감소를 공식 성공률 상승으로 바꾸어 설명하지 않는다. Report256, 추가 T/arm/sweep/S/R/L은 실행하지 않았다.

## 10. 재현 명령과 파일

실행 source: `3f1941b21538d6a7ad0afd756774ad6a0b605750`, tree `72503847266864bf5e1b0c96d63ba99aa03d846d`. 실행 lock SHA `b463293b23131b1453d1d3521a58f8f57e1804b0ca18d0fd8fcd0bdc64999147`. 원 M87f65ea2와 분석 source는 [source-evidence-manifest.json](source-evidence-manifest.json)에 분리한다. 최종 publication HEAD는 main push 후 별도 receipt로 인계한다.

```bash
python project/run_scripts/single_layer_edit_preserving_correction/server1_single_batch/analysis/review.py --input {out} --output {dest}
CUDA_VISIBLE_DEVICES='' python project/run_scripts/single_layer_edit_preserving_correction/server1_single_batch/analysis/artifacts.py --input {out} --output {dest}
python project/run_scripts/single_layer_edit_preserving_correction/server1_single_batch/analysis/plot.py --root {dest}
python project/run_scripts/single_layer_edit_preserving_correction/server1_single_batch/analysis/report.py --input {out} --output {dest}
```

실제 Python은 `/mnt/raid5/janghj/EasyEdit/.venv/bin/python`을 사용했다. PNG는 CSV 입력만으로 코드 생성했고 두 번 render한 bytes 일치를 [plot-reproduction.json](plot-reproduction.json)에 기록했다. 입력/출력SHA·환경·명령 포함. 이미지 육안 확인 및 Markdown 렌더/링크/열 검사 결과는 postcheck receipt에 별도 기록한다.

핵심 CSV: [첫 표](first-eight-arm-table.csv), [최종 표](final-eight-arm-table.csv), [paired 전이](paired-transitions.csv), [NLL 꼬리](request-tail.csv), [기계적 경로](mechanism.csv), [trial](trials.csv), [geometry](geometry.csv), [RAND](random-diagnostics.csv), [비용](compute-summary.csv), [누락](coverage-and-limitations.csv). 로컬 raw는 원 실행WT 및 그 imports에 보존하며 Git에는 코드/집계/해시만 게시한다.
'''
    (dest/'diagnostic-report-ko.md').write_text(text)
    print('REPORT_BUILT',sha(dest/'diagnostic-report-ko.md'))


def hashlib_sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


if __name__=='__main__':main()
