"""Detailed Korean report and raw-free rooted publication closure."""
import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path
from .identity import save,sha,digest,member,REPO,ROOT

def readcsv(p):
    with p.open() as f:return list(csv.DictReader(f))

def num(x):return '' if x in ('',None) else f'{float(x):.7g}'

def table(headers,rows):
    def cell(x):return str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(cell,row))+' |' for row in rows])+'\n'

def main(args):
    root=Path(args.root);summary=readcsv(root/'endpoint-summary.csv');base=readcsv(root/'base-preservation-versus-recovery.csv')
    runs=readcsv(root/'run-index.csv');parity=readcsv(root/'native-response-parity.csv');compute=readcsv(root/'compute-ledger.csv')
    trajectory=readcsv(root/'trajectory.csv');complete=json.loads((root/'completeness.json').read_text())
    generation=readcsv(root/'generation-summary.csv')
    calibration=readcsv(root/'calibration.csv')
    structural=readcsv(root/'structural-actions.csv');paths=readcsv(root/'physical-path-actions.csv')
    intermediate=readcsv(root/'intermediate-current.csv');coverage=readcsv(root/'requirements-evidence.csv')
    resources=json.loads((root/'resource-receipt.json').read_text())
    allcompute=json.loads((root/'all-attempt-compute-receipt.json').read_text())
    technical=json.loads((root/'technical-exclusions.json').read_text())
    peaks=readcsv(root/'peak-memory.csv')
    impact=readcsv(root/'repair-impact.csv')
    from .interpretation import findings
    answers=findings(root)
    save(root/'interpretation-findings.json',dict(answers=answers,claim_scope='descriptive fixed experiment',scientific_promotion=False))
    text=['# L4 two-memory conflict routing v2 — 상세 진단 보고서',
          '',f"새 경로 {complete['new_paths']}/16, B100 native reference 재사용 {complete['native_reused']}/3. "
          'We 기준 Base 보존, active-Past bank, full P* 및 시간당 비용·누적 anchor의 v2다. '
          'ABC/이전 EP 및 W0 복구 objective와 별도 실험이다. scientific_promotion=false.',
          '', '**기술 정정:** 최초 실행의 BF8 3개와 Middle Frozen-BF8은 큰 Gram 행렬 크기의 residual tolerance를 '
          'dual의 부호 검사에 잘못 적용하여 음수 dual/slack을 허용했다. 해당4경로만 제외·복구했고 N/OS/BF1 및 준비 자산은 보존했다. '
          'Middle-first/primary-three-entry 중간 보고의 BF8/Frozen 수치와 최초 전체-valid 표기는 이 보고서로 대체한다. '
          '새 solver는 기존 λ≥0 영역을 정확히 검사하며 목적식·bank·calibration·h·threshold를 바꾸지 않았다. '
          '37개 canonical node의 비음수 domain과 저장 QP replay를 재검산했다. 실패/취소 allocation도 아래 비용에 포함한다.',
          '16개는 고유한 예정 경로의 수다. 최초16개 실행 중4개를 기술 교체했으므로 실제 신규 endpoint/history action은 '
          '총20회이며, 정상 endpoint를 성능 때문에 재실행한 것은 아니다. 원3개 N reference는 별도 재사용이다.',
          '', '### 기술 정정의 수치 영향 — 이전 값은 과학 분모에서 제외', '',
          table(['entry','arm','Past old→fixed','Base KL old→fixed','BaseAudit KL old→fixed','Current NS old→fixed'],
              [[r['entry'],r['arm']]+[num(r['excluded_'+k])+'→'+num(r['canonical_'+k]) for k in ('Past','Base','BaseAudit')]+
              [r['excluded_NS_n']+'/'+r['excluded_NS_d']+'→'+r['canonical_NS_n']+'/'+r['canonical_NS_d']] for r in impact]),
          '', '## 1. 지표와 분모', '',
          'RS/PS는 target-new NLL < target-true NLL, NS는 반대 방향이며 동률은 실패다. '
          'NLL은 낮을수록 해당 target의 확률이 높다. TF strict와 token accuracy는 별도 열로 보존했다. '
          'Current B100/Fixed100/Past100은 각각 RS100·PS200·NS1000이다. B1/B7 Current는 실제 B·2B·10B이며 Fixed/Past는 각100을 유지한다. '
          'Token accuracy의 분모는 target token 수이므로1010처럼 더 클 수 있지만, NS 성공률의 분모1000과 혼용하지 않는다. '
          '같은 Fixed100의 entry 반복을 독립300으로 합치지 않았다. 성공률은 서로 다른 state의 pair-event를 합산한 온라인 점수가 아니다.',
          '', '## 2. 전체 endpoint — 절대 성공률', '']
    primary=[r for r in summary if r['reference']=='N']
    groups={}
    for r in primary:groups.setdefault((r['entry'],r['batch_raw'],r['arm'],r['panel']),{})[r['metric']]=r
    display=[]
    for (e,b,a,p),rr in groups.items():
        def rate(k):
            r=rr[k];return f"{r['numerator']}/{r['denominator']} ({float(r['rate']):.2f}%)"
        display.append([e,b,a,p,rate('RS'),rate('PS'),rate('NS'),num(rr['RS']['new_nll_mean']),num(rr['PS']['new_nll_mean'])])
    text += [table(['entry','B','arm','panel','RS','PS','NS','rewrite new NLL','rephrase new NLL'],display),
             'W0는 audit reference이며 controller teacher가 아니다. ENTRY는 We, N은 해당 effective B의 native endpoint다.',
             '', '## 3. NLL 분포와 N 대비 paired 변화', '']
    text.append(table(['entry','B','arm','metric','new mean/median/p90/max','true mean/median/p90/max','N 대비 new NLL Δ mean/median/p90/max','성공률 Δpp [95% CI]','N 성공→실패 / N 성공'],
        [[r['entry'],r['batch_raw'],r['arm'],r['metric'],
          '/'.join(num(r['new_nll_'+k]) for k in ('mean','median','p90','max')),
          '/'.join(num(r['true_nll_'+k]) for k in ('mean','median','p90','max')),
          '/'.join(num(r['paired_new_nll_delta_'+k]) for k in ('mean','median','p90','max')),
          f"{num(r['delta_pp'])} [{num(r['paired_ci_low'])},{num(r['paired_ci_high'])}]",
          f"{r['loss']}/{r['reference_success']}"] for r in primary if r['panel']=='Current100']))
    text += ['request-cluster paired bootstrap 2000회, seed20260911. 같은 요청의 prompt들을 함께 재표집했다. '
             '개발된 고정 패널에서의 descriptive association이며 새 독립 benchmark/인과 증명이 아니다. '
             'Fixed/Past의 전체 NLL 분포·ENTRY 대비 loss/recovery는 endpoint-summary.csv에 있다.', '',
             table(['entry','B','arm','metric','TF new exact','TF true exact','new token accuracy','true token accuracy'],
               [[r['entry'],r['batch_raw'],r['arm'],r['metric'],f"{r['strict_new_numerator']}/{r['denominator']}",
                 f"{r['strict_true_numerator']}/{r['denominator']}",f"{r['new_token_correct']}/{r['new_token_count']}",
                 f"{r['true_token_correct']}/{r['true_token_count']}"] for r in primary if r['panel']=='Current100']),
             '', '## 4. We 보존과 W0 복구 — 분리된 관측', '']
    text.append(table(['entry','B','arm','bank','F(controller)','raw mean','context raw median/p90/max','음수 KL token'],
        [[r['entry'],r['batch_raw'],r['arm'],r['bank'],num(r.get('controller_value')),num(r.get('raw_value')),
          '/'.join(num(r.get('per_context_raw_'+k)) for k in ('median','p90','max')),r.get('negative_kl_tokens','')]
         for r in base if r.get('controller_value','')!='']))
    text += ['Past의 ψ는 context target-token 평균 NLL 증가에 먼저 적용하고 request/context 평균한다. '
             'Base는 KL(p_We||p_W)이며 작은 음수 반올림은 raw/count를 보존한 뒤 controller에서0으로 처리했다. '
             'BaseW0/BaseAuditW0는 같은 위치에서의 별도 audit KL이다. BaseAudit는 Base candidate512 전체와 disjoint하다. '
             'dataset target_true NLL과 NS는 별도 모델 평가이며 KL 감소와 동일한 지식 복구로 간주하지 않는다.', '']
    text.append(table(['entry','B','arm','bank','Base NS','target_true rewrite NLL mean','NS true NLL mean','NS new NLL mean'],
        [[r['entry'],r['batch_raw'],r['arm'],r['bank'],f"{r.get('numerator')}/{r.get('denominator')}",
          num(r.get('rewrite_target_true_nll_mean')),num(r.get('NS_true_nll_mean')),num(r.get('NS_new_nll_mean'))]
          for r in base if r.get('metric')=='NS']))
    text += ['', '## 5. 실제 기하·누적 보정·joint conflict', '']
    text.append(table(['entry','B','λr','scalar constraint residual','FP32 scalar drift','span residual','projected stationarity'],
       [[r['entry'],r['batch_raw'],num(r['ridge']),num(r['algebraic_equality_residual']),num(r['actual_scalar_drift']),num(r['span_relative']),num(r['stationarity_allowed_norm'])] for r in parity]))
    text += ['P*는 sym(Praw)의 eigenvalue>.5 전체 공간이다. 저장된 작은 제외공간은 정확한 동일 operator 표현이며 Ub100/V256 제한이 아니다. '
             'N에는 원 Praw를 사용했다. OS의 대수적 scalar equality와 실제 FP32 weight 반올림은 별도 값이다. '
             'context-response.csv의 성분은 실제 저장 weight 차이를 고정 L4 입력에 적용한 구조적 응답이며 final logits의 선형성 주장이 아니다.', '']
    text.append(table(['entry','B','channel','a_A','sigma','q_ref','epsilon','terminal budget'],
        [[r['entry'],r['batch_raw'],r['channel']]+[num(r[k]) for k in ('anchor_action_aA','sigma','q_ref','epsilon','terminal_budget')] for r in calibration]))
    text.append(table(['entry','B','arm','node','Σ용 step action/h','cumulative Z H²','ξ/h','joint active','실제 normalized harm','actual f−c−ξ','terminal f−budget'],
      [[r['entry'],r['batch_raw'],r['arm'],r['node'],num(r.get('action_over_h')),num(r.get('cumulative_anchor_action')),
        r.get('xi_per_h',''),json.loads(r['kkt'])['active'] if r.get('kkt') else 'stationary',r.get('normalized_actual',''),
        r.get('actual_local_envelope_gap',''),r.get('actual_terminal_budget_excess','')]
       for r in trajectory]))
    text += ['conflict-map.csv는 같은 state 두 gradient의 H^-1 inner product와 joint dual/slack을 기록한다. '
             '작은 2×2 solve 시간은 전체 full-space/observer 비용이 아니다. 단순 Σ step norm을 net norm으로 바꾸지 않는다. '
             'FP32 runtime 표시용 first-order error와 저장된 FP64 scalar들에서 다시 산출한 진단값은 구분했다. '
             '후자는 추가 forward나 controller 변경 없이 산술적으로 유도한 값이다.',
             'actual f−c−ξ와 최종 f−budget는 실제 nonlinear 관측의 signed gap이다. 양수도 정상 과학 관측이며 '
             '재실행/탈락 조건이 아니다. Linearized KKT의 PASS와 실제10% bank 목표 달성은 서로 다르다.',
             '', '### 실제 물리 변화와 구조적 목적식', '',
             table(['entry','B','arm','||W−We||F²','||W−W0||F²','Past mapping','Base mapping','response deviation N','J_pres','native M action','entry C0 action'],
                [[r['entry'],r['batch_raw'],r['arm']]+[num(r[k]) for k in ('physical_from_entry_sq','physical_from_W0_sq','Past_mapping_action','Base_mapping_action','response_deviation_from_native_sq','J_pres','native_history_action','entry_C0_action')] for r in structural]),
             'native M action은 별도 진단이다. Routing Cp는 active bank이고 M_native를 혼합하지 않았다. '
             'J_pres와 functional F, Frobenius energy와 H action은 서로 다른 양이다. geometry-spectrum.csv는 Praw 전체 spectrum 및 projected data-Gram spectrum을 보존한다.',
             '', '### 저장된 중간 Current 관측', '',
             table(['entry','arm','completed node','reference','metric','success','new NLL mean/p90','paired Δpp'],
                [[r['entry'],r['arm'],r['node_completed'],r['reference'],r['metric'],f"{r['numerator']}/{r['denominator']}",num(r['new_nll_mean'])+'/'+num(r['new_nll_p90']),num(r['delta_pp'])] for r in intermediate]),
             's=.25/.5의 실제 관측만 사용한다. 시점 사이 NLL/성공률을 보간하지 않았다. '
             'per-context 목표 성분·수직 성분의 mean/median/p90/max는 context-response-summary.csv에 있다.',
             '', '## 6. 계산량·자원·실용성', '']
    text += [f"전체 승인 scope allocation: {resources['total_gpu_seconds']:,} GPU-seconds = {resources['total_gpu_hours']:.4f} GPUh. "
             '모든 실제 job/attempt의 model 상주·평가·생성 시간을 포함한다. CPU geometry/분석 시간은 별도이며 GPUh로 세지 않는다.', '',
             table(['job','status','exit','GPU seconds'],[[r['job'],r['status'],r['exit'],r['gpu_seconds']] for r in resources['jobs']])]
    text += [f"CPU geometry pass의 기록된 wall-time 합(이전 preview 포함)은 {allcompute['cpu_geometry_seconds_sum']:.3f}초다. "
             '병행 실행된 pass의 합이므로 campaign 경과시간이나 GPU시간이 아니다. cpu-geometry-ledger.csv로 결속했다.','']
    text.append(table(['entry','B','종류','component','actual'],[[r['entry'],r['batch_raw'],r['category'],r['component'],num(r['value'])] for r in compute]))
    text.append(table(['entry','B','peak allocated GPU bytes','GiB'],[[r['entry'],r['batch_raw'],r['peak_allocated_gpu_bytes'],f"{int(r['peak_allocated_gpu_bytes'])/2**30:.3f}"] for r in peaks]))
    text += ['위 scoped component 합은 scheduler GPU elapsed와 동일하지 않다. 모델 load·native/cache·candidate/key·We/W0 teacher·full P*·factor/inverse·'
             'functional gradient·actual risk·Full 평가·generation·저장/계측·추가 관측 비용을 분리한다. '
             '원 B100 N3의 cold z/native GPU 시간을 이번 실행 시간에 더하지 않았다. B1/B7 native joint solve는 새로 실행했다. '
             '5,568 backward는 bank128/128의 계획값이며 실제 count와 비교한다. 미배정 GPU-hour cap에 이전8h/48h를 상속하지 않았다.',
             'Global model.forward hook의 완전 계수와 node별 inverse-action 단독 timer는 NOT_RECORDED_SCHEMA_GAP이다. '
             '대신 실제 functional forward/backward, evaluator forward, generation token/sequence, static factor+inverse, 전체 node/할당 시간을 공개한다. '
             '이 누락을0이나 추정 FLOPs로 채우지 않았고 profiling-only 재실행도 하지 않았다.',
             'compute-ledger는 유효 preparation/path와 최종 관측의 scoped count다. technical-exclusions.json은 교체된4경로의 '
             '추가 소모를 따로 보존하며 job-gpu-ledger는 원시 시도·수리·중단된 관측을 모두 포함한다. '
             '따라서 유효5568 backward와 실제 낭비를 포함한 총연산량을 동일시하지 않는다.',
             'all-attempt-compute.csv는 allocation별 실제 cumulative counter를 중복 없이 기록한다. 중단된 관측2개는 '
             '마지막 저장 시점의 lower bound이며 이후 호출을 추정하지 않는다. 중단 전후 전체 GPU 상주 시간은 scheduler로 정확히 계상한다.',
             '', '## 7. 사용자의 아홉 질문에 대한 해석', '']
    text += ['### 고정 generation 패널', '',table(['entry','B','arm','literal prefix / prompts'],
        [[r['entry'],r['batch_raw'],r['arm'],f"{r['literal_prefix_numerator']}/{r['denominator']}"] for r in generation]),
        '기존 metadata-hash Current20×(rewrite+2rephrase), 작은 B는 min(20,B)의 실제 패널이다. '
        'Greedy32 tokens의 literal-prefix 관측이며 semantic accuracy가 아니다. 원 generation 문자열은 local-only다.','']
    questions=[
        ('OS의 Current/보호 절충','N 대비 표에서 Current의 RS/PS/NLL 손실과 Past/Base F를 함께 본다. N→OS는 공간·통계·목적식이 함께 바뀐 정적 방법 전체의 효과다.'),
        ('scalar 진척과 실제 NLL','scalar residual이 작아도 개별 context 응답과 NLL을 보장하지 않는다. endpoint paired loss와 context-response의 목표 수직 성분이 그 한계를 측정한다.'),
        ('BF1의 추가 가치','OS→BF1의 Base/Past F 차이뿐 아니라 BaseAudit/Fixed/Past NS·retention과 Current 손실을 함께 비교한다. 단일 bank 개선만으로 유용성을 확정하지 않는다.'),
        ('BF8의 비용 정당화','BF1→BF8은 반복 경로 전체 차이다. 작은 성공률 차이나 동일 endpoint라면 observer 비용 증가를 정당화하는 보편적 증거는 아니다.'),
        ('Frozen 대비 방향 갱신','Middle Frozen-BF8과 BF8은 같은 h/anchor/metric/calibration에서 방향 refresh만 달라진다. 다른 entry에 자동 일반화하지 않는다.'),
        ('held-out 전이','controller128과 disjoint audit128·Fixed/Past 패널의 방향을 분리한다. bank만 좋아지면 bank 적합이라는 설명과 양립한다.'),
        ('누적 보정·수직 성분·slack','trajectory 및 structural/context 표는 추가 action, net 보정, 목표 밖 성분, 시간당 slack을 구분한다. slack이나 위험 증가는 정상 진단값이며 제외 기준이 아니다.'),
        ('We 보존 대 W0 복구','Base KL(We) 감소, KL(W0) 변화, 정답 NLL/NS는 각각 다른 관측이다. We 보존 objective에는 W0 복구항이 없다.'),
        ('B1/B7의 의미','해당 B의 새 native joint solve를 사용했다. NLL/성공률 차이는 서로 다른 요청과 batch 구성을 포함하므로 batch-size만의 인과 효과가 아니다.')]
    for i,answer in enumerate(answers,1):
        text.extend([f"### {i}. {answer['question']}",''])
        text.extend(s+'\n' for s in answer['observations'])
        text.extend([answer['interpretation'],''])
    # Concrete per-entry primary contrasts, not a new decision gate.
    text += ['### 관측값에 묶인 핵심 비교', '']
    for e in ('Early','Middle','Late'):
        rr=[r for r in primary if r['entry']==e and int(r['batch_raw'])==100 and r['panel']=='Current100']
        by={(r['arm'],r['metric']):r for r in rr}
        if not rr:continue
        statements=[]
        for a in ('OS','BF1','BF8','Frozen-BF8'):
            if (a,'RS') in by:statements.append(f"{a}: N 대비 RS {num(by[a,'RS']['delta_pp'])}pp, PS {num(by[a,'PS']['delta_pp'])}pp, NS {num(by[a,'NS']['delta_pp'])}pp, rewrite new NLL Δ {num(by[a,'RS']['paired_new_nll_delta_mean'])}")
        text.extend([e+' — '+'; '.join(statements)+'.',''])
        harms={(r['arm'],r['bank']):float(r['controller_value']) for r in base
               if r['entry']==e and r['batch_raw']=='100' and r.get('controller_value','')!=''}
        mechanics={r['arm']:r for r in structural if r['entry']==e and r['batch_raw']=='100'}
        for left,right in [('N','OS'),('OS','BF1'),('BF1','BF8'),('Frozen-BF8','BF8')]:
            if (left,'Base') not in harms or (right,'Base') not in harms:continue
            d=[]
            for bank in ('Past','Base','BaseAudit'):
                a0=harms[left,bank];a1=harms[right,bank]
                d.append(f"{bank} {a0:.7g}→{a1:.7g} (Δ {a1-a0:+.7g})")
            text += [f"{e} {left}→{right}: "+'; '.join(d)+'.','']
        if 'N' in mechanics and 'OS' in mechanics:
            n=mechanics['N'];o=mechanics['OS']
            text += [f"{e} 정적 변화: J_pres {num(n['J_pres'])}→{num(o['J_pres'])}; "
                f"Past token-key mapping energy {num(n['Past_mapping_action'])}→{num(o['Past_mapping_action'])}; "
                f"Base token-key mapping energy {num(n['Base_mapping_action'])}→{num(o['Base_mapping_action'])}; "
                f"native M action {num(n['native_history_action'])}→{num(o['native_history_action'])}.", '']
    text += ['**설명과 미분리 요인:** 보호된 target 예측 위치의 고정 key에서 ΔWk가 작아도, '
        '다른 문맥 token의 L4 출력 변화가 이후 attention/비선형 경로에 영향을 줄 수 있다. 따라서 구조 mapping 비용 감소와 '
        '실제 KL/NLL 손상 증가가 공존하는 것은 구현상 모순이 아니다. 이 설명은 full-model 관측과 정합적인 가설이며, '
        '특정 token 경로를 분리한 causal intervention으로 확정하지 않았다. 또한 active bank Cp와 전체 native M의 보호 범위가 '
        '다르다. M action 증가를 그 자체로 실제 forgetting이나 causal 원인으로 단정하지 않는다.', '',
        'input-mode-conflict.csv와 input-mode-endpoint-change.csv는 P*로 투영·정규화한 **서로 직교하지 않는 named factor column**의 '
        'Current/Past/C0 에너지와 공통 OS gradient의 native-mode 내적을 제공한다. 이 열들을 합해 독립 에너지 share로 만들지 않는다. '
        '각 BF node의 전체 gradient tensor 또는 request별 native factor attribution은 NOT_RECORDED이며, '
        '저장된 joint Gram/dual 및 OS gradient 이상으로 원인 귀속을 꾸며내지 않는다.', '']
    text += ['이 수치와 비용은 이번 고정 entry/panel에 한정된다. OS가 같은 보존/품질을 더 적은 추가 비용으로 제공하면 one-shot을 우선한다. '
             'BF8/Frozen 차이가 없거나 부정적이면 방향 갱신의 실용 이득이 입증되지 않은 결과로 남긴다. '
             '현재 설계에서 미분리된 요인은 요청 내용·history·bank 표본·scalar response의 대리성이다. '
             '가장 중요한 후속 질문은 **동일한 실제 Current 품질에서 bank 밖 보존 이득이 추가 observer 비용을 정당화하는가**다. 추가 실험은 자동 제출하지 않는다.',
             '', '## 8. 재현·원본 보존·한계', '',
             'source/runtime/input identities는 run-index.csv 및 raw-input-manifest.json으로 연결된다. '
             '원 ABC/BLUE/EP bytes는 읽기 전용이며 technical attempt와 terminal-valid 경로를 구분한다. '
             '실제 B1000·sequential·multilayer·새 z·sweep·hard-key-null/V256 확장은 수행하지 않았다. '
             'NO_BROADCAST_NOT_REQUIRED: 같은 server1의 immutable raw 경로와 hash manifest를 참조한다.', '',
             table(['entry','B','execution HEAD','job','new paths','native reused'],
               [[r['entry'],r['batch_raw'],r['source_head'],r['job'],r['new_paths'],r['native_reused']] for r in runs]),
             '분석 source HEAD/tree와 member SHA는 analysis-manifest.json, input의 before/after SHA는 input-preservation-receipt.json에 있다. '
             'input.lock.json의 source_head는 커밋 전 준비 시점 base이며 실제 실행 HEAD는 위 표의 별도 봉인값이다. '
             '실행 원문 SHA=1dee69d0a203fd1816e24452ca614db79fdaaf63e7ad97eb1191a3ef5fdaa80f, '
             'v2 설계 SHA=4efe1063ea80684beafa791eb99e20c1aa7f3f61636dabab00f9f3805c32eb1d.', '',
             table(['requirement','expected','observed','status'],[[r['requirement'],r.get('expected',''),r.get('observed',''),r['status']] for r in coverage]),
             '```bash',f'python -m project.run_scripts.l4_two_memory_conflict_routing.plotting --input {root} --output <create-once-output>', '```','',
             '![Current endpoint](endpoint.png)','', '![Paired differences](paired.png)','', '![Bank harm](harm.png)','', '![Actual trajectory](trajectory.png)','', '![Compute](compute.png)','']
    path=root/'diagnostic-report-ko.md'
    with path.open('x') as f:f.write('\n'.join(text))
    print(str(path),sha(path),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args())
