"""Reproducible Python PNGs and Korean terminal diagnostic report."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .transfer import DEST,sha
from project.run_scripts.single_layer_cumulative_risk.records import save

def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def md(headers,values):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in values])
def number(x):return f'{float(x):.6g}'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();root=a.report
    assert not any((root/name).exists() for name in ('diagnostic-report-ko.md','trajectory.png','observed-tradeoff.png','middle-auxiliary.png','analysis-manifest.json','rooted-receipt.json')),'PUBLICATION_ALREADY_EXISTS'
    receipt=json.loads((root/'analysis-receipt.json').read_text());assert receipt['status']=='EXACT_REDUCER_PASS'
    agg=rows(root/'endpoint-and-curve-summary.csv');traj=rows(root/'trajectory.csv');run=rows(root/'run-index.csv');paired=rows(root/'paired-summary.csv');probes=rows(root/'same-state-probes.csv')
    lookup={(r['entry'],r['arm'],int(r['step']),r['panel'],r['metric']):r for r in agg}
    n_for={r['arm']:int(r['steps']) for r in run};n_for.update(N=0,W0=0,ENTRY=0)
    def terminal(e,arm,p,m):return lookup[e,arm,n_for[arm],p,m]
    def score(r):return f"{r['numerator']}/{r['denominator']} ({float(r['percent']):.2f}%)"
    def risk(e,arm):return next(float(r['actual_fp32_risk']) for r in run if (r['entry'],r['arm'])==(e,arm)) if arm!='N' else 0
    def main_table(panel):
        rr=[]
        for e in ('Early','Middle','Late'):
            for arm in ('N','H','R','EP'):
                rs,ps,ns=[terminal(e,arm,panel,m) for m in ('RS','PS','NS')]
                rr.append([e,arm,score(rs),score(ps),score(ns),number(rs['new_nll_mean']),number(ps['new_nll_mean']),number(ns['true_nll_mean']),number(ns['new_nll_mean']),number(risk(e,arm))])
        return md(['entry','arm','RS','PS','NS','rewrite new NLL','rephrase new NLL','NS true NLL','NS new NLL','actual normalized risk'],rr)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for col,e in enumerate(('Early','Middle','Late')):
        for arm in ('H','R','EP'):
            rr=[r for r in traj if r['entry']==e and r['arm']==arm]
            axes[0,col].plot([float(r['time']) for r in rr],[float(r['post_edit_nll']) for r in rr],marker='o',label=arm)
            axes[1,col].plot([float(r['time']) for r in rr],[float(r['risk_actual_fp32']) for r in rr],marker='o',label=arm)
        axes[0,col].set_title(e);axes[0,col].set_ylabel('Mean train edit NLL');axes[1,col].set_ylabel('Actual normalized risk');axes[1,col].set_xlabel('Fixed time')
        for ax in axes[:,col]:ax.legend();ax.grid(alpha=.25)
    fig.savefig(root/'trajectory.png',dpi=140,metadata={'Software':'blue_l4_progress_barrier.report'});plt.close(fig)
    curves=rows(root/'curve-common-inventory-summary.csv')
    curve_lookup={(r['entry'],r['arm'],int(r['step']),r['panel'],r['metric']):r for r in curves}
    fig,axes=plt.subplots(3,3,figsize=(14,11),constrained_layout=True)
    for col,e in enumerate(('Early','Middle','Late')):
        for arm in ('H','R','EP'):
            rr=[r for r in curves if r['entry']==e and r['arm']==arm and r['panel']=='Current100' and r['metric']=='RS']
            values=[]
            for r in rr:
                step=int(r['step']);ns=curve_lookup[e,arm,step,'Fixed100','NS'];assert int(ns['denominator'])==200
                ps=curve_lookup[e,arm,step,'Current100','PS']
                actual=next(float(z['risk_actual_fp32']) for z in traj if (z['entry'],z['arm'],int(z['step']))==(e,arm,step))
                values.append((float(r['new_nll_mean']),float(ps['percent']),actual,float(ns['percent'])))
            for row in range(3):axes[row,col].plot([v[row] for v in values],[v[3] for v in values],marker='o',label=arm)
        for row,label in enumerate(('Current rewrite NLL','Current PS (%)','Actual normalized risk')):
            ax=axes[row,col];ax.set_title(e);ax.set_xlabel(label);ax.set_ylabel('Fixed NS (%) [same 200 prompts]');ax.legend();ax.grid(alpha=.25)
    fig.savefig(root/'observed-tradeoff.png',dpi=140,metadata={'Software':'blue_l4_progress_barrier.report'});plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for arm in ('H','EP','EP-Free','EP-J4','EP-N16'):
        rr=[r for r in traj if r['entry']=='Middle' and r['arm']==arm]
        axes[0,0].plot([float(r['time']) for r in rr],[float(r['post_edit_nll']) for r in rr],marker='.',label=arm)
        axes[0,1].plot([float(r['time']) for r in rr],[float(r['risk_actual_fp32']) for r in rr],marker='.',label=arm)
        vv=[r for r in curves if r['entry']=='Middle' and r['arm']==arm and r['panel']=='Current100' and r['metric']=='RS']
        ns=[float(curve_lookup['Middle',arm,int(r['step']),'Fixed100','NS']['percent']) for r in vv]
        ps=[float(curve_lookup['Middle',arm,int(r['step']),'Current100','PS']['percent']) for r in vv]
        axes[1,0].plot([float(r['new_nll_mean']) for r in vv],ns,marker='o',label=arm)
        axes[1,1].plot(ps,ns,marker='o',label=arm)
    for ax,x,y in [(axes[0,0],'Fixed time','Mean train edit NLL'),(axes[0,1],'Fixed time','Actual normalized risk'),(axes[1,0],'Current rewrite NLL','Fixed NS (%) [same 200]'),(axes[1,1],'Current PS (%)','Fixed NS (%) [same 200]')]:
        ax.set_xlabel(x);ax.set_ylabel(y);ax.legend();ax.grid(alpha=.25)
    fig.savefig(root/'middle-auxiliary.png',dpi=140,metadata={'Software':'blue_l4_progress_barrier.report'});plt.close(fig)
    sections=[]
    sections.append('# BLUE L4-only progress-preserving barrier — 최종 진단 보고서\n\n'
      'instruction_id: ODEEDIT-S06-BLUE-L4-PROGRESS-PRESERVING-BARRIER-SH2-V1\n\n'
      '작성: Server2. 주 비교 9 trajectories/72 logical steps, 보조 3/32, 합계 **12/104**. '
      'N은 새 native 실행이 아니라 정확한 기존 N3/WN 평가다(이번 refinement의 t=0 reference). N8/N16은 새 refinement의 Euler step 수이며 기존 N3와 구분한다. 아래 표는 fixed terminal(N8 step8, 보조 N16 step16)이며, 성능으로 endpoint를 선택하지 않았다. scientific_promotion=false.\n\n'
      '## 1. 핵심 절대값 — Current100\n\n'+main_table('Current100'))
    facts=[]
    for e in ('Early','Middle','Late'):
        ep=terminal(e,'EP','Current100','PS');h=terminal(e,'H','Current100','PS');r=terminal(e,'R','Current100','PS')
        facts.append(f"{e}: EP−H PS {float(ep['percent'])-float(h['percent']):+.2f}pp, EP−R PS {float(ep['percent'])-float(r['percent']):+.2f}pp, EP actual risk {risk(e,'EP'):+.6g}")
    sections.append('관측 요약: '+'; '.join(facts)+'. 평균 일차 진척 조건과 실제 유한-step NLL/held-out 성공은 다른 관측이다. 아래 같은-state 및 요청 분포를 함께 읽어야 한다.')
    observed=[]
    for e in ('Early','Middle','Late'):
        for left,right in [('H','N'),('EP','H'),('EP','R')]:
            def delta(panel,metric,field):return float(terminal(e,left,panel,metric)[field])-float(terminal(e,right,panel,metric)[field])
            observed.append([e,left+'−'+right,number(delta('Current100','RS','new_nll_mean')),
               number(delta('Current100','PS','percent')),number(delta('Fixed100','NS','percent')),
               number(delta('Past100','NS','percent')),number(risk(e,left)-risk(e,right))])
    sections.append('### 핵심 비교의 동시 관측\n\n'+md(['entry','비교','Current rewrite NLL Δ','Current PS Δpp','Fixed NS Δpp','Past NS Δpp','actual risk Δ'],observed)+'\n\nNLL Δ는 음수가 낮은 손실, 성공률 Δ는 양수가 높은 선호 성공이다. Risk가 더 낮아도 NS가 같은 방향으로 움직이지 않으면 두 관측을 분리해 해석한다. 아래 수치는 최저 NLL 또는 최상 NS를 고른 결과가 아니라 사전 고정 terminal이다.')
    interpretations=[]
    for e in ('Early','Middle','Late'):
        dn=float(terminal(e,'H','Current100','RS')['new_nll_mean'])-float(terminal(e,'N','Current100','RS')['new_nll_mean'])
        identical=all(terminal(e,'H',p,m)['numerator']==terminal(e,arm,p,m)['numerator'] for arm in ('R','EP') for p in ('Current100','Fixed100','Past100') for m in ('RS','PS','NS'))
        interpretations.append(f"{e}: H−N Current rewrite NLL={dn:+.8g}. "+('H/R/EP의 세 패널 RS/PS/NS 절대 성공 count는 모두 동일하다. 이 entry에서는 barrier 추가의 성공률 이득이 관측되지 않았다.' if identical else 'H/R/EP의 성공 count가 일부 다르다. 위 panel별 delta를 사용하며 이를 단일 개선 점수로 합치지 않는다.'))
    sections.append('**관측에 근거한 해석:** '+' '.join(interpretations)+' 성공률이 같아도 NLL·risk·개별 문항 전환이 같다는 뜻은 아니다. Native 후 refinement의 효과와 barrier가 추가로 만드는 효과를 구분한다.')
    mechanism=[]
    for e in ('Early','Middle','Late'):
        ee=[r for r in traj if r['entry']==e and r['arm']=='EP'];active=sum(float(r['factor'])>0 for r in ee)
        hh=next(r['selected_weight_sha'] for r in run if r['entry']==e and r['arm']=='H')
        eh=next(r['selected_weight_sha'] for r in run if r['entry']==e and r['arm']=='EP')
        mechanism.append(f"{e}: EP 보정 활성 {active}/8, H/EP 최종 weight SHA "+('동일' if hh==eh else '다름')+f", actual cap 초과 {sum(r['actual_fp32_cap_exceeded']=='True' for r in ee)}/8.")
    sections.append('**활성과 실제 endpoint:** '+' '.join(mechanism)+' 비활성 경로에서는 이 입력에서 nominal이 이미 elastic 판정의 허용 방향에 있었다는 해석이 가능하다. 이것은 barrier의 일반적 무용성이나 다른 입력의 안전성 증명이 아니다. 양의 cap 초과/slack도 삭제하지 않았으며 hard safety PASS로 부르지 않는다.')
    if all(terminal(e,'N','Current100','RS')['numerator']=='100' for e in ('Early','Middle','Late')):
        sections.append('**가능한 설명(확정 원인 아님):** 세 N reference의 Current RS가 이미100/100이다. 높은 초기 선호 성공률 때문에 NLL 변화가 성공 count 변화로 이어지지 않을 수 있다. 이 개발 패널의 operating point와 낮은 barrier 활성 빈도를 함께 고려해야 하며, 이를 일반적인 layer capacity 또는 모델 전체 성능의 결론으로 확대하지 않는다.')
    baseline=[]
    for e in ('Early','Middle','Late'):
        for arm in ('W0','ENTRY','N'):
            for panel in ('Current100','Fixed100','Past100'):
                baseline.append([e,arm,panel,*[score(terminal(e,arm,panel,m)) for m in ('RS','PS','NS')]])
    sections.append('### 재사용 W0 / We(ENTRY) / WN(N)\n\n'+md(['entry','state','panel','RS','PS','NS'],baseline)+'\n\n이는 봉인된 기존 Full3900이며 신규 native 또는 W0 평가가 아니다. 동일 reference를 여러 대비에 사용해도 독립 실행/분모를 늘리지 않았다. W0의 낮은 RS는 지정 target-new 선호이며 모델 전체 능력 저하를 뜻하지 않는다.')
    sections.append('## 2. Fixed/Past retention과 locality\n\n### Fixed100\n\n'+main_table('Fixed100')+'\n\n### Past100\n\n'+main_table('Past100')+'\n\nFixed100은 세 entry에서 같은 100 requests다. 이를 독립 300개로 합산하지 않았다. Early/Middle/Late는 history뿐 아니라 Current batch도 다르므로 age-only 인과 비교가 아니다. NS는 해당 문항의 true/new 선호이며 일반 pretrained capability의 보존 증명이 아니다.')
    ranges=[]
    for e in ('Early','Middle','Late'):
        intervals=[]
        for arm in ('H','R','EP'):
            vv=[float(r['new_nll_mean']) for r in curves if r['entry']==e and r['arm']==arm and r['panel']=='Current100' and r['metric']=='RS']
            intervals.append((min(vv),max(vv)));ranges.append([e,arm,number(min(vv)),number(max(vv)),len(vv)])
        lo=max(v[0] for v in intervals);hi=min(v[1] for v in intervals)
        ranges.append([e,'interval intersection',number(lo) if lo<=hi else 'NONE',number(hi) if lo<=hi else 'NONE','range only'])
    sections.append('### 실제 snapshot의 strength 범위\n\n'+md(['entry','arm','관측 rewrite NLL min','max','실제 points'],ranges)+'\n\nRange가 겹치는 것은 동일 strength의 실제 checkpoint가 존재한다는 증명이 아니다. 그림은 실제 4개 snapshot만 연결한 시각적 가이드이며, 교차 구간의 NS를 보간하거나 strength-matched 점수를 만들지 않았다.')
    sections.append('## 3. 같은-state nominal shadow와 matched-risk probe\n\n'+md(['entry','검사','pre-step','EP−대조 mean NLL','median','p90','최대','악화/100','개선/100','일차 mean 차이'],[[r['entry'],r['kind'],r['pre_step'],number(r['mean']),number(r['median']),number(r['p90']),number(r['maximum']),r['worsened'],r['improved'],number(r['first_order_difference'])] for r in probes])+'\n\nNominal shadow는 실제 EP pre-state0/3/7에서 비교한다. pre0은 동일 WN의 H1을 재사용했다. Matched-risk는 동일 WN에서 EP 첫 step과 **일차 risk 감소량**만 맞춘 단발 비교다. 실제 risk 이차항·FP32 적용 차이가 남으므로 완전히 같은 nonlinear risk endpoint라고 부르지 않는다. Probe는 trajectory에 carry하지 않았다.')
    aux=[]
    for arm in ('N','H','R','EP','EP-Free','EP-J4','EP-N16'):
        aux.append([arm,*[score(terminal('Middle',arm,'Current100',m)) for m in ('RS','PS','NS')],number(risk('Middle',arm)),number(terminal('Middle',arm,'Fixed100','NS')['percent']),number(terminal('Middle',arm,'Past100','NS')['percent'])])
    sections.append('## 4. Middle 보조: Free/J4/N16\n\n'+md(['arm','Current RS','Current PS','Current NS','actual risk','Fixed NS%','Past NS%'],aux)+'\n\nFree는 기존 free 성분 삭제, J4는 고정 case-ID hash 4×25 observer, N16은 같은 ν/ε/field에서 h만 절반이다. J4/N16의 좋은 문항만 선택하지 않았으며 추가 threshold나 gain은 없다. 상세 CI는 paired-summary.csv의 각 보조−EP 비교다.')
    wh={r['arm']:r['selected_weight_sha'] for r in run if r['entry']=='Middle'}
    af=[r for r in traj if r['entry']=='Middle' and r['arm']=='EP-Free']
    sections.append(f"Free 보정 활성 {sum(float(r['factor'])>0 for r in af)}/8; H와 최종 weight SHA는 "+('동일하다.' if wh['H']==wh['EP-Free'] else '다르다.')+' 같은 weight와 같은 covariance에서도 별도 프로세스에서 저장한 FP64 reduction scalar의 미세 차이가 있을 수 있다. 동일 weight의 scalar 차이를 물리적 write 변화나 방법 성능 개선으로 해석하지 않는다. 별도 프로세스 전 경로의 bitwise gradient replay는 검증하지 않았다.')
    pm=rows(root/'matched-risk-curve-paired.csv')
    sections.append('### Matched-risk의 held-out 실제 관측\n\n'+md(['entry','panel','metric','EP1−probe Δpp','95% CI','prompts'],[[r['entry'],r['panel'],r['metric'],number(r['delta']),f"[{number(r['ci_low'])},{number(r['ci_high'])}]",r['prompts']] for r in pm if r['field']=='success'])+'\n\nMatched probe risk는 저장 reduced-coordinate risk다. EP의 actual FP32 risk와 동일 필드로 섞지 않는다. Probe의 full-weight FP32 risk는 NOT_RECORDED이며 matched-risk의 동등성 주장은 일차 감소량에 한정한다.')
    contrasts=[r for r in paired if r['metric'] in ('PS','NS') and r['field']=='success' and r['contrast'] in ('EP−H','EP−R','H−N')]
    sections.append('## 5. 요청 단위 paired 통계\n\n'+md(['entry','비교','panel','metric','Δpp','95% CI','requests/prompts','기존성공→실패','기존실패→성공'],[[r['entry'],r['contrast'],r['panel'],r['metric'],number(r['delta']),f"[{number(r['ci_low'])}, {number(r['ci_high'])}]",r['requests']+'/'+r['prompts'],r['success_to_failure'],r['failure_to_success']] for r in contrasts])+'\n\nBootstrap 2000회는 request를 cluster로 재표집하고 rewrite/rephrase/neighborhood 및 두 arm의 대응을 유지한다. 원시 prompt identity를 검증한 뒤 join했다. CI는 이 기존 개발 패널의 조건부 불확실성이며 새 독립 benchmark 일반화 증거가 아니다. NLL mean/median/p90, true와 new NLL, TF strict/token numerator/denominator는 endpoint-and-curve-summary.csv에 별도 기록했다. TF exact와 candidate preference를 혼합하지 않았다.')
    geometry_rows=[]
    for e in ('Early','Middle','Late'):
        for arm in ('H','R','EP'):
            rr=[r for r in traj if r['entry']==e and r['arm']==arm]
            geometry_rows.append([e,arm,number(max(float(r['leakage_max']) for r in rr)),number(np.mean([float(r['q_fraction']) for r in rr if r['q_fraction']])),number(max(float(r['slack']) for r in rr)),
                str(sum(float(r['factor'])>0 for r in rr))+'/'+str(len(rr)),
                number(np.mean([float(r['correction_native_norm'])/float(r['nominal_native_norm']) for r in rr if float(r['nominal_native_norm'])>0])),
                number(sum(float(r['actual_step_energy']) for r in rr)),number(max(float(r['materialization_off_support_norm']) for r in rr))])
    sections.append('## 6. 기하·실제 적용·분포 진단\n\n'+md(['entry','arm','최대 observer leakage','평균 q/q_all','최대 slack','보정 활성 step','평균 correction/nominal norm','Σ 실제 step norm²','최대 off-support rounding norm'],geometry_rows)+'\n\n활성은 저장 factor>0이라는 계산 분기이며 새 과학 threshold가 아니다. H는 설계상 보정0이다. R에는 progress equality가 요구되지 않으므로 R leakage는 구현 오류 판정값이 아니다. q/q_all은 현재 support/observer에서의 local 여력이며 전역 capacity 충분성/부족의 증명이 아니다. Saved Ub의 실제 Gram을 사용했다. 작은 FP32 materialization의 off-support rounding도 숨기지 않았으며 projected actual prediction과 이상적 coefficient prediction을 분리했다. 요청별 gradient는 기존 backward의 L4 per-sequence 관측으로 수집하여 추가 backward0이다; group dense pullback과의 부동소수점 차이는 원 raw step terms에 보존했다.')
    generation=rows(root/'generation-summary.csv')
    sections.append('## 7. 자유 생성과 관측 범위\n\n'+md(['entry','arm','literal prefix','denominator'],[[r['entry'],r['arm'],r['numerator'],r['denominator']] for r in generation])+'\n\n동일 Middle20 rewrite+40 rephrase, greedy/max_new_tokens32 패널이다. 새 H/R/EP180 sequences만 생성했고 기존 N은 재사용 reference다. Literal prefix는 의미 정확도가 아니며 전체 자유 생성 accuracy로 확대하지 않는다. 생성 문자열은 local-only다.')
    sections.append('## 8. 비용·실용성\n\n계획과 실측을 분리한다. 계획은 95 gradient sweeps/34,390 backward/41,692 training forwards, 89,700 새 prompt pairs/179,400 candidate sequences+180 generation이다. 실제 실행별 항목은 compute-summary.csv를 기준으로 한다. Teacher·load·geometry·paired 평가·generation·행렬 계측·저장 비용은 서로 다르다. Scheduler elapsed는 별도 job receipt로 결속한다. 초기 reference를 공유할 때에는 teacher/input/groups 해시가 동일해야 한다.\n\n이것은 **cached-native 뒤의 refinement 비용**이다. 새로운 cold compute-z/native writer/history 비용을 실제로 재측정하지 않았으므로 온라인 전체속도 개선이나 과거 다른 GPU의 native 시간 대비 우월성을 주장하지 않는다. EP의 실용성은 위 CurrentPS/Fixed·Past/NS 변화와 추가 비용을 함께 판단할 조건부 결과다.')
    reconciliation=json.loads((root/'compute-reconciliation.json').read_text());totals=reconciliation['totals']
    sections.append(md(['호출 계측','계획','실제'],[
       ['gradient sweep',95,totals['gradient_sweeps']],
       ['edit+essence backward',34390,totals['edit_backward']+totals['essence_backward']],
       ['training forward (gate 별도)',41692,totals['training_forward']-reconciliation['extra_overlay_gate_forward']],
       ['evaluation candidate sequences',179400,totals['evaluation_candidate_sequences']],
       ['새 generation sequences',180,totals['generation_sequences']],
       ['요청별 관측 추가 backward',0,0]])+'\n\ntraining_tokens/evaluation_input_tokens/generation_output_tokens를 분리한다. actual_model_forward_tokens는 attention-mask 합계이므로 generation의 cached context도 포함한다. 이를 새 input token 수라고 부르지 않는다.')
    sections.append('89,700은 새 **평가 pair-event 수**이며 서로 다른 요청 89,700개라는 뜻이 아니다. 별도 봉인 reference35100 pair-rows를 재사용했고, local request-metrics.csv는 신규+reference124800 state-bound rows다. 상태별 분모만 사용하며 이 행들을 하나의 pooled 성공률 분모로 합치지 않는다. Training 요청별 관측은10400행이다. Margin은 모든 metric에서 true−new로 보존하므로 RS/PS는 양수, NS는 음수일 때 성공한다.')
    ac=rows(root/'per-arm-compute.csv');arm_costs=[]
    for r in run:
        def value(ledger,name):return sum(float(z['value']) for z in ac if z['entry']==r['entry'] and z['arm']==r['arm'] and z['phase']=='arm_increment' and z['ledger']==ledger and z['name']==name)
        arm_costs.append([r['entry'],r['arm'],number(value('counts','gradient_sweeps')),number(value('counts','edit_backward')+value('counts','essence_backward')),
            number(value('counts','training_forward')),number(value('seconds','grouped_objective')),
            number(value('seconds','curve_evaluation')+value('seconds','full_evaluation')),number(value('seconds','generation'))])
    sections.append('### Arm별 실제 추가 비용\n\n'+md(['entry','arm','추가 gradient sweeps','B','training F','objective 초','panel eval 초','generation 초'],arm_costs)+'\n\n각 arm 첫 node 준비 후부터 terminal까지의 누적 ledger 차이다. G0/teacher/load/geometry/matched probe는 shared entry setup에 별도 배정하며 per-arm-compute.csv에 보존했다. 공유된 G0 비용을 숨겨 독립 단독 실행 비용이라고 부르지 않는다. 보조 arm에는 generation을 수행하지 않았으므로 generation0을 controller 속도 개선으로 해석하지 않는다. setup+arm increment의 모든 count 합이 전체 ledger와 정확히 일치하는지 검산했다.')
    sections.append('EP의 추가 training forward에는 pre3/pre7의 same-state shadow 진단도 포함된다. 따라서 위 측정값은 이번 진단 campaign 비용이며, 진단을 제거한 production-only controller latency는 NOT_RECORDED다. Small controller solve와 일부 tensor/SHA 관측은 전체 job overhead 안에 있으며 독립 timing replay를 추가하지 않았다.')
    costs=rows(root/'compute-summary.csv');seconds={}
    for r in costs:
        if r['ledger']=='seconds':seconds[r['name']]=seconds.get(r['name'],0)+float(r['value'])
    sections.append(md(['측정 component','합산 초'],[[k,number(v)] for k,v in sorted(seconds.items())])+'\n\n상기 ledger component를 중첩/누락 없이 전체 GPU elapsed라고 가정하지 않는다. 파일 저장·SHA·matrix/scalar 계측 및 예약/시작 overhead가 별도로 있다.')
    sections.append(md(['run','자원/저장 항목','관측값'],[[Path(r['run']).name,r['name'],r['value']] for r in costs if r['ledger'] in ('memory','storage')])+'\n\nGPU는 allocator peak allocated bytes, host는 프로세스 max RSS KiB다. 노드 전체 사용량이나 예약량과 동일하지 않다. Output bytes는 해당 terminal run 디렉터리 파일 합이며 공통 imported assets는 별도 보존한다.')
    accounting=root/'job-accounting.json'
    if accounting.exists():
        acc=json.loads(accounting.read_text());sections.append('실제 job GPU elapsed 합계: **'+number(acc['total_gpu_hours'])+' GPUh**. 대기시간은 제외했다.\n\n'+md(['job','상태','exit','elapsed초'],[[r['job'],r['state'],r['exit'],r['elapsed_seconds']] for r in acc['jobs']]))
    initial=root/'initial-free-energy.csv'
    if initial.exists():
        sections.append('### 고정 geometry 및 초기 free 성분\n\n'+md(['entry','observer','초기 free-energy 비중','rank'],[[r['entry'],r['observer'],number(r['free_energy_fraction']) if r['free_energy_fraction'] else 'NA',r['rank']] for r in rows(initial)])+'\n\n전체 H_N eigenvalue는 fixed-geometry-spectrum.csv에 CPU FP64 재구성으로 기록하고 runtime GPU min/max와 차이를 별도로 기록했다. U/K/M가 inner 동안 고정이므로 geometry spectrum도 고정이다. Free-energy 비중은 저장 G0로 계산한 entry pre0 값이며 이후 node 비중은 NOT_RECORDED다.')
    sections.append('## 9. 관측·가능한 설명·미분리 가설\n\n'
      '- **H 대 N:** endpoint 표의 H−N은 공통 loss와 저장 native support에서 추가 refinement한 실제 차이다. H가 나쁘면 barrier 효과 이전에 공통 refinement의 operating point/노출을 의심할 수 있으나, 원인을 확정한 실험은 아니다.\n'
      '- **EP 대 H/R:** 같은-state 평균 일차 leakage와 nonlinear request NLL 차이를 분리한다. matched-risk 표에서 EP−probe NLL이 음수인 경우 해당 같은-state 일차 risk 감소량에서는 EP 쪽 실제 edit 손실이 더 작았다. 이것만으로 모든 endpoint의 risk가 완전히 같거나 locality가 보장된다고 하지 않는다.\n'
      '- **평균 보호의 한계:** shadow의 악화 request 수, J4 및 CurrentPS/Fixed/Past를 함께 보아야 한다. J1 mean 조건은 요청별/held-out/과거 fact 보호 조건이 아니다.\n'
      '- **Risk와 locality:** 구조적 covariance risk와 실제 NS true/new NLL을 별도로 측정했다. 방향이 일치하지 않는 경우 proxy와 출력평가의 분리가 관측된 것이며 capacity failure로 바꿔 설명하지 않는다.\n'
      '- **Free/J4/N16:** 동일 field의 삭제/observer/시간격자 보조 비교다. N16은 추가 feedback 비용도 증가한다. 결과가 유사하면 이 세 entry에서 추가 비용 대비 차이가 제한된 관측이지 모든 문제의 충분한 step size 증명은 아니다.\n'
      '- **미검증:** natural sequential 5-batch, 다른 모델/레이어, cold-z 포함 전체비용, 새 독립 benchmark, 보편적 causality/lifelong safety는 수행하지 않았다. 어떤 후속도 자동 제출하지 않는다.\n\n'
      '해석은 이 instruction에 명시된 사용자 예외 범위이며 사실표와 가능한 설명을 분리했다. Scientific promotion은 false다.')
    sections.append('## 10. 정확한 계약·source·복원\n\n'
      '설계 SHA30be08eccc4b74b34acfeb9e3deb5cbffc9e6c4a02768e6f69fddf5593d99a8e, baseline main285d464361c82d033cbaf88c177fd6c5af8f83df. 실행 source/job별 정체성은 run-index.csv와 runtime/input/execution locks에 결속한다. Llama revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FULLFP32, L4 down_proj 단일 weight, 정확한 WN+XUbᵀ에서 시작했다. We teacher, pre-native M, fixed z/K/U/P/M, κ2/ε=.1q_ref/ν entry-once/d_ref1/8, N8/N16 matched time을 유지했다. 새 compute-z/native/history append0.\n\n'
      'actual-output functional/materialized gate는 대표 입력의 exact logits를 검증했고, 전체 모든 prompt의 cross-server bitwise replay라고 주장하지 않는다. Transformers/tokenizers runtime .py/.so1735개는 source와 exact이며 torch/CUDA/GPU actual runtime은 각 job에 별도로 기록했다. 기존 full 평가와 새 평가의 패널/packing/source를 결속하되 미검증 numerical parity는 동일성으로 부풀리지 않았다.\n\n'
      'Entry binding: Early=We B010/current B011, Middle=We B050/current B051, Late=We B090/current B091의 봉인 자산이다. 각 current의 기존 N3/WN을 공통 refinement 시작점으로 사용했다. 세 entry의 history만 바꾸고 Current를 고정한 실험은 아니다.\n\n'
      'Local X snapshots는 0/1/2/4/8(N16 대응)의 실제 저장 값과 Ub/WN reference를 포함한다. X와 reference로 재구성하며 full pretrained 모델을 중복 저장하지 않는다. 이전 ABC source/report 및 원본 raw는 수정하지 않았다. Native scalar J_N의 원 저장 FP32 값을 정규화에 재사용하고 FP64 재검산 차이를 calibration.json에 별도 기록했다.')
    sections.append('## 11. 산출물과 재현\n\n'
      '- `run-index.csv`, `trajectory.csv`, `same-state-probes.csv`, `paired-summary.csv`, `compute-summary.csv`, `endpoint-and-curve-summary.csv`.\n'
      '- 요청별 raw-free numeric CSV: request-metrics-reference.json의 local path/SHA. 원 prompt/generation/tensor는 Git 제외.\n'
      '- 실제 입력/출력 전수 SHA: raw-input-manifest.json. 소형 publication closure: analysis-manifest.json/rooted-receipt.json.\n'
      '- 그림은 아래 Python report code로 생성했으며 측정 snapshot 점만 연결한다. 보간된 endpoint/NS strength matching은 없다.\n\n'
      '```bash\npython -m project.run_scripts.blue_l4_progress_barrier.report --report <canonical_report_root>\n```\n\n'
      '재생성은 같은 집계 CSV를 복사한 새 디렉터리에서 실행한다(create-once). PNG source: trajectory.csv / curve-common-inventory-summary.csv. Codex imagegen/visualization 사용0.\n\n'
      '![실제 trajectory](trajectory.png)\n\n![관측 snapshot trade-off](observed-tradeoff.png)\n\n![Middle 보조 비교](middle-auxiliary.png)\n\n'
      'Raw preservation root: `/mnt/raid5/janghj/ODE-edit/local/blue-l4-progress-barrier/attempt-v1/`. Raw broadcast: **NO_BROADCAST_NOT_REQUIRED** — 이번 승인 예외에 따라 동일 프로젝트 보존 경로와 manifest만 공유. 다른 task의 monitoring pause/실행 변경0.')
    with (root/'diagnostic-report-ko.md').open('x') as f:f.write('\n\n'.join(sections)+'\n')
    members=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.iterdir()) if p.is_file()]
    save(root/'analysis-manifest.json',dict(members=members,status='PUBLICATION_SEALED',scientific_promotion=False))
    save(root/'rooted-receipt.json',dict(manifest_sha=sha(root/'analysis-manifest.json'),report_sha=sha(root/'diagnostic-report-ko.md'),
          source_input_lock_sha=sha(DEST/'input.lock.json'),raw_free=True,scientific_promotion=False))
    print('REPORT_SEALED',sha(root/'diagnostic-report-ko.md'))

if __name__=='__main__':main()
