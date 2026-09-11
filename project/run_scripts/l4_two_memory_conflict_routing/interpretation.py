"""Descriptive, data-bound answers; no outcome-dependent execution decisions."""
import csv
from pathlib import Path

def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def findings(root):
    endpoint=read(root/'endpoint-summary.csv');bank=read(root/'base-preservation-versus-recovery.csv')
    context=read(root/'context-response-summary.csv');paths=read(root/'physical-path-actions.csv')
    ep={(r['entry'],r['batch_raw'],r['arm'],r['panel'],r['metric']):r for r in endpoint if r['reference']=='N'}
    harm={(r['entry'],r['batch_raw'],r['arm'],r['bank']):float(r['controller_value']) for r in bank if r.get('controller_value','')!=''}
    ns={(r['entry'],r['batch_raw'],r['arm'],r['bank']):r for r in bank if r.get('metric')=='NS'}
    entries=('Early','Middle','Late');answers=[]
    def change(e,left,right,panel='Current100',b='100'):
        return ', '.join(f"{m} {float(ep[e,b,right,panel,m]['rate'])-float(ep[e,b,left,panel,m]['rate']):+.3f}pp" for m in ('RS','PS','NS'))
    def fchange(e,left,right,b='100'):
        return '; '.join(f"{k} {harm[e,b,left,k]:.7g}→{harm[e,b,right,k]:.7g}" for k in ('Past','Base','BaseAudit'))
    answers.append(dict(question='OS는 Native보다 Current를 유지하면서 보호했는가?',
        observations=[f"{e}: {change(e,'N','OS')}; {fchange(e,'N','OS')}" for e in entries],
        interpretation='Current rewrite success는 유지되지만 세 entry의 Base/Past functional harm은 Native보다 크다. 구조 목적식 개선을 기능적 보호 성공으로 부를 수 없다.'))
    arr=[]
    for e in entries:
        r=ep[e,'100','OS','Current100','RS'];p=ep[e,'100','OS','Current100','PS']
        c=next(x for x in context if x['entry']==e and x['batch_raw']=='100' and x['endpoint']=='OS' and x['reference']=='native')
        arr.append(f"{e}: N 대비 new NLL 악화 rewrite {r['new_nll_higher']}/{r['denominator']}, rephrase {p['new_nll_higher']}/{p['denominator']}; 목표 수직 response² type-weighted mean {float(c['perpendicular_sq_type_weighted_mean']):.7g}")
    answers.append(dict(question='Scalar 진척이 같아도 개별 반응/NLL은 달라지는가?',observations=arr,
        interpretation='실제 weight의 scalar residual과 개별 prompt NLL은 다른 조건이다. 좋은 평균 NLL도 모든 요청의 비악화를 의미하지 않는다.'))
    answers.append(dict(question='BF1의 실질적 추가 가치는?',
        observations=[f"{e}: OS→BF1 {change(e,'OS','BF1')}; {fchange(e,'OS','BF1')}" for e in entries],
        interpretation='관측한 controller-bank harm 변화와 audit 전이를 분리한다. 이 고정 실험에서 작은 bank 개선만으로 전체 edit/locality 우월성을 확정하지 않는다.'))
    answers.append(dict(question='BF8이 BF1보다 추가 비용을 정당화하는가?',
        observations=[f"{e}: BF1→BF8 {change(e,'BF1','BF8')}; {fchange(e,'BF1','BF8')}" for e in entries],
        interpretation='BF8은 entry당1536개의 추가 functional backward와 중간 평가를 사용한다. 성공률/보호 이득을 이 추가비용과 함께 판단해야 하며, 미세 bank 차이를 실용적 우월성으로 확대하지 않는다.'))
    answers.append(dict(question='Frozen 대비 방향 갱신의 이득은?',
        observations=[f"Middle: Frozen→BF8 {change('Middle','Frozen-BF8','BF8')}; {fchange('Middle','Frozen-BF8','BF8')}"],
        interpretation='공통 h/anchor/calibration에서 방향 refresh 차이다. 새로운1536 backward의 비용은 추가되며, 한 entry에서의 작은 차이를 다른 상태의 일반 법칙으로 확장하지 않는다.'))
    arr=[]
    for e in entries:
        for panel in ('Fixed100','Past100'):arr.append(f"{e} {panel}, OS→BF8: {change(e,'OS','BF8',panel)}")
        a=ns[e,'100','OS','BaseAudit'];b=ns[e,'100','BF8','BaseAudit']
        arr.append(f"{e} BaseAudit NS OS {a['numerator']}/{a['denominator']}→BF8 {b['numerator']}/{b['denominator']}")
    answers.append(dict(question='Bank 개선은 held-out 보호로 이어지는가?',observations=arr,
        interpretation='Fixed/Past retention과 candidate-disjoint BaseAudit에서 관측된 변화만 전이라고 부른다. Bank 향상이 전이되지 않으면 bank 적합이라는 설명과 양립한다.'))
    arr=[]
    for e in entries:
        pp=[r for r in paths if r['entry']==e and r['batch_raw']=='100' and r['arm']=='BF8']
        r=max(pp,key=lambda x:int(x['node']))
        arr.append(f"{e} BF8: end ||Z||F²={float(r['cumulative_Z_frob_sq']):.7g}, Σ||C||F²={float(r['sum_correction_frob_sq']):.7g}, Σ||C||H²/h={float(r['sum_action_H_over_h']):.7g}, physical net²={float(r['actual_net_from_entry_sq']):.7g}")
    answers.append(dict(question='보정/수직 성분/slack는 어떻게 읽는가?',observations=arr,
        interpretation='End Z, step action 합, 실제 net weight 차이는 구분했다. Slack와 nonlinear prediction error는 관측이며 hard safety certificate가 아니다. 수정 후 비음수 slack만 canonical에 포함한다.'))
    answers.append(dict(question='We Base 보존과 W0 복구는 같은가?',
        observations=[f"{e} BF8: We KL={harm[e,'100','BF8','Base']:.7g}, W0 KL={harm[e,'100','BF8','BaseW0']:.7g}; Base NS={ns[e,'100','BF8','Base']['numerator']}/{ns[e,'100','BF8','Base']['denominator']}" for e in entries],
        interpretation='서로 다른 teacher의 KL 절대값은 직접 같은 목적 달성률이 아니다. Preservation/recovery paired 표의 We−W0와 endpoint−We, 정답 NLL/NS를 각각 사용한다.'))
    arr=[]
    for b in ('1','7'):
        panel='Current100'
        # The inherited panel label stays Current100; its row denominator is B.
        for a in ('N','OS','BF1'):
            parts=[]
            for m in ('RS','PS','NS'):
                r=ep['Middle',b,a,panel,m];parts.append(f"{m} {r['numerator']}/{r['denominator']}")
            arr.append(f"B{b} {a}: "+', '.join(parts)+f"; BaseKL={harm['Middle',b,a,'Base']:.7g}")
    answers.append(dict(question='B1/B7는 무엇을 보여주는가?',observations=arr,
        interpretation='해당 B의 별도 native solve와 실제 분모를 사용했다. 실행 가능한 가변 B 구현이지 모든 B에서 충분한 보호/편집 용량의 증명이 아니다. B별 요청 구성도 달라 batch-size 인과 효과로 해석하지 않는다.'))
    return answers
