"""Measured, raw-free stage discussion; no automatic scientific winner selection."""
import argparse
import csv
import json
from pathlib import Path
from .report import table,rate
from .records import save
from .import_assets import sha

def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))
def num(x):return f'{float(x):.6g}' if x not in [None,''] else 'NOT_RECORDED'

def build(package,completion):
    stage=completion['stage'];rows=read_csv(package/'paired-summary.csv')
    rows=[r for r in rows if r.get('reference','') in ['', 'N']]
    lookup={(r['entry'],r['endpoint'],r['panel'],r['metric'],r['resolution']):r for r in rows}
    lines=[f'# 상위 {stage} diagnostic discussion','',
      '아래 해석은 이번에 관측한 checkpoint와 request에 한정된다. 결과를 본 뒤 만든 성능 통과 기준, arm 삭제, 보간 또는 scientific promotion은 없다.','']
    if stage=='A':
        lines+=['## 직접 최적화와 native-assisted support 비교','',
          'Direct-B는 native solve의 column support, Direct-C는 저장 P의 전체 orthobasis를 사용한다. Alpha와 eta는 native delta norm에 의존하므로 native-free 실험이 아니다. Middle에서 training objective만으로 선택한 alpha를 Early/Late에 옮겼다.','']
        data=[]
        for entry in ['Early','Middle','Late']:
            for method in ['N','B','C']:
                endpoint='N-full' if method=='N' else f"{method}-alpha-{completion['selections'][method]['alpha']}/eval-032"
                current=[lookup[(entry,endpoint,'Current100',m,'full')] for m in ['RS','PS','NS']]
                past=lookup[(entry,endpoint,'Past100','RS','full')]
                fixed=lookup[(entry,endpoint,'Fixed100','RS','full')]
                data.append([entry,method,*[rate(r) for r in current],num(current[0]['new_nll_mean']),
                    num(current[1]['new_nll_mean']),num(current[1]['new_nll_p90']),rate(fixed),rate(past),
                    f"{past['loss']}/{past['entry_success_denominator']}",
                    f"{past['recovery']}/{past['entry_failure_denominator']}"])
        lines += [table(['entry','method','Current RS','Current PS','Current NS','rewrite new NLL','rephrase new NLL','rephrase new p90','Fixed RS','Past RS','Past loss/entry-success','Past recovery/entry-failure'],data),'',
          '이 표의 loss/recovery는 We 대비이며, W0 대비 inherited degradation과 구분한다. Native와 Direct의 격차는 target fitting, support, regularization 및 update 크기가 함께 바뀐 결과다. 동일 RS라고 동일 NLL/PS/NS인 것도 아니다.','',
          '## 실제 strength 범위 — 외삽·보간 없음','']
        trajectory=read_csv(package/'trajectory.csv');data=[]
        for entry in ['Early','Middle','Late']:
            for family in ['N','B','C']:
                endpoints={r['endpoint'] for r in rows if r['entry']==entry and
                           (r['endpoint']=='N-full' or r['endpoint'].startswith('native-scale-') if family=='N' else r['endpoint'].startswith(family+'-alpha-'))}
                for metric in ['RS','PS']:
                    observed=[r for r in rows if r['entry']==entry and r['endpoint'] in endpoints and r['panel']=='Current100' and r['metric']==metric and r['resolution']=='curve']
                    if observed:data.append([entry,family,metric,len(observed),num(min(float(r['rate']) for r in observed)*100),num(max(float(r['rate']) for r in observed)*100),
                        num(min(float(r['new_nll_mean']) for r in observed)),num(max(float(r['new_nll_mean']) for r in observed))])
        lines += [table(['entry','family','metric','observed points','min success %','max success %','min new NLL','max new NLL'],data),'',
          '위 범위의 중첩은 정확한 strength-matched endpoint를 보장하지 않는다. 모든 관측점과 metric을 공개하며, NS가 좋아 보이는 점을 operating point로 선택하지 않는다. Curve neighbor는 2/request이고 Full NS 10/request와 합산하지 않는다.','',
          '## 최적화 진행과 native anchor','']
        data=[]
        for entry in ['Early','Middle','Late']:
            for support in ['B','C']:
                for endpoint in sorted({r['endpoint'] for r in trajectory if r['entry']==entry and r['endpoint'].startswith(support+'-alpha-')}):
                    points={int(r['step']):r for r in trajectory if r['entry']==entry and r['endpoint']==endpoint}
                    if 32 not in points:continue
                    first=points.get(1,{});last=points[32]
                    data.append([entry,endpoint,num(first.get('edit_nll')),num(last.get('edit_nll')),num(last.get('objective')),
                         num(last.get('normalized_native_action')),num(last.get('essence_kl_unweighted')),num(last.get('batch_net_norm')),num(last.get('path_length'))])
        lines += [table(['entry','candidate','step1 train NLL','step32 train NLL','step32 objective','native action ratio','essence KL','batch net norm','path length'],data),'',
          'Train 값은 post-update state에서 측정한다. Native/common observation은 optimizer가 없고 We teacher로 같은 six-context NLL/penalty/KL을 읽었다. 첫 gradient와 업데이트 후32step finite endpoint를 별도로 보존한다. 경로 길이의 합을 net weight norm으로 해석하지 않는다.','']
    elif stage=='B':
        lines += ['## 방향·amplitude의 관측 비교','',
          'GF−/GF+ 부호쌍은 공통 WN에서 시작하고, LF−는 batch-local contraction 설명을, Random1/2는 방향 특이성을 비교한다. OP−와 COV−는 서로 다른 위험함수이며 크기는 같은 native Frobenius 단위로 맞췄다. 수치적으로 unresolved/zero인 방향은 증폭하지 않는다.','']
        data=[]
        for r in rows:
            if r['resolution']=='full' and r['panel']=='Current100' and r['metric']=='RS':
                ns=lookup[(r['entry'],r['endpoint'],'Past100','NS','full')]
                past=lookup[(r['entry'],r['endpoint'],'Past100','RS','full')]
                data.append([r['entry'],r['endpoint'],rate(r),num(r['new_nll_delta_mean']),rate(past),rate(ns),num(ns['additional_margin_mean'])])
        lines += [table(['entry','predeclared .1 endpoint','Current RS','Current new NLL delta vs N','Past RS','Past NS','Past NS margin delta vs N'],data),'',
          '추가 방향이 현재 성능을 유지하지 못해도 관측값을 제외하지 않는다. Risk 1차항·2차항은 실제 FP32 delta로, intended delta는 별도 열로 기록한다. GF sign-pair 중심차분은 유한 amplitude의 대칭 관측이며 해석적 local derivative의 정확값은 아니다.','']
    else:
        lines += ['## 반복 feedback과 고정 방향','',
          '모든 arm은 같은 WN, nominal gradient refresh, fresh momentum을 공유한다. FrozenGlobal↔RefreshedGlobal 비교는 위험 gradient와 Current J의 공동 refresh이며 J 단독효과로 분리되지 않는다. RefreshedLocal은 reference가 We, Global은 W0다. SoftGlobal은 최초 한 번 norm 보정한 penalty이므로 매step fixed-length correction과 크기 궤적이 다를 수 있다.','']
        data=[]
        for r in rows:
            if r['resolution']=='full' and r['panel']=='Current100' and r['metric']=='RS':
                ps=lookup[(r['entry'],r['endpoint'],'Current100','PS','full')]
                past=lookup[(r['entry'],r['endpoint'],'Past100','RS','full')]
                ns=lookup[(r['entry'],r['endpoint'],'Past100','NS','full')]
                data.append([r['endpoint'],rate(r),rate(ps),num(r['new_nll_delta_mean']),rate(past),rate(ns)])
        lines += [table(['endpoint','Current RS','Current PS','Current new NLL delta vs N','Past RS','Past NS'],data),'',
          'Step별 실제 correction norm과 path를 함께 보며 성능 차이를 방향 정보만의 효과로 단정하지 않는다. 작은 correction, 음성 결과, finite defect 모두 같은 분모에 남는다. Classical CBF, monotonicity, 안전성 certificate를 주장하지 않는다.','']
    compute=read_csv(package/'auxiliary/compute-summary.csv')
    lines += ['## 실제 계산량','',table(['entry','process','actual forwards','backwards','training/observation sequences','objective wall s','eval wall s','peak allocated bytes'],[
       [r['entry'],r['unit'],r.get('counts.actual_model_forward_invocations','NOT_RECORDED'),r.get('counts.backward','NOT_RECORDED'),
        r.get('counts.training_sequences','NOT_RECORDED'),r.get('seconds.direct_objective','NOT_RECORDED'),
        r.get('seconds.full_evaluation','NOT_RECORDED'),r.get('peak_gpu_bytes','NOT_RECORDED')]
       for r in compute if r['scope']=='PROCESS_TOTAL_DO_NOT_SUM_WITH_CHILDREN' and r.get('execution_role')!='REUSED_A_REFERENCE']),'',
      'Process total만 합산 가능하다. Child cumulative ledger는 같은 process 내부 차분이며 process total과 다시 더하지 않는다. FLOPs는 측정하지 않아 NOT_RECORDED로 둔다. Native z는 정확한 sealed cache hit이며 cold compute-z 비용을 포함한 속도 비교가 아니다. GPU allocation의 실제 비용은 job ledger를 따로 본다.','']
    if stage=='A':
        lines += ['## 기록·실행 경계의 공개','',
          '- Middle Direct-B(r1)와 Direct-C(r2)는 같은 full logical objective지만 gradient accumulation의 FP32 reduction order가 다르다. 원래 유효 결과를 재실행하지 않았고 source identity를 분리한다.',
          '- Middle의 NumPy C0 diagnostic promotion을 발견하여 Torch FP32 covariance readout으로 37개 저장 state의 algebra-only 구조 관측을 보충했다. Target/writer/optimizer/선택에는 영향이 없다. 원래 raw는 보존한다.',
          '- Native/Early/Late W0 기준 평가는 cross-entry 일부 row가 중복 실행됐다. baseline-reuse receipt에 exact 수량/수치차를 기록한다. 성공분모 추가·평균·값 교체는 하지 않으며 중복 계산 비용을 숨기지 않는다.',
          '- R1/R2 cached-generation token counter는 attention-visible history이고 실제 query token 수가 아니다. 이후 source는 query/visible/padded count를 분리했다. 과거 token counter를 FLOP으로 변환하지 않는다.','']
    lines += ['## 해석 한계와 다음 질문','',
      '세 entry는 서로 다른 Current cohort와 history를 가지므로 시간 경과만의 인과효과가 아니다. Fixed/Past의 exact subject/relation conflict 및 neighbor overlap은 metadata strata로 공개하되 제거하지 않는다. 주된 bootstrap은 request-cluster2000회, subject/relation cluster는 보조이며 optimizer seed/order population uncertainty가 아니다.','',
      '가장 중요한 후속 질문: 직접 최적화의 현재 품질과 실제 추가 write 크기를 함께 고려했을 때, 누적위험 방향의 refresh가 단순 contraction보다 재현 가능한 과거 편집 보존 이득을 남기는가? 이번 관측만으로 분리되지 않는 항목은 다음 실험의 사실로 가장하지 않는다.','',
      'scientific_promotion=false.','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--completion',type=Path,required=True)
    a=p.parse_args();completion=json.loads(a.completion.read_text())
    if completion['status']!='COMPLETE':raise RuntimeError('INCOMPLETE_STAGE')
    path=a.package/'diagnostic-discussion-ko.md'
    with path.open('x') as f:f.write(build(a.package,completion))
    save(a.package/'discussion-receipt.json',dict(path=path.name,sha256=sha(path),
         inputs={str(p.relative_to(a.package)):sha(p) for p in [a.package/'paired-summary.csv',a.package/'trajectory.csv',a.package/'auxiliary/compute-summary.csv']},
         performance_selection=0,scientific_promotion=False))

if __name__=='__main__':main()
