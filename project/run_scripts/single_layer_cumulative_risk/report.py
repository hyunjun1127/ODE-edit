"""Korean factual report from measured tables; no model or external reviewer."""
import argparse
import csv
import hashlib
import json
import stat
import subprocess
from pathlib import Path
from .records import save,digest

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def table(headers,rows):
    if any(len(row)!=len(headers) for row in rows):raise ValueError('TABLE_COLUMN_COUNT_MISMATCH')
    def cell(x):return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(cell(x) for x in row)+' |' for row in rows])

def rate(row):return f"{row['numerator']}/{row['denominator']} ({100*float(row['rate']):.2f}%)"

def endpoint_order(endpoint):
    for index,prefix in enumerate(['W0','ENTRY','N-full','N_REUSED','B-alpha','C-alpha']):
        if endpoint.startswith(prefix):return index,endpoint
    return 6,endpoint

def build_a(package,completion):
    with (package/'paired-summary.csv').open() as f:rows=list(csv.DictReader(f))
    lines=['# 단일 layer 누적위험 진단 — A 사실 보고서','',
      f"상태: {completion['status']}. writers {completion['writers']}/13, logical full-batch steps {completion['fullbatch_steps']}/320, 추가 native scaling endpoints {completion['native_scales']}/12.",'',
      '이 문서는 Llama L4 단일 weight의 세 historical entry를 이용한 진단이다. 새 lifelong run, 안전성 보장, 인과적 locality 개선 또는 scientific promotion을 의미하지 않는다. 상위 B/C는 이 A의 Direct-B/Direct-C support와 다른 실험 단계다.','',
      '## 읽는 법과 분모','',
      '- W0는 원본 모델, ENTRY(We)는 현재 B100을 편집하기 전 historical checkpoint다. 둘을 같은 pre-edit로 부르지 않는다.',
      '- RS/PS는 target-new NLL < target-true NLL, NS는 target-true NLL < target-new NLL이다. Tie는 실패다. NLL은 target token별 평균이며 raw sequence joint NLL/확률과 구분한다. 낮을수록 해당 target의 token 평균 likelihood가 높다.',
      '- margin=true−new. NS에서 양의 margin 변화는 새 competing target 방향이다. inherited=ENTRY−W0, additional=post−ENTRY를 같은 prompt에서 계산한다.',
      '- Full은 panel마다 RS100/PS200/NS1000, 세 panel 합계3900쌍이다. Curve는 RS300/CurrentPS200/고정 neighbor600, 합계1100쌍이며 full NS로 부르지 않는다.',
      '- Teacher-forced exact는 모든 target token top-1 일치이며 자유 생성 의미적 정확도와 다르다.',
      '- 신뢰구간은 request별 prompt 묶음의 paired bootstrap 2000회다. Edit order/seed 변동의 신뢰구간이 아니다.','',
      '## 모든 full endpoint 성능','']
    for entry in ['Early','Middle','Late']:
        lines += [f'### {entry}','']
        selected=[r for r in rows if r['entry']==entry and r['resolution']=='full']
        lookup={(r['endpoint'],r['panel'],r['metric']):r for r in selected}
        data=[]
        for endpoint in sorted({r['endpoint'] for r in selected},key=endpoint_order):
            for panel in ['Current100','Fixed100','Past100']:
                metrics=[lookup.get((endpoint,panel,m)) for m in ['RS','PS','NS']]
                if not all(metrics):continue
                data.append([endpoint,panel]+[rate(r) for r in metrics]+
                     [f"{r['strict_numerator']}/{r['strict_denominator']}" for r in metrics])
        lines += [table(['endpoint','panel','RS','PS','NS','rewrite TF exact','rephrase TF exact','neighbor true TF exact'],data),'']
    lines += ['## NLL 및 손실/회복 — 전체 full endpoint','']
    data=[]
    for r in rows:
        if r['resolution']!='full':continue
        data.append([r['entry'],r['endpoint'],r['panel'],r['metric'],
           '/'.join(f"{float(r['new_nll_'+k]):.5f}" for k in ['mean','median','p90','max']),
           '/'.join(f"{float(r['true_nll_'+k]):.5f}" for k in ['mean','median','p90','max']),
           f"{r['loss']}/{r['entry_success_denominator']}",f"{r['recovery']}/{r['entry_failure_denominator']}"])
    lines += [table(['entry','endpoint','panel','metric','new mean/median/p90/max','true mean/median/p90/max','entry 성공→실패','entry 실패→성공'],data),'',
       '전체 분모는 성능이나 기존 실패 여부로 제외하지 않았다. Conditional loss의 분모는 entry 성공 수이며, 전체 request/prompt 분모와 다르다.','',
       '## 학습률 선택과 실행 경계','',
       '각 support는 Middle의 실제 finite W32 endpoint가 있는 후보 중 post-update W29..W32 공통 training objective 평균으로 alpha를 선택했다. PS/NS는 선택 입력이 아니다. Early/Late는 이 alpha를 재사용하되 각 entry의 native norm/초기 gradient로 eta를 한 번 계산한다.','',
       table(['support','selected alpha','training scores'],[[s,x['alpha'],json.dumps(x['scores'],ensure_ascii=False)] for s,x in completion.get('selections',{}).items()]),'',
       '목적함수는 context/request/target-token 평균 NLL + 0.1 native-action ratio + 0.0625 KL(student || frozen entry teacher)다. Canonical rewrite NLL 표는 이 six-context 전체 training objective와 구분한다.','',
       '## 그림과 재현','']
    for path in sorted(package.rglob('*.png')):
        lines += [f'![{path.stem}]({path.relative_to(package)})','']
    lines += ['모든 PNG는 저장소 plotting.py를 실행하여 생성한다. 입력/명령/환경/출력 SHA 및 동일 입력 두 번 렌더링의 byte 일치는 figures/plot-reproduction.json에 기록한다. 기존 output을 덮어쓰지 않고 재현 시 새 output directory를 사용한다.','',
       '## 해석과 제한','',
       '- Early/Middle/Late는 다음 B100이 서로 다르므로 age만의 인과효과로 읽지 않는다.',
       '- Native z cache hit 비용과 direct training 비용을 cold-native online speed 비교로 해석하지 않는다. 모델 load, preparation, evaluation, generation과 optimizer 시간을 분리한다.',
       '- Norm의 합은 net norm이 아니며 native metric action은 Frobenius 제곱합이 아니다. 위험량 감소와 실제 Past/Fixed 성능 변화는 함께 해석한다.',
       '- Curve에 없는 Past/Fixed PS는 미측정이다. 보간값이나 생성 점수로 채우지 않는다. Full endpoint에서 curve에 해당하는 실제 행만 추출한 경우 measurement_reuse=true이며 별도 model 평가나 추가 독립 분모가 아니다.',
       '- 그림의 marker는 W0=diamond, ENTRY=filled plus, Native=X, native scaling=square, Direct-B=circle, Direct-C=triangle이다. 각 점의 exact endpoint와 해상도는 CSV에 결속한다.',
       '- 손실 증가, risk 증가, 작은 gradient, strength 범위의 비중첩은 정상 관측이며 결과 제외나 다음 단계 차단 근거가 아니다.',
       '- A만으로 누적위험 방향이나 refresh의 정보 가치를 결론내리지 않는다. 상위 B/C의 공통 native 출발점 비교가 아직 별도로 필요하다.',
       '- 본 자동 factual table은 별도의 diagnostic discussion, direction/compute/metadata tables 및 requirements-evidence와 함께 읽는다. 없는 필드를 기록된 것으로 간주하지 않는다.','',
       'Scientific promotion=false.','']
    return '\n'.join(lines)

def build_later(package,completion):
    stage=completion['stage']
    with (package/'paired-summary.csv').open() as f:rows=list(csv.DictReader(f))
    lines=[f'# 단일 layer 누적위험 진단 — 상위 {stage} 사실 보고서','',
       f"상태: {completion['status']}; trials={completion['trials']}, nominal optimizer steps={completion['fullbatch_steps']}.",'',
       'W0(원본), We(현재 batch 직전 historical checkpoint), WN(같은 entry의 native endpoint)를 구분한다. 모든 비교는 동일 request/prompt identity로 결속한다. ENTRY 대비 변화와 N 대비 변화는 별도 행이며 두 reference를 합산하여 분모를 늘리지 않는다.','',
       'RS/PS=new NLL<true NLL; NS=true NLL<new NLL; tie 실패. Full은 panel별 RS100/PS200/NS1000, 총3900쌍; curve는 RS300/CurrentPS200/neighbor600, 총1100쌍이다.','',
       '상위 B는 Direct-B support가 아니며, 상위 C도 Direct-C support와 다른 단계다. 낮은 성능·risk 증가·0 또는 미해결 방향은 결과에 남기고 자동 성공/안전성 판정에 사용하지 않는다.','']
    if stage=='B':
        lines+=['## 방향 및 amplitude 계약','',
          'A에서 저장한 공통 WN에서 Current100을 metadata-only 10개 group으로 나누고 context-averaged NLL gradient를 한 번 계산한다. 정규화된 row/sqrt(10)로 J를 만든다. Sγ(g)=g−Jᵀ(JJᵀ+0.1I)⁻¹Jg는 soft filtering이며 Jd=0을 보장하지 않는다.','',
          'GF−/GF+/LF−/Random1/Random2/OP−/COV− 각각 native update Frobenius norm의 0.03/0.1/0.3으로 관측한다. GF+는 GF−의 정확한 반대다. Actual FP32 norm은 intended norm과 분리한다. 0 또는 수치적으로 미해결 방향을 강제로 증폭하지 않는다. N은 A의 측정값을 재사용하며 신규 trial로 중복 집계하지 않는다.','',
          '모든 trial의 curve를 관측하고, 미리 정한 amplitude0.1만 full로 관측한다. Full에서 포함된 curve rows를 다시 평가하지 않는다. Native scaling과 LF− 비교는 현재 update 축소 효과를 구분하는 보조 증거다.','',
          'GF 부호쌍의 중심차분과 curvature는 GF− 방향의 intended physical epsilon 기준이다. Actual rounded update norm과 함께 해석한다. RS/NS의 미분 또는 단일 amplitude로 일반적인 인과효과를 주장하지 않는다.','']
    else:
        lines+=['## Frozen/refresh 계약','',
          'Middle의 동일 WN에서 Continue/FrozenGlobal/RefreshedGlobal/RefreshedLocal/SoftGlobal을 각각 8 step 실행한다. We/M/essence teacher는 고정하고, 모든 arm의 nominal direct gradient와 momentum은 현재 상태에서 갱신한다. Momentum은 WN에서 새로 시작하며 eta는 A의 train-only 선택 Direct-C 값을 재사용한다.','',
          'FrozenGlobal은 처음 global filtered direction을 고정한다. RefreshedGlobal/Local은 각각 W0/We reference에서 risk gradient와 Current J를 함께 갱신한다. 따라서 차이를 J 하나만의 단일요인 인과효과로 읽지 않는다. Correction은 nominal momentum 밖에서 적용한다.','',
          'SoftGlobal의 penalty coefficient는 첫 correction의 물리 norm을 맞춰 한 번 계산하며 이후 재조정하지 않는다. Actual correction norm은 최종 FP32 state와 nominal-only FP32 state의 차이이고, intended coefficient-space correction norm과 다르다.','',
          'Step2/4는 curve, step8은 full 및 literal generation 관측이다. 미측정 Past/Fixed PS를 보간하지 않는다.','']
    lines+=['## Full endpoint — 실제 전체 분모','']
    for entry in ['Early','Middle','Late']:
        selected=[r for r in rows if r['entry']==entry and r['resolution']=='full' and r['reference']=='N']
        if not selected:continue
        lookup={(r['endpoint'],r['panel'],r['metric']):r for r in selected};data=[]
        for endpoint in sorted({r['endpoint'] for r in selected},key=endpoint_order):
            for panel in ['Current100','Fixed100','Past100']:
                metrics=[lookup.get((endpoint,panel,m)) for m in ['RS','PS','NS']]
                if all(metrics):data.append([endpoint,panel]+[rate(r) for r in metrics]+
                     [f"{r['strict_numerator']}/{r['strict_denominator']}" for r in metrics])
        lines += [f'### {entry}','',table(['endpoint','panel','RS','PS','NS','rewrite TF exact','rephrase TF exact','neighbor true TF exact'],data),'']
    lines+=['## Paired NLL와 loss/recovery','']
    data=[]
    for r in rows:
        if r['resolution']!='full':continue
        data.append([r['entry'],r['endpoint'],r['reference'],r['panel'],r['metric'],
             '/'.join(f"{float(r['new_nll_'+k]):.5f}" for k in ['mean','median','p90','max']),
             '/'.join(f"{float(r['true_nll_'+k]):.5f}" for k in ['mean','median','p90','max']),
             f"{r['loss']}/{r['entry_success_denominator']}",f"{r['recovery']}/{r['entry_failure_denominator']}"])
    lines += [table(['entry','endpoint','reference','panel','metric','new mean/median/p90/max','true mean/median/p90/max','성공→실패','실패→성공'],data),'',
         'Conditional loss의 분모와 all-prompt 분모를 혼동하지 않는다. Bootstrap은 request-cluster 2000회이며 edit order/seed 분포를 추정하지 않는다. Subject/relation과 overwrite strata는 exact metadata 기반 보조 분석이고 제외 기준이 아니다.','',
         '## 그림·계산량·한계','']
    for path in sorted(package.rglob('*.png')):lines += [f'![{path.stem}]({path.relative_to(package)})','']
    lines += ['PNG는 repository Python code로 생성하며 각 plot-reproduction receipt가 입력/명령/환경/SHA와 실제 byte 재현을 결속한다. Curve/full 해상도를 합산하지 않는다.','',
        '계산량은 actual forward/backward/sequence/token counts와 wall time이다. FLOPs는 측정하지 않았으며 시간에서 역산하지 않는다. Process total과 누적 child ledger를 중복 더하지 않는다.','',
        'Norm 합은 net norm이 아니고, Frobenius risk와 native history/L2 action은 서로 다른 양이다. Risk의 변화만으로 locality/retention 개선을 단정하지 않는다. Development checkpoint 세 개를 독립 lifelong benchmark로 부르지 않는다.','',
        '상세 해석, 미분리 원인, 방향 leakage와 비용은 diagnostic discussion 및 모든 원표를 함께 확인한다. Scientific promotion=false.','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--completion',type=Path,required=True)
    a=p.parse_args();completion=json.loads(a.completion.read_text())
    if completion['stage'] not in ['A','B','C'] or completion['status']!='COMPLETE':raise RuntimeError('STAGE_COMPLETENESS_REQUIRED')
    stage=completion['stage'];path=a.package/f'{stage}-factual-report-ko.md'
    with path.open('x') as f:f.write(build_a(a.package,completion) if stage=='A' else build_later(a.package,completion))
    members=[]
    for file in sorted(a.package.rglob('*')):
        if file.is_symlink():raise RuntimeError('PACKAGE_SYMLINK')
        if file.is_file():members.append(dict(path=str(file.relative_to(a.package)),sha256=sha(file),bytes=file.stat().st_size,mode=oct(stat.S_IMODE(file.stat().st_mode))))
    manifest=dict(stage=stage,members=members,members_root=digest(members),completion_path=str(a.completion),completion_sha=sha(a.completion),scientific_promotion=False)
    save(a.package/'package-manifest.json',manifest)
    source=Path(__file__).resolve().parents[3]
    receipt=dict(stage=stage,report_sha=sha(path),manifest_sha=sha(a.package/'package-manifest.json'),members_root=manifest['members_root'],scientific_promotion=False,
        analysis_source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        analysis_source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=source,text=True).strip())
    save(a.package/'rooted-receipt.json',dict(**receipt,identity=digest(receipt)))

if __name__=='__main__':main()
