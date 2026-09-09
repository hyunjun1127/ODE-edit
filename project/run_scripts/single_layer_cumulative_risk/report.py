"""Korean factual report from measured tables; no model or external reviewer."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from .records import save,digest

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def table(headers,rows):
    def cell(x):return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(cell(x) for x in row)+' |' for row in rows])

def rate(row):return f"{row['numerator']}/{row['denominator']} ({100*float(row['rate']):.2f}%)"

def build_a(package,completion):
    rows=list(csv.DictReader((package/'paired-summary.csv').open()))
    lines=['# 단일 layer 누적위험 진단 — A 사실 보고서','',
      f"상태: {completion['status']}. writers {completion['writers']}/13, logical full-batch steps {completion['fullbatch_steps']}/320, 추가 native scaling endpoints {completion['native_scales']}/12.",'',
      '이 문서는 Llama L4 단일 weight의 세 historical entry를 이용한 진단이다. 새 lifelong run, 안전성 보장, 인과적 locality 개선 또는 scientific promotion을 의미하지 않는다. 상위 B/C는 이 A의 Direct-B/Direct-C support와 다른 실험 단계다.','',
      '## 읽는 법과 분모','',
      '- W0는 원본 모델, ENTRY(We)는 현재 B100을 편집하기 전 historical checkpoint다. 둘을 같은 pre-edit로 부르지 않는다.',
      '- RS/PS는 target-new NLL < target-true NLL, NS는 target-true NLL < target-new NLL이다. Tie는 실패다. 각각의 NLL은 낮을수록 해당 target 확률이 높다.',
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
        for endpoint in sorted({r['endpoint'] for r in selected}):
            for panel in ['Current100','Fixed100','Past100']:
                metrics=[lookup.get((endpoint,panel,m)) for m in ['RS','PS','NS']]
                if not all(metrics):continue
                data.append([endpoint,panel]+[rate(r) for r in metrics])
        lines += [table(['endpoint','panel','RS','PS','NS'],data),'']
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
    for path in sorted((package/'figures').glob('*.png')):
        lines += [f'![{path.stem}](figures/{path.name})','']
    lines += ['모든 PNG는 저장소 plotting.py를 실행하여 생성한다. 입력/명령/환경/출력 SHA 및 동일 입력 두 번 렌더링의 byte 일치는 figures/plot-reproduction.json에 기록한다. 기존 output을 덮어쓰지 않고 재현 시 새 output directory를 사용한다.','',
       '## 해석과 제한','',
       '- Early/Middle/Late는 다음 B100이 서로 다르므로 age만의 인과효과로 읽지 않는다.',
       '- Native z cache hit 비용과 direct training 비용을 cold-native online speed 비교로 해석하지 않는다. 모델 load, preparation, evaluation, generation과 optimizer 시간을 분리한다.',
       '- Norm의 합은 net norm이 아니며 native metric action은 Frobenius 제곱합이 아니다. 위험량 감소와 실제 Past/Fixed 성능 변화는 함께 해석한다.',
       '- Curve에 없는 Past/Fixed PS는 미측정이다. 보간값이나 생성 점수로 채우지 않는다.',
       '- 손실 증가, risk 증가, 작은 gradient, strength 범위의 비중첩은 정상 관측이며 결과 제외나 다음 단계 차단 근거가 아니다.',
       '- A만으로 누적위험 방향이나 refresh의 정보 가치를 결론내리지 않는다. 상위 B/C의 공통 native 출발점 비교가 아직 별도로 필요하다.',
       '- 본 자동 factual table은 별도의 diagnostic discussion, direction/compute/metadata tables 및 requirements-evidence와 함께 읽는다. 없는 필드를 기록된 것으로 간주하지 않는다.','',
       'Scientific promotion=false.','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--completion',type=Path,required=True)
    a=p.parse_args();completion=json.loads(a.completion.read_text())
    if completion['stage']!='A' or completion['status']!='COMPLETE':raise RuntimeError('A_COMPLETENESS_REQUIRED')
    path=a.package/'A-factual-report-ko.md'
    with path.open('x') as f:f.write(build_a(a.package,completion))
    members=[]
    for file in sorted(a.package.rglob('*')):
        if file.is_symlink():raise RuntimeError('PACKAGE_SYMLINK')
        if file.is_file():members.append(dict(path=str(file.relative_to(a.package)),sha256=sha(file),bytes=file.stat().st_size))
    manifest=dict(stage='A',members=members,members_root=digest(members),completion_path=str(a.completion),completion_sha=sha(a.completion),scientific_promotion=False)
    save(a.package/'package-manifest.json',manifest)
    receipt=dict(stage='A',report_sha=sha(path),manifest_sha=sha(a.package/'package-manifest.json'),members_root=manifest['members_root'],scientific_promotion=False)
    save(a.package/'rooted-receipt.json',dict(**receipt,identity=digest(receipt)))

if __name__=='__main__':main()
