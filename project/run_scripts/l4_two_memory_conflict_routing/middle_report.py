"""Required Middle main-result milestone, only individually closed endpoints."""
import argparse
from pathlib import Path
from .analysis import read,summarize,csv_save
from .identity import save,member,digest,sha
from .report import table,num

def main(args):
    run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=[];members=[];bank=[]
    ref=read(run/'N/full.json')['rows']
    for arm in ('N','OS','BF1','BF8'):
        terminal=read(run/arm/'terminal.json')
        if terminal['status']!='TERMINAL_VALID':raise ValueError('MIDDLE_ENDPOINT_NOT_CLOSED')
        for name in ('terminal.json','full.json','harms.json'):members.append(member(run/arm/name))
        rows.extend(dict(arm=arm,**r) for r in summarize(read(run/arm/'full.json')['rows'],ref))
        harm=read(run/arm/'harms.json')
        bank.append(dict(arm=arm,**{k:harm[k]['value'] for k in ('Past','Base','BaseAudit','BaseW0','BaseAuditW0')}))
    csv_save(out/'middle-main-endpoints.csv',rows);csv_save(out/'middle-main-harm.csv',bank)
    lines=['# Middle B100 — 첫 주 비교 실측','',
        'N/OS/BF1/BF8 네 endpoint의 개별 terminal이 유효하다. 전체16개 완료 보고가 아니며 Frozen/다른 entry/작은 batch는 별도 진행한다. '
        '결과를 이유로 후속 경로를 제거하거나 설정을 바꾸지 않는다. scientific_promotion=false.','',
        table(['arm','panel','metric','success','NLL new mean','N 대비 Δpp','NLL new Δ'],
            [[r['arm'],r['panel'],r['metric'],f"{r['numerator']}/{r['denominator']}",num(r['new_nll_mean']),num(r['delta_pp']),num(r['paired_new_nll_delta_mean'])] for r in rows]),
        table(['arm','Past F','Base We KL','Base audit We KL','Base W0 KL','Base audit W0 KL'],
            [[r['arm']]+[num(r[k]) for k in ('Past','Base','BaseAudit','BaseW0','BaseAuditW0')] for r in bank]),
        'Base NS와 추가 context-response 상세 관측은 저장 endpoint-only pass에서 별도 측정한다. 현재 없다는 사실을0으로 대체하지 않는다.','']
    with (out/'middle-main-ko.md').open('x') as f:f.write('\n'.join(lines))
    save(out/'receipt.json',dict(status='MIDDLE_MAIN_ENDPOINTS_VALID',members=members,members_root=digest(members),
        report_sha=sha(out/'middle-main-ko.md'),report_path=str(out/'middle-main-ko.md'),full_campaign_complete=False,
        analysis_source=[member(Path(__file__)),member(Path(__file__).with_name('analysis.py')),member(Path(__file__).with_name('report.py'))]))
    print(str(out/'middle-main-ko.md'),sha(out/'middle-main-ko.md'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);main(p.parse_args())
