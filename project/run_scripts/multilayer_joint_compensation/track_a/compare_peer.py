"""Verify a Git-published peer package; join raw-free same-panel summaries.

This validates publication bytes, NOT the peer's private raw filesystem or
cross-hardware inference equivalence. No model/evaluator calls are imported.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

PEER_PATH='experiment-reports/servers/server2/multilayer-damage-compensation-b-2026-09-11-v1/partial-recall-r1'


def sha(data): return hashlib.sha256(data).hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: f.write(data)
def save(path,x): write(path,canonical(x)+b'\n')


def build(args):
    out=Path(args.output)
    args.peer_commit=subprocess.check_output(['git','rev-parse',args.peer_commit],text=True).strip()
    def blob(name):return subprocess.check_output(['git','show',args.peer_commit+':'+PEER_PATH+'/'+name])
    manifest_bytes=blob('analysis-manifest.json');manifest=json.loads(manifest_bytes)
    receipt_bytes=blob('rooted-receipt.json');receipt=json.loads(receipt_bytes)
    if receipt['manifest_sha']!=sha(manifest_bytes) or receipt['root']!=manifest['root']:
        raise ValueError('PEER_MANIFEST_BINDING')
    if sha(canonical(manifest['members']))!=manifest['root']:
        raise ValueError('PEER_ROOT')
    members={}
    for m in manifest['members']:
        data=blob(m['path'])
        if sha(data)!=m['sha256'] or len(data)!=m['bytes']:raise ValueError('PEER_MEMBER')
        members[m['path']]=data
    b=json.loads(members['partial-summary.json'])
    a0=Path(args.a0)
    common=json.loads((a0/'common-binding.json').read_text())
    a_ep=json.loads((a0/'endpoint-metrics.json').read_text())
    if b['common_ready_sha']!=common['ready']['sha256'] or b['common_root']!=common['members_root']:
        raise ValueError('DIFFERENT_COMMON_FIXTURE')
    if b['input_panel_sha']!=a_ep['panel_identity'] or b['current_effective']!=100:
        raise ValueError('DIFFERENT_EVALUATOR_PANEL')
    ap=Path(args.a_report)
    am=json.loads((ap/'manifest.json').read_text())
    for m in am['members']:
        path=Path(m['name'])
        if not path.is_absolute():path=ap/path
        data=path.read_bytes()
        if sha(data)!=m['sha256'] or len(data)!=m['bytes']:raise ValueError('A_REPORT_MEMBER')
    a_rows=list(csv.DictReader((ap/'endpoint-summary.csv').open()))
    rows=[]
    for r in a_rows:
        if r['state']!='A0':continue
        rows.append(dict(arm='A0',panel=r['panel'],metric=r['metric'],
            numerator=int(r['success_n']),denominator=int(r['denominator']),
            percent=100*float(r['success_rate']),new_nll=float(r['new_nll_mean']),
            true_nll=float(r['true_nll_mean']),evidence='SH1_LOCAL_RAW_VERIFIED'))
    for r in csv.DictReader(io.StringIO(members['endpoint-metrics.csv'].decode())):
        if r['state'] not in ('N4','B-OS'):continue
        rows.append(dict(arm=r['state'],panel=r['panel'],metric=r['metric'],
            numerator=int(r['numerator']),denominator=int(r['denominator']),
            percent=float(r['percent']),new_nll=float(r['mean_new_nll']),
            true_nll=float(r['mean_true_nll']),evidence='SH2_GIT_PUBLICATION_VERIFIED_RAW_OWNER_VERIFIED'))
    keys=[(r['arm'],r['panel'],r['metric']) for r in rows]
    if len(rows)!=27 or len(set(keys))!=27:raise ValueError('COMPARISON_COVERAGE')
    for r in rows:
        if r['denominator']!={'RS':100,'PS':200,'NS':1000}[r['metric']]:raise ValueError('PAIR_DENOMINATOR')
    order={'N4':0,'A0':1,'B-OS':2}
    rows.sort(key=lambda r:(r['panel'],r['metric'],order[r['arm']]))
    out.mkdir(parents=True,exist_ok=False)
    s=io.StringIO();writer=csv.DictWriter(s,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'same-entry-summary.csv',s.getvalue().encode())
    text='# Middle A0 / N4 / B-OS — 완료분 비교\n\n'
    text+='A-OS는 아직 이 표에 terminal이 없다. 전체32 endpoints/네 chain 완료가 아니다. '
    text+='동일 common READY/member root 및 평가 panel SHA를 확인했다. SH1 A0는 local raw 재검산, '
    text+='SH2 N4/B-OS는 owner의 raw 검산을 결속한 Git publication을 독립 재해시한 수준이다. '
    text+='SH2 private raw를 여기서 다시 읽거나 cross-hardware inference parity를 증명한 것은 아니다.\n\n'
    text+='| panel | metric | arm | n/d | % | new NLL | true NLL |\n|---|---|---|---:|---:|---:|---:|\n'
    for r in rows:text+=f"|{r['panel']}|{r['metric']}|{r['arm']}|{r['numerator']}/{r['denominator']}|{r['percent']:.3f}|{r['new_nll']:.6f}|{r['true_nll']:.6f}|\n"
    text+='\nRS/PS는 new NLL < true NLL, NS는 반대이며 tie는 실패다. '
    text+='A0 Current RS/PS가 N4보다 낮고, 같은-bank Base/Past 위험도 native보다 높았다. '
    text+='B-OS도 Base/Past 위험이 증가했으며 PCG 두 RHS의 finite 미수렴 상태를 함께 읽어야 한다. '
    text+='이 차이만으로 공동 편집이나 보정 공간의 불가능성을 결론내리지 않는다. '
    text+='동일 cohort aggregate 표이지 request별 원자료를 다시 join한 paired-delta 추론표는 아니다.\n\n'
    text+='## 비용과 미완료\n\nA0 allocation1561초 + 공통준비916초. '
    text+=f"SH2 B-OS allocation {b['cost']['scheduler_gpu_seconds']}초, 이전 실패 {b['cost']['prior_failed_gpu_seconds']}초는 별도다. "
    text+='서로 다른 host의 합계 wall speedup이나 forward 비용 동등성으로 해석하지 않는다. '
    text+='A0에는 PCG가 없고 B-OS는 a/u 각20회, relative residual0.1563698513/1.7954058935다. '
    text+='B independent Audit/attribution 미기록은 보간하지 않는다. A-OS 및 나머지 지정범위는 pending이다.\n\n'
    text+=f"Peer source publication `{args.peer_commit}`, report SHA `{sha(members['diagnostic-report-ko.md'])}`. "
    text+='Shared functional kernel은 a1fc35fc이고 execution/analysis source는 각 원패키지에서 구분했다. scientific_promotion=false.\n'
    write(out/'comparison-ko.md',text.encode())
    outputs=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(out.iterdir())]
    result=dict(status='SAME_COMMON_RAW_FREE_PARTIAL_COMPARISON',peer_commit=args.peer_commit,
        peer_path=PEER_PATH,peer_manifest_sha=sha(manifest_bytes),peer_receipt_sha=sha(receipt_bytes),
        peer_members=manifest['members'],peer_root=manifest['root'],
        a_report_manifest_sha=sha((ap/'manifest.json').read_bytes()),
        common_ready_sha=b['common_ready_sha'],input_panel_sha=b['input_panel_sha'],
        source_file_sha=sha(Path(__file__).read_bytes()),members=outputs,members_root=sha(canonical(outputs)),
        reproduce=['python3','-m','project.run_scripts.multilayer_joint_compensation.track_a.compare_peer',
                   '--peer-commit',args.peer_commit,'--a0',args.a0,'--a-report',args.a_report,
                   '--output','<new-create-once-output>'],
        remote_private_raw_rehash=False,cross_hardware_numeric_parity=False,
        model_or_evaluator_actions=0,scientific_promotion=False)
    save(out/'manifest.json',result)
    save(out/'rooted-receipt.json',dict(manifest_sha=sha((out/'manifest.json').read_bytes()),
        root=result['members_root'],status=result['status'],scientific_promotion=False))
    print(json.dumps(dict(path=str(out),report_sha=sha((out/'comparison-ko.md').read_bytes()))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('peer-commit','a0','a-report','output'):p.add_argument('--'+name,required=True)
    build(p.parse_args())
