"""Korean factual package from immutable S4 evidence. CPU only; no model calls."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from .review_server4 import review,csv_dump,dump,digest
from .report import report


def fmt(x):
    return 'NA' if x is None else f'{x:.6g}' if isinstance(x,float) else str(x)


def build(output,dest):
    output=Path(output);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    parent=output.parent
    manifest=report(output,dest)
    independent=review(output,dest)
    with (dest/'independent-metrics.csv').open() as handle:rows=list(csv.DictReader(handle))
    def read(name):
        p=output/name;return json.loads(p.read_text()) if p.is_file() else None
    lock=json.loads((parent/'execution.lock.json').read_text())
    source_evidence=[]
    frozen=Path(lock['execution']['source'])/'project/run_scripts/en_adaptive_nullspace'
    for path in sorted(frozen.glob('*.py')):
        parsed=ast.parse(path.read_text())
        definitions=[dict(name=node.name,line=node.lineno,kind=type(node).__name__)
            for node in ast.walk(parsed) if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
        source_evidence.append(dict(path=str(path.relative_to(frozen.parent.parent.parent)),
            bytes=path.stat().st_size,sha256=digest(path),definitions=definitions))
    dump(dest/'frozen-source-evidence.json',source_evidence)
    t0=read('T0-result.json') or read('T0/t0-observations.json') or read('T0/t0-failure.json') or {}
    technical=[]
    ad=t0.get('AD',{});fd=t0.get('directional_derivative',{});zc=t0.get('z_source_comparison',{})
    for item,value in [('cached_physical_objective_abs',ad.get('objective_abs')),
                       ('cached_physical_gradient_relative',ad.get('gradient',{}).get('relative_l2')),
                       ('FD_absolute_residual',fd.get('absolute_residual')),
                       ('FD_relative_residual',fd.get('relative_residual')),
                       ('T0_seconds',t0.get('seconds'))]:
        technical.append(dict(item=item,value=value,level='OBSERVED; global precision NOT_ESTABLISHED'))
    for key in ('native','cache_head_batch1','cache_head_batched'):
        for measure,value in zc.get(key,{}).items():
            technical.append(dict(item='z_'+key+'_'+measure,value=value,level='fixed four-request technical panel only'))
    for key in ('batch1_comparison','batched_comparison'):
        comp=zc.get(key,{})
        for measure in ('max_NLL_abs','max_gradient_relative','z_relative'):
            values=[r[measure] for r in comp.get('rows',[]) if measure in r]
            technical.append(dict(item='z_'+key+'_'+measure,value=max(values) if values else None,level='observed maximum; no new threshold'))
        technical.append(dict(item='z_'+key+'_historical_gate',value=comp.get('pass_inherited_NLL_gradient_and_stop_gate'),
            level='historical diagnostic gate, not a new admission waiver'))
    csv_dump(dest/'technical-observations.csv',technical)
    groups=[];references=[];coverage=[]
    for batch in (1,2,3):
        for group in (('SHARED',) if batch==1 else ('N4','EN_EXACT','EN_ADAPT')):
            spec=read(f'B{batch}/{group}-spectrum.json')
            if spec:
                g=spec['geometry'];s=spec['spectrum'];sel=spec['selection']['selected']
                for arm,row in sel.items():
                    executed=(batch==1 or arm==group)
                    groups.append(dict(batch=batch,group=group,arm=arm,actual_controller_executed=executed,
                        evidence_scope='ACTUAL_CONTROLLER' if executed else 'ALGEBRA_ONLY_NOT_EXECUTED',
                        rank=g['rank'],rank_status=g['rank_ambiguity_status'],
                        columns=g['columns'],logical_prefix_groups=g['representative_groups'],tau=g['rank_cutoff'],
                        numerical_cutoff=g['numerical_cutoff'],duplicate_witness=g['duplicate_difference_frobenius'],
                        numerical_released=s['numerical_released'],released_modes=row['released_modes'],blocked_rank=row['blocked_rank'],
                        eta=row['eta'],gradient_energy=row['gradient_energy'],predicted_decrease=row['predicted_decrease'],
                        ideal_norm=row['correction_norm'],ideal_response=row['response_norm'],active_caps=row['active_caps'],
                        native_norm=s['native_norm'],native_response=s['native_action']))
        for arm in ('N4','EN_EXACT','EN_NUM','EN_ADAPT') if batch==1 else ('N4','EN_EXACT','EN_ADAPT'):
            c=read(f'B{batch}/{arm}-controller.json')
            objs=[]
            if c:
                objs.append(('selected',c['objective']))
                for r in c['ledger']:objs.append((r['trial'],r['objective']))
            elif arm=='N4':
                anchor=read(f'B{batch}/SHARED-native-objective.json' if batch==1 else f'B{batch}/N4-reference-history-observer.json')
                if anchor:objs.append(('selected_native_baseline',anchor))
            dev=read(f'B{batch}/{arm}-Dev128.json')
            if dev:objs.append(('Dev128_postseal_observer',dev))
            for phase,obj in objs:
                refs=obj['rows']['reference'];hist=obj['rows']['history']
                references.append(dict(batch=batch,arm=arm,phase=phase,role=obj['role'],J=obj['J'],L_R=obj['L_R'],L_H=obj['L_H'],
                    documents=len(refs),positions=sum(r['positions'] for r in refs),history_requests=len(hist),
                    choice_mismatch_tokens=sum(r['choice_mismatches'] for r in refs),
                    safe_documents=sum(r['choice_mismatches']==0 for r in refs),
                    phi=sum(r['phi'] for r in refs)/len(refs),
                    diagnostic_phi_definition='mean_document_square(max(0,-minimum_position_choice_margin)); not controller objective',
                    exact_token_flip_identity='NOT_RECORDED_COUNTS_ONLY'))
    csv_dump(dest/'space-and-scale.csv',groups);csv_dump(dest/'reference-selected-and-trials.csv',references)
    conformance=[
        ('same W0/zero M4/fixed first300','runtime.py:Runtime; runner.py:main_run','runtime-load.json; native entry hashes; commits','Runtime guard + receipts; no saved W/M reconstruction'),
        ('fresh B1 native once; later own entries','native.py:NativeRunner.fit; runner.py','SHARED-native / arm-native.json','Target reuse between later arms forbidden in source'),
        ('request/unique-sequence/token weighting','current.py:weighted_columns','*-current-inputs.json aliases and weights','Captured bytes retained; no uniform-column substitute'),
        ('FP64 weighted TSQR/SVD; numerical witness','geometry.py:build_geometry','*-spectrum.json','One SVD per correction group; rank ambiguity reported'),
        ('ADAPT epsilon .05; .01/.10 algebra only','selector.py:frontier/select_arms','spectrum.frontiers and selection','No P/N metric input'),
        ('two candidates; ideal curvature / actual Armijo','controller.py:run_controller','controller ledger; selector-replay.json','CPU acceptance replay; not global optimum'),
        ('R512 + own at-write active history','objective.py:Objective; history registry','native-objective; atwrite-teacher; commit registry','Equal block means; at-write teacher tensors RAM only'),
        ('GPU accumulated gradient; final dense D2H once','objective.py:Objective.evaluate','objective sweep dense_gradient_D2H counters','No per-document dense D2H; physical PCIe profiler not run'),
        ('postseal official P/N and Dev','runner.py:SELECTIONS_SEALED; metrics.py:evaluate','seal followed by metrics; Dev only B1 N4/ADAPT','Additional TF metrics reuse same raw forwards'),
        ('final native history exactly once','native.py:finalize; runner.py commit','*-commit.json before/after M hash','No independent full tensor reload; noCP'),
        ('bounded T0, no inherited FD waiver','technical.py:run_t0','T0 observations and final state','Precision NOT_ESTABLISHED; actual residuals not promoted'),
        ('single lane ≤59GiB/noCP','run-server4.sbatch; freeze_server4.py','held inspection/accounting/complete peak','Exact crash resume NOT_AVAILABLE'),
    ]
    csv_dump(dest/'source-conformance.csv',[dict(requirement=a,implementation=b,evidence=c,level=d) for a,b,c,d in conformance])
    lines=['# Server4 EN adaptive-nullspace B300 상세 사실 보고','',f'상태: **{manifest["status"]}**. CPU 독립 reducer 완료 endpoint {independent["complete_endpoints"]}/10. 누락: {independent["missing_endpoints"]}.',
        '',f'실행 source `{lock["execution"]["commit"]}` / tree `{lock["execution"]["tree"]}`. Source/import/input 원본은 execution lock에 결속했다.',
        'SH3 source43904c13의 구현을 S4로 이관했다. SH3 actual submit0이며 test-only51258은 실행이 아니다. S4 teacher 원본을 재사용했고 역전송0이다.',
        f'실제 관측 unique requests={independent["unique_observed_requests"]}, current arm-request rows={independent["observed_current_arm_requests"]}. 완주 기대값은 각각300/1000(세 B300+NUM B1)이며 반복 all-seen 관측을 별도 고유 요청으로 합산하지 않는다.',
        '', '## 1. 독립 NLL 재집계', '',
        '|Batch|Arm|범위|RS|PS|NS|','|---:|---|---|---:|---:|---:|']
    for batch in (1,2,3):
        for arm in ('N4','EN_EXACT','EN_NUM','EN_ADAPT') if batch==1 else ('N4','EN_EXACT','EN_ADAPT'):
            for scope in ('current','active_past','all_seen','first100'):
                selected={r['family']:r for r in rows if int(r['batch'])==batch and r['arm']==arm and r['scope']==scope}
                if selected:
                    cells=[f'{selected[f]["numerator"]}/{selected[f]["denominator"]}' if int(selected[f]['denominator']) else 'N/A (0)' for f in ('RS','PS','NS')]
                    lines.append('| '+' | '.join([str(batch),arm,scope,*cells])+' |')
    lines+=['','RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL; 동률 실패. Fullseen과 current/active-past 분모를 분리했다. EN_NUM은 B1만 있으며 B300 결과로 대체하지 않는다.',
        'TF token-micro/prompt-macro/strict와 true/new NLL은 `independent-metrics.csv`, exact prompt/token-identity paired lost/gained와 NLL 변화는 `independent-paired.csv`이다. 동일 총점은 동일 성공집합을 뜻하지 않는다.',
        'Runtime request-cluster bootstrap 10,000회/seed20260920은 `official-paired.csv`에 별도로 보존했다. 이것은 한 fixed-order 개발 trajectory의 기술통계이며 독립 반복실험·보편적 비열화 인증이 아니다.',
        '', '## 2. 실제 공간·controller 동작', '',
        '|Batch/group|Arm|실행 범위|blocked rank|released|eta|predicted J 감소|','|---|---|---|---:|---:|---:|---:|']
    for r in groups:lines.append('| '+' | '.join([f'{r["batch"]}/{r["group"]}',r['arm'],r['evidence_scope'],str(r['blocked_rank']),str(r['released_modes']),fmt(r['eta']),fmt(r['predicted_decrease'])])+' |')
    lines+=['','`space-and-scale.csv`에 tau·numerical duplicate witness·전체 column·native norm/action·active cap을 수록했다. 수치 witness는 모든 플랫폼의 noise bound가 아니며 rank 모호성을 full precision PASS로 바꾸지 않는다.',
        '공통 native KL gradient와 weighted SVD를 B1 세 correction arm이 공유한다. B2 이후 각 arm은 자기 selected entry, native, history teacher, gradient를 사용한다. Algebra epsilon .01/.10을 새 모델 sweep으로 세지 않는다.',
        'Controller 최대2 actual 후보. 곡률은 ideal d=<G,D1>, Armijo는 실제 FP32 displacement 내적이다. 실제 감소·geometry·finite가 유효한 후보 중 최소 J를 선택하고 없으면 native fallback한다. Full-space 최적성이나 유한 후보 실패를 공간 전체 불가능성으로 해석하지 않는다.',
        'NUM/ADAPT에는 옛 Current NLL/PS strict gate나 DEC no-new-flip 조건을 넣지 않았다. Official 성능 저하는 observer 사실로 남기며 시행 제거·threshold tuning 이유로 쓰지 않았다.',
        '', '## 3. Reference/history와 독립 품질', '',
        '`reference-selected-and-trials.csv`는 selected와 rejected trial을 구분하며 R512 전체 문서/실제 생성 위치 참여, L_R+L_H, choice mismatch 수·safe document 수를 기록한다. Token별 mismatch ID가 저장되지 않은 경우 exact gained/lost token은 NOT_RECORDED다. Counts 감소를 동일 token 회복으로 만들지 않는다.',
        'History는 모든 받은 fact의 최신 유효 target 중 현재 overwrite를 제외한 active 요청이다. Own selected at-write full-vocab teacher를 사용하며 타 arm teacher 또는 현재 entry로 갱신하지 않는다. B1 history는 N/A. B1 Dev128은 N4/ADAPT postseal observer만; Report256 미개방.',
        '이 CSV의 phi는 문서별 최악 choice margin 음수부의 제곱을 문서 평균한 보조 진단이다. Controller 목적은 L_R+L_H이며 이 phi를 이전 DEC objective 또는 EN 선택 기준으로 해석하지 않는다. R512와 Dev128은 role/phase로 구분한다.',
        '', '## 4. T0 및 수치 한계', '',f'Fresh T0 finite/identity={t0.get("finite_identity_status","NOT_RECORDED")}; precision={t0.get("precision_status","NOT_ESTABLISHED")}.',
        '고정4reference/4current에서 실제 AD/physical/cache/단일 directional derivative/weighted geometry·FP32 materialization을 검사했다. 과거 FD waiver나 source-only CPU test를 실제 전체 수치 PASS로 사용하지 않았다. technical-observations.csv는 actual AD/FD와 z 시간·peak·최대 오차를 정리한다. 원 T0 receipt의 historical z gate=false도 그대로 보존한다.',
        'z 비교는 unhooked/cache+head batch1/고정4요청 batched를 구분한다. 요청별 native loss/Adam/clamp/stop 규칙 불변이며 과학 production chunk는16. 고정4요청 timing을 B100 batch16 전체 속도 보장으로 해석하지 않는다.',
        '', '## 5. 상태·저장·비용', '',
        f'관측 commit history append 합계 {independent["history_appends"]}; 예상10(네 B1+세 B2+세 B3). Runtime selected W/M hash와 ledger는 남겼지만 edited W/M/delta/resume checkpoint는 저장하지 않았다. exact crash-resume 및 independent GPU continuation은 NOT_AVAILABLE/NOT_TESTED다.',
        'Slurm parent allocation은 별도 accounting receipt로 결속하고 batch/extern 중복합산0. `costs.csv`의 shared actual은 B1 native/geometry/gradient 1회, method standalone core는 공유 필수비용 전액으로 구분한다. Objective wall은 teacher streaming/I-O를 포함하며 순수 neural compute와 NOT_SEPARATED다. Nested timer를 합산하지 않고 observer+history+CPU bootstrap 묶음도 NOT_SEPARATED다. Allocation은 utilization이 아니다.',
        '19GiB cache budget/52GiB host estimate/24h walltime은 계획이며 actual peak/시간과 다르다. 원 teacher 생성 비용·SH3 전송 비용은 신규 S4 allocation에 다시 청구하지 않는다.',
        '', '## 6. 재현과 검증 범위', '',
        'CPU 재현: `python -m project.run_scripts.en_adaptive_nullspace.publish_server4 --output <immutable-attempt/output> --destination <new-review-directory>`.',
        'Owner audit + 독립 NLL reducer + CPU selector/frontier replay. 별도 독립 red agent 미사용. 원 runtime/teacher/raw 무변경; 새 GPU/evaluator를 리뷰에서 호출하지 않는다. Source-conformance 표는 source의 의미와 실제 저장 검증 수준을 분리하며 frozen-source-evidence.json에 정확 file SHA와 함수 line을 결속했다.',
        'SH는 사실·산술·한계를 정리하며 인과해석·방법 우열·후속선택은 GH 검토 범위다. B300 이후 추가실험 권한0. NO_BROADCAST_NOT_REQUIRED.',
    ]
    if independent['complete_endpoints']:
        from .plot_server4 import plot
        figures=plot(dest)
        lines+=['','## 관측 endpoint 그림','',
            '점은 실제 완료 endpoint다. 연결선은 시각적 안내이며 중간 batch의 추가 관측이나 보간 추정값이 아니다.']
        for name in figures:lines+=['',f'![{name}]({name})']
    lines+=['','[독립 metric 표](independent-metrics.csv) · [paired 전이](independent-paired.csv) · [비용](costs.csv) · [source 대응](source-conformance.csv)',
        '', 'Markdown 표열·링크·UTF-8 및 PNG byte 재현은 CPU에서 검사한다. 별도 HTML renderer가 미설치이면 실제 HTML 렌더는 NOT_AVAILABLE이며 통과로 기록하지 않는다.']
    (dest/'report-ko.md').write_text('\n'.join(lines)+'\n')
    from .check_package_server4 import check
    publication_checks=check(dest)
    dump(dest/'publication-checks.json',publication_checks)
    with (dest/'report-ko.md').open('a') as handle:
        handle.write('\n이 생성 환경의 실제 HTML 렌더 상태: `'+publication_checks['actual_HTML_render']+'`.\n')
    inventory=[]
    for p in sorted(output.rglob('*')):
        if p.is_file():inventory.append(dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=digest(p)))
    dump(dest/'input-inventory.json',inventory)
    artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(dest.iterdir()) if p.is_file() and p.name not in ('analysis-manifest.json','rooted-receipt.json')}
    analysis=dict(execution_commit=lock['execution']['commit'],lock_sha256=digest(parent/'execution.lock.json'),
        source_sha256=digest(__file__),reducer_sha256=digest(Path(__file__).with_name('review_server4.py')),artifacts=artifacts,
        analysis_new_GPU=0,independent_reducer=True,independent_agent=False,save_checkpoints=False)
    dump(dest/'analysis-manifest.json',analysis)
    dump(dest/'rooted-receipt.json',dict(manifest_sha256=digest(dest/'analysis-manifest.json'),report_sha256=digest(dest/'report-ko.md'),
        inventory_sha256=digest(dest/'input-inventory.json'),status=manifest['status']))
    return manifest['status']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--destination',required=True);a=p.parse_args();print(build(a.output,a.destination))
