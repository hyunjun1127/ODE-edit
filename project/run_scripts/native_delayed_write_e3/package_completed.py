"""Publish only validated compact terminal results, never raw prompts or tensors."""
import argparse
import ast
import csv
import datetime
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
from .common import ROOT, INSTRUCTION, read, save, sha, file_record
from .reduce import write_csv


def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def source_entry(root, filename, function):
    path=Path(root)/'project/run_scripts/native_delayed_write_e3'/filename
    tree=ast.parse(path.read_text())
    matches=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==function]
    assert len(matches)==1
    return dict(file=str(path),function=function,line=matches[0].lineno,sha256=sha(path))


def main(attempt, review, figures):
    repo=Path(__file__).resolve().parents[3];attempt=Path(attempt);review=Path(review);figures=Path(figures)
    collector=attempt/'report';output=attempt/'output'
    terminal=read(collector/'terminal.json');audit=read(review/'review-receipt.json')
    assert terminal['status']=='COMPLETED' and audit['status']=='PASS_STORED_EVIDENCE'
    assert terminal['covered_pairs']==terminal['full_patch_pairs']==12
    destination=repo/'experiment-reports/servers/server4/native-delayed-write-e3-20260924-v1/completed-r1'
    destination.mkdir(parents=True,exist_ok=False)
    names=['endpoint-summary.csv','paired-transitions.csv','active-overwrite.csv','factorial-interaction.csv',
           'path-patch-paired.csv','coverage.csv','stage-status.csv','compute.json','lineage.json','terminal.json']
    for name in names:
        assert (collector/name).stat().st_size<5_000_000
        shutil.copyfile(collector/name,destination/name)
    for p in sorted(review.iterdir()):
        assert p.suffix in ('.csv','.json') and p.stat().st_size<5_000_000
        shutil.copyfile(p,destination/p.name)
    (destination/'figures').mkdir()
    for p in sorted(figures.iterdir()):
        assert p.suffix in ('.png','.json') and p.stat().st_size<2_000_000
        shutil.copyfile(p,destination/'figures'/p.name)
    submission=read(attempt/'submission.json');source=Path(submission['source'])
    specs=[
        ('source/input/24CP identity','stages.py','g00','G00','PASS: member SHA + CP stat; runtime tensor SHA/order/shape checks'),
        ('original first16 NLL/strict and repeat','stages.py','g10','G10','PASS: bounded actual panel only; not entire old evaluation replay'),
        ('FP32/eager/TF32 physical L4–8 mapping','backend.py','__init__','G10','PASS: source/config + runtime assertions'),
        ('E1 all25 endpoint/order/cardinality','stages.py','e1_validate','G21','PASS: exact3306 completion rows per endpoint'),
        ('E1 physical all-valid-token key action','stages.py','drift','G20','PASS: row module residual; not additive NLL fraction'),
        ('E3 same receiving-state factorial and A definition','backend.py','hybrid','G31/G51','PASS: exact own/upper keys + bounded FP32 module residual'),
        ('actual11 endpoint equality','stages.py','factorial','G30/G50','PASS: exact NLL/token predictions vs E1'),
        ('full valid-token dose/rotation patch','stages.py','patch','G40/G60','PASS: fixed doses/3seeds; pads excluded'),
        ('W0 restore and nonselected/RNG/input stat','backend.py','restore','G10/G20/G30/G40/G50/G60','PASS: runtime hash/guards; not GPU continuation'),
        ('no native fit/write/history/newCP','stages.py','run','science-terminal','PASS: endpoint-only source and actual counters0'),
        ('atomic ordered gate + scientific negative valid','stages.py','gate','all G00–G70','PASS: binding/predecessor SHA independently checked'),
        ('original vs active overwritten target','reduce.py','overwrite_metrics','active-overwrite.csv','PASS: fixed prefix ledger, separate denominators'),
        ('case/subject/duplicate-prompt cluster uncertainty','reduce.py','cluster_ci','paired/factorial/patch CSV','PASS: descriptive fixed trajectory; no causal promotion')]
    conformance=[]
    for requirement,filename,function,evidence,verdict in specs:
        conformance.append(dict(requirement=requirement,evidence=evidence,verdict=verdict,**source_entry(source,filename,function)))
    write_csv(destination/'source-conformance.csv',conformance)
    compact_index=dict(root=str(output),raw_inventory=file_record(collector/'artifact-index.json'),
        raw_member_count=len(read(collector/'artifact-index.json')['members']),
        total_raw_bytes=sum(m['bytes'] for m in read(collector/'artifact-index.json')['members']),
        raw_fullSHA_scope='new outputs only; common model/CP prior fullSHA receipts + current binding reused',
        raw_and_prompts_copied_to_Git=False)
    save(destination/'raw-index-root.json',compact_index)
    endpoint=rows(destination/'endpoint-summary.csv');module=rows(destination/'module-identity.csv')
    cost=read(collector/'compute.json');science=read(output/'science-terminal.json')
    accounting=cost['parent_accounting'].strip().splitlines();assert len(accounting)==1
    a=accounting[0].split('|');assert a[0]==submission['jobs']['science'] and a[3]=='COMPLETED' and a[4]=='0:0'
    elapsed=int(a[5]);gpu=int(re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',a[6]).group(1))
    assert gpu==1
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    text=['# BASE delayed-write E0(endpoint) → E1 → E3 최종 사실 보고','',
        f'완료 시각(보고 작성): {now}. 원 instruction `{INSTRUCTION}`.','',
        '## 완료 범위','',
        f'GPU {a[0]} parent `{a[3]} {a[4]}`와 scientific G00–G60/CPU G70를 별도 확인했다. '
        '공통 W0+Alpha/MEMIT 각12시점=25 logical endpoints, 고정 core4+확장8=12조합, '
        '48 factorial states와72 modified patch variants를 모두 완료했다. '
        'λ0는 같은 endpoint의 E1 관측을 재사용했다. actual11의 반복 forward는 key/항등식 검사용 비용에 포함한다.',
        '- 새 native z/solve/write/history append/전체 W·M checkpoint는 각각0. 기존24 checkpoint는 read-only 입력이다.',
        '- E0 새 batch continuation, E2, E4–E6, 추가 method/10k는 실행하지 않았다.',
        '- 과거 storage-block 보고와 원 lock을 보존하고 사용자 공간 확보 이후 동일 source로 정상 제출했다.',
        '', '## 입력·실행 identity','',
        f"- Source `{read(destination/'lineage.json')['source_sha256']}`; execution commit `3ebe0b07078940c2d46f9ea2226ccc20c0446162`.",
        f"- Lock `{sha(attempt/'execution.lock.json')}`; panel `{audit['source_binding']['panel_sha256']}`.",
        '- BASE_ALPHAEDIT42657/BASE_MEMIT42658 원 FP32 Llama3-8B-Instruct, blue=False, L4–8 down_proj. '
        'matmul TF32=false/cuDNN TF32=true/eager/TF4.44.2를 유지했다. 새 모델 revision/BLUE/native fitting으로 대체하지 않았다.',
        '- 원15파일245092B 수신 fullSHA; Alpha12CP는 prior receiver fullSHA+currentstat, MEMIT12CP는 이번 task receiver fullSHA. '
        'runtime에서는 두 family의120 weight tensor SHA/shape/finite/order를 확인했다. M은 forward에 불필요하여 materialize하지 않았다.',
        '', '## 고정 panel과 성능 정의','',
        '- N_diag1000=100case×10 prompts; H는 B1 original R100/P200. BaseEval256과 GeneralEval128은 별도 분모다. '
        '원 true/new 각각 TF completion을 계산하며 R/P는 new<true, N/Base는 true<new, tie는 실패다.',
        '- TF token-micro, prompt-macro, strict는 저장 token argmax로 CPU 재계산했다. 자유생성 accuracy가 아니다. '
        'm=true−new, g=−m; full-vocabulary forward KL은 General의 W0 행동 기준이며 사실 정답이 아니다.',
        '- H original과 ACTIVE/SUPERSEDED/active-target 결과를 분리했다. 보조66 completion rows를 원 H 분모에 합치지 않았다.',
        '- BaseEval은 canonical subject NFKC/case/punctuation와 fact/prompt disjoint. 외부 entity alias resolver NOT_AVAILABLE. '
        'same-relation high86/low85/zero85; 다른 relation 가용 pool0을 숨기지 않았다.',
        '- General은 C4 고정128문서, 문서당 natural target64/8192 positions. 새 generated teacher 실험이 아니다.',
        '', '## Endpoint 요약','', '| Endpoint | N /1000 | H R /100 | H P /200 | Base /256 |', '|---|---:|---:|---:|---:|']
    names=['W0']+[f'{f}_W{b:03d}' for f in ('BASE_ALPHAEDIT','BASE_MEMIT') for b in (1,5,10,20,30,40,50,60,70,80,90,100)]
    for name in names:
        nums=[]
        for panel,kind in [('N_diag1000','N'),('H_diag_B1_R100_P200','R'),('H_diag_B1_R100_P200','P'),('BaseEval256','BASE')]:
            nums.append(next(r['success'] for r in endpoint if r['endpoint']==name and r['panel']==panel and r['kind']==kind))
        text.append('| '+' | '.join([name,*nums])+' |')
    text+=['','세부 NLL/tail/TF 및 paired gained/lost는 [endpoint](endpoint-summary.csv), '
        '[독립 검산](independent-endpoints.csv), [paired](paired-transitions.csv), '
        '[active overwrite](active-overwrite.csv), [General](general-eval.csv)에 있다.',
        '', '## E3 factorial·patch와 기계적 검증','',
        '- A=W8(Ws)−W8(W0), B=W4:7(Wt)−W4:7(Ws). s=1은 B1 L8 성분, s>1은 누적 prefix다. '
        '나머지 endpoint weight는 고정했다. 00/10/01/11과실제 receiving-state를 source에 결속했다.',
        f"- 12조합 key equality 모두 exact. Module cross=AδK residual의 최대 relative={max(float(r['max_relative']) for r in module):.9g} "
        '이며 사전 ceiling1e−4와 비교했다. 분모는 실제 v11 norm이다. 작은 action 자체의 상대오차와 혼동하지 않는다.',
        '- 실제11 L8 output에 −λAδK를 모든 valid input-token 위치에 적용했다. λ=.5/1/−1 및3 fixed signed-permutation controls; '
        'token-RMS matching, pad0, λ0/E1 reuse를 기록했다. 토큰 평균 key로 대체하지 않았다.',
        '- signed paired full-panel 효과가 주표이며 W0→actual11 lost-only는 보조다. '
        '[Factorial NLL](factorial-interaction.csv), [patch](path-patch-paired.csv), '
        '[General paired](general-paired.csv), [module identity](module-identity.csv), [layer drift](layer-drift-summary.csv).',
        '- H1δK/F1tKt와 module cross는 선형 module 기여다. NLL 가산 기여율/집중 원인/방법 우월성으로 해석하지 않는다. '
        'query-specific inference hook은 배포 가능한 repair나 새 학습 trajectory가 아니다.',
        '', '## 비용·무결성·한계','',
        f'- GPU parent allocation {elapsed}GPU-sec ({elapsed/3600:.6f}GPU-h), program {science["cost"]["program_seconds"]:.3f}s. Allocation은 utilization이 아니다.',
        f'- GPU peak {science["cost"]["peak_gpu_bytes"]}B, host maxRSS {science["host_maxrss_bytes"]}B. 입력 CP/native 원 학습 비용은 이번 allocation에 재청구하지 않았다.',
        '- [compute.csv](compute.csv)는 누적 counter 차이이며 forward는 program에 포함된다. '
        'H2D bytes는 selected-weight copy 계측 범위만이다. 해시/geometry/CPU reduction/I-O의 순수 개별 시간은 NOT_SEPARATED다.',
        f'- {audit["score_datasets"]} score datasets/{audit["completion_rows"]} completion rows를 독립 검산했다. '
        '동일 endpoint hash만으로 평가를 대입하지 않았다. G00→G70 binding/predecessor SHA chain을 대조했다.',
        '- 신규 GPU continuation/exact 새 resume은 NOT_TESTED/NOT_AVAILABLE. 기존 입력 CP 보존과 이번 noCP 정책을 구분한다.',
        '- bootstrap은 case/subject/중복 prompt 연결 cluster,2000회/seed20260924. 고정 trajectory 기술통계이며 새 순서/모델/인과효과로 일반화하지 않는다.',
        '- 자기 source 검토+별도 CPU reducer이며 독립 agent red는 사용하지 않았다. '
        '[설계→코드→증거](source-conformance.csv), [coverage](coverage.csv), [raw inventory root](raw-index-root.json).',
        '', '## 그림','', '![Endpoint fixed observations](figures/endpoint-preferences.png)','',
        '![Measured patch NLL](figures/patch-neighborhood-nll.png)','',
        '## 재현·종료','',
        'Frozen GPU output은 local `attempt-v1/output`, 원 CPU collector는 `attempt-v1/report`다. '
        '새 scratch destination에서 `python -m project.run_scripts.native_delayed_write_e3.review_completed --attempt <attempt-v1> --destination <new-review>` '
        '및 `plot_completed --report <attempt-v1/report> --destination <new-figures>`로 CPU 검산/그림을 재현한다. 새 GPU 실행은 필요하지 않다.',
        '- 원 raw/prompt/tensor/fullstdout은 Git에 넣지 않는다. `NO_BROADCAST_NOT_REQUIRED`: 같은 host 완결자료이며 새 대형 원격 복제 없음.',
        '- 기전 해석/최종 claim은 GH 소유다. 승인 E3 완료 factual 인계 뒤 `TASK_COMPLETE_STOP`, monitoring_active=false, automatic_resume=false.']
    (destination/'report-ko.md').write_text('\n'.join(text)+'\n')
    links=[];table_rows=0
    for p in destination.glob('*.md'):
        n=None
        for line in p.read_text().splitlines():
            if line.startswith('|'):
                cols=len(line.split('|'))-2;n=n or cols;assert cols==n;table_rows+=1
            else:n=None
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in target:assert (p.parent/target).is_file();links.append(target)
    save(destination/'package-checks.json',dict(GFM_columns='PASS',table_rows=table_rows,links=links,
        actual_HTML_render='NOT_RUN_RENDERER_NOT_INSTALLED',
        renderer_available={m:importlib.util.find_spec(m) is not None for m in ('markdown','markdown_it','mistune')},
        independent_agent=False,raw_free=True,figures='code-only; separate regenerate SHA + visual check receipt required'))
    save(destination/'analysis-manifest.json',dict(instruction_id=INSTRUCTION,execution_source='3ebe0b07078940c2d46f9ea2226ccc20c0446162',
        analysis_commit_before_packaging=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        analysis_sources=[file_record(p) for p in sorted(Path(__file__).parent.glob('*completed*.py'))],
        execution_lock_sha256=sha(attempt/'execution.lock.json'),jobs=submission['jobs'],
        members=[dict(path=str(p.relative_to(destination)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(destination.rglob('*')) if p.is_file()]))
    print(destination)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--figures',type=Path,required=True)
    a=p.parse_args();main(a.attempt,a.review,a.figures)
