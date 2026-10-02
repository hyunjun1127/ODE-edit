"""Compact owner publication of already frozen receipts, no scheduler/model."""
import argparse
import json
from pathlib import Path
from .common import ROOT,INSTRUCTION,member,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True,type=Path);args=p.parse_args();r=args.attempt
    submission=json.loads((r/'submission.json').read_text());lock=json.loads((r/'execution.lock.json').read_text())
    receipt=dict(instruction_id=INSTRUCTION,status='SUBMITTED_INITIAL_NOT_OBSERVED',
        actor='SH4',jobs=submission['jobs'],source_commit=lock['source_commit'],source_tree=lock['source_tree'],
        lock=member(r/'execution.lock.json'),submission=member(r/'submission.json'),
        inspection=member(r/'held-inspection.json'),config=member(r/'config.json'),
        source_archive=lock['archive'],attempt=str(r),cap=2,save_checkpoints=False,exact_resume='NOT_AVAILABLE',
        monitoring_active=True,automatic_resume=False,new_baselines=0,prior_tasks='STOPPED_USER_UNCHANGED',
        broadcast='NO_BROADCAST_NOT_REQUIRED',설명='공통 W0 이후 두 pilot, 두 독립 timing/main 및 CPU collector 전체 등록/release. 실제 main 초기 연결은 미관측.')
    base='jlz-native-joint-v4-bs100x20-20261002-v1'
    write(ROOT/f'audits/servers/server4/{base}/submission.json',receipt)
    write(ROOT/f'tasks/status/{base}/server4.json',receipt)
    write(ROOT/'messages/server-heads/server4/2026-10-02-jlz-v4-compute-r1-2k.json',receipt)
    # Raw stdout and raw model arrays are intentionally excluded.
    report=ROOT/f'experiment-reports/servers/server4/{base}/report-ko.md'
    lines=['# JLZ v4 compute-r1 A/B 2k — 제출 인계','',
           '상태: SUBMITTED_INITIAL_NOT_OBSERVED. 완료·실제 main gate PASS를 뜻하지 않는다.',
           '',f'Instruction: `{INSTRUCTION}`.',f"Execution source: `{lock['source_commit']}`.",
           f"Lock SHA256: `{receipt['lock']['sha256']}`.",'',
           '| 단계 | Job | Dependency |','|---|---:|---|']
    ids=submission['jobs']
    for name,job in ids.items():
        dep='없음' if name=='shared-SHARED' else ('afterok:'+ids['shared-SHARED'] if name.startswith('pilot') else
            'afterok:'+ids['pilot-JLZ_A']+':'+ids['pilot-JLZ_B'] if name.startswith('main') else 'afterany:'+':'.join(v for k,v in ids.items() if k!='collector'))
        lines.append(f'| {name} | {job} | {dep} |')
    lines += ['', '전체 held owner/fullargv/resource/source/dependency 검사 후 release했다.',
        'pilot은 arm당 BS2 25후보/24 Adam commit 및 다음 BS2 entry만, timing은 arm당4후보이며 write0.',
        'main은 각 cold W0/H0 BS100×20, 총40 commit/1000후보/960 Adam이다. 신규 baseline0.',
        '', '## 검산과 자원', '',
        '정본 11파일 FULL_READ/SHA, 승인 archive33 regular member SHA/size, case2000행 및 평가 일정 검산.',
        'CPU tokenizer22 pack과 회귀6 tests PASS. 실제 GPU 정합은 아직 미관측이며 별도 독립 red는 사용하지 않았다.',
        'cap2; GPU job 각1GPU/8CPU/60416MiB, collector0GPU/8CPU/24576MiB.',
        'host 예상44GiB/GPU 예상65GiB, disk reserve30GiB. GPU wall7일은 ETA가 아니다.',
        '', '## 보존·한계', '',f'Local: `{r}`.',
        'save_checkpoints=false; exact_resume=NOT_AVAILABLE. Raw/teacher/tensor/prompt/fullstdout Git0.',
        '기존 v2 및 타 task의 STOP 유지. NO_BROADCAST_NOT_REQUIRED: 같은 서버의 소형 source/receipt만 게시.',
        '최종 수치/완료단계/실측비용은 아직 NOT_MEASURED. 등록된 runner/collector가 저장하며 사용자 recall 때 상세회수한다.']
    report.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(report),jobs=ids)))

if __name__=='__main__':main()
