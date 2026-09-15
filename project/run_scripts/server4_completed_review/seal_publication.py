"""Create-once raw-free CPU review manifests and own-scope handoff records."""
import argparse
import json
from pathlib import Path
import subprocess
from .begin import ROOT, LOCAL, WT, TASK, digest, save, query
from .publication import CAKE, CAP, OVER, copy, write
from .validate_publication import markdown_check, verify_sweep_seal

SESSION='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
AUDIT=WT/'audits/servers/server4/2026-09-16-completed-cake-cap-review'
RUN=WT/'runs/odeedit_server4_completed_cake_cap_review_20260916_v1'
STATUS=WT/'tasks/status/odeedit_server4_completed_cake_cap_review_20260916_v1/server4.json'


def seal(root, sources, inputs, extra):
    report=digest(root/'diagnostic-report-ko.md')
    members=[digest(p) for p in sorted(root.rglob('*')) if p.is_file()]
    manifest=dict(instruction_id=TASK,analysis_head=query(['git','rev-parse','HEAD'])['stdout'].strip(),
        analysis_tree=query(['git','rev-parse','HEAD^{tree}'])['stdout'].strip(),
        analysis_worktree=str(WT),source=sources,inputs=inputs,members=members,
        current_review_authority='CPU_COMPLETED_REVIEW_ONLY',new_GPU_seconds=0,
        no_model_forward=True,numerical_validation='NOT_ESTABLISHED_FOR_EP',
        verification_level='SINGLE_SH_SELF_AUDIT_AND_INDEPENDENT_RAW_NLL_REDUCER',
        raw_policy='LOCAL_ONLY_NO_BROADCAST_NOT_REQUIRED',**extra)
    save(root/'analysis-manifest.json',manifest)
    receipt=dict(instruction_id=TASK,status='REVIEW_COMPLETE_PUBLICATION_READY',report=report,
        manifest=digest(root/'analysis-manifest.json'),monitoring_active=False,
        automatic_resume=False,new_GPU_seconds=0,GPU_continuation='NOT_TESTED',
        completion_boundary='OWN_SCOPE_NONFORCE_MAIN_PUBLICATION_THEN_TASK_COMPLETE_STOP')
    save(root/'rooted-receipt.json',receipt)
    return digest(root/'rooted-receipt.json')


def main(validation):
    assert not AUDIT.exists() and not RUN.exists() and not STATUS.exists()
    checked=json.loads(validation.read_text())
    assert checked['status']=='CPU_PACKAGE_VALIDATED'
    assert len(checked['PNG_byte_reproduction'])==9
    verify_sweep_seal()
    AUDIT.mkdir(parents=True);RUN.mkdir(parents=True)
    for p in (LOCAL/'receipts/full-read.json',LOCAL/'receipts/scheduler-terminal-once.json',validation):
        copy(p,AUDIT/p.name)
    copy(validation.parent/'focused-tests.json',AUDIT/'focused-tests.json')
    root_boundary=query(['bash','scripts/check-session-boundary.sh',SESSION],ROOT)
    child_boundary=query(['bash','scripts/check-session-boundary.sh',SESSION])
    boundary=dict(session=SESSION,actual_host=query(['hostname'])['stdout'].strip(),
        root_origin=query(['git','remote','get-url','origin'],ROOT),
        root_boundary_helper=root_boundary,child_boundary_helper=child_boundary,
        root_dirty=query(['git','status','--short'],ROOT),
        old_sweep_dirty=query(['git','status','--short'],ROOT/'local/worktrees/server4-ep-tw1-alpha-cap-sweep-v1'),
        interpretation='Child helper missing local config is NOT PASS; explicit user clean-child scope and root boundary reused. No helper/config mutation.')
    assert boundary['actual_host']=='server4'
    assert root_boundary['returncode']==0,root_boundary
    assert boundary['root_dirty']['stdout']==''
    save(AUDIT/'boundary-checks.json',boundary)
    write(AUDIT/'analysis-repairs-ko.md',
        '# CPU 분석 수리와 검증 경계\n\n'
        'CAKE current의 weight_state/cache_sha256 및 정수0 비개입 필드, NORM_ONLY strict-lost ID의 문자열 정렬(동일3개 exact set), CAKE 실행 closure의 단축 경로를 분석기에 반영했다. 실패한 CPU analysis namespace는 local에 그대로 보존했다. 과학 runtime/raw/threshold 수정과 재실험은 0이다.\n\n'
        '21개 focused CPU tests, 9개 PNG byte-identical 재생성, GFM 표 열/헤더 간격/이미지 링크, package SHA와 파일형식을 검사했다. 독립 NLL reducer는 runtime 집계값 대신 저장 pair를 다시 계산하지만 별도 agent red/blue 검토는 실행하지 않았다. 이 보고는 단일 SH 자체 감사이며 model-level numerical PASS가 아니다.\n\n'
        'CAKE checkpoint0은 사용자 waiver, 신규cap checkpoint30은 실제 fullSHA/weights_only CPU reload 대상이다. CAP1 checkpoint 부재는 정본 관측 재사용이며 이 리뷰에서 fullresume를 주장하지 않는다.\n\n'
        'sweep manifest의 parent instruction과 SOURCE_INPUT_FROZEN_NOT_SUBMITTED는 과거 source-freeze 기록이다. 현재 권한은 본 완료리뷰 instruction이며 현재 terminal은 별도 정확4job snapshot으로 확인했다. CAKE summary의 Adam NOT_RECORDED는 native counter 부재를 뜻하며 별도 원 로그 경계로 도출한 66952회와 출처를 구분한다.')
    write(CAKE/'REPRODUCE.md',
        '# CPU 재현\n\n새 output만 사용한다. Scheduler 조회는 기존 receipt를 재사용한다.\n\n'
        '```bash\npython -B -m project.run_scripts.server4_completed_review.cake full --output NEW_CAKE\n'
        'python -B -m project.run_scripts.server4_completed_review.plot_cake --package CAKE_PACKAGE --output NEW_FIGURES\n'
        'python -B -m project.run_scripts.server4_completed_review.validate_publication --output NEW_VALIDATION\n```\n\n'
        '실제 환경 Python3.12.3 / EasyEdit .venv. 9개 그림은 같은 CSV/코드/Matplotlib 환경에서 byte-identical 재생성됐다. CAKE codePNG는 이 package figures/plot-manifest.json에 결속된다. W/M 미저장으로 GPU continuation 또는 tensor replay는 재현 범위가 아니다.')
    # No scientific payload is copied; only compact CPU evidence and source hashes.
    sources=[digest(p) for p in sorted((WT/'project/run_scripts/server4_completed_review').glob('*')) if p.is_file()]
    inputs=[digest(AUDIT/p) for p in ('full-read.json','scheduler-terminal-once.json','validation.json','focused-tests.json','boundary-checks.json')]
    for p in (CAKE/'REPRODUCE.md',AUDIT/'analysis-repairs-ko.md'):
        assert not markdown_check(p)['errors']
    cake_receipt=seal(CAKE,sources,inputs,dict(execution_head='7884aeb6000f8343139172825ec6c4ca24357fc0',
        execution_tree='4dabdb1e408726974ac0f91285ad35bac698b88e',batches=100,unique_requests=10000,
        weight_history_tensors='NOT_SAVED_USER_DIRECTED',new_CP_reload=0,
        measured_allocated_GPU_seconds=32194,reused_W0_evaluation_job=42673))
    cap_receipt=digest(CAP/'rooted-receipt.json')
    save(OVER/'family-receipts.json',dict(instruction_id=TASK,CAKE=cake_receipt,EP_SWEEP=cap_receipt,
        limitations=['CAKE no W/M tensor save','CAP1 CP absence is historical sealed reuse',
                     'EP user-skipped FD/direct/selfKL not established','No new model or GPU execution']))
    copy(AUDIT/'validation.json',OVER/'publication-validation.json')
    overall_receipt=seal(OVER,sources,inputs,dict(family_receipts=[cake_receipt,cap_receipt],
        jobs=[48101,48148,48149,48150],all_four_terminal='COMPLETED_0_0',
        measured_new_execution_GPU_seconds=55127,review_GPU_seconds=0,
        prior_reuse_GPU_seconds=dict(CAP1=7694,teacher=98,failed47884=473,failed47942=69)))
    common=dict(instruction_id=TASK,session=SESSION,host='server4',status='REVIEW_COMPLETE_PUBLICATION_READY',
        current_review='CPU_ONLY',reports=dict(overall=overall_receipt,CAKE=cake_receipt,caps=cap_receipt),
        terminal_jobs={str(j):'COMPLETED_0_0' for j in (48101,48148,48149,48150)},
        new_review_GPU_seconds=0,new_model_forwards=0,monitoring_active=False,automatic_resume=False,
        numerical_validation='SKIPPED_USER_DIRECTED / NOT_ESTABLISHED',
        resume_trigger='EXPLICIT_USER_CALL_ONLY',next='NONFORCE_MAIN_PUBLICATION_THEN_STOP')
    save(STATUS,common);save(RUN/'review-receipt.json',common)
    copy(LOCAL/'receipts/SH2-N4-path-ack.md',AUDIT/'SH2-N4-path-ack.md')
    write(WT/'messages/acks/server4/2026-09-16-completed-cake-cap-review.md',
        '# FULL_READ / M0 / completed review ACK\n\n'
        f'Instruction `{TASK}`. 새 main7ef4fc05와 최신 user pause를 결속했다. 정확4job만 한 번 조회해 COMPLETED0:0 확인 후 CPU 분석했다. 다른 paused task 재개와 새 GPU는0.\n\n'
        'full-read.json SHA8f51e8a7e37ecab5668c07f8b9293dc5a895b6cf11a939ab469b50ef673b7481. 최초표를 GH에 전달했고 정본 CAP1과 기존 역사 보고는 동일 SHA로 재사용했다. 별도 실행 source와 분석 source를 manifest로 구분했다.\n\n'
        'SH2 exact N4 seen-full 파일은6663948B/SHAe58ed53d6733415d9ae9b20c4f9f1cb82f0769d9cd749bf4af4d98ea138583e3/regular ownerjanghj로 확인했다. SH4 복사0. 직접 transport는 대상 Codex binary 부재로 COMMUNICATION_HOLD이며 수신 완료로 기록하지 않았다.')
    write(WT/'messages/server-heads/server4/2026-09-16-completed-cake-cap-review.md',
        '# CAKE/cap completed CPU review — publication ready\n\n'
        'CAKE48101:9840/10000,17755/20000,62935/100000. Cap W10: CAP1(reuse)998/1942/8056; CAP10 999/1947/8075; CAP100 998/1945/8052; NORM_ONLY999/1938/8042 (denominators1000/2000/10000). 10kと1kは別表。\n\n'
        'CAKE100commit/99metadata links/12fullseen、user no-CP. 新規cap30CP/27links/3000targets/30solve/30history。21CPU tests、9PNG byte reproduction、exact source/manifest/Markdown checks. Numerical validation NOT_ESTABLISHEDを維持。\n\n'
        f'Overall receipt SHA{overall_receipt["sha256"]}; CAKE{cake_receipt["sha256"]}; sweep{cap_receipt["sha256"]}.\n\n'
        'New execution allocation55127GPU-sec (CAKE32194/cap22933), prior CAP1/teacher/failures separate. ReviewGPU0. Ownscope nonforce main publication ready; no further experiments or monitoring scheduled.')
    access=query(['bash','scripts/check-agent-access.sh','--all-changed'])
    save(AUDIT/'access-check.json',dict(helper=access,
        explicit_user_authorized_exception='runs/odeedit_server4_completed_cake_cap_review_20260916_v1/ is explicitly authorized by section7; helper lacks generic server-head runs rule. No helper modification and no helper PASS claim for rejected path.'))
    return common


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validation',type=Path,required=True)
    print(json.dumps(main(p.parse_args().validation)))
