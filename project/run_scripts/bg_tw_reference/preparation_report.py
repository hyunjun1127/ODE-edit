"""Compact raw-free preparation/G0_BLOCKED publication; no job/result polling."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .control import identity, save, sha, TASK

PUB='experiment-reports/servers/server4/bg1-c4-ours-first-2026-09-15-v1'
AUD='audits/servers/server4/2026-09-15-bg1-c4-ours-first'


def text_once(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(text)


def csv_once(path,rows):
    with Path(path).open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def build(worktree,attempt,control=None):
    w,a=Path(worktree).resolve(),Path(attempt).resolve()
    c=Path(control).resolve() if control else a
    assert c.is_relative_to(a)
    if control:c.mkdir(parents=True,exist_ok=False)
    pub=w/PUB;pub.mkdir(parents=True,exist_ok=False)
    audit=w/AUD;audit.mkdir(parents=True,exist_ok=False)
    r=a/'reference-v1'
    load=lambda p:json.loads(Path(p).read_text())
    fullread=load(a/'full-read-receipt.json');built=load(r/'build-status.json')
    sources=load(r/'source-manifest.json');splits=load(r/'splits.json')
    lock=load(a/'teacher.lock.json');released=load(a/'teacher-submission-v1/release-receipt.json')
    resources=load(a/'teacher-submission-v1/resource-preflight.json')
    peer=load(a/'sh2-calibration-peer-receipt-r2.json')
    assert peer['status']=='DELIVERED_COMPLETED' and 'CALIBRATION_MISSING' in peer['gh_response']
    assert released['job_id']=='47592' and 'JobState=PENDING' in released['record']
    assert len(lock['members'])==1772
    for m in built['members']:assert sha(m['path'])==m['sha256']
    # Reused large asset fullSHA verification was completed on this frozen lock
    # before this report. This report rehashes compact/new reference members only.
    fixture_output=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(w/'project/run_scripts/bg_tw_reference'),top_level_dir=str(w))
    tested=unittest.TextTestRunner(stream=fixture_output,verbosity=1).run(suite)
    assert tested.wasSuccessful() and tested.testsRun==39
    tests=dict(status='PASS_CPU_FIXTURES',tests=tested.testsRun,failures=len(tested.failures),errors=len(tested.errors),
        gpu_model_tests='NOT_RUN',frozen_source_fullSHA_members=1772,
        frozen_source_verification='FROZEN_PREPARATION_CLOSURE_PASS; model load0/GPU0; separate completed CPU invocation',
        raw_model_off_on_parity='NOT_TESTED',teacher_same_logp_selfKL='code fixture only; real teacher NOT_OBSERVED')
    save(audit/'cpu-checks.json',tests)
    controlfix=save(c/'teacher-submission-v1/no-requeue-control-receipt.json',dict(job_id='47592',
        action='scontrol update JobId=47592 Requeue=0',exit_code=0,verified_fields=dict(JobState='PENDING',
        Requeue=0,Restarts=0,RunTime='00:00:00',AllocTRES='(null)',Dependency='(null)'),
        source_changed=False,scope='OWN_NEW_TEACHER_OPERATIONAL_SETTING_BEFORE_G0',
        note='Scheduler default Requeue=1 disabled; frozen source/lock unchanged; no other job mutation',
        time_recorded=datetime.now(timezone.utc).isoformat()))
    cal=dict(status='CALIBRATION_MISSING',scientific_submission_allowed=False,
        required_batches=list(range(1,11)),present_batches=[1,5,10],missing_batches=[2,3,4,6,7,8,9],
        trajectories=['AlphaEdit_BLUE_L4_ONLY lifelong main-cell-3 (39283_3)','BLUE L4 one-shot1000 (38997)'],
        owner_check='SH2:20 immutable commit metadata file SHA/size newly checked; missing7 each checkpoint=null',
        owner_peer_receipt=identity(a/'sh2-calibration-peer-receipt-r2.json'),
        cp_fullSHA_level='SH2 reused exact prior fullSHA + current stable stat; no new S4 tensor transfer/load',
        weight_delta_journal='NOT_AVAILABLE in either checked catalog/companion closure',
        native_targets_or_hashes_are_weight_deltas=False,
        calibration_D64='NOT_MEASURED',fixed_b=None,numerical_floor=None,
        partial_max_or_warmproxy_or_zero_fallback=False,baseline_editing_reruns=0,
        next_authority_needed='Actual missing seven compatible stored states/deltas, or explicit changed calibration/re-execution authority; no automatic proposal execution')
    save(pub/'calibration-availability.json',cal)
    report_refs=[w/'experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/diagnostic-report-ko.md',
        w/'experiment-reports/servers/server4/refit4-write-refresh-seq1000-2026-09-14-v1/completed-review-v1/diagnostic-report-ko.md']
    policies=['AlphaEdit','MEMIT','AlphaEdit-BLUE','MEMIT-BLUE','AlphaEdit-L4_only','REFIT4']
    reuse=dict(reports=[identity(p) for p in report_refs],policies=[dict(policy=p,status='REUSE_SCOPE_LIMITED',
        interpretation='W50/B51..60 auxiliary only; not W0 paired baseline' if p=='REFIT4' else
        'Existing W0 lifelong publication only; full10k is not this first1000 measurement; config/layers differ',
        new_editing=0,new_C4_evaluation=0) for p in policies],
        model_revision=lock['model_revision'],model_assets='1772-member frozen preparation closure verified; no new model download',
        full_read=identity(a/'full-read-receipt.json'),data_order_root='5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729',
        first1000_order_root='40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd',
        fields_not_reused=['warmW50/M50/M8','historical teacher probabilities','historical C4 budget','prior policy list'],
        remote_payload_transfers=0,raw_broadcast='NO_BROADCAST_NOT_REQUIRED')
    save(pub/'evidence-reuse-manifest.json',reuse)
    sr=[dict(split=x['split'],bytes=x['bytes'],sha256=x['sha256'],jsonl_rows=x['row_counts']['jsonl_rows'],
        crc=x['gzip_crc_validation']) for x in sources['source_files']]
    csv_once(pub/'reference-source-summary.csv',sr)
    composition=load(r/'reference-composition.json')
    rows=[dict(role=role,documents=val['documents'],domains=len(val['domains']),top10_share=val['top10_document_share'],
        wiki_domain_tagged=val['wiki_domain_tagged_documents'],wiki_mirror_tagged=val['wiki_mirror_phrase_tagged_documents'])
        for role,val in composition['roles'].items()]
    csv_once(pub/'reference-composition-summary.csv',rows)
    coverage=[('source/documents','FULL_READ','14 docs + PROTOCOL1269 + envelope + native/fitter source','NOT_A_GPU_GATE'),
        ('fixed10k prefix','CPU_VERIFIED','exact load_prefix1000; originalfull10000 bytes/order/inventory verified','no resampling'),
        ('C4 full gzip','NEW_FULL_SHA_CRC','359779975B;401893 JSONL rows; full EOF','only two pinned first shards, not all C4'),
        ('reference768','CPU_TOKEN_SEALED','S64/Dev128/Reserve320/Report256; actual768x257; token/member SHA','no losses used'),
        ('overlap','PARTIAL_SCOPE_VERIFIED','Wiki128 + first1000 R/P/N input-only13128; selected294528 pairs','mom2/MMLU/Audit/future NOT_INSPECTED'),
        ('teacher192','PENDING_NOT_VALIDATED','47592 released; last PENDING;24x8doc FP32 planned','12,608,077,824 tensor bytes estimate'),
        ('calibration10','BLOCKED','missing2/3/4/6/7/8/9; two trajectories checked by owner','no partial max/rerun'),
        ('native S(R)=RA','CPU_FIXTURE_PASS','nonSPD direct solve; map/direct FP32 comparison; FD/VJP','actual Llama/native model parity NOT_TESTED'),
        ('barrier/gradient','CPU_FIXTURE_PASS','C2 join/negative slack; one fullD64 slope; mass accumulation','real model gradient NOT_TESTED'),
        ('anchor/trust/materialization','CPU_FIXTURE_PASS','own anchors vsH; FP32; candidateorder/screen/parent','live model transaction NOT_TESTED'),
        ('teacher token/lock guard','CPU_VERIFIED','vocab/range/roles/IDs/source+tokenizer/shard coverage;1772 fullSHA','model forward NOT_RUN by agent'),
        ('BG model adapter/persistent runner','INCOMPLETE_NOT_GPU_VALIDATED','math and failclosed gate only; no executable scientific submission','not a completed BG implementation'),
        ('history/zero-write/restore/resume','NOT_TESTED_MODEL_LEVEL','no actual BG batch; no history append claim','needs model adapter and approved calibration'),
        ('scientific B1..B10','NOT_SUBMITTED','0 scientific jobs;0 new baselineediting','G0 firstB100 notrun'),
        ('Dev/Reserve/Report/Audit/MMLU eval','DEFERRED_NOT_EVALUATED','Dev teacher only prep; no general evaluation scores','no policy choice/claim')]
    csv_once(pub/'implementation-coverage.csv',[dict(component=c,status=s,evidence=e,limit=l) for c,s,e,l in coverage])
    red=dict(status='BLOCK_SCIENTIFIC_ALLOW_PREPARATION',independent_red='bounded read-only source/39 parent-final tests; no GPU',
        findings=[dict(item='dispatch warm/server/calibration bypass guards',status='FIXED_CPU_REGRESSION_PASS'),
        dict(item='token OOV and source ID bridge',status='FIXED_CPU_REGRESSION_PASS'),
        dict(item='teacher required closure/tokenizer/reference seal',status='FIXED_ACTUAL_FROZEN_1772_FULLSHA_PASS'),
        dict(item='transformers ambient4.57 vs pinned4.44.2',status='FIXED_FREEZE_ASSERT_AND_EXACT_PYTHONPATH'),
        dict(item='model-level BG flow not implemented/tested',status='UNRESOLVED_EXPLICIT'),
        dict(item='missing N4 seven endpoints',status='BLOCK_CALIBRATION_MISSING')],
        no_efficacy_selection=True,all_workers_stopped=True)
    save(audit/'red-preflight.json',red)
    code=[identity(p) for p in sorted((w/'project/run_scripts/bg_tw_reference').iterdir()) if p.is_file()]
    provenance=dict(authoritative_main=fullread['source_main'],authoritative_tree=fullread['source_tree'],
        execution_source_head=lock['source_head'],execution_source_tree=lock['source_tree'],
        execution_archive=lock['source_archive'],teacher_lock=identity(a/'teacher.lock.json'),
        builder_execution_git_head=sources['builder_git_head'],builder_exact_file=sources['builder_file'],
        builder_note='Executed before source commit; exact file SHA authoritative, dirty/new file not claimed present at priorHEAD',
        analysis_code=code,analysis_code_hash_level='exact file bytes; publication commit separately supplied in final handoff',
        operation_amendment=controlfix,existing_source_mutations=0)
    save(pub/'source-manifest.json',provenance)
    private_files={}
    for p in a.rglob('*'):
        if p.is_file():
            s=p.stat();private_files[(s.st_dev,s.st_ino)]=s.st_size
    summary=dict(reference_identity=splits['reference_identity_sha256'],reference_member_root=built['member_root'],
        source_total_bytes=sum(x['bytes'] for x in sr),reference_rows=768,teacher_rows_planned=192,
        cpu_build_elapsed_seconds=built['cpu_build_elapsed_seconds'],cpu_tests=39,
        teacher_last_job_state='PENDING',teacher_job_id='47592',teacher_measured_GPU_seconds_at_last_observation=0,
        teacher_final_cost='NOT_OBSERVED',teacher_estimated_GPU_hours=[.25,2.0],teacher_requested_wall_hours=2,
        teacher_tensor_payload_bytes=12608077824,task_reserved_estimate_bytes=60*(1<<30),
        filesystem_exclusive_reservation=False,disk_free_at_submit=resources['free_bytes'],
        local_unique_inode_logical_bytes=sum(private_files.values()),teacher_io_model_C4_metrics='NOT_OBSERVED',
        scientific_edits=0,baseline_reruns=0,calibration_forwards=0,teacher_NLL_in_Git=False)
    save(pub/'preparation-summary.json',summary)
    resume=dict(instruction_id=TASK,nonce='ODEEDIT-GH-SH4-BG1-C4-OURS-FIRST-20260915-R1',host='server4',
        owner_session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',run_root=str(a),
        status='WAITING_USER_RESUME',gate='G0_BLOCKED',reason='CALIBRATION_MISSING',
        monitoring_active=False,automatic_resume=False,resume_trigger='explicit_user_call',
        scientific_jobs=[],preparation_jobs=[dict(job_id='47592',phase='W0_TEACHER192',last_state='PENDING',
        last_observation=released['last_observation'],last_state_amendment=controlfix,dependencies=[],requeue=0,
        lock=identity(a/'teacher.lock.json'),source=lock['source_archive'],output=lock['output'])],
        reference=identity(r/'build-status.json'),teacher_manifest=dict(path=str(Path(lock['output'])/'teacher-manifest.json'),
        current_existence='NOT_OBSERVED; job PENDING at bounded submission check'),
        calibration=identity(pub/'calibration-availability.json'),reuse=identity(pub/'evidence-reuse-manifest.json'),
        selected_checkpoint=None,history=None,RNG=None,last_processed_batch=0,next_ordinal=0,
        expected_scientific_terminal=None,expected_scientific_evaluation=None,
        G0=dict(reference768=True,teacher192=False,calibration10=False,model_technical_validity=False,
        persistent_scientific_job=False,firstB100_finite=False,denominator100=False,finalization1=False,
        next_ordinal100=False,resume_state=False),
        pending_job_is_pass=False,whole1000_completed=False,scientific_promotion=False,
        deferred=['all baselineediting','BG B1..B10','Reserve/Report teacher','Audit/MMLU/Report scores','new policies'])
    resume_id=save(c/'resume-manifest.json',resume)
    save(pub/'resume-manifest.json',resume)
    report=f'''# BG-1 / C4 ours-first — 준비 및 G0_BLOCKED 사실 보고

Instruction: {TASK}. 상태 **G0_BLOCKED / CALIBRATION_MISSING / WAITING_USER_RESUME**.
이것은 BG-1 실행완료·성능 보고가 아니다. Scientific promotion=false.

## 결과와 blocker

동일 W0/order/native L4/L2=1의 N4 B1..B10 전 endpoint로 C4 D64 최대값을 고정해야 한다.
SH2가 lifelong main-cell-3 및 L4 one-shot38997의 봉인 commit20개를 확인했다.
두 경로 모두 B1/B5/B10만 저장됐고 **B2/3/4/6/7/8/9는 checkpoint=null**, 복원 가능한 weight delta journal도 없다.
Native target 또는 entry/endpoint hash만으로 누락 weight를 만들 수 없다.
기존 CP bytes 검증은 SH2 과거 fullSHA+현재 stat 결속 재사용이며 이번 S4의 신규 tensor검산/전송이 아니다.
일부 endpoint 최대값/따뜻한 W50 proxy/0 budget으로 치환하지 않았고 baseline editing rerun도0이다.
따라서 fixed b는 미정이고 **BG scientific job0, firstB100 미실행**이다.

## 완료된 독립 준비

|항목|관측|
|---|---|
|원문/참조|14문서, PROTOCOL1269줄, dispatch/native/fitter full-read; 원문 SHA 보존|
|C4 획득|정확2gzip 359,779,975B; train356317/validation45576행 전체 EOF·CRC·fullSHA|
|Reference|S64+Dev128+Reserve320+Report256=768, 실제 int64[768,257]|
|Token|자연256+BOS1; target[129,257), logits[128,256); decode-reencode0|
|Overlap|Wiki128+first1000 R/P/N 입력13128; 선택문서294528쌍 검사, 중복0|
|독립 CPU 테스트|39 PASS; native map/FD·VJP/C2/mass/ball/trust/candidate/guard|
|Teacher 준비|job47592 held검사→release; 마지막 PENDING; 정상기동/완료 아직 미관측|

Reference identity `{splits['reference_identity_sha256']}`. Member root `{built['member_root']}`.
모든 역할과 window는 첫 model loss 전에 CPU 봉인했다. Sampler는 답/score/future subject list를 받지 않는다.
mom2 원문, deferred MMLU/Audit 및 미래 stream overlap은 미검사이며 overlap-free로 주장하지 않는다.
PSL은 composition만 계산하고 선택에는 관여하지 않는다. 전체 C4 균등표본·semantic 중복부재도 주장하지 않는다.

## Source·구현 검증 경계

Teacher execution source `{lock['source_head']}`, tree `{lock['source_tree']}`;
archive `{lock['source_archive']['sha256']}`, lock `{sha(a/'teacher.lock.json')}`.
Builder는 commit 전 실제 파일 SHA를 결속했으며 그 당시 mainHEAD에 신규 builder가 있었다고 쓰지 않았다.
Frozen source/model/tokenizer/4shards/index/dependencies/kernel/reference 포함1772 members를 CPU fullSHA 검산했다.
실제 teacher는 같은 lock을 model load 전에 다시 검증한다. FP32/eager/V128256/MB1, 8docs×24 mmap shards,
KL(p0||pW)는 vocab합→128position평균→document평균이다. Reserve/Report teacher는 생성하지 않는다.
동일 logp의 selfKL=0은 독립 W0 재실행 parity가 아니다.

`native_map`은 고정 native 직접 RHS solve와 RA 경로의 FP32 차이를 구분한다. Cholesky/inverse/SPD 대체0.
`correction`은 전체 D64에서 정한 단일 slope, 문서 mass, own anchor ball, 실제 executable trust,
RAW1/CORR1/CORR.5/CORR.25와 parent fallback을 CPU fixture로 검사했다.
**BG model adapter/persistent controller는 미완성·미검증이다.** 실제 Llama post-write gradient,
all-token model parity, branch restore, history exactlyonce/zero-write, resume, G0는 검증하지 않았다.
Toy CPU 결과를 model-level PASS로 승격하지 않는다. 기준 b/b_num·실제 BG 수치 tolerance도 아직 lock하지 않았다.
Red가 찾은 W0/server/calibration 우회·OOV/ID·source/tokenizer seal·환경버전 guard는 수정 후 CPU 검산했다.
새로운 scientific 구현·수리·제출은 다음 명시적 recall 이후이며 누락 calibration 권한/자산 해결도 필요하다.

## 비용·자원

실제 C4 CPU build {built['cpu_build_elapsed_seconds']:.3f}s. 신규 editing/calibration forward0.
Teacher 마지막 관측 allocated/RunTime0; 완료 GPU시간/peakmemory/NLL/실제 teacher bytes는 **NOT_OBSERVED**.
계획 teacher tensor 12,608,077,824B(약11.7422GiB), 예상0.25–2GPUh/요청wall2h는 측정값이 아니다.
제출 시 project active/admitted0+teacher1≤cap2, 1GPU/8CPU/60416M, source별도/exportNONE.
당시 물리GPU8개는 다른 사용자에게 할당되어 있었다. 다른 job 선점/취소/변경0.
작업 디스크여유 {resources['free_bytes']}B, 계획60GiB는 독점 filesystem 예약이 아니다.
Scheduler 기본 Requeue=1은 이 새 teacher47592만 Requeue=0으로 제출제어 정정했다.
Frozen source/과학설정/output·기존 job은 바꾸지 않았다. 자동 retry/agent callback/monitoring0.

## 인계와 재현

Local root `{a}`. Resume manifest `{resume_id['path']}`, SHA `{resume_id['sha256']}`.
Teacher output 예상 `{lock['output']}`; 종료/manifest 존재를 추가 관측하지 않는다.
원본 보고서/코드/모델/P/stats/dirty/중지 ORBODE 및 다른 paused task는 그대로 보존한다.
Reference/teacher/raw text/tokens/log는 local-only, NO_BROADCAST_NOT_REQUIRED. PNG 생성0(성능결과 없음).

재현 코드: `project/run_scripts/bg_tw_reference/README.md`; builder의 원 CLI 입력은 source-manifest 및 full-read에 결속.
`preparation_report --worktree <fresh-cleanW> --attempt <A> --control <A/new-report-control>`로 기존 입력과 새 출력 namespace에서 재생성한다.
경로/기록시각 필드가 달라질 수 있으므로 report 전체 byte-identical 재현은 주장하지 않는다.
CPU tests: `PYTHONPATH=<deps-transformers-4.44.2>:<W> /data/janghj/EasyEdit/.venv/bin/python -m unittest discover -s project/run_scripts/bg_tw_reference -t . -v`.
사용자 호출 전 polling/terminal대기/자동 결과분석/후속제출은 없으며 준비 job은 스케줄러에서 그대로 진행한다.
'''
    text_once(pub/'g0-factual-report-ko.md',report)
    reportid=identity(pub/'g0-factual-report-ko.md')
    compact=dict(instruction_id=TASK,status='WAITING_USER_RESUME',gate='G0_BLOCKED',reason='CALIBRATION_MISSING',
        reference_documents=768,CPU_tests=39,teacher_job_id='47592',teacher_last_state='PENDING',
        scientific_job_ids=[],monitoring_active=False,automatic_resume=False,resume_trigger='explicit_user_call',
        report=reportid,resume_manifest=resume_id,source=lock['source_archive'],lock=identity(a/'teacher.lock.json'))
    save(w/'tasks/status/odeedit_bg1_c4_ours_first_s4_v1/server4.json',compact)
    save(w/'runs/odeedit_bg1_c4_ours_first_s4_v1/preparation-g0-receipt.json',compact)
    text_once(w/'messages/server-heads/server4/2026-09-15-bg1-c4-ours-first.md',
        '# SH4 G0_BLOCKED / WAITING_USER_RESUME\n\n'+
        f"{TASK}: reference768/CPU39 PASS, teacher47592 lastPENDING, scientific0. N4B2/3/4/6/7/8/9 CP/delta absent; no partialmax/rerun.\n\n"+
        f"Report `{PUB}/g0-factual-report-ko.md` SHA `{reportid['sha256']}`.\nResume `{resume_id['path']}` SHA `{resume_id['sha256']}`.\n\n"+
        'No automatic resume/monitoring/analysis. Teacher job unchanged after bounded submission control. Actual BG model/G0 validity not tested.\n')
    members=[identity(p) for p in sorted(pub.rglob('*')) if p.is_file()]
    manifest=save(pub/'analysis-manifest.json',dict(scope='PREPARATION_AND_G0_ONLY_NOT_RESULTS',members=members,
        rawfree=True,model_forward_analysis=0,reproduction_command='python -m project.run_scripts.bg_tw_reference.preparation_report --worktree <fresh-cleanW> --attempt <A> --control <A/new-report-control>'))
    rooted=save(pub/'rooted-receipt.json',dict(report=reportid,manifest=manifest,member_count=len(members),
        root_sha256=sha(pub/'analysis-manifest.json'),verification='Every published member SHA/size verified',
        status='G0_BLOCKED_WAITING_USER_RESUME',scientific_promotion=False))
    for m in members:assert identity(m['path'])==m
    return dict(report=reportid,manifest=manifest,receipt=rooted,resume=resume_id)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--attempt',required=True);p.add_argument('--control')
    a=p.parse_args();print(json.dumps(build(a.worktree,a.attempt,a.control),ensure_ascii=False))
