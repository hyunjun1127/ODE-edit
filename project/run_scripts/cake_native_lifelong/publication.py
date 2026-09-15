"""Compact create-once submission publication; reads no scientific output/job."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .preflight import TASK_ROOT, file_identity


def write(path, text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(text)


def js(path, value):
    write(path,json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);a=ap.parse_args()
    repo=a.repo;task=TASK_ROOT
    pre=json.loads((task/'preparation-v1/preparation.json').read_bytes())
    lock=json.loads((task/'execution.lock.json').read_bytes())
    release=json.loads((task/'release.json').read_bytes())
    held=json.loads((task/'submission-held.json').read_bytes())
    assert held['job_id']==release['job_id']=='48101'
    report=repo/'experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1'
    audit=repo/'audits/servers/server4/2026-09-15-cake-native-lifelong'
    report.mkdir(parents=True,exist_ok=False);audit.mkdir(parents=True,exist_ok=False)
    for src,name in [(task/'preparation-v1/preparation.json','preparation.json'),
                     (task/'preparation-v1/full-read.json','full-read.json'),
                     (task/'preparation-v1/upstream-inventory.json','upstream-inventory.json'),
                     (task/'preparation-v1/unused-notebooks-import.patch','unused-notebooks-import.patch'),
                     (task/'execution.lock.json','execution.lock.json'),
                     (task/'submission-held.json','submission-held.json'),
                     (task/'held-inspection.json','held-inspection.json'),
                     (task/'release.json','release.json')]:
        with (audit/name).open('xb') as f:f.write(src.read_bytes())
    resume=dict(instruction_id=lock['instruction_id'],job_id='48101',job_name='odeedit_cake_native_lifelong_s4',
        owner='SH4',session=pre['session'],host='server4',registered_cwd=pre['registered_cwd'],
        execution_source=lock['source'],source_archive=file_identity(task/'source-execution.tar'),
        execution_lock=file_identity(task/'execution.lock.json'),upstream=lock['upstream'],
        last_observed_utc=release['time_utc'],last_observed_job_state=release['status'],
        agent_status='MONITORING_PAUSED_AWAITING_USER',initial_native_batch='NOT_RUN_OR_NOT_OBSERVED_AT_PENDING_HANDOFF',
        monitoring_active=False,automatic_resume=False,resume_trigger='explicit_user_call',
        output=lock['output'],expected_terminal_path=str(Path(lock['output'])/'terminal.json'),
        expected_initial_path=str(Path(lock['output'])/'initial-execution.json'),
        output_existence='NOT_REOBSERVED_AFTER_PENDING; no live output inspection',
        committed_batches='NOT_OBSERVED',checkpoint_storage=lock['checkpoint_storage'],exact_restart_available=False,
        remaining='Submitted program B1..B100/evaluation; detailed analysis only explicit completion recall',
        dependencies='none; initial project active+admitted pending=0, requested1/cap2',scientific_promotion=False)
    js(task/'resume-manifest.json',resume);js(report/'resume-manifest.json',resume)
    cpu=dict(status='CPU_SOURCE_RESOURCE_PREPARATION_PASS_NOT_GPU_VALIDITY',fixtures=9,
        syntax_python_files=7,shell_syntax='PASS',memory_policy='PASS 60416MiB/1GPU',
        frozen_member_fullsha=dict(count=1959,bytes=8287575787,status='PASS'),cuda_initialized_during_import=False,
        import_compatibility=dict(original='ModuleNotFoundError: notebooks',execution='PASS after unused import removal + final LF',native_AST='otherwise identical'),
        data='official load_prefix10000/verify/full record SHA/order/inventory PASS',
        common_context='native get_context_templates AST equal; saved contexts33cec0ee reused; no regeneration',
        model='revision/config/shard readability+size reuse; new full model shard rehash0',
        projector='file fullSHA/current size verified, five-layer prior tensor identity bound; actual tensor shape/identity check remains in submitted runtime',
        native_gpu_execution='NOT_OBSERVED',continuation_replay='NOT_TESTED',
        focused_auditor='parent SH4; bounded worker spawn unavailable: agent thread limit; no independent red/GPU PASS claim',
        original_disk_hold='observed39,098,310,656B <63,417,876,480B tensor floor; user removed checkpoint storage',
        admission_available_bytes=held['available_bytes'],raw_reserve_bytes=lock['raw_reserve_bytes'],
        disk_change_attribution='external filesystem change, no SH4 deletion/transfer',
        active_and_pending_other_gpus=held['other_active_plus_pending_reserved_gpus'],
        deferred='first native batch/history/actualstate; full completion and scientific analysis',
        broadcast='NO_BROADCAST_NOT_REQUIRED',source_and_raw_of_other_tasks_changed=False)
    js(audit/'cpu-resource-review.json',cpu)
    compatibility={k:pre[k] for k in ['hparam_differences','planned_environment','upstream_readme_environment','projector_mapping','context','context_function_ast_equal','baseline_lock','baseline_runtime','baseline_source_head','baseline_source_tree']}
    js(report/'compatibility.json',compatibility)
    text=f'''# CAKE native lifelong 제출 사실 보고

상태: **48101 제출·held 검사·release, 마지막 관측 PENDING; MONITORING_PAUSED_AWAITING_USER**.
GPU 실제 native batch/성능/전체 완료를 아직 관측하지 않았다. 다른 EP/BLUE/REFIT4 작업은 재개하지 않았다.

## 최신 사용자 저장 override

“CAKE 부분은 weight 저장 하지 말고 그냥 올려라”를 적용했다.
W/M tensor checkpoint를 디스크에 저장하지 않는다. 5개 down_proj weight와 5개 cache_c는
단일 process 메모리에서 자기 이전 batch state를 이어받으며, 열린 batch rollback도 메모리에서만 수행한다.
원래12개 checkpoint 시점은 **평가 시점으로 유지**한다. Hash/context/RNG/commit 기록은 남지만
가중치/history 자체가 없으므로 재구성 가능한 checkpoint·정확 restart·continuation PASS가 아니다.
기존 CP 삭제·이관·공유자산 변경은0이다. 원래63,417,876,480B tensor floor 디스크 부족 관측은 보존했다.
최신 admission available={held['available_bytes']:,}B, raw/source/log reserve=17,179,869,184B(16GiB, 추정여유)이다.
그 사이 filesystem 가용량 증가는 SH4가 삭제한 결과가 아니다. 이후 공간이 늘어도 최신 no-checkpoint 지시는 유지한다.

## 원본 method와 source

- 원본 CAKE c8243e1d7e43ca9cf64d552f96221fcb9561aac2 / tree4f59249bb23c7cacf0f6490bce8127c74ff7b111 / MIT.
- 실행 ODE hook {lock['source']['head']} / tree{lock['source']['tree']}. 분석·제출 publication commit과 구분한다.
- Archive983589f0b6ca5dcca85508bdaa2d5a3a1d38128697ec89dedf54f5ede1f16ebf (4,003,840B).
- Lockf2ade3ee9dbdabdc6a1ac00a9d36b0e902710a44cf5e2a6a49c6d861e18eb401.
- 원 `Cake_main.apply_Cake_to_model`을 batch당1회 직접 호출한다. 최종L8 target100→L4..8 현재 residual/native directsolve→최종 각층 history1.
  wrapper의 추가 finalize/history append, 다른 method solver, target cache 재사용은0이다.
- 원본에 없는 notebooks.util의 미사용 import로 CPU import가 실패했다. 실행 복사본에서 그 한 줄 제거와 EOF LF만 변경했다.
  원본 clone은 clean 보존했고 함수/나머지AST 동일을 확인했다. 호환 patch는 audit에 별도 봉인했다.
- Original layers4..8/L2=10/decay.4/clamp.5/temperature.1와 원 causal_scores0..4를 유지한다.
  Physical4..8→P asset0..4→local0..4. 점수 key를 physical layer로 재색인하지 않았다.

원본 README 환경torch2.6.0/transformers4.51.3과 실제 재사용환경torch2.9.1+cu128/transformers4.44.2는 다르다.
기존 baseline과 FP32/eager, matmulTF32false/cudnnTF32true, seed20260907을 결속했다.
Writer는 기존 add_bos_token=false/right padding; evaluator는 기존 별도 tokenizer 및 kernel의 manual left-padding/MB16 그대로다.
CAKE 원 context 생성함수와 기존 native 생성함수AST가 같아 기존 context bytes를 제공한다(재생성0).
이는 CAKE README 환경 전체와 같다는 주장이 아니다. BaseAlphaEdit clamp.75/decay.5와도 다르다.
Layer weighting 하나만 바꾼 통제 비교로 해석하지 않는다.

## 고정 stream·평가·저장

Llama revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, freshW0/coldcache_c0.
고정 counterfact SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1,
whole orderedroot5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729를 공식loader로 검증했다.
B1=[0,100), …, B100=[9900,10000);10000unique,100×B100. 각batch case/request/target/prompt/order hash를 lock에 저장했다.

매batch actual editedW에서 CurrentR/P/N 및 기존 baseline의 all-seen rewrite를 유지한다.
B1/5/10/20/30/40/50/60/70/80/90/100에서 actual full-seenR/P/N.
최종W100 분모10000/20000/100000을 실제cardinality로 확인하고 한번만 평가한다.
Current와 fullseen의 같은state rows는 identity로 합쳐 중복forward/분모를 피한다.
RS/PS newNLL<trueNLL, NS trueNLL<newNLL, ties=failure; TFstrict/token은별도다.
평가결과는 target/causal allocation 입력이 아니다. W0는 기존42673 publication 재사용, 새GPU W0평가0.
GLUE/MMLU/downstream/다른모델/다른baseline/새 stats 생성은0이다.

## 최소 준비·자원·미검증

CPU9 fixtures, Python AST/shellsyntax/import/config/unused-import native AST, 공식data/order,
선택closure1959 members/8,287,575,787B fullSHA 검산 PASS. Import 중 CUDA initialized=false.
모델전체shard는 기존 revision/closure와 가독성·size를 재사용했으며 새 전체content rehash는 하지 않았다.
P/stats fileSHA는 이번선택검산에 포함, P tensor schema/identity 및 실제 native write/history 검사는 제출 프로그램의 첫batch에서 확인한다.
별도GPU smoke/FD/ULP/gradient parity gate는 실행하지 않았다. 기존타task 수치PASS를 CAKE로 전용하지 않는다.

요청1GPU/8CPU/60416M/exportNONE/Requeue0/48h; GPUhour hardcap=null, cap2.
Fresh admission의 다른 project active+admitted pending=0. 하나의 chain만 등록했다.
Held 상태에서 owner/job/command/source/args/memory/GPU/48h/no-requeue를 확인해 release했다.
마지막 관측은 {release['time_utc']} / {release['status']}; 이후 scheduler/log/output을 다시조회하지 않았다.

계획상10000 request-z, 최대250000 loss/240000 Adam,500solve,100whole-history passes=500layerappends.
실제조기종료/시간/메모리/성능은 아직미측정이다. 총 wall은48h 제한으로 요청했으나 CAKE speedup 추정이나48GPUh 예산으로 부르지 않는다.
Target/keys/solve는 전체write wall 내부 nested component이며 단순합산하지 않는다.
평가계획의 repeated state-population 관측은 RS505000/PS128800/NS644000 rows, NLL pair2배=2,555,600 teacherforced sequences이며
unique sample수가 아니다. CAKE 실제초기실행/완료시 비용계측으로 확인해야 한다.

## 경로·인계

Local `{task}`. Output `output/main/`, logs `logs/48101.out` 및`.err`.
예상 terminal `output/main/terminal.json`; 최초 marker `output/main/initial-execution.json`.
현재 산출물 존재/commit은 pending인계 후 재조회하지 않았다.
`resume-manifest.json`: monitoring_active=false, automatic_resume=false, explicit_user_call만 재개.
100batch 프로그램/평가/기록은 자연진행하며 agent heartbeat/callback/자동분석/추가submit0.
Git에는 source/raw-free metadata만, rawNLL/로그는local. NO_BROADCAST_NOT_REQUIRED.

후속비교는 `blue-native-lifelong-comprehensive-review-2026-09-11-v1/diagnostic-report-ko.md`와
W0/BASE_ALPHAEDIT/BASE_MEMIT/BLUE/가용single-layer를 동일10kW100·호환성 범위에서 연결할 예정이며
사용자 완료recall 전에는 결과분석을 시작하지 않는다. Scientific promotion=false.
'''
    write(report/'submission-factual-report-ko.md',text)
    write(audit/'focused-review-ko.md','''# SH4 focused CPU/source/resource 검토

최신 사용자 no-weight-storage override는 W/M checkpoint 파일 생략으로 명시했고, history 실행 누적/평가규약은 유지했다.
원 source의 import 한 줄+EOF LF 외 AST 변화0. 실제 first native 실행은 NOT_OBSERVED이며 CPU9/import/fullSHA를 GPU PASS로 승격하지 않았다.
Cap2에 기존 active+pending까지 계수한0+1을 확인하고 held-inspect-release했다. 별도GPU gate0.
원native/config/sample/model/P/hparams 공유변경0. source/코드·smallmetadata 외 raw Git0; 자동모니터링0.
Bounded 독립 worker는 thread limit으로 생성되지 않아 parent SH4가 focused검사를 수행했다. 독립 red PASS라고 하지 않는다.
Generic agent-access script가 runs/<task>/ 경로를 server-head 허용표에 포함하지 않는 부분은 이번 명시 envelope의 정확 own runs 경로만 예외로 기록한다. 공유정책 수정0.
Runtime commit7884aeb와 후속 submission/publication source를 구분한다. Reproduction은 CPU preparation/tests/hash검산에 한정하며 Slurm/GPU 재실행 지시가 아니다.
''')
    note=f'''# SH4 → GH CAKE 제출 / monitoring pause

Instruction {lock['instruction_id']}. 사용자 최신“weight 저장하지 말고 제출” 적용: W/M checkpoint0, history는메모리에서누적, 평가schedule변경0, exactrestart불가.
48101/odeedit_cake_native_lifelong_s4 held검사→release, 마지막{release['status']},초기GPU관측NOT_RUN/NOT_OBSERVED.
Source{lock['source']['head']}, lockf2ade3ee9dbdabdc6a1ac00a9d36b0e902710a44cf5e2a6a49c6d861e18eb401.
원CAKEc8243e1/layers4..8/L2=10/decay.4/clamp.5/temp.1, fixed10k B100×100, 1GPU8CPU60416M48h/cap2.
Report `{report.relative_to(repo)}/submission-factual-report-ko.md`; local `{task}`.
CPU9/import/source-assetSHA1959PASS;GPU모델write/결과미관측. MONITORING_PAUSED_AWAITING_USER/automatic_resume=false. 원EP/다른task변경0.
'''
    write(repo/'messages/acks/server4/2026-09-15-cake-native-lifelong.md',note)
    write(repo/'messages/server-heads/server4/2026-09-15-cake-native-lifelong.md',note)
    js(repo/'tasks/status/odeedit_cake_native_lifelong_s4_v1/server4.json',resume)
    js(repo/'runs/odeedit_cake_native_lifelong_s4_v1/submission.json',resume)
    write(report/'reproduce.md','''# CPU 재현만 (새 GPU/submit 지시 아님)

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/source-tech-r2 /data/janghj/EasyEdit/.venv/bin/python -B -m unittest project.run_scripts.cake_native_lifelong.test_preflight project.run_scripts.cake_native_lifelong.test_runtime`

`preflight.py --repo <worktree> --output <task root의 새 preparation namespace>`는 source/data/hash/storage CPU검사만 한다. timestamp/available수치는 새관측이므로 동일bytes 재생성주장0.
Full-read/upstream/lock/held/release identity는 audit JSON으로 검산한다. publication.py는 새create-once 패키지를 만들며 기존패키지 overwrite0.
Runtime/source archive는 원native 실행commit을pin한다. 최초GPU관측/완료는 사용자명시recall때만 확인한다.
''')
    paths=[p for root in [report,audit,repo/'project/run_scripts/cake_native_lifelong'] for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    manifest=dict(scope='CAKE own source + raw-free preparation/submission only',execution_source=lock['source'],
                  members=[dict(path=str(p.relative_to(repo)),bytes=p.stat().st_size,sha256=file_identity(p)['sha256']) for p in sorted(paths)])
    js(report/'analysis-manifest.json',manifest)
    js(report/'rooted-receipt.json',dict(manifest=file_identity(report/'analysis-manifest.json'),
        report=file_identity(report/'submission-factual-report-ko.md'),resume=file_identity(task/'resume-manifest.json'),
        status='SUBMITTED_PENDING_MONITORING_PAUSED',scope_complete='compact submission only, scientific results pending',
        raw_git=0,new_scientific_jobs=1,scientific_promotion=False))
    print(json.dumps(dict(report=file_identity(report/'submission-factual-report-ko.md'),
                         manifest=file_identity(report/'analysis-manifest.json'),receipt=file_identity(report/'rooted-receipt.json'),
                         resume=file_identity(task/'resume-manifest.json')),ensure_ascii=False))


if __name__=='__main__':main()
