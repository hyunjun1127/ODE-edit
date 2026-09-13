# SH2 소유 branch main 통합 보고 — 2026-09-14

## 결과와 범위

검증한 자체 scope만 main 게시 대상으로 통합했다. **모든 branch 통합 완료가 아니다.**
이번은 Git/CPU 작업이며 GPU/model/evaluator/Slurm/rsync/실험 monitoring=0이다.
기존 B-BF4 및 다른 실험 pause는 유지한다. scientific_promotion=false.

초기 remote main `7d5bae2e3be8dba87c92b86a727d5de9ed549af3` / tree `32f695a3cae93a0bef391093ba74990bba6bc1e4`.
통합 내용 commit `3bf9d1310c0809c3ff0ba424e3f6515cfa306a8b` / tree `1dc16cf125489760744bd86d2091c20975154d20`.
최종 audit를 포함하는 publication HEAD/tree는 이 보고서 자체 hash와 순환 참조하지 않도록
Git commit 및 최종 전송 ACK에서 결속한다. 실행 source를 publication SHA로 대체하지 않는다.

55개 local/remote ref, 48개 고유 HEAD를 조사했다. remote/local 중복을 독립 실험으로 세지 않았다.

| 분류 | ref 수 |
|---|---:|
| ALREADY_IN_MAIN | 39 |
| CONFLICT_REQUIRES_DECISION | 8 |
| INTEGRATED | 5 |
| NOT_READY | 1 |
| PARTIALLY_INTEGRATED | 2 |

전체 경로·HEAD/tree·원래 base·task 식별 근거는 `branch-inventory.csv`, 파일별 잔여는
`remaining-files.csv`, 원 commit→게시 commit은 `source-commit-map.csv`에 있다.
원래 reflog가 없는 base는 추정값으로 채우지 않았다. 작업 시작 snapshot 이후 새 remote branch는
이 inventory의 과거 범위에 소급 추가하지 않는다.

## 이번에 포함한 검증 범위

- fixed10k PRE_EDIT: exact12ad4cbe observer/prepare/launcher/tests. 기존 평가 결과를 새로 읽거나 재평가하지 않았다.
- Alpha JV sequential: a036641b의 세 가지 기존 receipt 포함 로직만 추가. 완료 main report와 L8 이관 사실을 바꾸지 않았다.
- FzCB: HA0 backend, Official HA1 adapter/preflight와 수치 HOLD package. Llama HA1 유효/Qwen numerical HOLD,
  HA2 미실행이라는 원래 경계를 유지했다. HOLD를 실험 PASS로 올리지 않았다.
- A/B shared functional/PCG/elastic, B protocol/runtime/FD refinement/BF4 initial-gate 재사용 코드,
  native baseline binding 및 이미 봉인된 Middle B-OS partial report. finite PCG nonconvergence와 미완료 범위 그대로다.
  SH1 common/A는 본 통합에서 제외했다. 독립 shared kernel 검증과 전체 runtime 의존성은 별개다.
- P1R52 same-B100 Official native-z 진단의 역할 추가는 자동 3-way 충돌 없이 적용했다.
  기존 runtime 다른 역할을 덮지 않았다. historical TECH-R6 비교 report는 단일 B100이며 10×B100 결과가 아니다.

## byte-preserving 역사 보고서 게시

`experiment-reports/servers/server2/fzcb-hard-alpha-densec-ha0-ha1-numerical-hold-20260914-publication-v1/`
및 `experiment-reports/servers/server2/p1r52-joint-pc-tech-r6-baseline-comparison-20260914-publication-v1/`에
각 원본 4파일을 Git blob과 byte-exact 비교하여 게시했다. 원 sealed 파일/수치/manifest는 수정하지 않았다.
각 `publication-relocation-receipt.json`이 옛 local 경로와 새 게시 경로를 연결한다.
원 manifest의 mode0600은 역사적 filesystem 사실이며 Git의100644 게시 mode와 구분한다.
원 raw 재해시나 과학 결과 재검증은 이번 Git-only task에서 하지 않았다.

## 검사와 한계

- 새/재사용 CPU fixture 47 tests PASS: B shared/controller/FD/적용/계측, FzCB model-free11, PRE_EDIT3 포함.
  SH1 소유 common은 원 6f5e5456에 결속된 read-only dependency로만 사용했다. 새 SH2 파일을 우선 import했다.
  이 환경 결속 검증은 SH1 source가 아직 없는 main 전체 runtime의 standalone 실행 PASS와 다르다.
- P1R52 기존 sequential/FZ-independent/native-z 및 Slurm memory-policy CPU tests23 PASS.
- FzCB Official import 전용2 tests는 새로 실행하지 않았다. 동일 source blob의 역사 HA0 13/13 receipt를 재사용했다.
- 최초 직접 suite에서 common 미포함에 따른 import error2가 확인되었고, 별도 dependency harness의
  parent namespace wiring error10도 보존했다(`cpu-checks.json`). harness만 수정 후47 PASS(`cpu-checks-r2.json`).
  production 수식/tolerance 변경은 없다. 초기 실패를 삭제하거나 GPU 검증으로 바꾸지 않았다.
- compile/bash/JSON/보고서 member SHA 및 47 original worktree HEAD·dirty/untracked 보존 검사는
  `publication-checks.json`. shared dirty root의 `agents/server2/`와 기존 untracked 보고서는 포함하지 않았다.
- 게시본 launcher2개만 과거65000M→현재60416M으로 resource-policy 정합화했다.
  옛 branch/source/locks/job/result는 불변. 옛 session/namespace guard가 남으므로 새 실행 허가가 아니다.
- source/config의 dataset/model 경로는 재현 참조일 뿐 이번 자산 접근/재전송이 아니다.
  새 그림·PNG나 통계 분석이 필요한 작업이 아니므로 생성0; 기존 code-generated PNG도 재생성0.

## 미통합 잔여 및 필요한 결정

8개 legacy ref는 p1_runtime/p1_scalable_batched_experiment/p1r52_sequential_runtime 등의 실제
3-way conflict가 있다. 동일패치로 이미 포함됐다고 표시하지 않았고 blanket ours/theirs도 사용하지 않았다.
일부 branch의 .codex agent 설정은 현재 위임 source scope 밖이다. P1R52 옛 report 값이 최신 sealed
main 파일과 다른 경우도 덮어쓰지 않았다. 경로별로 legacy port 또는 역사 archive 필요 여부를 정해야 한다.

R13 P1R18 branch1개는 구 launcher/test가65000M을 요구하며 그 bytes가 source manifest에 봉인되어 있다.
단순 merge 후 current resource audit PASS라고 할 수 없어 NOT_READY다. versioned lock/resource 전환이 필요하다.
두 B local/remote ref는 SH2 자체 scope가 포함되었으나 SH1 공통/A 부분은 별도 owner 통합 대상이므로 PARTIALLY_INTEGRATED다.
이 의존성을 이유로 SH1 실험을 깨우거나 다른 SH 파일을 복사하지 않았다.

## 안전성과 재현

Red pre/post: own-scope만, 오래된 main 파일 의미 보존, 원 source bytes 재사용, 잔여 공개,
raw tensor/model/checkpoint/dataset/prompt/cache/full-log/credential Git0. 보존 경로/Git reference만 게시한다.
`NO_BROADCAST_NOT_REQUIRED`: Git control/source/report 외 새 raw broadcast 없다.

재현(모두 CPU/Git-only):

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /mnt/raid5/janghj/EasyEdit/.venv/bin/python audits/servers/server2/2026-09-14-owned-branches-main-integration/cpu_checks.py
python3 audits/servers/server2/2026-09-14-owned-branches-main-integration/publication_checks.py
python3 scripts/slurm_memory_policy.py audit
```

Receipt 재생성은 새 review worktree에서만 수행한다. 원 봉인 package는 덮어쓰지 않는다.
완료 상태: 본 Git 통합 scope TASK_COMPLETE_STOP(잔여 공개); 실험 전체 완료 주장이 아니다.
