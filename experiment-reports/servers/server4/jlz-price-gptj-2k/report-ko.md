# GPT-J PRICE 6-arm — EasyEdit hparam 교정·재제출

최신 사용자 지시에 따라 EasyEdit GPT-J 현물 hparam을 채택해 6개 cold first2000 BS100×20을 새 source로 등록·held 검사·release했다. 초기 snapshot은 모두 PENDING이다. 실제 B1/W20 및 새 W&B 원격 identity는 NOT_OBSERVED다.

| 항목 | 취소된 이전 설정 | 새 실제 설정 |
|---|---|---|
| 편집 층 | L4–L8 | L3–L8 |
| optimizer lr | 0.1 | 0.5 |
| AlphaEdit L2 | 1 | 10 |
| H append 예상/arm | 100 | 120 |

| writer | CAP075 | CAP100 | FREE100 |
|---|---:|---:|---:|
| MEMIT | 60134 | 60135 | 60136 |
| AlphaEdit | 60137 | 60138 | 60139 |

GPU0 collector는 **60140**이다. MEMIT 60134→60135→60136, Alpha 60134→60137→60138→60139의 afterany DAG다. 첫 MEMIT은 공통 native context 입력을 준비하며 각 arm은 독립 cold W0/H0다. collector는 여섯 새 GPU ID 모두 afterany다. 취소된 old ID를 dependency로 재사용하지 않았다.

## 취소와 수리 근거

EasyEdit `/data/janghj/EasyEdit/hparams/MEMIT/gpt-j-6B.yaml`과 `hparams/AlphaEdit/gpt-j-6B.yaml`은 실제 L3–L8/lr .5/Alpha L2 10이다. 이전 구현은 YAML을 읽었지만 ours override를 유지했고, fit도 optimizer default .1을 사용했다. 최신 사용자 지시로 이 세 hparam을 ours PRICE method에 연결했다. stock residual/divisor/native Adam 등 다른 방법은 수입하지 않았다.

원 GPU 60112–60117와 collector 60118은 owner/source/argv/script/zero elapsed/accounting no-start/no-allocation을 확인했다. Slurm Backfill의 미래 StartTime 예상은 실제 시작 이력과 구분했다. collector/후속부터 hold 후 reverse-topological 취소했으며 모두 CANCELLED·allocated GPU-sec 0이다. source `52263371`, config/archive/raw는 그대로 보존한다. **60001 및 Llama 60102–60108 변경 0**, Qwen 재개 0이다.

새 source **`ae507064421876af6fbf04231cf43abcb2cb0a81`**, config SHA **`c93fdcab97b0686fce3ae5d2b98caacc6ed0a9306807516f4325f21061c4e00c`**, lock SHA `acb37a03142d197adb4c7fba315596d9494a0a05e1f8bbace0cc4e8c96e2bfbc`다. 원 model/projector full-SHA+fresh stat를 재사용하고 새 L3 C0 및 cold weight만 추가 결속했다. 새 GPT-J namespace에 Alpha `10I+NH`/thin Woodbury/`10Q+N(HQ+KKᵀQ)` residual/LOO를 맞췄고 commit·reducer도 같은 λ/lr를 검산한다. Llama 공유 코드 수정 0이다.

6층 host peak **54.3615GiB**, GPU peak **65.9788GiB**는 source/input 추정이며 실측이 아니다. 각 1GPU/8CPU/59392MiB/48h 요청, hard60416MiB, exportNONE/Requeue0. 실제 기존 Llama 자원 DAG 폭1+새 GPT-J 폭2≤cap3을 검산했다. 동시 disk reserve 36,590,583,808B, preflight free 39,293,767,680B 수준으로 여유가 작다. 기존 batch-boundary guard를 유지하며 부족하면 typed RESOURCE_BLOCKED_STORAGE; 무관 삭제/로그 생략/자동 retry 없음.

## 검토 수준과 완료 경계

소스 33개 AST/import/CLI/profile·입력·자원 검산과 좁은 독립 정적 리뷰(`gptj_hparam_paths`, `gptj_pending_replace`)를 수행했다. 확인된 blocker 0이며 **실제 GPU 수치/gradient/성능 PASS는 아니다**. 별도 toy/pilot/fit/새 온라인 smoke는 0이다. 실제 endpoint/KKT/feasibility/payload/H 검사는 승인 main B1에 통합한다. 20 commits/19 joins/120 H per arm, 전체 H720을 기대하나 아직 관측하지 않았다.

W&B producer/helper 및 기존 B4/B5/W0 CPU 검산을 exact-byte 재사용한다. 새 run은 current100과 실측 all-seen, 모델/작가/arm/실제 job ID, percent/NLL/축/immutable identity를 구분한다. SDK 접수≠remote 인증, pending≠B1 완료다. 세 cap/base 및 PRICE 수식·KL/norm/C0·25eval24update·분모·NoCP는 그대로다. source/publication SHA와 executed source를 혼동하지 않는다.

[제출·취소 compact receipt](../../../../audits/servers/server4/jlz-price-gptj-2k/easyedit-hparams-20261007/submission-receipt.json), [새 job/dependency 기록](../../../../runs/jlz-price-gptj-2k/easyedit-hparams-20261007/submission.json), [상태](../../../../tasks/status/jlz-price-gptj-2k/server4.json), [Llama/GPT-J 등록 이력 ledger](current-cell-ledger.csv). Llama 행의 성능/상태는 과거 snapshot이며 이번에 재조회하지 않았다. 원자료와 이전 receipt KEEP, NO_BROADCAST_NOT_REQUIRED: 소형 source/보고/manifest만 Git. 등록 뒤 단일 초기 snapshot에서 인계했고 recurring monitor/heartbeat/autoretry=0이다.

## 이전 60112–60118 제출 기록 (역사)

아래는 교정 이전 source의 이력이며 현재 실행 설정/현재 job으로 해석하지 않는다. [이전 상세 제출 보고](../price-model-runs-tracking/report-ko.md)는 당시 역사 기록이다.

## 구현과 확인

- EasyEdit GPT-J 모델·native stats·projector·hparams를 결속하고 native parallel attention/MLP·fc_out bias·readout27에 ours hook을 적용했다. 기존 과학 계수와 cap 설정은 유지한다.
- native context 5개는 첫 실제 arm의 입력 준비에서 한 번 생성·고정한다. 별도 fit/pilot 없음. GPT-J exact W0가 없어 첫 arm에서 first2000을 관측하고 나머지는 identity가 일치할 때만 재사용한다.
- 기존 60001의 봉인 B4/B5/W0 raw CPU 검산에서 current 분모 R100/P200/N1000, B5 누적 R500/P1000/N5000, W0 R2000/P4000/N20000을 확인했다. 원 raw/stored aggregate/payload 일치. B4에는 all-seen key 없음. 기존 job이나 run 변경 없음.
- 새 caller는 scientific run에 current/pre·current/post·실측 all_seen/post·W0_first2000 및 W0 같은-cohort N 비교를 직접 기록한다. R/P/N 경로 분리, 백분율, harmonic, true-new margin, edits 축과 pre-state 위치를 구분한다. 중복 companion daemon을 자동 실행하지 않는다.
- SH1 job-identity helper `bc63425e` 채택. 실제 job 이름/Config 검증을 유지한다. immutable startup identity와 transport 상태를 분리하고 finish 후 bounded readback을 준비했다. 실온라인 검증은 NOT_OBSERVED.
- CPU fake-SDK/metadata 검사 25개 중 24 PASS, 1 SDK환경 검사 SKIP 후 task-local readback 검사 1개 추가 PASS. 소스 31개 AST/import/config 검사 통과. 이는 과학 toy/GPU/model PASS가 아니다. owner audit이며 별도 reviewer 없음.

## 제출 결속

최신 사용자 권한이 중앙 helper 표식 대기 조건을 해제했다. 동시에 실제 게시된 SH1 helper와 caller를 CPU 원자료/fakeSDK 검산하고 원자적으로 봉인했다. SH4가 공통 helper를 수정하지 않았다. helper receipt의 SDK 접수와 실제 remote readback은 구분한다.

실행 source `5226337121fd2c90c595f26297c9927400f8f0af`, config SHA `7331db77a872095f49472345de69f94ed8ff9d790796823b9d255ea3d563c6dc`다. fresh cap3·task2·storage/resource admission 후 여섯 GPU job과 GPU0 collector를 held 검사/release했다. MEMIT_CAP075가 native 입력을 준비하고 MEMIT 및 Alpha 각각의 자원 lane은 afterany로 이어진다. Llama 자원 폭1과 GPT-J 폭2의 합은 최대3이다. 과거 임시 freeze는 stale evaluator SHA로 **Slurm 호출 전** 실패했으며 원본은 보존했다. 최신 source/config/lock/archive는 새 attempt로 봉인되어 immutable이다.

noCP/exact resume NOT_AVAILABLE. 기존 Qwen STOP·Llama job 보존. 원자료/credential/model/SDK spool Git 업로드 없음. NO_BROADCAST_NOT_REQUIRED: 소형 source/receipt만 공유한다.
