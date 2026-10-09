# Qwen 미시작 작업 평가 코드 갱신

직접 USER의 미실행 PENDING 취소·최신 평가 코드 반영·재등록 지시를 적용한다.

## 취소 및 보존

원 source `69bfbb2cdffe24072733950c671e47597ff9fbd5`의 현재 owner/Command/WorkDir/source/state를 검산했다. 61899 및 61901–61922의 미시작 23개 작업을 hold 후 후속부터 취소했다. 실행 중 FT CF 61898 / zsRE 61900은 그대로 보존한다. 원 source/raw/checkpoint/history를 변경하지 않았다.

GPT-J 평가 전용 61942–61948은 취소하지 않았다. afterany 해제 경합 방지를 위한 임시 resource hold이며 새 Qwen frontier에 연결 후 같은 ID/source로 release한다.

## 변경과 검증

새 10개 본실험만 공통 `zsre_paper.evaluate`의 Qwen public query 및 Loc request-macro를 사용한다. CF scorer/생성 일정, native fit/hparams/precision/seed/BS100×20, checkpoint는 유지한다. 기존 W0는 original receipt/raw SHA와 물리 자산/runtime/tokenizer를 확인하고 별도 consumer binding으로 참조한다. 구 관측을 새 public-query W0로 relabel하지 않는다. CF W0+W20 생성 일정은 변경하지 않는다.

CPU 43 tests PASS. source166 SHA 및 외부 task import0. 실제 Qwen tokenizer/전체2000 요청 query24858의 public AST 비교 input/target mismatch0 (R6691/P6691/N11476). native 편집 loop AST는 기존 최소 indentation repair와 동일. GPU qualification은 `NOT_RUN_USER_DISABLED`; 실제 새 GPU/온라인 PASS가 아니다.

4개 resource lane, 원 FT 2개 및 GPT-J eval-only 후속을 포함해 cap4를 지킨다. 최신 source 게시/봉인 후 신규 10 GPU + 12 GPU0 보존 단계 + collector를 held 검사/release한다. FT 보존 단계는 원 provenance의 보존만 기록하며 새 source로 소급 adoption하지 않는다. receiver 미결속/consumer 미완료면 KEEP; 전송·삭제0.

## 실제 등록·release

2026-10-09 09:14:51 UTC 전량 held 검사 후 release했다. 실행 source `5503935821b0ececb4aef09a5bccb5308879a6b5`; 등록 control `1d9fa5cf35dfff469ccccdafab03543f9f3b09a5`. source lock SHA `6c5a1ab219f4efab3d0b6b2635ff66b4a7dc77dd7f90f2ec2889e28dcfa6bb9d`, input lock SHA `f86db774f44206859794b269b640996a0fc2d9ff5bb73a36961502076c5734ec`.

| Method | CF 새 GPU | zsRE 새 GPU |
|---|---:|---:|
| MEMIT | 61954 | 61956 |
| AlphaEdit | 61958 | 61960 |
| BLUE | 61962 | 61964 |
| FE | 61966 | 61968 |
| SPHERE | 61970 | 61972 |

FT 61898/61900은 기존 RUNNING 유지. 새 GPU0 archive/KEEP 단계 61952/61953/61955/61957/61959/61961/61963/61965/61967/61969/61971/61973, collector 61974. 전량 초기 snapshot은 dependency PENDING이며 영구 user hold 없음. 정확 jobname/config/dependency는 `submission.json`/`submission.csv`에 기록했다.

최종 admission에서 새 무관 RUNNING `61951 t1-sink-patch`를 발견해 변경하지 않고 포함했다. 첫 등록 시도는 sbatch 이전 fail-closed(no ID)였고, 별도 control 수정으로 61954에 afterany:61951을 추가했다. 이미 봉인한 과학 source/archive는 변경하지 않았다. 네 lane은 기존 FT 및 이 보호 작업까지 합산 cap4를 보장한다. GPU1/CPU6/59392MiB/48h, CPU 단계 GPU0/CPU2/4096MiB/4h, exportNONE/requeue0.

기존 GPT-J eval-only 61942→61970, 61943→61972, 61944→61966, 61945→61968 resource edge만 갱신했다. 61946/61947/61948의 의존성은 보존했고 61942–61948 전부 release했다. 이들의 source/config/jobID는 그대로다.

실제 자산 preflight ready=true/blocker0, 보존 CP+네 lane atomic+reserve 계획 115,580,338,176B 충족. CPU caller→공통 W&B schema 추가 검산 W5/10/15/20 4 endpoint PASS. W&B online/실제 job ID/새 UUID 설정은 봉인했으나 dependency 대기이므로 새 run startup/remote history는 아직 관측하지 않았다. GPU actual PASS 또는 W20 완료 주장 없음.

로컬 실행 root: `/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/registration-native-eval-r2`; 로그는 그 아래 `logs/`. 원 raw/CP 및 기존 archive 모두 KEEP, 전송/삭제0. README는 GH 단독 통합. 대규모 raw 공유 불필요: `NO_BROADCAST_NOT_REQUIRED`. 이후 반복 모니터/자동 retry 없이 인계한다.
