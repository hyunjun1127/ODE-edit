# JLZ two-arm: 구현·제출 인계

본 문서는 새 A/B 과학 결과 보고가 아니라 승인된 실행의 준비·등록 사실이다.
η=0/1 외의 목적함수·reference·route·예산은 공통이며 각 arm은 fresh W0/H0에서 독립 진행한다.

## 범위와 상태

- 상태: `INITIAL_GATE_PASS_MONITORING_STOPPED`
- 실제 job mapping: `{'collector': '56962', 'main-A': '56960', 'main-B': '56961', 'pilot-A': '56958', 'pilot-B': '56959', 'prep': '56957'}`
- Actual GPU 검증: `REPRESENTATIVE_MAIN_B1_FINAL_COMMIT_5_HISTORY_OBSERVER_RESTORE_B2_OWN_ENTRY`

공통 W0 teacher/기술 준비 뒤 BS4×2 A/B pilot과 BS100×20 A/B main을 각각 두 독립 1GPU lane으로 구성한다.
각 GPU job은 8CPU/60416MiB, project/task cap2, exportNONE/Requeue0이다. 과거 모든 task·job은 불변이다.
source freeze/held 검사/release와 actual scientific validity는 별개다.

## 정독·입력·구현

정본7개 및 실행/2GPU overlay를 전체 읽고 exact SHA/size 검산했다. 같은 S4 모델 revision/C0/context/data를
CPU fullSHA/shape로 결속했다. Reference pool은 11919→11544→11134→10715문항/34relation;
2pilot+20main의 문항·reference order는 outcome 전에 고정했다. 새 prompt/teacher/tensor는 Git에 넣지 않는다.

구현은 inactive current NLL/KL을 포함하며 reference token→prompt 평균에 B를 곱한다.
모든 token의 실제 다층 FP32 weight 경로와 dX를 보존한다. M=PᵀAP의 small/direct 검산과 ηΩ가 독립 CPU 검사를 갖는다.
SPG/BB/prox/linesearch는 원 수학을 유지하며 final original oracle 1회를 동일 cap32/120 안에 예약한다.
유한 수치차이는 원 PASS/FAIL을 보존한 기록 전용이다. Identity/IO/state/clamp/초기·최종 비유한 값은 여전히 차단한다.

## 재사용 baseline 첫 독립 CPU 표

| W20 baseline | R (2000) | P (4000) | N (20000) | 재사용 경계 |
|---|---:|---:|---:|---|
| BASE_ALPHAEDIT | 1986 | 3729 | 13718 | local paired raw + B020 endpoint exact |
| BASE_MEMIT | 1295 | 2468 | 10365 | local paired raw + B020 endpoint exact |
| MEMIT-H | 1988 | 3640 | 15809 | S3 completed B020 source/commit bridge; raw state field 미저장 |

각 raw26000행의 case/prompt/target/order/finite/strict tie-failure/TF 분모를 독립 CPU reducer로 확인했다.
이 표는 새 JLZ 결과가 아니다. 역사 TF4.44.2와 이번4.57.1, H200/Blackwell 및 TF32 차이는 그대로 기록한다.
따라서 matched speedup은 주장하지 않는다. 비교용 새2k baseline fitting은0, 같은 first8 BS4×2 기능 확인만 별도 비용이다.

## 저장·자원·미측정

신규 W/H/R/delta/optimizer checkpoint0, exact_resume=NOT_AVAILABLE. RAM transaction만 사용한다.
저장 teacher는 immutable W0 reference input이며 edited-state checkpoint가 아니다.
GPU peak 계획80GiB/host48GiB는 **예상**이며 actual은 Slurm program receipt로 별도 남긴다.
host 요청은59GiB 이하, disk reserve20GiB, finite wall7days는 ETA나 과학 callbudget이 아니다.
최초 actual fixed-candidate comparison 뒤 공통 route를 성능결과 아닌 시간/메모리로 고정한다.
기술≤6 B100 calls, pilot≤128, main≤4800 및 teacher/native/observer 비용은 분리한다.

대표 main B1 finalcommit+5history+observer restore→B2 ownentry가 실제 확인되거나,
전체등록/release 뒤 실제 resource부족으로 시작하지 못하면 agent monitoring을 중단한다.
등록된 runner/collector만 자연 진행하며 이후 상세 리뷰는 사용자 recall에서 수행한다.

NO_BROADCAST_NOT_REQUIRED: 동일 S4 실행이며 승인된 작은 completed-baseline 파일 이외 대형 전송0.
자기검산과 독립 red CPU/source 감사를 사용했다. 별도 red GPU 감사는 수행하지 않았으며 actual 범위는 상태 receipt에 한정한다.

## 재현

`python -m unittest discover -s project/run_scripts/jlz_two_arm -t . -p 'test_*.py'`

실행은 전용 immutable local attempt의 config/lock/archive/launcher와 제출 receipt를 따른다.
원 실행/평가 source는 수정하지 않고 새 task-local namespace만 구현했다.

## 대표 MAIN B1 독립 집계 / 초기 인계

대표 `main-A` / job `56960`. B1 실제 commit/5history/observer 복원과 B2 own entry만 확인했다. 두20batch 완료가 아니다.

| 패널 | 성공/분모 | TF token-micro | TF strict | true NLL | new NLL |
|---|---:|---:|---:|---:|---:|
| N | 887/1000 | 0.194175 | 171/1000 | 5.462566 | 11.537661 |
| P | 136/200 | 0.346535 | 68/200 | 7.002502 | 3.889782 |
| R | 100/100 | 1.000000 | 100/100 | 12.691668 | 0.003230 |

R/P 성공은 new NLL<true NLL, N은 true NLL<new NLL이며 tie는 실패다. TF는 원하는 completion의 teacher-forced accuracy이며 자유생성 정확도가 아니다.

초기 인계 이후 monitoring_active=false/automatic_resume=false. 이미 등록된 두 main/collector만 자연 진행하며 완료 상세리뷰는 사용자 recall에서 수행한다.

## 준비 실패와 최소 수리

원 prep56918은 native fitting 전 source inventory의 가상 상대 `_ops.py` 경로 오류로 실패했다. 종료 inventory에도 동일 오류가 발생해 terminal은 미저장이나 첫 실패/restore/stdout/stderr는 보존했다. 원 비용은 parent1116 GPU초이며 후속4개는 시작 없이 dependency 취소됐다. 원 W0 teacher/3회 비교/26000행 W0 관측은 exact source/input/state와 별도 reuse bridge로 재사용한다. 새 실행은 공유 계산을 반복하지 않으며 baseline fitting은 이전0회다. CPU 회귀는 actual 모델 성공을 뜻하지 않는다.

## GH 전달 경계

GH 직접 메시지 호출은 도구 제공 중단으로 전달되지 않았다. 별도 전송 성공/ACK를 주장하지 않는다. SH4 server-heads 메시지와 본 보고서·receipt를 own-scope main에 게시한 Git 인계를 남겼다. 직접 전달 실패 때문에 실험 조회나 자동 재개를 하지 않는다.
