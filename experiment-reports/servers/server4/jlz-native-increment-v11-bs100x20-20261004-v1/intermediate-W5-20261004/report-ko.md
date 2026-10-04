# JLZ v11 — W5 누적 중간 결과 (2026-10-04)

사용자 직접 요청: “v11의 중간 결과, v9의 2000edit 모두 main에 push해”.
이 보고는 진행 중 실험의 고정된 W5 snapshot이다. MAIN/NOALLOC의 B1–B5만 게시하며,
현재 live 진행률 또는 W20 완료를 주장하지 않는다. 실행/자원/의존성 변경은 없다.

## 저장 범위

MAIN58034 / NOALLOC58035 각각 W5 500요청·6500행
(R500/P1000/N5000)가 저장돼 있다. 이때 summary는 누적500,
current는 동일 W5 raw에서 추린 마지막100개다. 두 필드를 혼동하지 않는다.
각 arm의 첫5 batch는 5commit/125후보/120update/history25회이며 earlystop0.
각 commit에 observer 비변이와 noCP가 기록돼 있다.
신규 matched MEMIT-H58036의 W5 결과는 이 snapshot에 없으며 NOT_MEASURED다.

## W5 누적 비교

단위 %, NLL strict preference(동률 실패). TF accuracy와 구분한다.

| 방법 | RS /500 | PS /1000 | NS /5000 |
|---|---:|---:|---:|
| V11 MAIN | 99.80 | 94.40 | 82.72 |
| V11 NOALLOC | 99.80 | 94.30 | 82.74 |
| V9 A (기존) | 99.60 | 94.90 | 82.86 |
| BASE_ALPHAEDIT (기존) | 98.60 | 89.10 | 82.20 |
| AlphaEdit-BLUE L4+L8 (기존) | 100.00 | 96.20 | 83.80 |
| CAKE (기존) | 97.80 | 80.40 | 85.02 |
| MEMIT-H S3 (기존) | 99.00 | 87.30 | 86.28 |

MAIN−NOALLOC의 성공 건수 차이는 R0/P+1/N−1이다.
MAIN−V9 A의 차이는 RS+0.20/PS−0.50/NS−0.14 percentage point이다.
과학적 우열/원인/후속 선정은 이 사실 보고의 범위 밖이다.

기존 baseline은 같은 first500 W5의 aggregate 재사용이며 runtime·층·평가기 차이가 있다.
BLUE는 L4+L8/L2=1이며 v11은 L4–L8이다.
S3 MEMIT-H를 신규 matched MEMIT-H 경로의 결과로 바꾸지 않는다.
[comparison-W5.csv](comparison-W5.csv)에 분자·분모와 evidence 수준을 구분했다.

## TF와 NLL

| 경로/종류 | TF strict | TF token-micro | desired NLL 평균 |
|---|---:|---:|---:|
| MAIN R | 499/500 | 506/507 | 0.030410 |
| MAIN P | 637/1000 | 651/1014 | 1.865783 |
| MAIN N | 961/5000 | 1031/5070 | 5.048827 |
| NOALLOC R | 499/500 | 506/507 | 0.028792 |
| NOALLOC P | 634/1000 | 648/1014 | 1.871817 |
| NOALLOC N | 963/5000 | 1033/5070 | 5.053778 |

Desired target은 R/P=new, N=true다. TF는 teacher forcing이며 자유생성 정확도가 아니다.
[metrics-through-W5.csv](metrics-through-W5.csv)에 매 batch pre/post current와 W5 all-seen을 구분했다.
[batch-accounting-through-W5.csv](batch-accounting-through-W5.csv)의 seconds는 commit에 기록된
batch 시간이며 allocated GPU초 또는 순수 fit 시간으로 바꾸지 않는다.

## 검산·provenance

CPU 재집계로 두 W5 원 raw 각6500행의 중복/누락/finite NLL,
preference 성공 건수, desired TF strict/token 분모가 저장 summary와 일치함을 확인했다.
원 raw를 Git으로 옮기지 않았으며 새로운 평가 forward/GPU실행은0이다.
별도 독립 reviewer는 사용하지 않았다.

실행 source156dd16c9341727504d0136c9a879f380adce442,
tree6556e6fbf0395b685d7e9624a2abed5bd1f504a3.
source/config/실행lock 및 실제 읽은 raw chunk의 SHA/size는
[입력 manifest](../../../../../audits/servers/server4/v9-v11-result-publication-20261004/input-manifest.json)에 있다.
모델 Llama3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2,
CounterFact fixed10k first2000, seed20261002, 각 coldW0/H0 BS100×20 계획 중 첫5batch 관측이다.
[설계](../../../../../plans/global/2026-10-04-jlz-native-increment-v11/experiment-2k/experiment-ko.md).

Local:
`/data/janghj/ODE-edit/local/jlz-native-increment-v11/20261004-v1/attempt-r2-cap3/`.
실행 code/runner는 기존 main에 게시된 그대로 유지한다. 원 source/raw/log/CPU receipts KEEP,
raw/tensor/prompt/fullstdout Git0, NO_BROADCAST_NOT_REQUIRED.
진행 중 task status를 완료로 덮지 않는다. monitoring_active=false/automatic_resume=false는
이 side 게시 작업의 경계이며 sealed runner를 중지하거나 재개하는 명령이 아니다.
