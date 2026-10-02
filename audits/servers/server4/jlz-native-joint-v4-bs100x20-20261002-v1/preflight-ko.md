# JLZ v4 구현 전·제출 전 owner 검산

Instruction: `ODEEDIT-USER-GH-SH4-JLZ-V4-COMPUTE-R1-2K-20261002-R1`.

## 정본·입력

authority `8ab5d0ad13d43e4594ff97bc3cea2c5673a501e2` 전용 clean worktree를 사용했다.
원 root의 1390개 tracked 삭제는 변경하지 않았다. 원 method/TeX/portability와
별도 experiment, native 입력 정의를 전체 읽었다. `receiver-full-read.json`은
archive 249315B와 33개 regular member 수신 SHA/size 검증이다. 과거 정책/reference는
package 안에서만 보존했고 현재 repo를 덮어쓰지 않았다.

`build_plan.py --check --dataset <S4 fixed10k>`는 2000개 전행/전필드, 순서,
20개 batch 경계, active1984/superseded16 및 동일 evaluation schedule을 대조했다.
CPU tokenizer는 pilot 2개 + main 20개 pack의 teacher-forced/target-free prefix
identity를 확인했다. 모델·C0·context·dataset 17개 자산의 이전 fullSHA 재사용과
현재 fullSHA를 config에서 구별했다. 다른 baseline 실행·이전 W/H 재사용은 없다.

## 소스 대조

| 계약 | 구현 | 검산 수준 |
|---|---|---|
| native six-context 평균/요청 합, current‖entry KL | inputs/oracle | pinned preparer 재사용·CPU synthetic |
| δ=0 graph·전층 joint gradient | adapter/oracle | zero/nonzero gradient CPU |
| C0 FP32 raw/count→FP64 | allocation.system | source 및 schema |
| same-A right solve/비대각 결합/가변 B | allocation | B1/3/24/31 CPU direct 비교 |
| 25 candidate/24 Adam/native clamp | optimize | 두 eta CPU 전체 25회 |
| strict prefix subject 미포함·RoPE/mask | entry | 작은 causal Llama CPU full-reference 비교 |
| 전체 token current key writer/H 한 번 | writer | CPU materialization/final key/H 대조 |
| 5층 W/H RAM rollback | writer.Transaction | 축소 3층 fixture의 bitwise 복원 |
| observer 전용·tie 실패·finite | observe | CPU reducer/source |
| single schedule/noCP | run/collect | source 대조, actual 미실행 |

실제 검사는 owner source 검산과 CPU 6 tests다. **별도 독립 red를 실행했다고
기록하지 않는다. CPU toy를 actual Llama/GPU PASS로 확대하지 않는다.** 실제 zero/
nonzero 후보 reference·geometry·writer/history 비교는 승인된 작은 pilot에서 수행한다.
수치 초과는 원값을 기록하고 같은 목적 reference로 전환한다. 낮은 성능·집중도·
realization residual은 중단 기준이 아니다.

## 자원·운영

새 task GPU 동시 cap2. 1GPU/8CPU/60416MiB, CPU collector 8CPU/24576MiB.
공통 W0→A/B pilot→A/B 독립 timing+main→afterany collector 구조다.
2026-10-02 제출 준비 시 server4 own queue는 비어 있었고 resource helper는
active_project_gpus=0/cap2로 ALLOW였다. 이는 실제 job 할당 보장이 아니다.
등록 직전·release 직전에 다시 admission을 확인한다. 다른 job은 변경하지 않는다.

host 예상 peak44GiB, GPU 예상65GiB; 실제 P2 수치는 아직 미측정이다.
disk 관측 여유 약192GiB, raw/temp 안전 reserve30GiB. 새 W/H/optimizer/factor
checkpoint는 저장하지 않는다. source/input은 immutable, rollback/cache는 RAM만 사용한다.
GPU wall7일은 유한 scheduler 상한 내 요청이며 ETA가 아니다.

미완료: 실제 GPU qualification, held inspection, release, representative main 초기 연결.
등록만으로 actual gate PASS를 기록하지 않는다.
