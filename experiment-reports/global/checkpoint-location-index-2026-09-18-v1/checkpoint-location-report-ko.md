# 전체 서버 checkpoint 위치 인덱스 — 2026-09-18

## 결론과 사용법

GH가 server1·2·3·4의 사용자 연구 저장소를 직접 읽기 전용 조사했다. **이 보고서는 위치 인덱스이며 복원 성공 인증 또는 삭제 승인서가 아니다.** 원본/이관/실험/작업 상태를 변경하지 않았다.

조사 시작 2026-09-18T04:01:29.926953+00:00, 최종 경로 재확인 2026-09-18T04:09:04.725391+00:00 (UTC; 한국시간은 +9시간). 이관·새 결과 생성 중의 비원자적 snapshot이므로 이후 경로 변화는 반영하지 않는다.

- [checkpoint 전체 위치 CSV](checkpoint-index.csv): 모델 weight 지표·endpoint/snapshot 이름·native 준비 capsule·학습 adapter를 포괄하는 **후보 인덱스**. 실제 서버와 절대 경로로 조회한다.
- [lifelong/sequential 위치 CSV](lifelong-sequential-index.csv): 경로 이름으로 추린 부분집합이며 전체 chain 성공을 뜻하지 않는다.
- [실험 × 서버 위치표](experiment-server-matrix.csv), [실험별 위치 요약](experiment-summary.csv), [서버 요약](server-summary.csv).
- [delta/history/optimizer 구성요소](checkpoint-components.csv): 단독 완전 checkpoint와 분리.
- [모든 tensor 파일 후보](all-tensor-artifacts.csv): target·teacher·pretrained·미분류·부분 파일도 누락 없이 남긴 확장 목록.
- [이관 출발지→최종/임시 목적지](migration-status.csv): 기록된 SHA와 **이번 stat/size 관측**을 구분.

## 서버별 결과

| 서버 | 목록 경로 | 최종 존재 | 고유 inode | 고유 논리 GiB | scan 오류 |
| --- | --- | --- | --- | --- | --- |
| server1 | 804 | 804 | 756 | 498.668 | 0 |
| server2 | 326 | 326 | 326 | 435.413 | 0 |
| server3 | 0 | 0 | 0 | 0.0 | 0 |
| server4 | 196 | 196 | 196 | 153.457 | 0 |

`목록 경로`는 완전 checkpoint 개수가 아니다. 명시적 weight schema 지표, 이름만 있는 미확인 snapshot, 준비 capsule, LoRA adapter를 포함한다. 파일명만 같은 사본을 합치지 않았다. 고유 inode는 같은 서버 내 hardlink/동일 파일 경로 중복만 제외한다. GiB는 파일 논리 크기이며 실제 회수 가능한 디스크 공간이 아니다. 역할별 분류는 [classification-counts.json](classification-counts.json)에 있다.

server3도 SSH/디렉터리 조사에 성공했다. 지정 범위에서 실험 checkpoint 후보는 발견되지 않았고 pretrained·projector·demo vector만 확인했다. 이를 서버 전체 디스크에 checkpoint가 전혀 없다는 주장으로 확대하지 않는다.

## 진행 중인 server4 → server2 이관

| 이관 목록 | CP 수 | GiB | S4 원본 존재 | S2 최종 존재/크기일치 | S2 incoming 존재/크기일치 |
| --- | --- | --- | --- | --- | --- |
| 2026-09-11 | 183 | 223.681 | 0 | 183/183 | 0/0 |
| 2026-09-18 | 34 | 113.682 | 34 | 0/0 | 15/15 |

2026-09-18 대상은 `official-layer-realization-debt-lifelong-b100-v1`이다. source manifest의 tensor checkpoint 행만 집계했다. server4 원본은 `/data/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/`, server2 최종 예정지는 `/mnt/raid5/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/`, 임시 수신지는 같은 경로에 `.incoming-server4-20260918-r1`을 붙인 디렉터리다.

**incoming 파일이 존재하거나 크기가 일치해도 이관 완료/검증 완료가 아니다.** 전송 담당자의 full SHA·closure·seal receipt 확인은 이 조사에서 대체하지 않았다. `.state-*.pt.<suffix>` 같은 임시 파일은 checkpoint 본수에서 제외하고 확장 목록에 보존했다. 현재 이관에는 명령·삭제·이동·대기·재시도를 추가하지 않았다.

2026-09-11 기존 183개 migration-map도 현재 출발/목적지 stat에 대조했다. 과거 map의 SHA는 과거 기록이지 이번 새 재해시가 아니다. 경로가 사라졌다면 다른 위치 이동/삭제/재보관 원인은 이 인덱스만으로 판단하지 않는다. 서로 다른 서버의 실제 byte equality나 마지막 유일 사본 여부를 보장하지 않는다.

## 연구 계열별 위치

아래는 ODE 관련 계열 중심 요약이다. 전체 학습 adapter·별도 연구 디렉터리는 experiment-summary.csv에 함께 있다. archive/payload 안의 원 실험명은 논리적 계열로 묶되 **현재 위치는 CSV의 server/path가 정본**이다.

| 서버 | 실험/보관 계열 | 존재 경로 | 고유 GiB |
| --- | --- | --- | --- |
| server1 | alpha-jv-llama-diagnosis-sweep | 159 | 148.478 |
| server1 | baseline-mechanism-first-e01 | 64 | 63.002 |
| server1 | l4-two-memory-conflict-routing | 142 | 55.344 |
| server1 | multilayer-joint-compensation | 5 | 5.364 |
| server1 | native-response-ode-v31-parallel-pilot | 20 | 44.61 |
| server1 | reflection-based-ke/03-trajectory-sft-path-b | 183 | 35.834 |
| server1 | reflection-based-ke/04-sft-v1-teacher-check | 6 | 0.903 |
| server1 | reflection-based-ke/08-sft-v3-adapter-arms | 13 | 1.956 |
| server1 | single-layer-cumulative-risk-abc | 189 | 139.717 |
| server2 | alpha-native-response-v31-sequential-routing | 18 | 115.84 |
| server2 | blue-alphaedit-l4-oneshot-sequential | 3 | 2.953 |
| server2 | blue-alphaedit-l8-oneshot-sequential | 3 | 2.953 |
| server2 | blue-alphaedit-sequential-comparison | 3 | 5.906 |
| server2 | blue-l4-progress-barrier | 66 | 12.612 |
| server2 | blue-lifelong-b100x100 | 72 | 57.752 |
| server2 | blue-lifelong-b100x100-l567 | 72 | 43.315 |
| server2 | fixed10k-native-baselines | 24 | 72.188 |
| server2 | multilayer-joint-compensation | 8 | 9.736 |
| server2 | results/official-layer-realization-debt-lifelong-b100-v1.incoming-server4-20260918-r1 | 15 | 59.077 |
| server2 | single-layer-zflow | 12 | 10.966 |
| server2 | state/alpha-jv-migration-server4-20260907 | 6 | 38.613 |
| server4 | blue-lifelong-b100x100 | 2 | 0.875 |
| server4 | local-z-adaptive-allocation | 157 | 38.225 |
| server4 | results/official-layer-realization-debt-lifelong-b100-v1 | 34 | 113.682 |
| server4 | single-layer-edit-preserving-correction | 3 | 0.675 |

## 분류와 복원 한계

- `WEIGHT_STATE_INDICATORS`: 제한된 ZIP data.pkl opcode에서 W/weights/state_dict 등의 문자열 지표가 발견됨. top-level schema 인증이나 tensor finite/shape 검증은 아니다.
- `NAME_ONLY_CHECKPOINT_CANDIDATE`: endpoint/state/node/snapshot 이름 근거이며 내용 확인 부족. target-only일 수 있다.
- `PREPARED_OR_NATIVE_CAPSULE`: 준비/원 native endpoint 보조 묶음. commit 완료 또는 정확 sequential resume 가능 상태라는 뜻이 아니다.
- `TRAINED_ADAPTER`: adapter 파일. base model/config/tokenizer 등 companion이 추가 필요하다.
- delta/history/optimizer/target/key/teacher/statistics는 별도 목록으로 남겼다. `actual-increments`, `selected-delta`에 weight 지표가 있더라도 재구성 component로 보수적으로 분류했다.
- checkpoint 개수는 실험 수·독립 endpoint 수와 다르다. 실패/취소/technical/copy/backup 역시 제외하지 않고 경로 그대로 표시했다. 성능에 따른 선택은 하지 않았다.
- 명시적 CPU/synthetic/test fixture는 checkpoint 본수에서 제외하고 확장 목록의 `TEST_FIXTURE`로 보존했다. `recorded_arm/batch/payload_sha256`은 기존 migration manifest로 결속되는 파일에만 채웠으며, 나머지 `batch_step_path_hint`는 경로 힌트일 뿐이다. 실제 shape나 완료 여부를 추정하지 않았다.
- schema 읽기는 torch/pickle load 없이 pickle opcode만 사용했다. 대형 payload SHA·tensor 로딩·GPU·scheduler·모델 평가·resume 검증은 0회다. 일부 schema의 `NOT_INSPECTED`/오류는 CSV에 그대로 있다.

## 조사 범위와 사각지대

server1/2의 `/mnt/raid5/janghj`, server3/4의 `/data/janghj` 아래 비숨김 연구 디렉터리 전체, `.codex/worktrees`, `.cache/huggingface`를 조사했다. 반대 root 존재 여부도 확인했다. 접근 오류·제외 규칙·시작/종료 시각은 [scan-coverage.json](scan-coverage.json), symlink 대체 경로는 [directory-aliases.csv](directory-aliases.csv)에 기록했다.

`.git`, 설치 환경·package/cache 디렉터리, 별도 숨김 디렉터리, 사용자 root 밖 symlink, 다른 사용자/별도 미등록 mount는 포함하지 않는다. `.pt/.pth/.ckpt/.safetensors/.bin/.npz/.npy` 및 일부 backup/partial 확장자를 대상으로 했으며, 임의 확장자·확장자 없는 모델·다른 포맷·메타데이터만 있는 checkpoint는 완전 탐지하지 못한다. 따라서 **확인된 네 서버의 명시적 연구 범위 전수 파일 탐색**이며 모든 저장장치 무제한 전수조사 주장은 아니다.

수집 중 파일 변화는 `stable_during_inspection`, `changed_since_scan`, `last_stat_status`로 구분한다. manifest를 통해 최종 재확인에서 새로 발견한 수신 파일은 `PRESENT_AT_FINAL_STAT_ONLY`/schema 미검사로 표시한다. 새로 생성됐지만 첫 scan과 manifest 양쪽에 없던 파일은 다음 갱신 대상이다. 본 보고 이후 상태를 자동 추적하지 않는다.

## 재현과 증거

수집기 [checkpoint_inventory.py](checkpoint_inventory.py), 보고 빌더 [build_index.py](build_index.py). 최초 read-only scan은 아래 명령으로 새 디렉터리에 수행할 수 있다. 수집기는 create-once이며 기존 scan 덮어쓰기를 거부한다.

```bash
python3 checkpoint_inventory.py --collect /absolute/new-scan-directory
```

이번 raw metadata(모델/프롬프트 payload 아님)는 `/mnt/raid5/janghj/ODE-edit/local/checkpoint-location-index/20260918-v1/`에 보존했다. transfer manifest 두 개만 추가 읽었으며 Git의 기존 migration CSV를 재사용했다. 최종 재확인은 명시적 경로 stat만 수행했다. 빌더는 이 날짜의 고정 입력을 사용하므로 새 조사 시 SCANS/LOCAL과 날짜를 바꿔 별도 version으로 생성한다. 보고 파일/수집 입력 SHA는 [report-manifest.json](report-manifest.json)에 있다.

기존 실험 결과·worktree·이관 작업·Slurm·raw·checkpoint를 수정/삭제하지 않았으며 main push는 수행하지 않았다.
