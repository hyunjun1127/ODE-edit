# Native delayed-write E3 — 구현 및 제출 전 차단 인계

상태: **BLOCKED_STORAGE_NOT_SUBMITTED**. 실제 job ID는 없으며, `sbatch` 호출0,
새 GPU/model 실행0이다. 실제 E0/E1/E3 완료 보고가 아니다.

## 완료한 준비

- 정본 publication `11a2d883`, instruction `GH-SH4-NATIVE-DELAYED-WRITE-E3-20260924-V1`.
- 승인 소형15파일245092B를 원문 그대로 수신하고 receiver size/fullSHA15/15 PASS.
- 별도 necessary ordinary artifact 절차로 BASE_MEMIT12CP를 server2 보관본에서 exactpull,
  원 terminal SHA와 receiver fullSHA/size12/12 대조. Alpha12CP는 로컬 검증 자산 재사용.
  원본KEEP, 삭제/overwrite0, 큰 teacher/PDF/root broadcast0.
- N_diag1000/H_B1 R100·P200/Base256/General128을 성능 미열람 고정 hash로 선정했다.
  추가 active-target11 variant는 원래 H분모와 분리한다. 총TF completion rows3306.
- Base256은 같은 relation object exposure high86/low85/zero85다. 다른 relation의
  적합 후보는0이므로 있다고 주장하지 않는다. Canonical subject/fact/prompt 배제 및
  NFKC/case/punctuation alias 확인을 했고, 외부 entity-alias resolver는 NOT_AVAILABLE.
- 새 endpoint/hybrid/hook/CPU reducer 및 단일 GPU internal DAG를 구현했다.
  CPU9검사 PASS: create-once/nonfinite/행 순서/factorial 부호/rotation RMS/
  native tensor hash schema/tie-failure/실패 dependency 차단 등. 실제 모델 PASS는 아니다.

## 정확한 차단

Source `3ebe0b07078940c2d46f9ea2226ccc20c0446162`,
tree `7dece41be8ed2666d05a96cdb8432828c88fb5c0`.

Archive SHA `b2438bead1d154692ad3292bb7c5e9b98e186165609fa9dec03b35638f392616`.
Execution lock SHA `ff2c8fd6b685a933b980b2255504200b1744a8827dbae128ee1aa4da38bd3b71`.

제출 직전 `control.submit`의 disk admission이
`AssertionError: ('BLOCKED_STORAGE', 4236369920)`으로 종료했다.
관측 free4,236,369,920B(3.945GiB), output/temp/safety reserve4,294,967,296B(4GiB),
부족58,597,376B(약55.88MiB)였다. 이 검사는 sbatch 및 registration journal 생성보다
앞에 있으므로 held job이나 누락된 dependent job도 없다. 단순 PENDING/GPU부족이 아니다.

처음8GiB reserve를 잡았으나 General128의 W0 FP32 full-vocab teacher 약3.91GiB를
disk가 아닌 persistent process RAM에 두는 동등 계산 구조로 바꾸어4GiB로 산정했다.
문서/token/vocabulary/KL을 줄이거나 저장 waiver를 적용한 것이 아니다. 이후 free가
더 감소했으므로 같은 조건을 반복해서 낮추지 않았다. Shared filesystem 변화의
독점 원인은 확인하지 않았고, reserve는 실제 exclusive allocation이 아니다.

## 자율 실행 구조와 아직 미실행인 범위

GPU runner의 `G00→G10→G20→G21→G30→G31→G40→G50→G51→G60`는 이전 atomic PASS의
instruction/attempt/source/data/panelSHA까지 확인한다. CPUcollector/G70는 afterany로
연결한다. 별도 미래 agent submit 없이 전체 승인 범위를 진행하도록 구현했다.
음성 효과도 정상 완료이며 scientific-positive gate는 없다.

| 항목 | 현재 수준 |
|---|---|
| 입력15개 및 추가 MEMIT12CP 수신 | receiver SHA/size 확인 |
| 구현·CPU 검사 | 9 PASS, actual GPU 미검증 |
| G00/G10 및 E1/E3 | NOT_RUN |
| GPU runner·CPU collector 등록 | NOT_SUBMITTED, IDs=[] |
| R/P/N·General·factorial·patch 결과 | NOT_MEASURED |
| memory/time | host44GiB·GPU52GiB 예상; 상한 host59GiB, wall24h; 실제peak 미측정 |
| 신규 checkpoint | 저장0, exact new crash-resume NOT_AVAILABLE |
| 새 native z/write/history | 0, 이번 scope에서 금지 |
| E0 continuation/E2/E4–E6 | NOT_AUTHORIZED |

## 재개 경로

공간이 확보된 뒤 기존 immutable source/lock과 input identity를 확인하여 다음을 실행할 수 있다.
현재는 자동 대기·polling·예약 재개를 만들지 않는다.

```bash
cd /data/janghj/ODE-edit/local/native-delayed-write-e3/20260924-v1/execution-source-r1
/data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.native_delayed_write_e3.control --source /data/janghj/ODE-edit/local/native-delayed-write-e3/20260924-v1/execution-source-r1
```

실제 공간 외 다른 완결성 검사나 cap을 생략하는 명령이 아니다. 원 lock을 덮어쓰거나
새 checkpoint를 만들지 않는다. Projectcap2/taskcap1, 1GPU/8CPU/60416MiB/exportNONE/
Requeue0 유지. 기존 다른 task/source/raw/job은 손대지 않았다.

## 증거 및 한계

[상태](terminal.json), [CPU·자원 preflight](preflight.json), [패널](panel-summary.json),
[인계 manifest](handoff-manifest.json).
별도 독립 agent red는 사용하지 않았으며 owner 검토+CPU tests이다.
실제 결과가 없어 과학 그림/paired 성능표/후기 효과 해석은 생성하지 않았다.
GFM table 열수와 상대링크는 정적 검사하며, 현재 Python 환경의 Markdown/HTML renderer는
미설치여서 실제 렌더 검사는 NOT_RUN이다. 미설치 검사를 PASS로 바꾸지 않았다.
NO_BROADCAST_NOT_REQUIRED. Monitoring=false, automatic_resume=false.
