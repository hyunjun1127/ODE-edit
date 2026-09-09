# CounterFact 고정 10k / prefix 정책

2026-09-09 사용자 지시. GH 소유 공통 정책이며 모든 서버의 향후 CounterFact 비교 실험에 적용한다.

- 데이터셋 ID: `counterfact-fixed-10k-v1`.
- 기준은 완료된 BLUE lifelong의 **실제 10,000개와 순서**다. 다시 추출하지 않는다.
- ordered root: `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`.
- 원래 sample.lock SHA256: `a8d22c230611b6c14740d1d00658a53856bbde26d060189d5ddf30f2ffdbac92`.
- 원본 전체 CounterFact SHA256: `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`.
- 전용 `counterfact.json`: 16,679,956 bytes, SHA256 `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`.
- 원본 raw 레코드와 평가 prompt를 그대로 보존하고, 고정 순서 JSON 배열로 별도 추출한다.
- Git에는 raw 데이터가 아닌 정책과 생성·검증 코드만 보관한다.

## 각 서버의 전용 데이터 경로

| 서버 | 전용 파일 |
|---|---|
| Server1 | `/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json` |
| Server2 | `/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json` |
| Server4 | `/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json` |

같은 폴더의 `source-sample.lock.json`과 `receipt.json`도 반드시 함께 검증한다.
Server3는 아직 세션/배포 경로가 지정되지 않았으며, 향후 지정 시 동일 자산을 배포한다.
작업 worktree가 달라도 위 서버별 고정 자산 경로를 사용한다. 공유 원본 전체 데이터셋은 삭제하지 않는다.

## 실험 크기와 순서

- 10k 실험: 전체 10,000개.
- 3k 실험: 앞 3,000개 `records[:3000]`.
- 1k 실험: 앞 1,000개 `records[:1000]` (기존 JVP 표본 순서와 동일).
- 다른 N도 `1 <= N <= 10000`에서 앞 N개만 사용한다. 셔플·새 seed 추출·성공 기준 교체를 하지 않는다.
- 사용자 표현 `3000k/1000k`는 이번 문맥에서 **3,000개/1,000개**로 해석했다. 10k 초과 실험은 별도 사용자 결정이 필요하다.
- B100이면 0..99, 100..199 순서로 나눈다. batch 경계 때문에 임의 재정렬하지 않는다.
- 이미 실행 중이거나 봉인된 과거 실험은 변경하지 않는다. 새 실행부터 이 정책을 따른다.
- 별도 holdout/audit가 필요하면 사용자에게 독립 데이터 정책을 먼저 요청한다. 이 고정 평가 집합을 독립 미사용 audit이라고 부르지 않는다.

## 사용과 검증

`scripts/fixed_counterfact.py verify --root <전용 폴더>`를 모델 로딩 전에 실행한다.
runner는 `load_prefix(root, n)`을 호출하거나 CLI `prefix --root ... --count 3000 --output <새 파일>`로 만든 prefix를 사용한다.
기존 full CounterFact를 로드해 seed/필터로 새로 고르는 경로를 사용하지 않는다.
입력 개수·순서·record SHA·sample root를 runtime lock에 결속한다.

전용 자산 최초 생성은 다음과 같다. 기존 파일이 다르면 덮어쓰지 않고 실패한다.

```bash
python3 scripts/fixed_counterfact.py build --source <봉인된 전체 CounterFact> --sample-lock <원래 BLUE 10k sample.lock.json> --output <전용 폴더>
```

새 baseline 승인 범위: Llama의 pre-edit W0 전체10k 평가 1개, base MEMIT와 base AlphaEdit B100x100 각각 1개.
이들은 BLUE가 아닌 원본 경로로 구분하고 원본 설정 차이를 기록한다. 신규 3개 job도 이 전용 자산을 사용한다.
