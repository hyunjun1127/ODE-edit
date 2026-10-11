# Qwen FE: CF 데이터 구간만 바꾸는 2K 진단

사용자 요청: Qwen FE에서 데이터를 조금 다르게 하여 현재 모델 기준 z의 붕괴를 다시 확인한다.
실행 전에 선택을 고정했다. 기존 CF fixed-10K 파일의 **2,001–4,000번째 2,000건**을 사용한다.
기존 63385는 같은 파일의 1–2,000번째 요청이다. case ID 중복은 0건이다.
순서를 섞거나 성능·모델 정답 여부로 필터링하지 않았다.

## 고정 조건

- 모델: Qwen2.5-7B-Instruct, **BF16**. FP16 실험이 아니다.
- 공식 FE `main.py`, 현재 배치 진입 모델에서 현재 100건의 z만 계산하고 history를 계속 누적한다.
- 원저자 Qwen hparams: L4–L8, clamp 1.0, 최대 35 steps, lr 0.5, batch 100, seed 0.
- C0 원본 그대로, 보정 없음. context는 기존과 같은 native 생성 절차를 쓴다.
- 편집 코드 171개 및 평가 코드 524개가 기존 대조군의 frozen 파일과 byte 단위로 동일하다.
- 실제 resolved 설정의 차이는 데이터·출력 디렉터리와 실행 identity뿐이다.
- W0·500·1000·1500·2000건에서 weights/history/RNG/context를 저장하고 같은 공통 평가기로 평가한다.
- server1의 기존 FE 진단 cap 예외 안에서 편집 → 평가 한 lane만 추가하며, 기존 job을 변경하지 않는다.

## 비교와 상태

| 데이터 구간 | 편집 → 평가 job | CF 2K Score | Eff | Gen | Loc | 상태 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| fixed10K 1–2,000 | 63385 → 63387 | 50.1831 | 50.900 | 49.950 | 49.715 | 완료, 붕괴 |
| fixed10K 2,001–4,000 | 63561 → 63562 | — | — | — | — | PENDING, GPU 결과 미관측 |

2026-10-11 21:21 KST 제출 및 held 검산·release를 완료했다.
`63561 / s1-qwen25-cf-fe-next2k-train`은 server1 GPU 배정 대기,
`63562 / s1-qwen25-cf-fe-next2k-evaluate`는 `afterany:63561`이다.
등록 당시 server1의 전체 사용자 GPU DAG 최대 폭은3이며 두 새 job은 같은 한 lane이다.
서버 GPU8개가 Slurm에서 모두 할당되어 있고 실행 전이다. 30초마다 상태·checkpoint·지표를 읽어 기록한다.
[실제 등록 영수증](../../../audits/global/fe-qwen-cf-altdata-20261011/submission.json).

데이터가 다른 진단이므로 본표의 기존 CF 성능을 이 결과로 교체하지 않는다.
새 데이터에서 정상이어도 단일 구간 결과이며, 기존 구간의 실패를 지우지 않는다.
Score는 공통 평가기의 Eff/Gen/Loc 조화평균이며, W20에서 편집한 전체 2,000건을 평가한다.

## 재현 정보

- CF 원본: `local/datasets/counterfact-fixed-10k-v1/counterfact.json`
- 원본 SHA: `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`
- 실제 선택: Python slice `[2000:4000]`, 기존 first-2K 원본 일치 및 case 중복 0건 검증.
- FE author commit: `478134dfb24b43f4e18b47e8500893ce3f9cc50f`
- native patch source: `60eea6e2f7aeec94743f27cd5df1e67cc160fe84`
- 공통 evaluator source: `c400c2beaad62eab6e052d841e088d3cb3e547a4`
- 실행 config identity: `742f8c3a028056ee3f0062d569d939943b5a7d776e0ab95a036ea0cd7ec63e63`
- 패키지: `/mnt/raid5/janghj/ODE-edit/local/fe-qwen-cf-altdata-20261011/native-r1/`

[선택 규칙·전체 case ID](../../../audits/global/fe-qwen-cf-altdata-20261011/stream-proof.json) ·
[코드·설정 비교](../../../audits/global/fe-qwen-cf-altdata-20261011/preparation.json) ·
[실행 config](../../../audits/global/fe-qwen-cf-altdata-20261011/config.json).
