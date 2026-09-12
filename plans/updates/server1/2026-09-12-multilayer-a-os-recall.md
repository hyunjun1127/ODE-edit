# A0 완료분 확인과 A-OS 재개

instruction_id: ODEEDIT-S06-MULTILAYER-JOINT-EDIT-A-SH1-V1

2026-09-12 사용자 recall로 같은 task만 재개한다. 이전 실행 source
5f916fa7ffbee8793e09fc21029df453c7644989와 A0/common 원본은 변경하지 않는다.
44952는 COMPLETED/0:0/916 GPU초, 44970은 COMPLETED/0:0/1561 GPU초다.
전체 32 endpoint 또는 네 short chain 완료를 의미하지 않는다.

## 다음 실행

새 단일 A-OS는 완성된 Middle A0 endpoint를 정확히 읽고 재사용한다.
A0 재학습, 새 native z, bank/M8 재생성은 하지 않는다. 저장된 WA에서 Current
teacher만 한 번 capture한다. Base/Past teacher는 기존 We이며 calibration은
공통 native WN이다. full P*와 joint L4–L8 GGN, joint equality 하나를 유지한다.
Balance의 gradient/curvature에는 정규화H가 아니라 raw S를 쓴다.
직접 저장된 WA를 anchor로 사용하고 Dcum=WA−We의 FP32 표현을 따로 기록한다.

공유 functional/PCG/elastic은 SH2가 공개한 a1fc35fc의 변함없는 수학을 재사용한다.
소유자 SH2는 이후 B runner/FD 계측 commit만 있으며 shared kernel 변경이 없다고
확인했다. SH2 결과값은 독립 수신검증 전까지 peer-reported로만 구분한다.

## 메모리·비용

장비는 48GiB RTX A6000이고 앞선 A0 peak allocated는 40,691,670,528 bytes다.
따라서 새 solver vectors와 selected state는 CPU FP32에 두고 매 full forward에
differentiable device transfer를 사용한다. 모델/forward FP32, native factor와
contraction FP64는 그대로다. 물리 microbatch2, logical Current100/600contexts,
Base128/Past128의 모든 고정 context를 유지한다. rank 절단이나 근사 metric은 없다.
Host 사용 예상은 182272MiB 안이며 디스크 여유는 준비시 약937GiB였다.

OS의 두 RHS는 각각 PCG 최대20 iteration이다. true residual 평가를 포함해
최대43 full K actions(2×21+stationarity1)가 필요하다. 각 K는 Current/Base/Past
전체 JVP/VJP이며 984 contexts를 microbatch2로 처리하므로 약492 JVP와492 VJP,
984 model forwards다. 실제 길이/token·early convergence·메모리 transfer 비용은
실제 ledger로 남긴다. 작은 equality solve 비용으로 대체하지 않는다.
SH2의 단일-layer B-OS가 보고한 14284초는 아직 peer 결과이며 A-OS 실측이 아니다.
두-layer A-OS 계획치는 대략4–12시간 범위이고 실제 첫 full operator 비용을 통해
갱신한다. 이 범위는 새 GPU-hour 예산이나 성능 gate가 아니다.

1GPU/8CPU/mem182272M/12h/exportNONE, project cap2를 submit 직전 다시 확인한다.
다른 사용자 GPU와 job은 변경하지 않는다. 최초 실제 host/device AD 및 full K
finite 결과를 확인하면 agent는 MONITORING_PAUSED_AWAITING_USER로 종료한다.
이미 제출된 프로그램은 두 RHS solve, terminal materialization/history once,
Full3900 평가, generation, signed attribution을 자체 순서대로 계속한다.
PCG finite 미수렴은 approximate로 보존하고 tolerance 변경/성과 제외는 없다.
후속 BF/Early/Late/chain 자동 제출은 다음 사용자 recall 전에는 없다.

## 완료분 보고

이번 recall의 CPU 분석은 기존 A0/common bytes만 재검산한다. A0의 Current 결과와
기능적 위험, audit, signed attribution, 실제 비용, 미측정 항목을 partial report에
기록한다. source와 분석 source는 분리하며 scientific_promotion=false다.
