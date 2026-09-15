# EP-TW-1 v3 한정 red preflight

상태: **PENDING_IMPLEMENTATION / 실제 모델 G0 미검증**. 신규 EP-TW-1의 독립 감사이며 이전 paused 작업을 재개하지 않았다.

## 검토 범위

- 2026-09-15 SH4 envelope, v3 설계, method/dispatch/C4 reference 계약.
- `ep_tw/control.py`의 새 admission 및 teacher 재사용 결속.
- read-only `native_map.py`, native singleton fitting/finalizer, teacher score/reduction source.
- 이 시점 policy/model/persistent runner는 작성 중이므로 미존재 연결부를 실제 오류나 PASS로 분류하지 않는다.
- GPU/model/Slurm/peer/다른 실험 출력 접근·수정 및 이전 task 재개 0. 실행 성능 기준은 기술 gate로 사용하지 않는다.

## 현재 확인

1. 새 admission은 W0/L4/L2=1/B100×10/first1000/EP-TW-1 한 경로를 고정하고 N4 calibration·preservation budget 없이 구성된다. 이전 BG selector/admission을 재사용하지 않는다.
2. 기존 FrozenNativeMap은 일반 선형 solve를 사용한다. SPD/Cholesky 가정을 넣지 않으며 source RHS solve와 A-factorized map의 FP32 비동일성을 별도 기록한다. 실제 native Vp를 A-map 재구성으로 대체하지 않는지 새 model wiring에서 확인해야 한다.
3. 기존 singleton fitter의 fitting은 native 마지막 history loop만 제거하고 동일 loop를 endpoint finalizer로 분리한다. 새 runner의 processed B100당 finalizer 1회·inner append 0은 아직 실제 연결 미검증이다.
4. 기존 teacher는 full-vocab FP32, input[129:257] / logits[128:256] 위치와 vocab합→128 position평균→문서평균 KL이다. 기존 self-KL=0은 동일 logp 자신 비교이므로 독립 actual W0 forward 및 cross-kernel 해상도 검증을 대신하지 않는다.

## 최종 wiring에서 필수 확인할 항목

- Actual stored Vp가 RAW/C0의 byte 기준이며 모든 candidate는 독립 Vp+beta·CA. 보정만 축소하고 실제 byte 중복을 제거한다.
- canonical request token평균→100평균 gE와 S64 문서평균 gD를 별도 sweep으로 누적. 두 gradient를 scalar E+muD로 대체하지 않는다.
- L4 functional weight가 모든 token 위치에 작동하고 실제 materialized forward와 방향미분이 일치하는지 기술 해상도 범위에서 확인한다.
- finite E≤Ep와 정확한 Ap ID subset이 admission 기준이며 양의 품질 allowance가 없다. 성공 count만 같은 후보는 충분하지 않다.
- native anchor/radius, actual native delta norm 기준 trust, projection 후 내적 및 실제 FP32 correction norm을 기록한다.
- min actual D64, numerical ambiguity에서 RAW 우선, parent fallback 없음. Native 비유한 proposal은 technical failure로 남긴다.
- accepted-label ledger는 최종 strict 성공 요청만 갱신하며 unaccepted intent가 과거 accepted label을 supersede하지 않는다.
- W/M/RNG/hooks/cache/gradient 부작용 복원, durable next ordinal 및 B1→B2 연결; cap2/admitted pending·explicit memory는 제출 직전 parent가 결속한다.

현재 확인된 실행 차단 결함은 없다. 이는 미완성 구현의 실제 correctness PASS나 G0_PASS를 뜻하지 않는다. Parent의 최종 wiring 재검토 요청 후 이 파일에 구분된 결과를 추가한다. G0_PASS/확정 blocked/pending handoff 후 모든 agent 작업을 중지한다.

## 최종 wiring 한정 재검토

이 절은 위 초기 작성 중 상태 이후의 검토다. 검토 snapshot은 runner `ad449ff18005500a24ac0b0644d77b77a6438d789c17f1a95a9a8540bbb3a6b2`, model adapter `aaf857a16359cb1cf4b901a9737dd16b32e9d2fc62fde2ea86232485b81bbdc5`, policy `56b28d7b1398aa71c23ac11bb5164216c5c921d80ed287940b48d03652a3797d`다. Parent의 후속 변경은 최종 source lock과 구분한다.

### 수정 확인

- 최초 model adapter는 비유한 gE/gD를 즉시 throw하여 policy의 `GRADIENT_NONFINITE_RAW_ONLY`에 도달하지 못했다. 현재는 gradients_finite 진단과 실제 gradient를 반환한다. Native Vp/원 E/D 비유한 값은 여전히 기술 실패다.
- 최초 candidate observer 비유한 값도 전체 batch 실패로 전파됐다. 현재 runner는 보정 후보의 `NONFINITE_*`만 제외하며 finally에서 Vp를 복원한다. 상태·identity 오류를 이 catch로 숨기지 않는다.
- 첫 B1에서 비유한 gradient 때문에 필수 방향미분 검증을 할 수 없다면 G0_BLOCKED가 가능하다. 이는 finite poor quality에 대한 탈락이나 정책 성공/RAW fallback 증명이 아니다. 이후 batch의 finite raw + invalid correction은 RAW fallback이어야 한다.

### 잔여 edge finding — 부모에게 수정 요청

Candidate C1 관측이 비유한 값이면 C1은 제외되지만, C05/C025가 동일 bytes의 `duplicate_of=C1`인 경우 alias는 아직 trust_valid이고 관측은 생략된다. Selector가 존재하지 않는 obs[C1]을 요구해 전체 batch를 중단할 수 있다. 동일 bytes root의 관측 제외 사유를 aliases에도 전파해야 한다. 수정 전 해당 edge는 BLOCK이며 원본 RAW 비유한 실패와 혼동하지 않는다.

### 코드상 확인 및 경계

- `.parents[2]`는 `project/run_scripts`이며 readonly token contracts 경로가 맞다. freeze 목록에 해당 contracts 파일이 포함된다.
- C0 custom autograd forward는 실제 RAW clone을 반환하고 backward만 고정 A VJP를 사용한다. Functional weight는 L4 모든 token 위치에 적용된다. 실제 parity는 G0에서 확인할 대상이다.
- gE는 각 request token평균의 group합/N100, gD는 문서128-position KL평균/N64를 따로 누적한다. S64 외 gradient 접근을 거부한다.
- Quality screen은 E≤Ep와 정확한 Ap subset이다. 양의 E 허용량이 없고 correction-only 후보, actual byte dedup, RAW 우선 ambiguity 처리다.
- Entry/raw/selected P/N 및 accepted-old 관측은 policy 인자로 전달되지 않는다. Ledger는 final strict IDs만 accepted로 append하고 unaccepted intent가 과거 accepted target을 바꾸지 않는다.
- Finalizer는 candidate 선택 후 1회다. CP의 W/M/RNG/ledger 저장·CPU load 및 실제 selected state 복원을 확인하는 경로가 있고 다음 batch chronology를 비교한 후 G0를 기록한다.
- 실패 시 selected W/M/RNG/ledger 복원과 nonselected/hooks/cache guard가 있다. Nonselected tensor나 hook의 임의 훼손을 실제로 복구했다고 확대 주장하지 않는다. Guard 실패는 RESTORE_FAILURE로 남는다.
- Source launcher는 1GPU/8CPU, explicit mem60416M, exportNONE이며 12h는 GPU-hour cap이 아니다. Aggregate cap2/현재 admitted pending 확인은 parent 제출 receipt 소유이고 이 red 검토에서 scheduler를 조회하지 않았다.

이 검토는 read-only source 검토이며 실제 Llama/GPU gate PASS가 아니다. 새 GPU·Slurm·원격·실험 output 접근은 없었다.

## 최종 edge 수정 재확인

Runner SHA `f4bb79794052f67aeba7b946649a0cd2fcb9e1ca7002218436549e0790bca464`에서 selector 호출 전 동일-byte root의 trust_valid=False를 aliases에 전파하고 `DUPLICATE_OF_INVALID_OBSERVATION_<root>` 사유를 남긴다. 위 마지막 edge BLOCK은 해소됐다. 이 한정 source 검토 범위에서 남은 알려진 BLOCK은 없다.

상태는 **SOURCE_REVIEW_NO_KNOWN_BLOCK / ACTUAL_GPU_G0_NOT_TESTED**다. Parent의 CPU test 결과는 별도 receipt로 결속하며 이 red agent가 실행한 것으로 표시하지 않는다. Native/raw 비유한 실패, 첫 B1 방향미분 증거 부족, 실제 모델/source/state 오류가 있으면 G0_PASS로 승격하지 않는다. 그 이후의 작업은 사용자 초기-gate pause 정책을 따른다.
