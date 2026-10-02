# JLZ v8 구현 계약

이 문서는 v7 실행 source를 참조하는 변경 명세다. Production 구현을 수정했다는 보고가 아니다. 새 구현 namespace는 `project/run_scripts/jlz_native_writer/`로 분리하고 v7 source·raw·job을 보존한다. 의미 정본은 [method-ko.md](method-ko.md), 고정 설정은 [contract.json](contract.json)이다.

## 변경할 함수와 데이터

| v7 모듈 | v8 책임 |
|---|---|
| inputs.py | Native 문장·그룹 경계·owner ID 유지. 원래 context 순서 복원 후 native_mean_keys 사용 |
| entry.py | Native anchors/teachers/C0/H 유지. replay reference preparation 제거. q scale s 계산·기록 |
| causal_builder.py | 전체 native context 수집 후 FP32 nested mean κ. Current lower write와 whole-B barrier 유지 |
| dynamic_solve.py | M=A+κκᵀ, RHS=κ. Dense reference와 B-dual. G/E와 VJP 교체 |
| allocation.py | 기존 A/B root 집계·계수 유지, mean-key G/E 입력 사용 |
| subject.py | Native 목적 그대로 D leaf에 backward. q norm으로 목적을 대체하지 않음 |
| physical_aux.py | Current 정합·KL만 유지. Past branch/resident/teacher/NLL hinge 및 reference forward 제거 |
| optimize.py | q Adam, 후보별 D leaf bridge, 모든 D gradient 완료 뒤 q.grad=sD.grad. Same physical clamp |
| writer.py | Final actual mean key의 native CPU FP32 Gram 누적. Memory admit 제거 |
| memory.py | Plain 실행에서 생성·import·호출하지 않음 |
| run.py | Sample/prepare_reference/memory state 제거. W/H/RNG rollback, cold W0/H0, candidate/state binding 유지 |
| observe.py·collect.py | 평가 의미 유지. Memory 의존 검사 대신 replay_disabled=true와 W/H 비변이 검사 |

## Native writer 경계

`native_mean_keys(raw_keys, request_ids, context_ids, group_slices)`는 actual-B와 원래 문맥 순서가 검증된 FP32 tensor에서 native와 같은 group mean, stack mean을 수행한다. 길이 정렬이나 microbatch 변경 뒤에도 canonical owner/context 대응을 복원한다. NLL에는 모든 rewrite context를 동등한 native 가중치로 사용하며, writer 그룹 가중치와 섞지 않는다.

Reference는 `torch.linalg.solve(A64 + kappa64 @ kappa64.T, kappa64)`다. SPD-qualified fast path는 A=LLᵀ, F=L⁻¹κ, S=I_B+FᵀF, V=S⁻¹I_B, T=FV, P=L⁻ᵀT를 사용한다. G=TᵀT, E=(Pᵀκ−I)(Pᵀκ−I)ᵀ다. A factor는 고정하지만 κ/F/S/T/P는 상층에서 매 후보 새로 구성한다.

Q=∂L/∂P, Y=(A+κκᵀ)⁻ᵀQ이면 solve의 κ adjoint는 `Y @ (I-P.T@kappa) - P @ (Y.T@kappa)`다. E의 explicit κ gradient, G의 P/T gradient와 actual activation 경로를 추가한다. T=LᵀP로 계산한 custom backward는 T adjoint를 P adjoint에 `L @ grad_T`로 합산한다. 첫 구현은 dense/dual autograd reference를 보유한다.

실제 linear의 D/P/input direct VJP와 FP64 materialization→FP32 cast/add는 v7 정의를 유지한다. Mean key로 바뀌어도 전체 weight gradient를 만들 필요는 없다. Last-layer write까지 actual all-token 적용이라는 사실은 바뀌지 않는다.

## q Adam의 안전한 구현

q·s·D·D.grad·q.grad·Adam moment는 FP32로 고정한다. s는 `s=(a.double()/sqrt(d_out*m)).float()`로 한 번 구성한다. Geometry는 기존 FP64다. 첫-step 제곱 상대 norm의 이론 상한 η² 검사는 FP32 반올림을 고려하여 `measured <= eta**2 + 1e-8 + 1e-5*eta**2`로 검증한다. 이는 물리 clamp나 목적에 추가하는 ε가 아니다.

q가 유일한 optimizer parameter다. s는 `[d_out,B]`에 broadcast 가능한 `[1,B]` 고정 tensor로 layer별 보관한다. `s=a/sqrt(d_out*m)`이며 gradient를 갖지 않는다. Model profile의 m은 처음 정한 전체 eligible layer 수다.

각 후보에서 `D_l=(s_l*q_l).detach().requires_grad_(backward)`를 만든다. 전체 native·policy·actual gradient를 기존 D leaf buffer에 누적한다. 이후 `q_l.grad=(s_l*D_l.grad).detach()`를 딱 한 번 지정한다. D/P에 대한 auxiliary chunk adjoint는 이 mapping보다 먼저 causal builder로 되돌린다. q에 loss를 다시 backward하여 같은 gradient를 두 번 적용하지 않는다.

Adam moment는 q 단위이며 D-space moment로 해석하지 않는다. qspace eps=1e-8도 고정한다. Post-step D의 norm으로 clamp scale을 계산하여 q에 적용한다. Equivalent q cap은 c√(d*m)다. s>0와 fixed entry anchor를 검증한다. Projection으로 moment를 초기화하거나 outward component를 몰래 제거하지 않는다.

## Replay 제거와 history 보존

Current-only four pulse는 v7과 같은 seed/purpose/partition 생성 방식을 사용한다. Replay RNG의 제거가 current partition identity를 바꾸지 않도록 독립 RNG를 유지한다. Capacity0의 dummy memory 객체를 만들어 경로를 남겨두는 대신 plain code path에서 past preparation/forward/gradient/admission을 제거한다.

History는 FP32 CPU tensor다. Terminal actual context keys를 native nested mean하여 `h.add_(kappa_cpu @ kappa_cpu.T)`를 한 번 실행한다. Native dense reference에 없는 mirror, weight decay, division, deduplication을 추가하지 않는다. W와 H가 동일 transaction 안에서 완료되며 실패 시 둘 다 복구한다. Persistent checkpoint·과거 문장 저장을 새로 추가하지 않는다.

## Telemetry

기존 후보 timing·native NLL/KL·norm·policy·g/e·write norm·solver residual에 다음을 추가한다. 이들은 진단이며 gradient·후보 선택의 입력이 아니다.

- Layer/request별 actual `rho=||D||/a`, clamp 전후 rho, clamp scale, q norm. qnorm과 Dnorm을 별도 필드로 구분한다.
- 첫 update의 `sum_layer rho^2`와 이론 상한 η². 이 검사는 optimizer 구현 일치 확인이다.
- 실제 optimizer가 제안한 `DeltaD=s*(q_proposed-q_before)`의 radial component와 최종 projection이 버린 양.
- 현재 total gradient의 `dot(D, grad_D)/||D||`를 D≠0에서 기록한다. 양수인데 proposal이 바깥으로 향하면 현재 gradient와 Adam 제안의 반경 방향이 다르다는 근거다. Momentum과 좌표별 preconditioning을 분리 측정하지 않고 이를 stale momentum만의 원인으로 단정하지 않는다. D=0은 null로 기록한다.
- 각 layer의 합성 D gradient norm 및 q gradient norm. Pulse에서는 이미 계산된 current actual direct/P 기여를 합칠 때 중복 전달 여부를 검사한다. 목적항별 gradient를 재계산하는 추가 backward를 자동 도입하지 않는다.
- Terminal actual native KL과 실제 current NLL을 저장한다. 기존 `native_kl_mean`은 virtual임을 명시한다. `native_terminal_nll`처럼 branch가 불분명한 이름 대신 `terminal_actual_context_nll`을 쓴다.
- `replay_enabled=false`, `past_rows=0`, `past_forward_count=0`, `past_loss=0`, history mode=`native_mean_key_cpu_fp32`.

추가 telemetry는 detached vector로 GPU에서 모아 후보 경계에서 일괄 CPU 전송한다. 요청별 Python scalar 동기화나 추가 backward를 반복하지 않는다.

Clipping 비율, layer 집중 또는 NS/PS를 기반으로 강제 조기 종료·후보 재선택·계수 변경을 하지 않는다. First-step bound 밖의 clipping은 구현상 허용되는 결과다.

## 검증 순서와 허용오차

1. CPU tiny algebra: FP32 native nested mean 일치, mean-key ridge dense/dual 및 G/E identity, solve VJP·direct E·causal lower→upper gradient, q leaf bridge와 monolithic chain rule, first-step bound와 후반 포화 반례.
2. 실제 작은 LM pilot: B2–4, 길이·subject 위치·복수 target token·context group 길이를 바꾸고 두 arm을 확인한다. 동일 physical D에서 v8 reference/optimized writer P/U/loss/전체 D gradient를 비교한다. v7 full-context writer와 출력이 같아야 한다고 요구하지 않는다.
3. Terminal native mean-key/weight commit/history parity와 다음 batch own-entry 연결, failed transaction rollback을 확인한다. Current-only 학습·과거 forward0을 instrument하여 확인한다.
4. B100 smoke가 필요하면 고정된 작은 후보 예산으로 시간·GPU/host peak·전체 key barrier를 검증한다. 기존 BS100×5/500 scope를 이어 사용할 수 있으나 이 설계 revision 자체가 job 제출이나 v7 취소 명령은 아니다.

CPU FP64 direct/dual 시작 기준은 atol1e-9,rtol1e-8, finite difference 전체 gradient atol1e-6,rtol2e-4다. 실제 LM은 v7의 동일-D reference 기준인 output/component atol1e-4+rtol1e-5, 전체 gradient RMS(error)≤1e-6+2e-3 RMS(reference), solve relative residual1e-8을 시작점으로 사용하고 사전 qualification 결과를 기록한다. Native nested mean은 동일 tensor·순서·dtype에서 exact 비교한다. Hardware·BLAS·shape가 다른 경우 bitwise 동일성을 주장하지 않는다. 허용오차를 성능 결과에 맞춰 완화하지 않는다.

실제 pilot에서 PS 유지나 locality 개선은 아직 확인하지 않았다. CPU toy의 서로 다른 크기 배분은 optimizer가 해당 상태를 표현·갱신할 수 있다는 제한된 검증이며 실제 편집의 optimal allocation 증거가 아니다.
