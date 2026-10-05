# Causal joint MEMIT 수학 참조

사용자가 선택한 **actual candidate-weight forward + MEMIT writer 한 경로**의 작은 FP64 CPU 증거다. NumPy만 사용하며 모델·데이터·GPU를 읽거나 production 실험을 실행하지 않는다. 이전 두 arm/exact 비교를 구현한 것이 아니다.

각 candidate에서 모든 eligible layer를 순서대로 통과한다. 하층 candidate write가 반영된 상태로 상층 native mean key를 계산하고, `P=solve(A+K K^T,K)`, `U=R P^T`를 구한다. 모든 요청의 `R=a*u`는 공동 변수로 남는다. 목적은 그 실제 candidate weights의 logits에서 계산하는 synthetic NLL 및 KL(current||entry)와 ideal writer 비용이다.

`A`는 batch entry에서 고정한 대칭 SPD 입력이며 key는 후보에 따라 변한다. Native rewrite key 평균은 명시된 context weights를 사용하고 KL row는 writer key 평균에서 제외한다. KL 문장은 task forward에 남는다. Hidden 변화는 전체 candidate weight forward에서 계산한다. 별도의 subject-only virtual teacher를 학습 결과로 사용하지 않는다.

`Q_dense=sum tr(U A U^T)`와 `Q_compact=sum tr[R(P^T A P)R^T]`를 모두 계산한다. Compact 식의 `P`를 detach하면 안 된다. `writer_jvp`와 `candidate_jvp`는 상층 key가 하층 write에 의존하는 solve 경로까지 미분한다. 고정 key 방식의 누락을 잡는 명시적 반례 검사를 포함한다.

Norm과 공유 budget은 **pre-writer 요청량** `u`에 유지한다. 실제 응답 `U K`로 대체하지 않는다. 모든 layer가 0인 c0에서는 Q와 full gradient가 0이다. 첫 nonzero ideal FP64 `U=R P^T` 후보에서 full-u Q gradient norm과 native norm-gradient norm을 맞추는 coefficient 계산식만 제공한다. FP32 effective update의 0 여부와 이 판정을 혼동하지 않는다. 추가 fitting loop나 실험 coefficient는 제공하지 않는다.

실행:

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python run_audit.py
```

NumPy가 설치된 다른 Python으로도 실행할 수 있다. `audit.json`은 synthetic 검사의 상태·오차·source hash만 기록한다.

검사 범위:

- Native 평균 key support와 all-context equality의 차이.
- MEMIT ridge 응답, dense/compact 비용과 모든 key 경유 gradient.
- 실제 candidate NLL/KL·Q gradient의 finite difference.
- 하층 변수가 상층 writer에 미치는 gradient를 잘못 끊으면 실패하는 증거.
- 같은 candidate weights의 functional forward와 명시적 commit forward 일치.
- c0 Q-gradient=0 및 단일 coefficient 단위 규칙.
- 0 layer의 gradient 경로·예산 내 재진입 가능성.
- Logical batch 1/2/4/작은 tail, 가변 context 개수.
- 전체 logical candidate의 native loss를 microbatch 1/2/4+tail로 합산하고 regularizer는 한 번만 적용.
- Requested norm과 realized-action norm의 구분, 잘못된 metric의 명시적 실패.

이 참조는 FP64 수학을 검사한다. 실제 FP32 materialization/cast/add, 저장 weight increment, dtype별 gradients, 모델 parity 및 메모리·시간은 별도 production qualification 대상이다. Reference는 비대칭 A를 임의로 대칭화하거나 jitter를 추가하지 않는다. 기존 common stop 및 max25 evaluation/24 update 정책은 이 CPU 목적함수 증거의 범위 밖이며 변경하지 않는다.
