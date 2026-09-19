# EN R512/G256: 공동 보존 capacity와 배치 계산 비용 감사

작성일: 2026-09-19. 사용자 지정 server4 완료 보고서를 원격에서 읽고 기존 로컬 사본과 SHA256을 대조했다. 두 파일 모두 `4ac7db3bf3a2f70ed6426459ae702f8c0f3b1539d0566136ffa535c9413d85ab`다. 실행 commit은 `5574f2c63a355043ba28c8e557383be3075a5e48`이다.

이번 작업은 보고서·frozen source·기존 tensor 감사 receipt 검토와 CPU raw 재집계다. 신규 모델 생성/forward/backward, GPU 실험, sequential 실행은 없다. 기존 보고서·raw·code를 수정하지 않았다. 기존 독립 감사의 대용량 tensor 검산을 이번에 다시 실행했다고 주장하지 않는다.

- [지정 보고서의 동일 SHA 로컬 사본](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-19-en-execution-reuse-failure-review/report-snapshot/completed-review-v1/diagnostic-report-ko.md)
- [기존 상세 실패 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-19-en-execution-reuse-failure-review/review-ko.md)
- [이번 CPU 재집계 코드](/mnt/raid5/janghj/layer_allocation/empirical/en_r512_g256_capacity_audit_20260919.py)
- [이번 재집계 receipt](/mnt/raid5/janghj/layer_allocation/empirical/en_r512_g256_capacity_audit_20260919.json)

## 1. 판정

**EN은 single-layer의 정확한 response 보존과 실행 재사용 구조를 반영했다. 그러나 신규 편집·과거 편집·원래 지식·일반 능력을 동시에 유지하는 capacity를 충분히 반영하거나 검증한 실험은 아니다.**

실제로 실행한 것은 강한 native L4 endpoint를 먼저 만든 뒤, 그 endpoint의 특정 full-token responses를 고정하고 **C4 기반 W0 평균 분포 KL을 한 번의 projected-gradient 방향으로 감소**시키는 방법이다. Reference별 판단의 보호 제약, factual QA 보호, history loss, general-ability floor는 이번 B1에서 실행/검증되지 않았다.

Legacy↔Reuse의 목적은 같은 수치 결과를 더 적은 중복 실행으로 얻는 것이다. 이 비교의 exactness 성공과 N4↔EN의 효능 부재를 구별해야 한다. 이번 matched arms는 같은 R512/G256을 사용한 두 실행 schedule이며, S64↔R512 또는 G128↔G256의 요인 분리 실험이 아니다.

## 2. 실제 반영 범위

|요구|이번 실제 구현·관측|판정|
|---|---|---|
|현재 편집 보존|Current 전체 token key null space, 개별 NLL/ID guard, 실제 logit invariant|보호한 입력·수치 tolerance 내 확인|
|Reference512 전체 사용|Train512, 실제130,235 생성 위치, full vocabulary128,256, 모든 문서 backward|확인|
|Base 최대 생성256|BOS+자연128 prompt 뒤 W0 생성, EOS 또는256|확인; 전부 완결 답변이라는 뜻은 아님|
|Reference별 판단 보호|평균 forward KL만 감소시킴|구현되지 않음|
|사실 reference 직접 보호|전체640 입력이 C4 문서 prefix|factual QA bank가 아님|
|과거 편집·overwrite 관리 검증|Cold B1, Past 없음, 확정 후 history append만 수행|검증 불가|
|일반 능력 보존|Dev128 KL·R/P/N·canonical greedy32 관측|광범위 일반 능력 회복은 미측정|
|Single-layer 보존 가능한 해 탐색|한 gradient와 그 방향의4개 scale만 평가|해 공간 전체·수렴해 검증 아님|

R512의 구성은 S64 64+Reserve320 320+AdditionalTrain128 128, 독립 Dev128 128이다. 실제 train EOS 종료11/512, 상한 종료501/512다. 모든 문서를 사용했다는 것은 모든 문서를 개별적으로 보호했다는 뜻이 아니다. GSS는 이번 실행에 없다.

## 3. 실제 효능과 복구량

|Endpoint|RS|PS|NS|
|---|---:|---:|---:|
|W0|5/100|20/200|886/1000|
|Native L4|100/100|194/200|865/1000|
|Selected EN-F|100/100|194/200|865/1000|

N4→EN은 R/P/N 성공 ID까지 동일하다. W0에서 맞았다가 N4에서 잃은24개(17 requests)를 하나도 복구하지 못했다. W0에서 틀렸다가 N4에서 얻은3개도 유지됐다. PS97%는 new/true NLL preference이며 PS teacher-forced all-token strict는125/200이다.

이산 score만 변화하지 않은 것도 아니다. Native에서 실패한 N135개 중60개는 margin 개선,75개는 악화했다. 손상24개 중18개는 개선 방향이지만 평균 부족 margin2.858894 대비 평균 이동은0.000154484였다. 실패 문항의 자기 부족 margin 중 회복한 비율의 최대는0.52345%다. 따라서 실제 복구에 필요한 크기와 큰 차이가 있다.

N true-target 평균 NLL은+5.1934e-6로 미세하게 악화했다. N new-target NLL이 더 악화하여 상대 margin이 소폭 개선된 것이며, 정답 확률의 평균 회복과 동일하지 않다. P new-target NLL도+2.3300e-5다.

|Reference 관측|Train512|Dev128|
|---|---:|---:|
|Native KL|0.0011201773|0.0013293532|
|Selected KL|0.0010561484|0.0013283442|
|상대 감소|5.71596%|0.0758985%|
|개선/악화 문서|511/1|107/21|

Train 순개선의80.03%는 상위10개 개선 문서에서 나온다. Dev의 작은 평균 개선은 문서별 상쇄에 민감하며, 기존 문서 bootstrap95% 구간은0을 포함한다. 이는 고정 B1에서의 탐색적 문서 변동성이며 seed·stream 재현성 평가가 아니다.

## 4. 보정은 적용됐지만 한 방향 탐색으로 종료됐다

실제 목적은

\[
L(W)=\frac1{512}\sum_i\frac1{T_i}\sum_s
D_{\mathrm{KL}}(p_{W_0}(\cdot\mid c_{is})\|p_W(\cdot\mid c_{is})).
\]

알고리즘은 native WN에서

\[
G=\nabla L(W_N),\quad H=GQ_E,\quad
\eta_0=L(W_N)/\|H\|_F^2,\quad D_j=-2^{-j}\eta_0 H
\]

를 계산한다. 수용 후 gradient budget1 소진으로 종료한다. 최종 gradient를 재측정하지 않았으므로 stationarity나 수렴을 주장할 수 없다.

|trial|eta|Train KL|개선/악화 문서|판정|
|---|---:|---:|---:|---|
|native|—|0.0011201773|—|시작점|
|0|4.245243|0.0030560983|501/11|Armijo 탈락|
|1|2.122622|0.0016900066|506/6|Armijo 탈락|
|2|1.061311|0.0011811946|510/2|Armijo 탈락|
|3|0.530655|0.0010561484|511/1|수용|

처음3개는 guard/invariant 이전에 KL 조건에서 탈락했다. 따라서 이번 scale 축소를 guard 탓으로 돌릴 수 없다. 반대로 큰 후보는 공식 P/N·Dev를 평가하지 않았으므로 그 후보들도 NS 복구에 실패했다고 말할 수 없다.

문서73은 native KL0.0183706에서 큰 세 step에서1.008216/0.346521/0.080262가 됐다. 네 번째에서는0.0059539로 개선된다. Trial2의510개 개선을73·163 두 문서의 악화가 뒤집었다. 최종 수용된 후보에서도163은0.0034205→0.0082464로141.1% 악화한다. **평균 KL 감소는 개별 보존의 보장이 아니라는 실제 사례**다. 해당 문서 삭제·reference 축소를 권하는 분석이 아니다.

이 결과는 방향별 finite-step 민감도 차이를 보여준다. 문서별 gradient나 Hessian은 저장되지 않아 gradient conflict와 curvature를 분리하지 못한다. 민감 문서의 teacher 데이터는 기존 bounded 검사에서 정상이나, 해당 문서의 physical AD/FD를 새로 검사한 것은 아니다.

## 5. Edit-null 공간의 장점과 제한

- Allowed14326, blocked4596, remaining9730.
- Pstar를 통과하는 원 gradient energy92.9114%.
- Edit-null까지 통과한 원 gradient energy54.8394%.
- 즉 edit-lock만으로 이미 allowed인 energy의40.9767%가 추가 제거된다.
- Actual correction norm0.00861995, native norm7.61165의0.11325%.
- Rounding delta는 correction norm의약0.0228%; FP32 update 소실이 아니다.
- Correction/native cosine은−3.03e−8로 단순 native shrinkage도 아니다.

여기서 정당한 결론은 **공간이 비어 있지 않고 평균 Train KL을 줄이는 방향이 남았다**는 것이다. “Functional preservation에 충분한 공간이 남았으므로 constraint는 원인이 아니다”라는 결론은 나오지 않는다. C4 gradient의 남은 energy는 neighborhood margin gradient의 남은 energy가 아니다.

더 근본적으로,

\[
\{D:DK_E=0\}\subseteq
\{D:\text{지정 edit 판단이 허용 기준을 유지}\}
\]

는 필요한 전체 input keys를 포함한 이상적 정확연산에서의 포함관계다. Exact module-response 보존은 최종 정답 선택 유지보다 강하다. EN은 WN의 realized hidden responses를 고정하므로 다른 hidden responses로 같은 edit를 성공시키면서 locality를 개선하는 해를 탐색하지 않는다.

만약 특정 reference key가 보호한 key span에 있으면 그 key에서는 correction 반응을 만들 수 없다. 다만 실제 neighborhood의 projected keys/gradient를 측정하지 않았으므로 이것이 이번 실패의 원인이라고 확정할 수 없다. 전체 full-token hard lock의 비용과 평균 KL objective의 한계, one-gradient 제한을 분리해야 한다.

## 6. Single batch 비용: 기존 중복 제거가 놓친 경로

Reuse correction controller는2798.45초=46.64분이다. 과거 same-native 기록285.91초와 합하면 약51.41분, native 시간의10.79배다. 이 비율은 서로 다른 시점의 계측을 회계상 합친 값이며 동시 matched speed benchmark가 아니다. 준비15472.14초=4.30시간은 별도 공통 비용이다.

Legacy3313.26→Reuse2798.45초의15.54% 감소와 Current suffix3120→1248은 관측됐다. 다만 Legacy→Reuse 고정 실행 순서와 page-cache 차이가 있으므로 전체 wall 감소를 dedup의 순수 인과효과로 할당할 수 없다. 두 schedule의 endpoint는 exact하게 같아야 하는 비교다.

|Reuse 계측|시간|정확한 해석|
|---|---:|---|
|Teacher 처리|1213.98초|순수 disk I/O가 아니라 파일·수치검증과CPU복사 포함|
|Reference prefix hash|218.84초|이미 CPU resident인640개 key/residual cache를5sweep 전후10회 검사|
|Reference suffix|270.32초|Reference model forward 구간|
|Reference backward|225.90초|autograd+FP64변환+512회D2H+CPU누적 포함|
|Session hash|106.48초|CPU/GPU owner 검증; GPU→CPU39회 포함|
|Teacher H2D|31.91초|별도 device 전송 구간|

상위·하위 timer가 존재하므로 이 표를 독립 wall components로 단순 합산하지 않는다.

### Teacher-read라는 이름이 실제 병목을 가린다

`generated_teacher.document()`는 문서마다 진입·퇴장에 capsule/logp/keys/residual 전체 SHA를 확인한다. 모든 payload를 mmap한 뒤 finite 검사를 하고, logp에는 argmax와 FP64 logsumexp 정규화 검사를 다시 한다. 이미 upstream cache로 보관한 keys/residual 파일도 재검증한다.

따라서 teacher 처리1213.98초는 순수 disk read가 아니다. `teacher_bytes_read=334.07GB`는5sweep에서 복사한 logp payload 크기만 집계한다. 파일 SHA를 위한 반복 접근까지 포함한 물리 disk byte 계측이 아니다. Reference cache hash도180.42GB의 논리적 hash 접근을 기록한다.

근거: [teacher 문서 접근](/mnt/raid5/janghj/ODE-edit/local/en-reuse-failure-review-20260919/en_execution_reuse/generated_teacher.py:387), [수치 검증](/mnt/raid5/janghj/ODE-edit/local/en-reuse-failure-review-20260919/en_execution_reuse/generated_teacher.py:342), [read timer](/mnt/raid5/janghj/ODE-edit/local/en-reuse-failure-review-20260919/en_execution_reuse/generated_oracle.py:171).

### 줄일 여지가 있는 비용과 남는 비용

1. 불변 teacher의 검증과 소비 버퍼를 정확히 결속하고 검증 결과를 재사용한다. 미사용 upstream 파일과 finite/argmax/정규화 검사를 매문서·매후보마다 다시 수행할 필요를 재설계한다. 기존 검사를 그냥 끄는 변경이 아니라 byte identity·mutation 감지 범위를 유지하는 새 실행 계약이 필요하다.
2. 각 문서의 FP32 gradient를 GPU FP64 accumulator에 원래 문서 순서대로 더한 뒤 마지막448MiB만 전송하는 경로를 검증한다. 현재512회240.52GB D2H를1회로 줄일 수 있는 구조다. GPU/CPU arithmetic 동등성은 G/H·후보·선택 endpoint까지 확인해야 하며,225.90초가 전부 전송 시간은 아니다.
3. 이미 소유한 immutable prefix와 session weight의 반복 hash를 수명·소유권에 맞춰 재사용한다. Reference hash와 GPU weight readback은 다른 경로다.
4. Preparation generation은 KV cache 없이162708 decoder forward,14704.69초를 사용했다. KV generation은 별도 token/EOS parity 검증 대상이며, 고정W0/reference teacher는 후속 batch에서도 재사용한다. Prep 최적화가 현재46.64분 controller를 없애는 것은 아니다.

Train logp만66.81GB여서 현재59GiB host budget 아래에서 전체 teacher를 RAM에 올리자는 처방은 맞지 않는다. Bounded immutable buffer와 검증 재사용의 수명 설계가 필요하다. 이 문서에서 새 최적화를 구현하거나 시간을 실측하지 않았다.

R512/G256/full vocabulary 자체의 suffix/backward 비용은 별도로 남는다. 따라서 단순 trial 상한8→2나reference 축소로 해결할 문제가 아니다. 이번 수용은4번째 trial이므로 상한2를 적용하면 현재 경로에서는 fallback된다.

## 7. 결론을 제한하는 요소와 판단 우선순위

이번 결과는 다음을 배제한다: 보정 미실행, native fallback, null space 완전 소멸, FP32 반올림으로 보정 소실, 처음3후보의 품질 guard 탈락.

다음은 아직 분리되지 않았다: factual locality에 대한 generic mean-KL의 목적 불일치, exact full-token lock이 제거한 유용한 방향, one-gradient 제한으로 인한 미수렴, 민감 reference의 곡률/충돌, finite reference 밖 일반화.

기존 single-layer 분석에서 제시한 구조적 장점은 이번 EN에 상당 부분 사용됐다. 그 장점만으로 기능적 공동 보존이나 저비용 처리가 보장되지 않는다는 사실이 이번 결과에서 드러난다. 더 많은 reference와 긴 생성 길이로 데이터량을 늘린 것만으로는 이 두 문제가 해결되지 않았다. 단, 확대의 독립 효과를 분리한 실험은 아니다.

추가 gradient를 늘리거나 긴 sequential을 돌리기 전에 반복 검증·전송 비용을 줄이는 것이 우선이다. 이후에도 평균 KL 감소를 capacity 개선으로 인정해서는 안 된다. 현재24개 손상 복구0, Dev 개선0.0759%, correction46.64분이라는 효능·비용을 함께 기준으로 삼아야 한다.

원격 전체 tensor의 새 재검산이나 sensitive-reference AD/FD, reference별 선택 제약, 새로운 optimizer, sequential 실험은 이번에 수행하지 않았다. 새 CPU 재집계는 기존 audit manifest56개 파일을 재해시하고 raw 문서별 loss와 공식 성공집합을 다시 계산해 일치함을 확인했다.
