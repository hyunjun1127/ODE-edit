# Server4 BLUE·native lifelong 실험 독립 점검

작성: 2026-09-11. 대상은 `blue-native-lifelong-comprehensive-review-2026-09-11-v1`의 공개 산출물 전체와 14개 실행 경로의 원시 산출물이다. 원보고서 SHA256은 `60a4bcf69f5a63cf581b1b9ee55f299694a786d070b5a0ddb738d544f254e6ce`다. 원본 실험·보고서는 수정하지 않았다. 이번 점검에서 모델 forward, edit, GPU 실행은 0회다.

## 1. 판단

**주요 집계와 산출물의 일관성은 확인됐다. 결과는 AlphaEdit BLUE-L4의 편집 유지 능력이 강하지만 Base locality는 여전히 크게 훼손된다는 해석을 지지한다.** 동시에 single-layer의 공동 보존 capacity가 충분하다거나, BLUE가 두 layer에 편집 부담을 충분히 분산했다고 결론내릴 근거는 없다.

이번 독립 점검에서 추가한 결과는 다음과 같다.

- 기존 6개와 신규 8개를 모두 포함해 원시 평가를 다시 집계했다. 성공률·NLL·cohort·망각 전이·비용의 대조에서 불일치를 찾지 못했다.
- 원보고서에서 비어 있던 신규 8개 arm의 W0 대비 문항 전이를 server2의 봉인된 원본 W0 평가로 보완했다. 기존 6개 전이도 다시 일치함을 확인했다.
- AlphaEdit BLUE-L4가 잃은 neighborhood 문항 26,583개 중 4,242개는 true NLL이 악화되지 않았는데 competing-new가 더 강해져 선호가 뒤집혔다. Base 보호를 true NLL 하나로 정의할 때의 한계다.
- AlphaEdit BLUE의 L4 평균 update norm은 L4-only의 99.913%다. Frobenius 크기로 본 L4의 부담은 거의 줄지 않았다.

**명칭 구분:** 이 보고서의 `BASE_ALPHAEDIT`/`BASE_MEMIT`은 `blue=False`인 원본 5층 편집 arm이다. 우리가 설계한 Base 지식 보호 channel과 다른 의미다. 또한 새 설계의 Native proposal N은 BLUE-L4 endpoint이므로, 아래 표의 원본 5층 native arm과 혼동하지 않는다.

## 2. 무엇을 실제로 확인했는가

|대상|범위|이번 확인|
|---|---:|---|
|공개 package|79개 파일|전체 size/SHA, manifest/receipt 관계 확인|
|CSV|58개, 합계 82,090행|전수 파싱, 분모·rate·전이 합·분위수 순서·family 분할·관련 표 간 집계 확인|
|PNG|8개|전부 열어 축·범례·수치 방향·결측 처리 확인, 생성 코드와 원자료 연결 확인|
|분석 코드|봉인 목록 16개|전체 SHA 대조, 주요 reducer·aggregation·audit·plot·package 로직 검토|
|실행 source 근거|7개 파일, 보고된 44개 위치|파일 SHA와 인용한 코드 행 일치, 실제 write/target/history 정책 검토|
|실행 원시 파일|8,978개, 195,400,329,632 bytes / 181.981 GiB|기존 6개·신규 8개를 모두 전수 재해시; 불일치 0|
|원시 tensor 파일|168 checkpoint + 1,400 target 파일|위 전수 파일 SHA에 포함; 이번에 tensor를 별도 역직렬화하거나 GPU 복원한 것은 아님|
|실행 source archive|서로 다른 4개|전체 SHA 재확인|
|원시 JSON 재집계|4,582개, 약 6.66 GB|14 runtime, 1,400 entry/commit/current, 168 seen-full, native seen-rewrite 200개|
|평가 row 확인|12,949,200 prompt-state 관측|NLL로 성공 bit와 margin 재계산, order/identity/분모 확인|
|연속 state|14 × 99 연결|이전 endpoint와 다음 entry의 weight/history signature 연결 확인|
|W0 추가 pairing|14 × 130,000 prompt 관측|전이 42행; 기존 18행 재확인, 신규 24행 추가|

12,949,200은 여러 state에서 반복 관측한 row 수이며 고유 문항 수가 아니다. 기본 edit 요청은 고정된 10,000개다. 모든 결과는 하나의 seed/order에서 나온 기술통계다.

원시 재집계에서 수치 대조의 최대 절대 차이는 2.78×10^-17이었다. 최종 RS/PS/NS 정수 numerator는 전부 정확히 일치했다. 이 검증은 저장된 평가의 집계 신뢰성을 확인하며, evaluator 자체의 모델 forward를 다시 실행한 검증은 아니다.

## 3. 최종 W100의 결과

모든 arm은 100 requests × 100 batches다. 마지막 실제 W100에서 전체 10,000개를 다시 평가했다. RS/PS/NS 분모는 각각 10,000 / 20,000 / 100,000이다.

|방식|MEMIT RS|MEMIT PS|MEMIT NS|AlphaEdit RS|AlphaEdit PS|AlphaEdit NS|
|---|---:|---:|---:|---:|---:|---:|
|원본 5층, blue=False|64.530|57.035|49.838|73.430|62.885|55.285|
|BLUE, L4+L8|71.830|67.650|53.287|98.880|95.775|63.726|
|BLUE-L4|77.220|74.830|57.848|99.390|95.680|65.348|
|BLUE-L5|80.840|75.040|54.285|99.340|94.195|62.822|
|BLUE-L6|82.770|73.805|50.488|96.830|85.740|58.396|
|BLUE-L7|84.440|74.705|48.547|92.400|80.970|56.277|
|BLUE-L8|81.770|72.300|48.425|93.960|77.780|54.703|

단위는 %. W0는 RS 7.910%, PS 9.985%, NS 89.212%다. 편집 전 RS/PS가 낮다는 것은 새 target을 선호하지 않는다는 뜻이다.

AlphaEdit에서는 L4가 단일층 중 가장 강하다. BLUE와 비교하면 L4-only는 RS +0.510pp, NS +1.622pp지만 PS는 -0.095pp다. 따라서 두 baseline을 유지할 이유가 있으며, 모든 지표에서 L4가 BLUE를 지배한다고 쓰면 부정확하다. 두 arm은 일부 같은 문항을 서로 다르게 잃고 얻는다. NS의 경우 BLUE→L4 비교에서 7,968개를 잃고 9,590개를 얻어 순증 1,622개다.

MEMIT에서는 L7이 RS, L5가 PS, L4가 NS에서 가장 높다. Layer가 뒤로 갈수록 모든 지표가 단조 개선된다는 결과는 아니다. MEMIT의 후기 layer 선택은 편집 유지와 locality 사이의 절충을 바꾼다.

## 4. 처음 못 쓴 것과 나중에 잃은 것을 분리하면

각 요청을 처리한 직후의 RS와 최종 RS를 동일 요청별로 비교했다. At-write는 서로 다른 W에서 관측한 값의 집계이며 최종 W 하나의 성능으로 대체해서 쓰지 않는다.

|arm|At-write RS 성공|최종 RS 성공|후속 lost|후속 gained|성공했던 요청 중 lost|
|---|---:|---:|---:|---:|---:|
|원본 MEMIT 5층|9,004|6,453|2,728|177|30.298%|
|MEMIT BLUE|9,999|7,183|2,816|0|28.163%|
|MEMIT BLUE-L4|9,934|7,722|2,233|21|22.478%|
|원본 AlphaEdit 5층|9,990|7,343|2,649|2|26.517%|
|AlphaEdit BLUE|9,999|9,888|111|0|1.110%|
|AlphaEdit BLUE-L4|9,993|9,939|55|1|0.550%|
|AlphaEdit BLUE-L5|9,989|9,934|57|2|0.571%|
|AlphaEdit BLUE-L8|9,988|9,396|595|3|5.957%|

원본 AlphaEdit의 낮은 최종 성능은 단순히 최초 write가 성립하지 않은 문제가 아니다. 최초 99.90% 성공 뒤 상당수를 잃었다. 반대로 BLUE-L4는 신규 write와 장기 retention 모두 강하다.

이 차이는 오래된 cohort에서도 확인된다. 마지막 W100에서 최초 1,000개 RS는 원본 AlphaEdit 55.4%, BLUE 96.6%, BLUE-L4 98.7%다. 마지막 1,000개는 각각 99.5%, 99.9%, 99.6%다. L4의 최종 성능이 최근 요청만으로 높아진 것은 아니다.

의도적 재편집과도 구분했다. 이후 동일 subject/relation에 다른 target이 오는 요청은 212개, 동일 target만 오는 요청은 5개, 이후 같은 fact가 없는 요청은 9,783개다. AlphaEdit BLUE-L4의 lost 55개 중 19개는 이후 다른 target이 오는 집합에 속하며, 36개는 이후 동일 fact가 없는 집합이다. 명시적 overwrite가 일부 영향을 주지만 전체 관측을 설명하지는 않는다. 반대로 이후 다른 target이 생긴 모든 과거 성공을 끝까지 유지해야 한다는 목표도 적절하지 않다.

## 5. Base locality: 순점수보다 실제 잃은 문항이 더 크다

W0의 NS 성공은 89,212개다. 원보고서에서 신규 8개에는 없던 W0 문항 전이를 이번에 추가 계산했다. W0 원본은 server2의 `preedit-full10000.json`이며 봉인 inventory와 SHA/size가 정확히 일치했다. 모든 arm에서 case_id, prompt_index, prompt/target identity를 대조했다.

|arm|W0에서 맞다가 잃은 NS|W0에서 틀렸다가 얻은 NS|W0 성공 중 손실률|
|---|---:|---:|---:|
|원본 MEMIT 5층|43,747|4,373|49.037%|
|MEMIT BLUE|39,306|3,381|44.059%|
|MEMIT BLUE-L4|34,527|3,163|38.702%|
|MEMIT BLUE-L5|38,084|3,157|42.689%|
|MEMIT BLUE-L6|42,004|3,280|47.083%|
|MEMIT BLUE-L7|43,877|3,212|49.183%|
|MEMIT BLUE-L8|43,929|3,142|49.241%|
|원본 AlphaEdit 5층|37,874|3,947|42.454%|
|AlphaEdit BLUE|28,306|2,820|31.729%|
|AlphaEdit BLUE-L4|26,583|2,719|29.798%|
|AlphaEdit BLUE-L5|29,169|2,779|32.696%|
|AlphaEdit BLUE-L6|33,825|3,009|37.915%|
|AlphaEdit BLUE-L7|35,969|3,034|40.319%|
|AlphaEdit BLUE-L8|37,772|3,263|42.340%|

AlphaEdit BLUE-L4는 NS가 89.212→65.348%, 즉 -23.864pp다. 순감 23,864개보다 실제 lost 26,583개가 크며, 2,719개 gained가 일부를 상쇄한다. 편집 유지 실패 55개와 비교할 때 Base 손상이 별도 연구 대상이라는 근거가 분명하다. 두 수의 원분모가 다르므로 raw count 비율로 문제 크기를 직접 비교하지는 않는다.

### 5.1 정답 약화와 경쟁 target 강화

AlphaEdit BLUE-L4의 전체 neighborhood에서 W0 대비 평균 true NLL은 **+1.455814**, competing-new NLL은 **-2.903066** nats/token 변했다. 기존 정답 확률 저하와 새 target의 주변 문맥 확률 상승이 함께 있다.

W0 성공→실패인 26,583개를 나누면 다음과 같다.

|손실 문항에서 관측한 변화|수|lost 중 비중|
|---|---:|---:|
|true NLL 증가와 competing-new NLL 감소가 함께 있음|19,832|74.604%|
|true NLL 증가만 있음|2,509|9.438%|
|true NLL은 증가하지 않았고 competing-new NLL이 감소함|4,242|15.958%|

따라서 true NLL의 추가 악화만 막는 보호는 마지막 유형을 놓칠 수 있다. 이는 Base의 full-distribution reference를 고려하는 이유다. 다만 이 결과에서 full-vocabulary KL barrier의 효과를 측정한 것은 아니며, 계획한 We-teacher KL의 유효성은 새 실험으로 확인해야 한다.

### 5.2 평가 neighborhood가 명시적 edit와 겹치는 경우

고정 dataset의 relation_id와 canonical rewrite prompt를 사용해 Unicode NFC·공백 정규화 뒤 정확히 일치하는 경우를 검사했다. NS 100,000개 중 2,432개가 10,000개 edit의 canonical prompt와 겹쳤다. 이런 경우 일부는 의도적으로 바꾼 지식을 locality 손실로 셀 수 있다. 의미적 alias나 paraphrase까지 찾아낸 전체 충돌 검사는 아니다.

이 2,432개를 제외한 97,568개에서도 AlphaEdit BLUE-L4의 NS는 **89.521→65.731%, -23.791pp**다. W0 성공 중 25,699개를 잃고 2,487개를 얻었다. 정확히 겹치는 문항을 제외해도 큰 손상이라는 결론은 유지된다. 남은 모든 문항이 의미적으로 미편집 지식이라고 증명된 것은 아니다. 주표를 이 subset으로 바꾸지 않고 민감도 분석으로 별도 보존했다.

## 6. BLUE가 실제로 L4 부담을 분산했는가

BLUE는 L4에서 local z를 계산하고 전체 residual을 쓴 뒤, 바뀐 모델에서 L8의 local z를 다시 계산한다. 고정된 하나의 residual을 반씩 나누는 구현이 아니다.

|계열|BLUE L4 평균 batch-net norm|L4-only 평균 norm|BLUE / L4-only|BLUE의 평균 L4/L8 norm share|
|---|---:|---:|---:|---:|
|MEMIT|9.780381|8.964529|109.101%|62.684% / 37.316%|
|AlphaEdit|10.372295|10.381361|99.913%|73.803% / 26.197%|

AlphaEdit BLUE는 L8에도 실제 update가 있지만, L4 norm은 L4-only와 거의 같다. MEMIT에서는 오히려 BLUE의 L4 norm이 더 크다. **이 지표에서 BLUE를 L4 부담을 줄인 대조군으로 보기 어렵다.** 사용자의 기존 지적이 수치로 뒷받침된다.

동시에 norm share를 지식량·편집 기여·residual 부담의 정확한 비율이라고 해석하지 않는다. 활성 key의 크기와 downstream sensitivity가 다르면 같은 norm의 효과가 다르다. 여기서 비교한 것은 actual stored-weight batch-net Frobenius norm이다. 공통 target에 대한 layer별 functional contribution 또는 native action은 이 산출물에 없다.

원본 5층 AlphaEdit의 평균 norm share는 L4 약 11.79%, L8 약 36.64%이며 중간층도 쓰지만 최종 retention은 낮다. 이것 역시 layer 수가 늘면 공동 보존 능력이 자동으로 좋아진다는 결과가 아니다. Write/target 정책과 L2가 함께 달라 capacity만의 대조가 되지 않는다.

## 7. P·Past history·capacity의 해석

**AlphaEdit이 이전 edit 전부의 exact null space로 들어간다는 설명은 이 구현과 다르다.** 고정된 P는 준비된 원래 지식의 covariance 기반 projector다. 과거 편집은 누적 key Gram M을 선형계에 넣어 보호한다. 실행 코드의 구조는 `P(KKᵀ+M)+L2 I`, RHS `PKRᵀ`이며, batch 후 실제 선택 layer의 key로 M을 한 번 갱신한다. 이전 edit에 대한 functional equality를 hard constraint로 푸는 것은 아니다.

이 실험이 직접 지지하는 것은 **BLUE-L4의 특정 P/history/write 조합에서 이전 edit의 실제 선호가 매우 잘 유지됐다**는 사실이다. 그 성공을 P 하나 또는 history 하나의 인과 효과로 분리하지 못했다. 원본 AlphaEdit에서도 같은 종류의 history를 사용하지만 retention이 나쁜 점을 함께 설명해야 한다.

Single L4에서는 고정된 teacher-forced 입력에 대해 해당 down-projection의 input key가 후속 L4 write로 변하지 않는다. 다중 layer에서는 하위 layer의 후속 write가 상위 layer input을 바꿀 수 있다. 따라서 저장된 Past key와 현재 model input 사이의 차이가 유지 성능에 영향을 줄 수 있다. 이는 source 구조에서 도출한 가설이며, 이번 실험이 과거 key를 재측정해 그 drift의 원인 효과를 검증한 것은 아니다.

Capacity는 다음처럼 구분해야 한다.

|질문|이번 결과가 말해 주는 것|
|---|---|
|L4 하나에서 새 edit를 성립시킬 수 있는가|이 fixed10k/B100 순서에서 AlphaEdit BLUE-L4의 at-write RS 99.93%로 강하게 지지|
|후속 edit 이후 앞선 edit를 유지할 수 있는가|최초 1k 최종 RS 98.7%, 전체 lost 55개로 강하게 지지|
|Base knowledge도 같이 보존할 수 있는가|현재 알고리즘에서는 큰 NS 손상. 보존 가능한 대체 write가 존재하는지는 미측정|
|Single-layer 공동 보존 capacity가 부족한가|손상만으로 불가능성을 증명하지 못함. 알고리즘이 손상이 작은 방향을 못 찾았을 가능성도 남음|
|Single-layer capacity 문제가 없는가|입증되지 않음. BLUE는 L4 부담을 실질적으로 줄인 통제군이 아님|
|임의 batch size에서도 같은 결과인가|실모델 결과는 B100뿐. B 독립 API/수치 정의와 성능 일반화를 구분|

따라서 단일 L4에서 conflict routing을 먼저 확인하는 연구 순서는 타당하다. 성공하면 해당 보호 지표에서 우회 여력이 존재한다는 근거이고, 실패하면 제약·proxy·관측 공간·최적화 문제와 진짜 capacity 한계를 구분해야 한다.

## 8. 높은 RS/NS를 읽을 때 필요한 구분

RS/PS는 새 target과 원래 target 두 후보 사이의 평균-token NLL 선호다. 모든 target token의 top-1 일치나 자유 생성 정답률이 아니다.

|AlphaEdit BLUE-L4|NLL-pair 성공|해당 target의 teacher-forced strict|
|---|---:|---:|
|Rewrite/new|99.390%|95.290%|
|Rephrase/new|95.680%|66.810%|
|Neighborhood/true|65.348%|8.210%|

W0 neighborhood true strict는 21.567%였다. 따라서 NS 하락은 비교 후보만의 숫자로 끝나는 관측이 아니며, 정답의 절대적인 top-1 지표에서도 악화가 보인다. 다만 TF strict도 자유 생성 전체 문장의 정확도와 같지 않다.

Family 간 Base 보존 순위도 지표에 따라 다르다. MEMIT BLUE-L4는 NS 57.848%로 AlphaEdit보다 낮지만, neighborhood true NLL은 6.024746으로 AlphaEdit의 6.458119보다 낮고 true strict는 9.516%로 AlphaEdit의 8.210%보다 높다. **NS가 높다는 이유만으로 기존 정답의 절대 확률까지 더 잘 보존했다고 주장할 수 없다.** Current edit 성능과 competing-new의 변화까지 함께 보아야 한다.

AlphaEdit BLUE-L4의 online-at-write NS 72.505%와 최종 NS 65.348%는 서로 다른 관측이다. 각 요청의 at-write까지는 그 이전 batch들의 변화도 이미 누적됐다. 따라서 W0→at-write 차이 전부를 해당 요청이 속한 batch 하나의 즉시 손상으로 귀속할 수 없다. 이번 batch의 추가 손상을 확인하려면 동일 보호 문항을 같은 entry We와 endpoint 양쪽에서 평가해야 한다.

## 9. 비용을 읽으면 one-shot의 의미도 분명해진다

AlphaEdit 세 baseline의 실제 기록 시간은 다음과 같다. 단위는 시간이다.

|arm|z 계산 호출|edit 전체|그중 target/z|그중 key|그중 solve|evaluation|
|---|---:|---:|---:|---:|---:|---:|
|원본 5층|10,000|4.7375|3.4675|1.0295|0.01668|4.8950|
|BLUE L4+L8|20,000|8.7126|8.1977|0.4136|0.00669|3.0370|
|BLUE-L4|10,000|7.9046|7.6480|0.2054|0.00333|2.8734|

BLUE-L4에서는 target 계산이 edit 시간의 96.753%다. 기존 solve는 100 batches 합계 약 12초다. Layer 수와 z 호출 수만으로 runtime 배수를 정할 수 없으며, z 최적화의 위치·난도·실제 반복 경로가 함께 작용한다. 호출 두 배인 BLUE가 z 시간까지 두 배인 것은 아니다.

이로부터 다음을 구분해야 한다.

- One-shot refinement가 절약하는 것은 모델의 반복 손상 관측·보정 비용이다. 기존 z 최적화가 사라지는 것은 아니다.
- 새 OS의 보호 bank 준비와 full-P metric factorization 비용을 기존 native solve의 약 0.12초/batch로 대체 추정하지 않는다.
- BF8은 dense solve 횟수뿐 아니라 보호 bank의 forward/backward와 full-space preconditioning 비용이 중요하다.
- 원본 5층은 매 batch seen-rewrite 평가를 추가했으므로 전체 wall/GPUh를 동일 평가 부하의 속도 비교로 사용하지 않는다. Edit 시간에도 target/key/solve가 이미 포함돼 중복 합산하지 않는다.

W0는 server2 A6000, 편집은 server4 PRO6000이다. 이 보고서의 W0/편집 시간으로 hardware-independent 속도 효과를 추정하지 않는다.

## 10. 원보고서에서 유지할 결론과 유보할 결론

유지할 결론:

1. 이번 설정의 강한 편집 baseline은 AlphaEdit BLUE 및 BLUE-L4다.
2. AlphaEdit BLUE-L4의 Past retention과 Base locality 사이에는 큰 간극이 있다.
3. 단일층 선택은 단순한 parameter 수 감소 문제가 아니며 target/write 위치를 함께 바꾼다.
4. Current/online만으로 lifelong 성능을 대표하면 MEMIT 및 원본 AlphaEdit의 후속 손실을 놓친다.

유보할 결론:

1. Native→BLUE 향상을 local z 또는 layer 수 감소 하나의 효과로 귀속하기. 원본 AlphaEdit L2=10, BLUE L2=1이며 residual 배분과 target 시점도 다르다.
2. 단일층 우위로 capacity 문제를 부정하기, 또는 NS 손상만으로 single-layer 불가능성을 선언하기.
3. M이나 weight norm을 실제 functional 지식 보호량으로 읽기.
4. 동일 order의 많은 prompt를 독립적인 실험 반복으로 취급하기.
5. 이번 결과로 barrier 또는 closed-form refinement의 추가 이득이 이미 입증됐다고 말하기. 그런 arm은 이 실험에 없다.

재현 한계도 원보고서와 구분 없이 덮지 않는다. 미선택 전체 parameter의 독립적인 byte parity, GPU checkpoint continuation replay, 공통 fixed-z의 post-write activation realization/native action은 미측정이다. 이번에는 tensor file 전수 SHA와 기존 CPU tensor 감사 코드를 확인했으며 tensor 전체를 새로 역직렬화해 검증했다고 주장하지 않는다.

그림 8개는 수치 방향과 원자료 연결이 맞다. 다만 cohort heatmap에는 색상 척도 막대를 추가하면 단독 사용 시 읽기 쉬워지고, NS의 target-new NLL 곡선은 낮아짐을 개선으로 오독할 수 있으므로 true/margin과 함께 읽어야 한다. 원그림은 수정하지 않았다.

## 11. 현재 L4 두 보호 집합 설계에 주는 의미

이 결과는 Base의 추가 변화를 먼저 줄이겠다는 선택을 지지한다. 이미 높은 Past retention을 더 높이는 것보다, Current/Past 성능을 관측하면서 남은 Base 손상을 줄일 여지가 중요한 연구 질문이다. 다만 Base를 개선하기 위해 Current를 무조건 희생하는 우선순위를 뜻하지 않는다.

Base의 We 기준 full-distribution KL은 true NLL만으로 놓치는 경쟁 target 강화도 관측하려는 선택이다. 이번 분석은 그 필요성을 설명하지만 KL barrier의 효과를 검증하지 않았다. W0 대비 누적 손상과 정답 지표는 계속 별도 평가해야 한다.

OS/BF1/BF8/Frozen-BF8의 비교는 그대로 필요하다. OS로 이득이 나면 정적 우회 여력을 발견한 것이고, Frozen 대비 BF8의 추가 이득이 있을 때 실제 output 방향 갱신의 비용을 평가할 수 있다. 이번 결과만으로 실험 arm이나 hard gate를 늘릴 이유는 없다.

Current 보호는 scalar 진척뿐 아니라 실제 NLL-pair와 기존 TF strict 지표로 읽는다. Base 평가에는 true NLL·competing-new NLL·동일 문항 lost/gained를 함께 남긴다. 모두 관측 항목이며 후속 step을 막는 gate가 아니다.

## 12. 근거와 추가 산출물

- [전체 감사 receipt](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/audit-receipt.json)
- [원 package 79개 파일 목록과 SHA](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package-inventory.csv)
- [W0→14개 arm 전이 42행](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/w0-all14-paired-transitions.csv)
- [Base 손상 성분 분해 56행](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/w0-locality-damage-components.csv)
- [명시적 edit prompt 중복 제외 민감도 28행](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/w0-locality-exact-overlap-sensitivity.csv)
- [원보고서의 로컬 봉인 복사본](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/diagnostic-report-ko.md)
- [원본 layer summary](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/layer-summary.csv)
- [원본 overwrite strata](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/overwrite-strata.csv)
- [원본 compute summary](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/compute-summary.csv)

전체 원 package는 `local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/`에 그대로 보존했다. 원시 모델 weight와 prompt 결과는 공개 보고 디렉터리에 복제하지 않았다. 계산 코드·상세 receipt·중복 문항 index는 같은 local review 디렉터리에 보존했다. 다시 계산하는 데 필요한 서버 입력과 경로 대응은 [재현 입력 안내](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/README.md)에 명시했다.
