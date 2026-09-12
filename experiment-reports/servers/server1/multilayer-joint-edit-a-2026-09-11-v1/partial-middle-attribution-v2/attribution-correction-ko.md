# Middle A0 대칭 attribution 정정 보충 — 설계 §13

이 보충은 이미 게시된 `partial-middle-v1`의 결과·소스·raw bytes를 변경하지 않는다. 기존 네 state의 NLL 관측 자체는 그대로이며, 기여 명칭과 빠진 대칭 attribution을 정정한다. 새 모델 실행·평가·GPU·tensor 읽기·PNG 생성은 0이다.

## 1. 정정 범위

v1 `signed-attribution.csv`의 `e4`, `e8`, `e48` 및 해당 그림은 실제로 각각 **standalone E(4), E(8), E(4,8)** 값이다. v1에서 `e4/e8`라고 부른 standalone 효과를 설계 §13의 대칭 layer 기여로 읽으면 안 된다. 이 보충에서는 혼동 없이 `standalone_E4/E8/E48`로 명시한다.

설계의 실제 signed 기여는 `signed_L4=.5*(E4+E48−E8)`, `signed_L8=.5*(E8+E48−E4)`다. 합은 총 NLL 개선 E48이다. Interaction=E48−E4−E8은 기존 값과 동일하다. Standalone 효과는 각 block 단독 적용, symmetric 기여는 두 block 제거 순서의 평균적인 한계 효과라는 서로 다른 양이다. 이는 선택한 두 block intervention에 한정되며 fact 저장 위치의 증명이 아니다.

## 2. 정정 수치

| metric | quantity | mean | median | p90 | max | positive / zero / negative | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RS | standalone_E4 | 6.91166371 | 7.06459248 | 11.2720486 | 16.5137939 | 97 / 0 / 3 | 100 |
| RS | standalone_E8 | 4.40139622 | 4.94401244 | 13.1585078 | 19.1324366 | 72 / 0 / 28 | 100 |
| RS | standalone_E48 | 8.52062984 | 8.47367526 | 14.0986899 | 20.4607768 | 100 / 0 / 0 | 100 |
| RS | signed_L4 | 5.51544867 | 5.18341783 | 9.209633 | 14.6448295 | 97 / 0 / 3 | 100 |
| RS | signed_L8 | 3.00518118 | 3.02283359 | 8.09181523 | 11.5397098 | 74 / 0 / 26 | 100 |
| RS | interaction | -2.79243008 | -2.54249133 | 4.57615557 | 18.4239001 | 35 / 0 / 65 | 100 |
| PS | standalone_E4 | 5.05293738 | 4.81856728 | 9.41784874 | 13.738965 | 187 / 0 / 13 | 200 |
| PS | standalone_E8 | 3.42852767 | 2.63680649 | 10.3953864 | 17.8684854 | 146 / 0 / 54 | 200 |
| PS | standalone_E48 | 6.60601471 | 6.40328902 | 11.797275 | 17.8546278 | 193 / 0 / 7 | 200 |
| PS | signed_L4 | 4.11521221 | 3.91468734 | 8.23437764 | 11.9696151 | 189 / 0 / 11 | 200 |
| PS | signed_L8 | 2.4908025 | 1.78392066 | 6.71336852 | 12.9353258 | 158 / 0 / 42 | 200 |
| PS | interaction | -1.87545033 | -1.09800036 | 3.39413898 | 14.2087282 | 73 / 0 / 127 | 200 |

RS는 Current rewrite100 requests, PS는 동일100 requests의 두 paraphrase prompt200개다. 평균은 해당 prompt-row 분모이며 PS를 독립200 requests라고 주장하지 않는다. Request/prompt별 signed 원값은 `symmetric-attribution.csv`에 있다. 총 개선≤0인 행도 제거하지 않으며 기여율은 계산하지 않았다.

## 3. Identity·해석 경계

300행에서 대칭 합 identity failure=0. FP64 최대 절대 residual=0, backward-roundoff bound 대비 최대 비율=0. 수학적 항등식과 부동소수점 bitwise equality는 구분했다. 수치 상한은 `8 eps64 max(1, |E4|+|E8|+|E48|)`이며 성능 gate나 조정된 threshold가 아니다.

두 layer의 평균 대칭 기여가 양수여도 50:50 할당 성공 또는 native-L4 대비 유용한 부담 분산을 입증하지 않는다. A0-L4/A0-bal0 및 동일 Current 품질의 OS/BF·audit 비교가 필요하다. 음의 interaction은 개별 효과의 중복/비선형성을 포함하므로 이를 단독으로 유해 상호작용이라고 단정하지 않는다. 기존 v1의 높은 Current RS/PS와 큰 Base/Past harm 및 balance 포함 objective 증가 관측은 바뀌지 않는다.

v1 물리 magnitude 표의 `||·||F` 헤더는 Markdown separator와 충돌하는 표현이었다. 새 생성기는 헤더의 pipe를 escape하도록 수정했으며, 게시된 v1 bytes는 그대로 보존한다. 이 보충은 물리 magnitude 값을 재평가하거나 변경하지 않는다.

## 4. Provenance와 재현

입력 CSV SHA256 `ebe9ac0e8d0bda4ccc5e4027f69e8a9b2d9e00d71dedf4132c51c035346e763a`. v1 manifest SHA `097450bca315856fbc70b2b7a4bf631eba4a18bc58bb1e7d2e55da26d3f4971e`, receipt identity `8c6ae516ec778637c57de6734009509333a742b0cbe8cd77374fdd7a83fdbab6`. 이 보충은 이 CSV의 세 standalone scalar만 사용했으며 원본 NLL·tensor·모델을 다시 계산하지 않았다.

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.multilayer_joint_compensation.track_a.analyze_partial --attribution-supplement /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-multilayer-joint-edit-a-v1/experiment-reports/servers/server1/multilayer-joint-edit-a-2026-09-11-v1/partial-middle-v1 --output <new-create-once-supplement>
```

Source SHA, 입력 identity, 출력 members root는 `manifest.json`과 `rooted-receipt.json`에 있다. scientific_promotion=false.
