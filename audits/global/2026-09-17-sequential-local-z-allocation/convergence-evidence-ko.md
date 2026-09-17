# Local-z L8 최적화 종료·수렴 근거 감사

2026-09-17. 기존 cold 7-arm 중 LD의 저장 로그·per-target counter·작은 receipt와 B1 fit capture를 CPU로 검산한 기록이다. 새 실험, 모델 forward, GPU 호출, 원 산출물 변경은 없다. 이 문서와 [구조화 근거 JSON](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-17-sequential-local-z-allocation/convergence-evidence.json)만 새로 작성했다.

## 1. 확인된 핵심

**L8의 평균 Adam 4.176회는 각 요청이 약 4회에 수렴했다는 뜻이 아니다.** a4=.75 뒤 L8은 826개가0회, 174개가24회이며 중간1–23회는 없다. 선택된9batch만 보면740/900이0회,160/900이24회다.

24회를 사용한 요청은 총loss<.05 조건을 충족하지 못했다. 그러나 대부분 최종 target NLL은 이미 작고, 총loss의 큰 부분은 decay와 KL이다. 따라서 cap 종료만으로 target이 덜 학습됐다고 판단하거나, 추가 반복이 PS/locality를 개선한다고 주장할 수 없다. 반대로 최종 gradient/KKT가 없으므로 수렴했다고 단정할 수도 없다.

## 2. 입력·결속·검증 범위

원격 파일은 모두 server4의 기존 파일을 read-only로 읽었다. 원격 경로를 이 workspace의 로컬 파일 링크처럼 표시하지 않는다.

- stdout: `/data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1/submission-v1/science-48680_0.out`, 3,260,708bytes /50,164줄. SHA256 `558810a1aaf32fbcaaf8ecf9af41ac6138e08a367ec7903dc21ec73c0511be7e`.
- per-target CSV: `/data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1/review-20260917-v1/tensor-audit/target-counters.csv`, 전체12,000행. SHA256 `ec99a4ca230f0a91ff36642928fdd843f247888a73b020b2a8216808390fa6f6`.
- B1 fit: `/data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1/arms/LD/attempt-v1/output/B001/proposals/local8-a4-0.75.pt` 및 `local8-a4-1.pt`. 기존 full-SHA inventory를 재사용하고 이번에는 CPU `weights_only/mmap`으로 작은 capture들을 읽었다. 큰 파일 전체 SHA를 새로 계산하지 않았다.
- 각batch의 `local8-a4-{0.75,1}.json` receipt20개에서 full-fit 실제 weight delta norm을 읽었다.

stdout를 source의 `L4→L8@.75→L8@1` 순서와 각 fit의 요청 순서에 따라 기존 per-target CSV에 대응시켰다. **30fits×100=3,000요청,31,584 loss 관측**의 수가 일치했다. 모든 요청의 loss 관측 수, 최종 delta norm(차이<1e-5), 최종 반올림 loss(원값과 차이≤.000501)를 교차 검산했다. 로그 자체에 case_id가 직접 찍힌다고 주장하지 않는다. ID는 저장 capture에서 추출된 CSV와 고정 호출 순서를 통해 결속했다.

B1 target tensor는 compute_z capture 및 target_observations의 target과 exact 일치했고, zero-step delta는 모든 원소가 정확히0이었다. CPU tensor 검사 후 `torch.cuda.is_initialized()==False`를 확인했다.

검산에 사용한 [기존 per-target 추출 코드](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/project/run_scripts/local_z_adaptive_allocation/analysis/tensors.py:62), [실제 fit 순서](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/project/run_scripts/local_z_adaptive_allocation/engine.py:28), [공개 counter 표](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/completed-review-20260917-v1/diagnostic-report-ko.md:361), [기존 input manifest](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/completed-review-20260917-v1/input-manifest.json), [기존 raw inventory](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/completed-review-20260917-v1/raw-inventory.json)를 구분한다. stdout와 per-target CSV의 새 집계값을 기존 공개 표에 이미 기재된 값처럼 쓰지 않는다.

## 3. 반복 횟수와 두 L8 branch의 차이

| LD fit | 요청 | 0 Adam | 24 Adam | Adam 합 | 평균 | 중앙값 |
|---|---:|---:|---:|---:|---:|---:|
| own L4 |1000|1|999|23976|23.976|24|
| L8 @ a4=.75 |1000|826|174|4176|4.176|0|
| L8 @ a4=1 |1000|982|18|432|.432|0|
| 합계 |3000|1809|1191|28584|—|—|

두 L8 branch를 합친 zero 비율은1808/2000=90.4%다. 실제 선택된 `(.75,.5)`9batch는740/900=82.22%이므로 두 분모를 섞지 않는다. L8 full-L4 branch에서는 이미 편집된 상태라 대부분 zero-step이며, 이 비율을 약한 L4 뒤의 보완 write에 그대로 대입하면 안 된다.

| Batch | L8@.75 zero/nonzero | L8@1 zero/nonzero | .75 뒤 full-fit ΔW8 norm | 실제 선택 ΔW8 norm |
|---|---:|---:|---:|---:|
|B1|64/36|100/0|7.398135|3.699067|
|B2|78/22|95/5|5.820525|2.910263|
|B3|78/22|99/1|5.897822|2.948911|
|B4|86/14|98/2|4.820722|0 (N4 선택)|
|B5|89/11|98/2|4.415904|2.207952|
|B6|89/11|100/0|4.506561|2.253280|
|B7|84/16|98/2|5.457063|2.728532|
|B8|87/13|100/0|4.857310|2.428655|
|B9|89/11|99/1|4.546962|2.273481|
|B10|82/18|95/5|5.882052|2.941026|

nonzero는 모두24 Adam이다. full-fit norm과 half-write 실제 norm을 구분한다. [기존 실제 선택 delta 표](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/audits/global/2026-09-17-local-z-independent-review/layer-delta-by-batch.csv)의 norm은 weight 변화량이다. `native-fit-tensors.csv`의 weight 자체 norm과 같지 않다.

## 4. 중단 기준·loss 구성·최종 clamp 상태

실제 hparam은 Adam lr=.1, decay=.5, clamp=.75, max25 loss evaluations/24 Adam updates, KLfactor=.0625다. [h8 설정은 h4를 deepcopy하고 layers만 변경한다](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/project/run_scripts/local_z_adaptive_allocation/model.py:68).

원격 실제 source `AlphaEdit/compute_z.py:159–189`는 다음 목적을 사용한다.

`total = NLL + weighted_KL + .5 * ||delta|| / ||anchor||²`

총loss<.05 검사와 마지막 iteration 검사를 backward 전에 수행하고, step 뒤 `||delta||≤.75||anchor||`로 투영한다. source SHA `a941a492d9e9e45f2aa7b88ab0137ae20910b423d7e59186b20c85bb285ff79f`, config4 SHA `2392ab8392476ed019985e4292a8c0aa2a3957f36a06a704eb7590e8e8c99c5d`를 직접 다시 읽어 기존 manifest와 일치함을 확인했다.

| 집단 | zero-step 최종 total 범위 | 24-step 최종 total 범위 |
|---|---:|---:|
|L8@.75|.000379–.049944|.050768–.148862|
|L8@1|.000091–.038644|.058933–.131977|

모든 zero-step은 처음 평가부터 기준을 통과했다. 모든24-step은 마지막에도 기준을 통과하지 못했다.

| 최종 항 | 실제 선택 active160: 평균 [최소,최대] | 전체 active174: 평균 [최소,최대] |
|---|---|---|
|총loss, 저장 원값|.0697681 [.0507676,.1488624]|.0708336 [.0507676,.1488624]|
|target NLL, 반올림 로그|.0004813 [0,.004]|.0004713 [0,.004]|
|KL항, 반올림 로그|.0070563 [0,.082]|.0080517 [0,.084]|
|decay항, norm 재계산|.0621377 [.0443351,.0864383]|.0622182 [.0425395,.0864383]|
|native 평균 target probability|.9994355 [.9964411,.9999706]|.9994442 [.9964411,.9999706]|

NLL/KL은 stdout 소수3자리 반올림이므로0을 정확0으로 해석하지 않는다. Decay는 stdout의 init/delta norm으로 CPU 재계산한 값이며 GPU FP32 component bit-parity 주장이 아니다. 평균 target probability는 native의 `exp(-nll_loss_each).mean()`이며 strict token accuracy가 아니다.

최종 `abs(||delta||/radius−1)≤1e−6`는 선택144/160, 전체155/174였다. 현재 delta의 decay 항만 .05 이상인 요청은 선택156/160, 전체169/174였다. **현재 delta에서는 total-loss 중단 기준이 막혀 있지만, 모든 feasible delta의 최적loss가 .05 이상이라는 하한은 아니다.** 이 검사는 최종 boundary 근접 여부이며 반복 중 clamp-hit 횟수를 복원하지 않았다.

## 5. 마지막 다섯 관측의 변화

마지막 다섯 loss 관측은 update20 이후부터 update24 이후까지로, 네 번의 optimizer update 간격이다. 아래 변화는 소수3자리 로그에서 계산했다.

| 집단 | total 감소/동일/증가 | total 평균 변화 | NLL | KL | decay |
|---|---:|---:|---:|---:|---:|
|선택 active160|113/46/1|−.0020438|−.0001313|−.0014563|−.0004563|
|전체 active174|122/51/1|−.0024310|−.0003218|−.0015920|−.0005287|
|B1 active36|21/14/1|−.0012778|−.0001667|−.0006389|−.0004444|

반올림상 동일은 수렴 증명이 아니다. 각 loss 관측은 약±.0005, 두 관측 차이는 약±.001의 반올림 범위를 가진다. 일부 감소는 남으며 평균 변화는 NLL보다 KL/decay에서 크다.

stdout의 B1 첫 active 요청 case16186은3134–3164행, 마지막 다섯 loss는3159–3163행이다. 총loss는 모두 반올림 .067, NLL은0, KL .005, decay .061로 표시되지만 평균 target probability는 .999589→.999658로 증가한다. 이 사례도 반올림 plateau와 정확한 정지를 구분해야 함을 보여준다.

## 6. B1의 zero-step과 실제 residual·write 크기

B1 저장 `Z`와 writer readout `H`는[4096,100]이며 `R=Z−H`를 CPU로 재계산했다. 차분은 저장 FP32 tensor, norm/energy는 FP64다.

| B1 L8 fit | a4=.75 | a4=1 |
|---|---:|---:|
|zero/nonzero 요청|64/36|100/0|
|activation ||δ||F|27.7451112|정확0|
|writer ||Z−H||F|27.7451122|.000208167|
|full-fit ||ΔW8||F|7.3981346|.0000551086|
|실제 half-write ||ΔW8||F|3.6990673|—|

a4=.75의 zero64개는 δ가 정확0이고 R column norm은1.30e−5–3.64e−5였다. 이들의 residual energy 비중은4.11e−11이다. nonzero36개 R column norm은3.466–6.344, 평균4.592이며 energy 비중은99.9999999959%다. `||R−δ||F=.000223719`이고 `||anchor−writer H||F=.000223719`로 미세한 readout 차이에 해당하는 크기다.

따라서 B1의 큰 target residual이 zero-step 요청에서 나왔다는 설명은 맞지 않는다. 다만 residual energy 집중은 요청별 weight/성능 기여의 ablation이 아니며, zero-residual 요청의 key도 공동 solve geometry에 들어간다.

[기존 B1 후보표](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/completed-review-20260917-v1/diagnostic-report-ko.md:263)에서 .75 L4-only→.75 L4+.5 L8은 E .182800→.0124145, current strict99→100이다. 이때 S64 KL은 .000951336→.001252665로 증가하지만 full L4 ownN4의 .001720908보다 낮다. 이는 약한 L4 뒤의 edit 보완으로 읽으며, partial L4의 손상을 직접 repair한 관측으로 쓰지 않는다.

## 7. 판정의 한계

- 최종 gradient norm, projected-gradient/KKT residual, optimizer의 수렴 진단은 미저장이다.
- 정확한 반복별 component tensor와 반복별 clamp-hit 횟수는 없다. stdout의 반올림 값과 최종 capture를 구분한다.
- 추가 반복을 실행하지 않았으므로 더 긴 z 최적화의 actual write, PS, locality 이득은 미측정이다.
- Direct activation target에서 낮은 NLL을 얻어도 선형 solve와 half gate를 거친 실제 모델이 같은 likelihood를 재현한다는 보장은 없다.
- 선택160개가 low-NLL이라는 사실은 paraphrase PS 무손실 근거가 아니다. 기존 LD 최종은 N4 대비 PS−25/2000, NS+284/10000이었다.
- 현재 δ에서 decay≥.05라는 사실을 모든 δ의 목적함수 하한이나 추가 최적화 불가능성으로 확대하지 않는다.

이 기록은 종료 이유와 남은 불확실성을 구분하는 근거다. 새 convergence 정책의 성능 검증이나 실험 결과가 아니다.

