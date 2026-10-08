# SH2 native FLU/CON 재등록

`USER-DIRECT-SH2-GPTJ-NATIVE-FLUCON-REREGISTER-20261008-R1` 전용 새 entry이며 기존 source/archive/job을 수정하지 않는다.

실행은 `generation_native_bind` → `generation_native_cpu_checks` → source commit → `generation_native_submit` 순서다. 이미 존재하는 준비·CPU·등록 receipt는 덮어쓰지 않는다. Source/API 검사는 실제 GPU 관측이나 online delivery 인증이 아니다.

각 경로의 cold first2000 BS100×20 native fit/write/history/RPN은 기존 구현을 재사용한다. `generation_native_run`은 매 batch 및 W5/10/15/20 RPN을 유지하고, 실제 commit.json 20개 저장 이후 W20 first2000에서 `NativeGenerationObserver`를 한 번 호출한다. F/C는 동일 생성 text의 점수다. W0·중간 생성, 옛 MB qualification, 공유 W0 generation READY, fallback, old partial generation 재사용은 없다.

SH1 `adb244e6f9c86b54f73bd6d8fb833b338f470ded`의 native profile `cf-cake-native-casebatch-kv-total100-globalrng-v1`을 재사용한다. Per-case padded KV/topk5/batch multinomial, prompt-inclusive total100/noEOS, endpoint seed once/finally restore는 명시적인 새 의미이며 옛 파생 경로의 bitwise 동등성이나 속도 우위를 주장하지 않는다. 공통 generation/tracking namespace는 읽기전용이다.

Stock AlphaEdit는 B1 이전 H={}이며 B1 native 호출에서 초기화된다. CAKE/BLUE의 caller-owned H와 구별하며 B1 이후 각 arm의 실제 H와 W 연속성을 검산한다. PRUNE terminal base fix, RECT dense planning/masked commit, native bias 및 원 hparams/precision은 변경하지 않는다.

합산 cap2 아래 기존 61428을 보호한다. 해당 job이 유일한 정확 RUNNING/allocated1 frontier이면 새 MEMIT는 그 뒤, 새 AlphaEdit는 두 번째 lane이다. 이어 MEMIT→CAKE→PRUNE, AlphaEdit→BLUE→RECT를 `afterany`로 묶는다. 추가 frontier이면 두 head가 모두 그 뒤로 대기한다. 더 엄격한 cap1이면 단일 lane으로 직렬화한다. Collector는 여섯 새 GPU job 뒤 CPU-only다. 전량 held owner/source/full argv/script/resource/dependency 검사 후 release하며 반복 제출·취소·재시도는 없다.

공통 online scalar logger의 새 native profile 허용값을 결속한다. 실제 Slurm ID와 run.name job 번호, immutable run/source/config identity, 별도 fit·generation progress 축을 유지한다. Network delivery와 scientific completion은 구별한다. Raw/text/token/tensor/model/code/secret 업로드, checkpoint 저장 및 exact resume는 없다. 등록 후 한정 snapshot 인계만 하며 runner/collector는 자연 진행한다.
