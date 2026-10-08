# GPT2-XL AlphaEdit-BLUE / PRUNE / RECT W20-only generation

Source·compact receipt/report를 own branch와 main에 비강제 게시했고 첫 원격 main `cc03e93406883b5d6c0143d3f2c8e10d5a95928d`를 exact 확인했다. 이는 publication commit이며 실행 source `9a8c7ebf`/봉인 archive는 변경하지 않았다.

현재 단계는 실제 세 GPU job+target-only CPU collector held 검사 및 release 완료다. 최종 owner preflight 122개 PASS, skip/failure/error0. 실제 GPU qualification·W20 점수·새 W&B 원격 기록은 미관측이다. Source `9a8c7ebfae197cc0d8dba994ff1208cdb24635f8`, config SHA `ea4b2aa08b99e4b5dd097855a7d006aa53df3a75402dd320a36de30af39906e0`, lock SHA `0616974ac0e809b177d58d395c6c12cfe6eb825d470162b6e7586a00a4a0e6c7`.

| Method | Actual job | Resource afterany | Bounded initial state |
|---|---|---|---|
| AlphaEdit-BLUE | 61436 | 없음 | PENDING / ReqNodeNotAvail |
| PRUNE | 61437 | 없음 | PENDING / ReqNodeNotAvail |
| RECT | 61438 | 61436 | PENDING / Dependency |
| GPU0 collector | 61439 | 61436,61437,61438 | PENDING / Dependency |

소유 project active/admitted GPU0에서 새 DAG width2/cap2를 검산했고 release 직전 재검산도 PASS다. 모든4 owner/fullargv/script/source/input/ref/PLAN/resources/dependencies 검산 및 reverse release 성공. Fresh root available130369314816B, RAID1368186036224B, inodes335242110; reserve16GiB. 현재 물리 노드 가용성 문제로 정상 PENDING이며 완료·실패 지점 통과를 기다리지 않았다.

사용자 권한 nonce `USER-GH-SH1-GPT2XL-BLUE-PRUNE-RECT-W20-GENERATION-20261008-R1`, authority `4e8a77004aef3cfff43f17b795f2b920c78e51ea`, envelope SHA `2728196c31779c98fd599ec9a5acf9ff0424c87c37dc6f24efa8fdbc5c05d54f`를 전체 읽고 검산했다. App thread/session/host와 실제 root CWD `/mnt/raid5/janghj/ODE-edit`를 확인했다. 역사 registry29e4는 강제 CWD로 쓰지 않았고, 원 dirty root는 보존했다.

## 정확 기존 대상 및 보존

61170 AlphaEdit-BLUE, 61171 PRUNE, 61172 RECT는 실제 accounting에서 이미 CANCELLED/elapsed0/start없음/allocation없음이다. 기존 제출 receipt·source bb86a6ca·Slurm WorkDir/SubmitLine을 대조했다. 이번 취소0. 공유61173은 COMPLETED이지만 실패 보고 collector일 뿐 과학적 성공이 아니며 그대로 보존한다. 보호한 MEMIT61167 FAILED(allocated GPU28015초), AlphaEdit61168/CAKE61169 CANCELLED 및 OURS/W0/자산 준비/다른 서버 job은 변경·재제출하지 않았다.

새 세 arm은 독립 coldW0/H0, exact ordered first2000, BS100×20이다. GPT2 revision `15ea56dee5df4983c59b2538573817e1667135e2`, case-order digest `0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4`, 기존 native bundle/model/tokenizer/C0/P/context/YAML/evaluator/runtime/reference를 읽기전용 결속했다. 큰 모델/자산은 기존 SHA와 현재 size/inode/mtime 증빙을 사용하며 모든 모델 byte 재해시/GPU 검증을 수행했다는 주장은 하지 않는다. stats/P 재계산·다운로드0.

## 구현 및 검산 범위

새 task-private `w20_*` caller/control/collector는 old runtime의 global을 바꾸지 않는다. W0/current/pre/post/W5/10/15/first500 generation은 없다. 기존 R/P/N W0·current100 pre/post·all-seen W5/10/15/20·paired retention·TF·NLL 일정은 그대로 유지한다. W20 R2000/P4000/N20000. PRUNE은 기존 saved coldW0+compressed cumulative delta 수리 후 같은 실제 state를 평가한다. native 알고리즘/계수/층/dtype/solver/편집 예산 변경0.

실제 W20 first2000에서 한 번 생성해 fluency와 consistency를 함께 계산한다. 최종 planned2000/valid count/결측 이유를 검산하고 관측 없는 평균은 생략한다. 진행률은 `phase=W20_generation`과 `generation_progress/step` 별도 축, 최종 generation은 `all_seen/post`·edits2000에서만 기록한다. 실제 job 번호/name·immutable UUID/URL/config receipt·scalar privacy를 유지한다. SDK 접수와 remote readback은 다르다.

원 사전고정 qualification PLAN SHA `83cee93d1b462f8e0b503f4692dd537430ee9084971aaa3a9150bcf4a2a4590b`, 동일 최대8 prompt/허용값을 그대로 봉인한다. 각 arm의 최초 승인 job 안에서 실제 receipt를 작성한다. 이전 cached singleton의 strict 실패를 PASS로 바꾸지 않았고, reference fallback을 속도 향상으로 주장하지 않는다. 별도 편집 pilot/fit/sweep0.

최근 보호 MEMIT 실패의 실제 원인은 `observer.subset()`의 digest 변수와 member 이름 변수 충돌이었다. 새 공통 source에서 이름만 분리했고, 두 subset·반복 호출·digest basename CPU 회귀12검사 PASS. 원 source/archive는 불변이다. Owner audit와 독립 `/root/cache_repair_red` source/CPU 검토를 실제 수행했다. 세 실제 production native_loop의 20회 호출 CPU fixture와 W20 실패 시19commit 보존, flattened W20 receipt/qualified binding/saved raw2000의 collector 통합을 확인했다. RPN/native seam의 일부는 명시적 mock이며 실제 모델/GPU PASS가 아니다. Raw reducer 검산은 counts/sums/cohort/source 산술이며 생산 텍스트의 TF-IDF 재평가와 구분한다.

## 자원·산출물·인계

Server1 combined cap2 또는 더 엄격 정책, 각 GPU1/CPU8/65536MiB/48h 상한, target-only collector GPU0/CPU8/24576MiB/4h. 기존 source·현재 frontier는 새 admission에서 한정 재확인한다. 새 DAG는 resource `afterany`이며 W0 generation READY·성능 `afterok` gate가 없다. Free GPU agent polling 없이 실제 scheduler PENDING을 인계한다. wall은 ETA가 아니다.

프로그램 시간/RSS/Slurm allocation은 별도 기록한다. qualification에서 CUDA peak를 route마다 reset하므로 terminal CUDA peak를 전체 프로그램 peak로 표현하지 않고, qualification route별 peak를 함께 보존한다. NoCP/editedW/H/optimizer durable tensor0, KV는 RAM만. 원 raw/source/실패/비용/취소 이력 KEEP. `NO_BROADCAST_NOT_REQUIRED`: 같은 host 자산·raw를 복제하지 않으며 compact source/manifest/report만 Git 공유한다. 신규 recurring monitor/heartbeat/automatic retry0.
