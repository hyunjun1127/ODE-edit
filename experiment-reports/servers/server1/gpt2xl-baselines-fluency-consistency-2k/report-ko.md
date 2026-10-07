# GPT2-XL native baseline generation rerun 사실보고

상태: **구현/CPU 검산/immutable source freeze 완료, 실제 Slurm 등록은 저장공간 오류로 차단**. 신규 job ID 없음, held 검사·release 미실행, 새 GPU 관측·W&B run·generation W0 미실행. ACK/CPU PASS를 실행 완료로 쓰지 않는다.

## 권한과 경계

Nonce `USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1`, authority `b0cee1a302c10d48e206a31c7c5cf318ac36d8c3`. Actual app CWD `/mnt/raid5/janghj/ODE-edit`, session `01a04939-f93a-7b50-bca0-65438eab2062`, host devbox/server1. 과거 registry29e4는 역사값이다. 전용 non-main WT/branch에서 작업했고 dirty root, 다른 agent/source, OURS/PRICE/W0/setup/model/C0/P 및 기존 raw를 보존했다.

Baseline-only 취소는 source/owner/argv/script SHA와 실제 상태를 대조하여 collector→pending successor→active parent 순서로 수행했다. `60759,60741,60758,60757,60740` 취소 성공. 최종 정확9-ID active queue는 비어 있었다. `60150,60151,60152,60739`는 이미 COMPLETED여서 보존했다. 기존 60740의 과거 GPU 비용4703초, terminal GPU 비용은 MEMIT5663초/Alpha6483초/CAKE7173초이며 신규 실험 비용과 섞지 않는다. Pending 취소는 GPU 할당0. 이전 결과/부분결과/출력 삭제0, rescue CP0.

## 구현·공통 결속

공통 source `83535c6a47c552cc4e5c6385f3a587d752820150`, 최초 main 게시 `2828ab0938184ae863831c6e5e93fcf7a021eca4`. API는 `project/run_scripts/experiment_generation_eval/README.md`. `GenerationObserver.observe`는 exact state/runtime/occurrence별 실제 생성1회, `subset/read_observed`는 identity-checked CPU 재사용이다. 호출자는 ordered occurrence와 actual physical W identity 및 H/context guard를 결속한다. 원 native fit/writer/context/precision은 기존 sealed source를 읽기 재사용했다. 추가 fit/pilot/품질 gate/B21 없음.

Declared profile `cf-cake-prompt-inclusive-total100-eos-corrected-v1`: unpadded single-row full-prefix no-cache, topk5/temp1/topp1/총prompt+continuation100/native EOS. 원 CAKE generator의 byte/published-performance parity 또는 빠른 route라고 주장하지 않는다. Prompt truncate/overwrite0, 이미100이상이면 그대로 보존·continuation0·typed reason. Python/NumPy/CPU/이미 초기화된 CUDA RNG, W/H/context/hooks/parameter-buffer/training mode 비변이와 예외 restore를 검산했다.

Fluency는 native NLTK H2/3+2H3/3 bits, per-text→per-occurrence→valid macro. Consistency는 fixed vocab/IDF cosine, relation/target 전체 reference이다. 재fit/subject-only essence 필터 없음. Missing/zero vector/nonfinite는 typed missing, 유효한 measured0과 구분한다. Raw sums/counts/overlap costs는 local 보존; 공개는 scalar mean/count/reason만이다. R/P/N/harmonic/분모는 불변. W&B strict key/config, job/name/immutable identity, edits/pre/post state와 fit 축을 결속했다. SDK 접수와 remote delivery는 별도이며 이번 새 run의 실제 online 검증은 아직 없다.

W0 generation은 첫 실제 BASE_MEMIT cold run에서 fresh first2000을1회 측정한 후 atomic READY를 공유할 계획이다. 과거 RPN-only W0를 generation evidence로 대체하지 않는다. 모든 baseline current/pre·post100, milestone W5/10/15/20 seen prefix를 평가하며 overlap은 같은 raw CPU subset으로 재사용한다. 새 생성 case 상한 계획은 arm별8500 + 공유W0 2000 = 여섯 arm53000이며 실제 비용/속도/ETA 측정값은 아니다. 각 raw/phase에 forward/token/time 비용을 별도 보존하고 실패 전 반환된 관측도 double-count 없이 집계한다.

## Reference READY와 CPU 증거

Reference actual manifest `/mnt/raid5/janghj/ODE-edit/local/baseline-generation-eval-assets/20261007/reference-ready-r1/manifest.json`,6769B SHA `6d9a713ab7eaa10871f277e10a3974e0bd5b140c265be80052c279f258876ca8`. READY999B SHA `634ca5c70f54f7af5346849b3594ed92280cfbc3853d0def16a7eef86ae64203`, identity `75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6`. 3개 공식 공개 reference를 단회 다운로드·SHA/schema 검산했으며 total968966959B. 정확 경로/size/SHA는 audit reference manifest에 있다. first2k prompt/reference coverage 각각2000/2000, missing0은 coverage 증거이지 과학 점수 PASS가 아니다. Vocabulary/finite IDF1380255, TF-IDF fitting0. NumPy2.2.6/SciPy1.15.3/sklearn1.7.2/NLTK3.6.5와 기존 active English punkt bytes/source를 결속했다. Peer는 exact files를 단회 pull하고 path override 가능; 다른 science env pin 변경0. Raw/model/stat/P/credential 대형전송0, NO_BROADCAST_NOT_REQUIRED 및 exact reference-pull 예외만 기록.

최종 owner CPU production fixtures154개 중152 PASS,2 SDK 의존 skip, 실패0. 테스트 source SHA receipt 보존. Independent reviewer1은 다른 작성자의 generation/tracking/source를 읽고19 unit·7 metadata observer·2 DAG checks PASS; 자신의 assets 구현은 제외했다. Reviewer2는 다른 작성자의 assets/native runner/collector32 CPU checks PASS; 자신의 공통 generation 수리는 다른 reviewer가 재확인했다. 실제 GPU/native fit/online 검증과 분리한다.

발견·수리한 기술 경계: 혼합-state endpoint/selected-row stale hash 재사용, startup manifest pointer/member/identity, native cosine1.0000000000000002의 FP64 endpoint rounding, `/tmp` ENOSPC와 CPU 준비의 잘못된 runtime key. 원 score/formula는 보존했고 cosine은 clamp하지 않았다. 고정4-ULP transport endpoint 허용만 명시하며1.000001은 거절한다. 원 품질/numerical threshold를 성능에 맞춰 바꾼 것이 아니다. RAID task-local TMPDIR를 제한 SDK sidecar에도 연결했다. 이전 CPU 준비/검산 기록은 남기며 scientific repeat-to-PASS0.

## 실제 등록 결과·차단 원인

실행 source `6bc51602632b5a2dfb4c832479002b30b604b8eb`.
Immutable attempt `/mnt/raid5/janghj/ODE-edit/local/gpt2xl-baselines-fluency-consistency-2k/20261007-v1/attempt-v1`.
Lock SHA `dd1690924118251ab302aeb885ffdd396976b6b863a731b7f54b120e5b135f35`, config SHA `713e164ce793ea7231d3b08b8bfab1a55adc08d74a8de62c290e6bd999f583aa`, source archive SHA `e502cb6a9e2e259aa074944dc834047a49a730750b7f404e90553ad5190f3fa9`. Archive CPU runtime binding332 source/137 native/3 original config members 검산 PASS, model forward/native apply0.

| Arm | 신규 job/dep | 실제 상태 |
|---|---|---|
| BASE_MEMIT | 없음 / 첫 head 계획 | 첫 sbatch I/O 실패 |
| BASE_ALPHAEDIT | 없음 / 새 MEMIT afterany 계획 | 미등록 |
| CAKE | 없음 / 새 MEMIT afterany 계획 | 미등록 |
| ALPHAEDIT_BLUE | 없음 / 새 Alpha afterany 계획 | 미등록 |
| PRUNE | 없음 / 새 CAKE afterany 계획 | 미등록 |
| RECT | 없음 / 새 AlphaBLUE afterany 계획 | 미등록 |
| CPU collector | 없음 / 새6 afterany 계획 | 미등록 |

Fresh own server1 GPU allocation/admitted queue0, prospective DAG width2/cap2. Other users' physical GPU2는 own project allocation으로 오인하지 않는다. 각GPU1/CPU8/65536MiB/48h request, CPUcollectorGPU0/CPU8/24576MiB/4h. Memory ceiling183296MiB, actual node/partition/QoS/disk/inodes 확인. 더 엄격 cap1이면 전체 직렬화하며 oldcancelled IDs를 resource gate로 사용하지 않는다. OURS/W0/타task source/job 변경0.

실제 제출1회에서 `sbatch: Batch job submission failed: I/O error writing script/environment to file`. 신규 ID 반환 없음, submitted receipt0, exact new task squeue empty. Slurmctld는 UP이나 StateSaveLocation `/var/spool/slurmctld`, SlurmdSpoolDir `/var/spool/slurmd`가 있는 `/dev/nvme0n1p2`의 일반 사용자 available bytes0/100%이다. Inodes는 소진되지 않았고 RAID에는 약1.2TB가 있다. `sacct`도 No space left on device로 실패하므로 내부 ID 소비 여부까지 인증하지 않는다. 이는 수치/방법/모델/C0/P failure나 GPU cap 부족이 아니다.

허가 없는 root/spool 삭제·이동·권한·서비스/config 변경은 하지 않았다. Sealed source/config/failure receipt 보존, 자동 재제출/monitor/heartbeat 없음. 관리자 권한으로 scheduler root 저장공간을 복구한 뒤 exact duplicate/immutable-attempt 검산을 거친 수동 owner recall이 필요하다. 기존 source/raw, 승인된새 C0/P와 W&B spool 보존. 새 실험을 완료 또는 Slurm PENDING이라고 쓰지 않는다.
