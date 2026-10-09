# server2 저장 zsRE W20 weights 재평가 준비

수신 nonce: USER-GH-ZSRE-SAVED-WEIGHTS-2K-REEVAL-20261009-R1. 정본 a81e4daf 전체 확인. 실제 CWD /mnt/raid5/janghj/ODE-edit 및 origin hyunjun1127/ODE-edit 확인. 전용 non-main WT odeeditsh2-zsre-2k-reeval-20261009에서 준비.

## Checkpoint inventory

- FT: 원 job 61726, 268474909 bytes, 최종 batch20/identity/fullSHA 검증 완료.
- MEMIT: 원 job 61728, 1610637581 bytes, 최종 batch20/identity/fullSHA 검증 완료.
- ALPHAEDIT: 원 job 61730, 8053090197 bytes, 최종 batch20/identity/fullSHA 검증 완료.
- ALPHAEDIT_BLUE: 원 job 61732, 2684378777 bytes, 최종 batch20/identity/fullSHA 검증 완료.
- MEMIT_FE: 원 job 61734, 1610637709 bytes, 최종 batch20/identity/fullSHA 검증 완료.
- SPHERE: 원 job 61735, 8053090069 bytes, 최종 batch20/identity/fullSHA 검증 완료.

6개는 서로 다른 dev/inode의 실제 원본이다. CP 역직렬화/모델 로드 없이 전체 SHA를 계산하고 result SCIENTIFIC_COMPLETE/20 commits 및 latest final_W20/원 identity와 대조했다. 원 checkpoint/source/config/tokenizer/stream hash는 inventory.json에 보존한다. Qwen 진행 trajectory는 대상에서 제외한다.

## 준비와 미완료 구분

- eval-only selected FP32 weight restore 모듈 작성. FT bias 포함 정확 keyset/shape/finite, 복원값 exact equality, nonedited parameter version/pointer guard. NativeEngine/fit/H/C0/P/context generator를 만들지 않는다.
- CPU fixture 4 PASS: 정확 선택복원, bias 누락 차단, batch19 차단, nonfinite 사전 차단. 실제 GPT-J/GPU restore PASS가 아니다.
- base model의 기존 fullSHA와 현재 unchanged dev/inode/size/mtime를 대조. tokenizer/config 및 원 hparams/2000 stream은 현재 fullSHA 대조. Base 대형 모델은 이번에 fullSHA를 재계산하지 않았으며 prior SHA+current unchanged stat 증거이다.
- GH 공통 official/evaluation/zsre_paper.py와 eval-only tracking authority/API는 아직 SOURCE_INPUT_PENDING. 마지막 fetch origin/main3827f644에서 zsre_paper.py 미게시 확인. 공통 evaluator/logger를 복제하거나 기존 잘못된 query evaluator로 대체하지 않는다.
- native full2000 query CPU parity 및 최종 runner/collector/API 결속, sourcefreeze/실제 제출은 공통 exact input 이후. 현재 신규 jobID 없음, Slurm PENDING 아님.
- 원본 CP/weights/raw/frozen job 변경·취소·전송·삭제0. 새 편집/fit/W0/CF/FluCon/qualification0. 원본 replica 복제 없음. NO_BROADCAST_NOT_REQUIRED: 동일 host 기존 자산을 읽기 전용 사용.

위 SOURCE_INPUT_PENDING은 최초 준비 시점의 역사 상태다. 이후 GH READY 4533756e의 공통 evaluator/query parity/tracking exact SHA를 결속해 해당 입력 blocker를 해소했다.

## 실제 READY 후 구현

전체 2000 request / 20808 query CPU native AST parity 통과: Eff 5557, Gen 5557, Loc 9694; input/target mismatch0, model forward0. Query SHA 2d27e4fc4445e709586e2d80f92dee76dfc1b6ce6cd7d333b6ca318ccb328a4b. 이는 실제 pretrained 출력 수치 parity가 아니다.

Eval-only runner/독립 CPU collector/held submitter를 추가했다. 최종 all_seen/post edits2000만 기록하고 진행률은 별도 scalar로 전송한다. 원 CP를 mmap read-only로 읽어 선택 W만 복원하며 원 CP 파일 SHA를 평가 후에도 확인한다. 현재 runtime 안의 finite/count/parameter/RNG guard를 유지하고 추가 qualification은 실행하지 않는다.

준비 영수증의 mtime_ns가 JSON 소비자에서 IEEE754 반올림된 것을 fail-closed stat 검사로 발견했다. 원본 6개를 다시 전체 SHA 재검증했고 모든 payload hash는 같았다. timestamp를 decimal string으로 바꿔 정확히 보존했다. 원 파일을 수정하거나 stat 허용값을 완화하지 않았다.

최소 CPU 15 PASS, source166/Python320/externalimports0. 별도 GPU/온라인 PASS는 아니다. 원 Qwen source69bfbb2c 및 jobs는 KEEP.

## 실제 등록/release

실행 source ce8d536fa7eb4dfdd38f7024381e68d212f44ea0. GPU6+CPUcollector1 전량 held 상태에서 owner/Command/WorkDir/full argv/source/config/input/CP/resource/dependency/tracking 계약을 검사하고 release했다.

| method | job | afterany resource dependency |
|---|---|---|
| FT | 61942 | 61918 |
| MEMIT | 61943 | 61920 |
| AlphaEdit | 61944 | 61914 |
| BLUE | 61945 | 61916 |
| FE | 61946 | 61942 |
| SPHERE | 61947 | 61943 |
| CPU collector | 61948 | 61942..61947 |

2026-10-09T08:52:45.504597+00:00 release 직후 단발 snapshot: 전부 dependency PENDING. 기존 Qwen 4개 resource lane을 연장하여 cap4를 보장하며 타 job 변경0. GPU 각1/CPU6/59392MiB/4h, collector GPU0/CPU2/4096MiB/4h; exportNONE/Requeue0. Runtime cap4 directUSER와 local4/QoS4를 결속, 구 canonical TSV2는 역사 상태이며 global cap파일 변경0.

W&B 현재 NOT_STARTED_UNTIL_ACTUAL_JOB_INIT. 실제 실행 때 새 immutable UUID/actualjob/name/provenance로 online 기록한다. 현재 source/API/CPU/등록 성공은 actual GPU 평가 또는 online delivery PASS가 아니다. 원 CP에 checkpoint write/전송/삭제0; 평가 raw는 ignored local만 보존.

Local root/log: /mnt/raid5/janghj/ODE-edit/local/official-baselines/server2/zsre-2k-reeval-20261009/registration-r1/ (logs/<METHOD>-<job>.out 및 .err). 결과 GPU 완료 대기/장기 polling/자동 retry 없음. README는 GH가 제출 영수증으로 통합한다.
