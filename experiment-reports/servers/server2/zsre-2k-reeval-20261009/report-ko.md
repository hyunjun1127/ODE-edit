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

현재 정확 blocker는 사용자 승인이 아닌 공통 evaluator 및 tracking source 입력이다. 실제 평가 output/온라인 PASS 또는 완료 지표를 아직 주장하지 않는다.
