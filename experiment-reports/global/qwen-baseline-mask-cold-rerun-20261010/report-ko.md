# Qwen baseline context-mask 수정 및 cold rerun

nonce USER-GH-QWEN-BASELINE-MASK-COLD-RERUN-20261010-R1.
공통 source ac19db2f, 공식 EasyEdit cached context generator의 mask만 fullprefix로 변경.
샘플링/길이/seed/hparams/수식 변경 없음. 원 upstream SHA는 SOURCES.json에 보존.
Qwen/GPTJ/Llama 작은 CPU FP32/eager 모델로 실제 generator·서로 다른 길이 rightpadding,
무편집 weights 불변, 잘못된 Qwen mask negative control, 네 native import/FEhistory 경로 검사3개 PASS.
source166 검증 PASS. pretrained/GPU 성공 및 실험 완료를 의미하지 않는다.

11개 cold 편집: SH2 본표8, SH1 FE_HISTORY1, SH4 heldout500 두개.
SH2의 FT61900은 편집 오류 대상이 아니며 최신공식 evaluator 최종CP 평가-only를 별도 등록/중복 확인.
OURS/PRICE/tuning, BLUE/FT 편집, 기존 Llama/GPTJ 실행 보존. 무관자원 cap 증액0.
각SH는 exactowner/현재상태 확인후 scopedcancel/의존성재연결/새freeze/held검사/release를 수행한다.
CF FLUCON은 최신 사용자 명령으로 DEFERRED, 최종2K(heldout500)CP 보존.

원 수치/원source/raw/CP는 유지하지만 영향받은 기존 점수는 수정된 본표 성능에서 제외했다.
새 ID 도착 전 README는 RERUN_REQUIRED로 표시하며 Slurm PENDING을 가정하지 않는다.
현재 단계: 공통 source 게시 및 SH1/SH2/SH4 직접 수락 완료.
SH1은 old61975를 보존하고 새 62061(official-s1-cf-qwen25-memit-fe-history)을
held 검사 후 release했다. 최초 관측 PENDING, dependency 없음, 실행 source
8da621a9e12fd341a4a70fba95531af6fe4fdb80. W20 및 온라인 기록은 아직 미관측이다.
새 context SHA는 실제 B1에서 기록되며 지금 생성 완료로 주장하지 않는다.
SH2 8개 cold 및 FT eval-only, SH4 heldout 2개는 제출 receipt 회수 중이다.
SH1 근거: audits/servers/server1/qwen-baseline-mask-cold-rerun-20261010/submission.json.
직접 수락과 실제 제출/결과 완료는 별개다. 장기 GPU 완료 대기/새 recurring monitor 없음.

[정본 실행 envelope](../../../messages/head/2026-10-10-qwen-baseline-mask-cold-rerun.md) ·
[공통 검산·원 조사 SHA](../../../audits/global/qwen-baseline-mask-cold-rerun-20261010/common-repair.json).
