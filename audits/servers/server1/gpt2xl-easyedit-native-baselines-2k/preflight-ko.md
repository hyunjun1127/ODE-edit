# Native baseline 사전 점검

권한 nonce는 `USER-GH-SH1-GPT2XL-EASYEDIT-BASELINES-2K-20261007-R1`입니다. 실제 server1/devbox/SH1 session 및 dedicated non-main WT를 `check-session-boundary.sh`로 확인했습니다. root dirty와 과거 registry29e4는 유지하며 공유 identity/config를 변경하지 않았습니다.

최종 CPU gate: **65 tests PASS**, CUDA 미초기화, model load0, 실제 native apply0, 신규 online smoke0입니다. local 봉인 receipt는 `/mnt/raid5/janghj/ODE-edit/local/gpt2xl-native-baselines/20261007-v1/preflight-final-v1/cpu-preflight.json`이며 source/실제 helper 파일 SHA를 포함합니다. 초기 test fixture의 일부 H층 누락과 준비 전 존재하지 않는 disk parent 경로는 CPU 단계에서 확인·수정했고 GPU 계산은 수행하지 않았습니다.

독립 작업자 `gpt2_runtime_preflight`는 native17 source 직접 import/실제 YAML parser·요청 문자열·원 apply 위임·Alpha 첫 reset/5층 history·관측 counter/콜백 격리를 구현·검산했습니다. `gpt2_submission_preflight`는 독립 저장-row reducer와 16개 관측/계수/accounting fixture를 구현했습니다. 후자는 owner 작성 prepare/run/submit을 읽기전용 검토하여 추가 source blocker를 발견하지 않았습니다. 실제 PRE→native apply→POST, MEMIT 무H/Alpha native history, RAM rollback/noCP, release 직전 fresh DAG cap 재검사와 봉인 bytes 재검증을 확인했습니다.

CPU/source 점검을 native GPU 수치 PASS 또는 W&B online PASS로 확대하지 않습니다. native compute_z가 실제 수행하는 earlystop 때문에 총 backward/Adam 횟수는 실측 없으면 NOT_RECORDED입니다. fit 축은 native z 완료 호출 누적이며 optimizer 내부 candidate 횟수를 추정하지 않습니다.

모델/C0/P는 기존 SHA+현재 stat seal을 사용하고 tokenizer/dataset/context/evaluator/runtime/native closure의 필요한 bytes를 fresh 검산했습니다. 5 C0 실제 cache 경로, P 절대경로, native logical model `gpt2-xl`, MEMIT/Alpha context generator AST 동일성과 cold/RNG restore provenance를 CPU에서 결속했습니다. native 입력 target/cache 정합만 변경하고 native 수식/계수/cast/solve 순서는 유지합니다.

최초 제출과 release 직전 exact resource-only DAG에서 기존 ours 두 writer lane 뒤에 baseline을 연결하고 합산 cap2 이하를 확인합니다. 다른 job 취소/자원 변경/과학 결과·로그 조회는 수행하지 않습니다. 새 checkpoint/Z cache0, raw local KEEP, `NO_BROADCAST_NOT_REQUIRED`입니다.
