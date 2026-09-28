# MEMIT history M0

Nonce: ODEEDIT-GH-SH3-MEMIT-HISTORY-FIXED10K-20260928-R1.
추가 ACK: ODEEDIT-GH-SH3-MEMIT-HISTORY-USE-BLUE-20260928-R1.
상태 IMPLEMENTING_NOT_SUBMITTED. GPU model forward 및 Slurm submit 0.

지정 review 전체 FULL_READ, SHA256 6b8a692ed344dc8f3679b8fe0886b018f7cb3eaf386c1bcc532a5c9b0f684508 일치.
원문 authority main0ab6c79f 및 추가 f05082cb. pinned BLUE311b076의
memit.memit_seq_main.apply_memit_seq_to_model을 실제 import/call한다.
파일 SHA f84fcf4b388ff1e5c5c9d520202d926e5b16314b8269063b57d8d25243bc716a.
returned model/cache_c object를 다음 batch에 그대로 연결한다. 별도 writer 재구현0.

|항목|BASE_MEMIT42658|이번 실행|
|---|---|---|
|entrypoint|memit_main.apply_memit_to_model|memit_seq_main.apply_memit_seq_to_model|
|history|없음|H0=0, 계수1, prior-H solve|
|history key|없음|5층 temporary write 완료 뒤 post-key, 층별1회 append|
|layers/분배|L4–8 / 5,4,3,2,1|동일|
|z/FP64solve/FP32 W|원 native|동일 pinned source|
|seed/TF32|20260907/matmul false/cudnn true|명시 동일|
|checkpoint|과거12개|false, RAM continuation only|
|hardware|S4 PRO6000|S3 H200; 수치동등성 미확립|

모델/fixed10k/C0는 S3 로컬 재사용 검산 중. BASE_MEMIT source archive61440B,
lock529205B, B1 context270B 선택수신 후 SHA 일치. Runtime 사본065dccfe는
과거 실행 표cef9e07b와 달라 original runtime이라고 인증하지 않는다.
Native compute_z/ks/seq 및 evaluator/contracts는 과거 lock SHA와 일치.
Archive SHA582a3789, lockf6d1d401, context33cec0ee. 원본 KEEP.
삭제C4 복원0, EN/GSS 재개0, 모델/CP 전송0.

Runner/상태/observer/noCP/launcher 작성, CPU 작은 회귀7개 PASS.
남은 작업: 전체 source/input/token/resource lock, 추가 CPU observer 및 reducer,
freeze, cap1 admission, held exact 검사 및 release. 실제 initial/terminal은 미관측.
예상 GPU시간은 BASE_MEMIT의 과거12.2133GPUh와 history/host 차이를 기준으로
12–48h 계획 범위, wall7d 여유 요청(실측 아님). H 3.828125GiB와 H rollback동량,
C0 3.828125GiB, selected W0/entry 각1.09375GiB, FP64 solve 일시행렬 등을 분리.
Host≤119GiB/1GPU/8CPU, 디스크 신규10GiB+임시2GiB+여유10GiB 계획.
실제 resource plan에서 전체 model/RAM/solve peak를 추가 검산한다.
과거 baseline의 raw-free 수치/동일 evaluator만 재사용; target/update/state는 재사용0.
최종 CPU reducer는 current/at-write/all-seen, active/superseded, first500 W5→W100,
RSPSNS와 TF micro/macro/strict/NLL을 기록하며 외부 baseline raw 미보유를 구분한다.
독립 red agent 미사용; owner CPU/source 검산. NO_BROADCAST_NOT_REQUIRED.
