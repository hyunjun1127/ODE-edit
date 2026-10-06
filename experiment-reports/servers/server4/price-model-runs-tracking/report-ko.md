# Llama/GPT-J PRICE 제출 및 W&B 정합

현재 단계는 소스·CPU 검산 완료, **새 작업 미제출**이다. 60001은 사용자 명시 KEEP으로 source/config/W&B/RAM 진행을 보존한다. 기존 60002/60003/60011/60012/60013 및 collector 60007/60017만 정확 owner/source/argv와 elapsed=0, StartTime 없음, 할당 없음인 PENDING으로 확인하여 downstream 먼저 hold했다. 아직 취소를 완료했다는 보고가 아니다.

새 Llama lane은 60001 → MEMIT CAP100 → FREE100 → Alpha CAP075 → CAP100 → FREE100의 afterany 자원 순서다. GPT-J는 MEMIT/Alpha × CAP075/CAP100/FREE100 여섯 cold first2000 BS100×20이다. 전체 서버 cap은 최신 사용자 3과 현재 더 엄격한 정책의 최솟값이며, Qwen 재개·baseline/pilot/추가 fit은 없다. 60001은 취소·hold·hotpatch·재실행·중복 제출하지 않는다.

W&B는 current/pre와 current/post를 항상 R100/P200/N1000으로 유지하고, all_seen/post는 실제 W5/10/15/20만 기록한다. 기존 B4/B5/W0 원 row로 재집계한 CPU 검산과 fake SDK 검사에서 pct/nats, N desired=true, harmonic, edits/state/fit 단조 축, 실제 job 식별자의 전달, immutable identity 및 privacy가 통과했다. fake SDK는 실제 온라인/GPU PASS가 아니다. 새 job의 startup/finish에서 bounded readback을 별도로 남긴다. 기존 60001에 새 W&B writer나 backfill을 붙이지 않는다.

원 Llama config/input/runtime/model/C0/projector 및 정확 W0 재사용 자격을 유지한다. GPT-J는 기존 준비 자산과 native adapter를 결속하며 첫 승인 main에서 native context 입력을 한 번 준비한다. 실행 중 데이터를 저장한 checkpoint에서 재개하지 않으며 모든 새 arm은 cold W0/H0다.

Collector는 새 MEMIT 두 cell과 보존한 60001을 **각자의 실행 source/config/lock**으로 검산한다. Alpha는 새 Llama 세 cell만 집계한다. 제외 Qwen을 실패로 세지 않는다. 기존 nested native NLL의 context→owner 집계 결함을 CPU reducer에서 수정했다. 대응하는 새 MEMIT/Alpha 결과 비교는 정확 endpoint 검산 전 PENDING_COMPARISON이다.

소스 검토는 owner와 별도 위임 worker/explorer의 한정 검토이며 독립적인 전체 과학 감사나 새 실제 모델 검증을 주장하지 않는다. 모든 원 archive/raw 및 실패·취소 이력은 KEEP이다. Git에는 소형 source/report/manifest만 게시한다. `NO_BROADCAST_NOT_REQUIRED`: 대형 raw와 W&B credential/spool을 옮길 필요가 없다.

최종 실제 job IDs, held 검사, release 및 1회 초기 snapshot은 제출 후 이 보고서에 별도 기록한다. 장기 GPU 대기·recurring monitoring·heartbeat·자동 Slurm retry는 없다.
