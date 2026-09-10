# L4 progress-preserving barrier 최소 사전 확인

instruction_id: ODEEDIT-S06-BLUE-L4-PROGRESS-PRESERVING-BARRIER-SH2-V1

- 설계 425행/37031 bytes 전체 읽음. SHA256 `30be08eccc4b74b34acfeb9e3deb5cbffc9e6c4a02768e6f69fddf5593d99a8e`.
- 기준 main `285d464361c82d033cbaf88c177fd6c5af8f83df`, tree `2b07c98890a2d8d769323b1bb6e2d99b662b69ed`. 원 root HEAD 979b4606 및 untracked agents/server2 보존.
- 독립 branch `codex/server2-blue-l4-progress-barrier-v1`; 기존 ABC package는 d2c80801과 byte 변경 없음. 기존 optimizer/soft_filter는 호출하지 않음.
- 승인된 server1 입력 89파일/13,461,868,847 bytes 전체 SHA 수신 검증 후 imports 봉인. 원본 변경/덮어쓰기 0.
- P/C0 원본 SHA 일치. 정확한 WN/We/M/Ub/K·패널·native target/checkpoint 연결 CPU 검증 통과.
- 로컬 Transformers4.44.2/tokenizers0.19.1 runtime .py/.so 1735/1735가 server1 원본과 일치. 41개 타입 stub/다른 모델의 kernel source는 로컬 closure에 없으며 eager Llama에서 사용하지 않음. torch/CUDA/hardware parity는 별도 runtime receipt에 실제 관측하며 동일하다고 가정하지 않음.
- 합성 수식·rank/zero·risk expansion·N16 동일 field·단위변환·group mapping·요청별 backward 관측 fixture: 8 tests PASS. 관측 hook은 기존 312 edit backward/50 essence backward를 그대로 쓰며 새 backward를 추가하지 않음.
- GPU 최초 공통 상태에서 functional/actual FP32 weight logits exact 비교 예정. 본 문서는 GPU gate 통과를 주장하지 않음.
- resource: server2 cap2, 1 GPU/8 CPU/60416M, export NONE; 숫자 GPU-hour 상한 없음. 신규 제출마다 allocation 재확인.
- 중지된 다른 task의 monitoring/실행/결과 변경 0. 성능/슬랙/risk는 scientific outcome으로 보존. scientific_promotion=false.

로컬 evidence root: `/mnt/raid5/janghj/ODE-edit/local/blue-l4-progress-barrier/attempt-v1/`.
`input.lock.json` SHA256 `97250e1466a856844bca76394dd8b5b0cf9b27bd8eeb7f70d53dee622df763e6`.
`execution.lock.json`은 clean source commit 뒤 create-once 생성하며 실행에 직접 결속한다.
