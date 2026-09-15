# Technical r1 telemetry failure / 최소 수리

Job48294: FAILED1:0, allocated31GPU-seconds, MaxRSS14504360K. Source1fb000484c75f14b5171362c43bab64ab128d209, input.lock77ec0350d365d65a22aa37cc75857f2bc7223f3a964a0d17c1919b5894bf0678. 원본 output/log/source는 보존한다.

4개 pretrained shard 로드 후 runtime metadata를 만드는 동안 `PreTrainedTokenizerFast.add_bos_token` 속성 직접 접근이 AttributeError를 일으켰다. Oracle/actual parity/편집/확정 checkpoint/scientific batch는 0이다. 성능 또는 derivative 실패가 아니다.

새 clean tech-r2는 optional attribute를 getattr로 관측하고 tokenizer class, 실제 고정 probe의 default/no-special IDs를 기록한다. 기존 writer의 add_bos_token=False 할당, 입력 구성, native source, model dtype/backend, 목적식, 수치 tolerance 정책은 변경하지 않는다. Fast tokenizer의 편의 attribute와 실제 backend special-token 처리를 동일시하지 않는다.

Control-only release 검사도 기존 local r1 스크립트를 보존하고 r2로 수정했다. Slurm `squeue -w server2`는 미배정 held job을 반환하지 않아 release 전 assert에서 멈췄으며, 전체 own-user project pending/active 확인으로 보완 후 exact48294만 release했다. 과학 source 변경과 무관하다.
