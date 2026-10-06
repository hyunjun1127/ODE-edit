# GPT2-XL W0-only 제출 전 점검

Nonce `USER-GH-SH1-SH2-BASEMODEL-W0-20261007-R1-SERVER1`의 명시 사용자 승인 범위다. 실제 root는 `/mnt/raid5/janghj/ODE-edit`, SH1 session은 `01a04939-f93a-7b50-bca0-65438eab2062`다. registry29e4는 역사 경로이고 전용 non-main WT의 ignored boundary만 결속했다. 원 root dirty, 모델, 기존 job/source/raw와 공통 설정은 보존했다.

한 fresh cold GPT2-XL W0의 first2000을 실제로 평가한다. R/P/N 분모는 2000/4000/20000, prompt-pair26000과 candidate-row52000을 구분한다. targetfit/edit/solve/history/C0/P/edited checkpoint는 없다. 기존 W0 raw는 CPU identity/추후 paired 비교용이지 신규 관측을 대체하지 않는다.

기존 canonical GPT2 scorer의 MB2 adjacent new/true, left padding, GPT2 default learned position, transformer final LN1회 및 tied head를 재사용한다. FP32/eager/eval/no-grad/autocastoff/TF32off/use_cachefalse이며 pointer/version/hook 및 선택층 hash 비변이 검사를 실제 관측 내부에서 수행한다. 전체 모델의 byte 비변이 인증을 주장하지 않는다.

준비 config SHA는 `6700bdcaf5969abbc26cfd1805d1beb4e0453b9779fc47b98d0e0db42fed7135`다. 원 model payload exact seal과 unchanged stat를 재사용했고 tokenizer/config/source/runtime/ordered first2k/evaluation26000 token-row를 fresh 결속했다. GPU/model load, C0/P load와 새 W&B smoke는 하지 않았다. CPU 검사 결과는 모델 PASS가 아니다.

제출 요청은 GPU1/CPU8/65536MiB/4h, collector GPU0/CPU8/24576MiB/2h다. 실제 Slurm partition/QoS/memory/disk 및 held owner/fullargv/source/script/dependency를 확인한다. W0 GPU는 dependency=null의 USER_SCOPED_CAP_EXCEPTION, collector는 afterany 실제 W0 ID다. 기존 method cap2는 그대로이고 global cap helper PASS를 꾸미지 않는다. 물리 부족은 정상 PENDING으로 넘기며 타 job을 선점하지 않는다.

W&B는 `wkdguswns2256` / `layer allocation`, writer=none, scientific, actual job/name/config/runID/URL을 결속한다. `W0_first2000`만 edits0에 기록하며 fit/current/post/milestone을 만들지 않는다. SDK 접수와 remote readback을 구분한다. 실제 online 상태는 job startup/finish에서만 확인하며 제출 전에는 미관측이다.

source/compact receipt만 Git에 게시한다. raw/model/dataset/token/credential/fullstdout은 Git/W&B에 넣지 않는다. NoCP와 exact_resume=NOT_AVAILABLE, NO_BROADCAST_NOT_REQUIRED를 유지한다. 제출 후 한정 registration snapshot에서 인계하며 장기 polling/retry/heartbeat는 하지 않는다.
