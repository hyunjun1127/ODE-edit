# GPT-J PRICE 6-arm — 등록·release

사용자가 W&B 비교 요건 반영 후 제출을 승인했다. MEMIT CAP075/CAP100/FREE100은 **60112/60113/60114**, AlphaEdit는 **60115/60116/60117**, GPU0 collector는 **60118**이다. 독립 cold first2000 BS100×20의 실제 held 검사/release를 완료했다. 초기 snapshot 모두 PENDING이며 actual B1/W20과 새 온라인 W&B identity는 NOT_OBSERVED다. [현재 상세 제출 보고](../price-model-runs-tracking/report-ko.md)가 최신 정본이다.

## 구현과 확인

- EasyEdit GPT-J 모델·native stats·projector·hparams를 결속하고 native parallel attention/MLP·fc_out bias·readout27에 ours hook을 적용했다. 기존 과학 계수와 cap 설정은 유지한다.
- native context 5개는 첫 실제 arm의 입력 준비에서 한 번 생성·고정한다. 별도 fit/pilot 없음. GPT-J exact W0가 없어 첫 arm에서 first2000을 관측하고 나머지는 identity가 일치할 때만 재사용한다.
- 기존 60001의 봉인 B4/B5/W0 raw CPU 검산에서 current 분모 R100/P200/N1000, B5 누적 R500/P1000/N5000, W0 R2000/P4000/N20000을 확인했다. 원 raw/stored aggregate/payload 일치. B4에는 all-seen key 없음. 기존 job이나 run 변경 없음.
- 새 caller는 scientific run에 current/pre·current/post·실측 all_seen/post·W0_first2000 및 W0 같은-cohort N 비교를 직접 기록한다. R/P/N 경로 분리, 백분율, harmonic, true-new margin, edits 축과 pre-state 위치를 구분한다. 중복 companion daemon을 자동 실행하지 않는다.
- SH1 job-identity helper `bc63425e` 채택. 실제 job 이름/Config 검증을 유지한다. immutable startup identity와 transport 상태를 분리하고 finish 후 bounded readback을 준비했다. 실온라인 검증은 NOT_OBSERVED.
- CPU fake-SDK/metadata 검사 25개 중 24 PASS, 1 SDK환경 검사 SKIP 후 task-local readback 검사 1개 추가 PASS. 소스 31개 AST/import/config 검사 통과. 이는 과학 toy/GPU/model PASS가 아니다. owner audit이며 별도 reviewer 없음.

## 제출 결속

최신 사용자 권한이 중앙 helper 표식 대기 조건을 해제했다. 동시에 실제 게시된 SH1 helper와 caller를 CPU 원자료/fakeSDK 검산하고 원자적으로 봉인했다. SH4가 공통 helper를 수정하지 않았다. helper receipt의 SDK 접수와 실제 remote readback은 구분한다.

실행 source `5226337121fd2c90c595f26297c9927400f8f0af`, config SHA `7331db77a872095f49472345de69f94ed8ff9d790796823b9d255ea3d563c6dc`다. fresh cap3·task2·storage/resource admission 후 여섯 GPU job과 GPU0 collector를 held 검사/release했다. MEMIT_CAP075가 native 입력을 준비하고 MEMIT 및 Alpha 각각의 자원 lane은 afterany로 이어진다. Llama 자원 폭1과 GPT-J 폭2의 합은 최대3이다. 과거 임시 freeze는 stale evaluator SHA로 **Slurm 호출 전** 실패했으며 원본은 보존했다. 최신 source/config/lock/archive는 새 attempt로 봉인되어 immutable이다.

noCP/exact resume NOT_AVAILABLE. 기존 Qwen STOP·Llama job 보존. 원자료/credential/model/SDK spool Git 업로드 없음. NO_BROADCAST_NOT_REQUIRED: 소형 source/receipt만 공유한다.
