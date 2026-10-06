# GPT2-XL cold W0-only 평가

이 문서는 제출 전 source 봉인 시점 기록이다. 실제 등록/held 검사/release 및 초기 상태는 제출 후 같은 task의 별도 audit/status와 문서 갱신으로 공개한다. 현재 모델 성능, 완료 또는 online 검증을 주장하지 않는다.

first2000 cold W0의 R2000/P4000/N20000을 새로 관측한다. 편집·fit·solve·H append·C0/P 준비·checkpoint 저장은 없다. 기존 ours/native baseline은 그대로 유지한다. W0-only 한 GPU에 한해 명시 사용자 cap 예외를 적용하되 원 method cap2 및 물리 scheduler 제약은 유지한다.

준비 및 평가 정의는 [제출 전 점검](../../../../audits/servers/server1/base-model-gpt2xl-w0/preflight-ko.md)에 기록했다. CPU 입력/스칼라 검사는 actual GPU 모델 검증과 구분한다. 실제 W&B는 신규 job의 startup/finish에 확인하며 raw local KEEP/noCP/원 소스 보존을 유지한다.
