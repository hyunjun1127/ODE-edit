# CF 표시용 Score 전송 수리

61711 AlphaEdit와 61712 SPHERE는 W0 factual 완료 후 첫 요약 전송에서 실패했다. B0 checkpoint/원 raw 보존, native commit 0. OOM/학습 실패가 아니다.

원 evaluator는 NumPy request/cohort mean 후 around(2), transport는 독립 fsum 원점수에 Python round(2)를 적용했다. 실제 Specificity 88.55499999999999에서 표시값 88.56/88.55로 갈라져 OFFICIAL_DISPLAY_SCORE_MISMATCH가 재현됐다.

수리: 원 evaluator/raw/Score/hparams/solver 불변. SH1 caller가 실제 observation의 strict preference bits를 request-macro NumPy mean/around로 집계해 Efficacy_AlphaEdit_display, Generalization_AlphaEdit_display, Specificity_AlphaEdit_display를 official endpoint scalar로 함께 전송한다. validator는 세 필드의 완전성·2자리 양자화·원점수 대비 nearest-cent(16 ULP 부동소수점 경계만)와 기존 harmonic 허용오차를 확인한다. Legacy companion 없는 payload 검증은 유지한다. whitelist 전체 해제/점수 tolerance 완화/extra model forward 없음.

CPU 83 tests PASS: 새 반올림 경계/오류 거절/원 raw 비변이/fake-SDK worker/4-lane DAG 및 기존 tracking/noqual/submit. 실패한 두 실제 W0 raw도 새 caller→real schema 수락 확인. source157 SHA 검증 PASS, external imports0. 실제 새 GPU/online 성공 증거가 아니다.

취소: 원 submission/launcher SHA/owner/Command/WorkDir/state 재확인 및 exact pending hold 후 61715→61714→61713→61709. 61711/61712는 이미 FAILED라 추가 취소0. 61710 및 zsRE61716..21와 공유 CPU collector61722 KEEP. 기존 collector는 원 CF 실패/취소와 zsRE coverage를 구분한 PARTIAL을 기록하며 새 CF collector와 혼합하지 않는다. 옛 source/raw/CP/online/cost 이력 모두 KEEP.

새 제출 범위: CF W0와 FT/MEMIT/MEMIT_FE/ALPHAEDIT/SPHERE의 cold2k 5chain 및 GPU0 collector. qualification 및 GPU resume 동등성은 NOT_RUN_USER_DISABLED. W20 FLUCON deferred 및 최신 checkpoint/W20 보존 계약 유지. zsRE/native math/다른 job 변경0. 각 유지 zsRE lane 말단별 resource dependency로 cap4를 강제한다. 실제 제출 정보는 submission receipt로 후속 기록한다.

Same-host 원 raw는 NO_BROADCAST_NOT_REQUIRED. Git에는 source/compact receipt만 게시한다. 독립 subagent review 미사용, owner source/CPU 검산 수행. 자동재시도/장기 monitor 없음.

## 실제 등록 및 인계

실행 source `34e4d52d821113d56ee6c59b71d553dd01768e73`, official tree `db0354edebde9517c200140dbaa717605c4d527e`. 2026-10-09 04:56:54 KST held 등록 및 owner/source/argv/input/resource/dependency 검산 후 전량 release. 초기 상태 모두 PENDING/dependency. 별도 GPU 검증과 새 W&B startup은 미관측이며 CPU PASS로 대체하지 않는다.

| 역할 | 새 job | dependency |
| --- | --- | --- |
| CF W0 | 61768 | afterany:61718 |
| AlphaEdit | 61769 | afterany:61719 |
| SPHERE | 61770 | afterany:61720 |
| FT | 61771 | afterok:61768, afterany:61721 |
| MEMIT | 61772 | afterok:61768 |
| MEMIT-FE | 61773 | afterok:61768, afterany:61769 |
| CPU collector | 61774 | afterany:61768:61769:61770:61771:61772:61773 |

cap4: 보존된 zsRE 4lane과 신규 CF를 함께 검산했다. 취소된 옛 CF ID를 새 resource edge에 사용하지 않았다. afterok는 필수 actual W0 입력 gate이며 성능 gate가 아니다. zsRE61716..19는 04:57:44 KST RUNNING,61720/21 PENDING으로 보존했다.

실제 lock SHA `6f820fd90fcfbbf3469980ee298a13525b3bc226f6455613699f08ce06682491`. 원본 receipt/log는 `/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/cf-display-score-repair-r1/registration-r1/` 아래이며 source tree는 frozen `source/`, 로그는 `logs/`이다. compact source/config/job receipt는 같은 task audit `submission.json`이다. GPU/W20 완료를 기다리지 않고 bounded handoff한다.
