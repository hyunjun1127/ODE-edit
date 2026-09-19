# SH2 checkpoint mechanism M0

nonce=ODEEDIT-GH-SH2-CHECKPOINT-MECHANISM-20260920-R1
cap_override_nonce=ODEEDIT-GH-SH2-CHECKPOINT-MECHANISM-CAP2-20260920-R1

서버2/session01a0493a-074c-7f91-9a13-769116326fef/CWD/origin 확인. 기존 root HEAD730d8cc와 untracked agents/server2/ 보존. 별도 codex/server2-checkpoint-mechanism-audit-20260920-v1 worktree에서 authority4440e111을 ff-only 수신했다. 원 지시문 SHA f207a7c4c28ea5b5ce4a2dc29f7b1f2950979747816f9ca105afbc3b2c66dad1 및 10개 정본 SHA/size 일치, 전체 읽음. PROTOCOL/registry/session/memory/cap 정책도 읽었다.

2,304 입력 members 중 2,084 기존 정확 복사본 재사용, 220files/450,763,288B만 승인된 source에서 선택 수신하여 양측 full SHA/size 검산. SOURCE_KEEP, W/M 재전송0. 입력 source-map SHA202a1bf65ddce6fb031294254bab7c69f0761a2fbfed3e2e9c0130fedc1213b0. 경로 local/checkpoint-mechanism-audit/20260920-v1/attempt-v1/inputs/. 별도 source assembly 34files와 Torch2.9.1+cu128/Transformers4.44.2 original import 확인. 이는 actual Llama PASS가 아니다.

CPU 실제 W0 hash 및 기존12CP full SHA/size/W/M·commit 결속 PASS. 13상태/11구간 공통256-vector sketch 완료(128.346s). 보존평가100current/12seen의 독립 scalar 검산 진행, B1 원래100/190/867 확인. 실제 모델 gate는 아직 미실행.

cap2에서 공통 B1 full-forward/prefix/원 평가 gate까지 직렬. 이후 history/operator의 독립 single-writer partition과 activation/demand 분석을 두1GPU lane으로 배분한다. 각1GPU/8CPU/60416M/exportNONE/Requeue0, 다른 allocation+admitted pending 제출직전 재계수. 중복 작업으로 slot을 채우지 않는다. 실제 gate 비용 후 확장 wall/storage를 산정한다. 새 z/편집/history append/CP 저장0. 기존 다른 task pause 유지.

root session helper PASS. 새 worktree helper는 local config 부재로 거부했고 전역 변경하지 않았다. task envelope의 전용 worktree 권한 + canonical root helper + exact frozen-source lock으로 경계를 기록한다. 전역 Slurm 정적 감사의 과거 server4 6파일 초과는 미수정; 신규 launcher 개별 감사로 구분한다.
