# Server2 CF official 6-arm 재준비 — 2026-10-09

## 완료한 중단

사용자가 실행/대기 전부 중단을 지시하고 재등록 arm을 **FT/MEMIT/AlphaEdit/BLUE/FE/SPHERE**로 확인했다.
Server2 exact owner/node/Command/WorkDir/현재 상태를 확인한 12개를 후속부터 취소했다.

- 실행 중 61534 MEMIT, 61535 AlphaEdit: CANCELLED. 각각 01:50:16, 04:09:32 GPU1 할당.
- 대기 61536/61537/61540, 61619/61624/61625/61626/61627/61628/61629: CANCELLED, 할당0.
- 후속 Server2 queue는 빈 상태를 관측했다. Server4 61418/61421/61618은 그대로 유지했다.
- 옛 source/archive/raw/log/기존 checkpoint를 삭제하거나 수정하지 않았다. 중단한 noCP RAM trajectory를 재개 가능하다고 주장하지 않는다.

## 새 caller 구현

전용 branch `codex/server2-cf-six-checkpoint-only-20261009`에서 own runner만 변경했다.
직접 USER cap3를 exact 새 profile에 명시했으며 root 및 전용 WT의 ignored local cap도 3이다.
다른 서버/공통 cap파일/공통 알고리즘/공통 logger는 수정하지 않았다.

새 `cf_checkpoint` stage는 여섯 GPU pipeline과 GPU0 collector를 등록하도록 구현했다.
첫 FT job이 shared factual W0와 독립 native scorer 검증을 한 번 수행한다. 각 arm은 기존 승인된 실제
연속B3 대 durableB2→B3 검증 후 별도 fresh model process로 cold first2000 BS100×20을 실행한다.
첫 FT 종료 후 최대 3개 lane으로 배치한다. 과학 quality gate/주기 polling/자동 retry는 없다.
기술 proof 미관측은 PASS로 간주하지 않는다.

FLU/CON은 W0와 W20 모두 생성하지 않는다. factual/native/RPN 정의는 유지한다.
W0 및 각 완료 batch마다 official latest1 checkpoint, 마지막 W20을 보존한다.
재구성할 native weights/H/RNG/context/cursor와 원 base/revision/token/input/source를 결속한다.
나중 2K checkpoint 평가가 명시적으로 예정되어 있으므로 consumer 완료 전 archive/삭제는 금지다.
평가 미측정은 `DEFERRED_NOT_MEASURED`이며 점수0/전체 평가완료가 아니다.

## 실제 제출은 아직 안 됨

현재 공통 `official.tracking`은 CF에서 `W0_AND_W20_FIRST2000`을 강제한다.
caller의 정직한 `DEFERRED_CHECKPOINT_EVALUATION` config는 CPU 재현에서
`ValueError: GENERATION_SCHEDULE`로 차단되었다. 켜진 일정으로 위장하지 않았다.
공통 소유권을 지켜 logger 복제/공통 코드 임의 변경 없이 좁은 API 보완을 요청했다.
등록된 GH의 server1 app-server 응답은 `COMMUNICATION_HOLD / UNRELATED_ACTIVE_NO_STEER`였으며
수신 ACK나 공통 입력 도착으로 보고하지 않는다.

남은 일은 공통 deferred CF schedule의 게시 API 결속, 새 source/main exact freeze,
현물 source/asset/manifest 재결속, 실제 held 등록/검사/release다. 새 job ID 없음.
실제 GPU/native/resume/online PASS 또는 W20 checkpoint 생성은 아직 관측되지 않았다.
별도 독립 reviewer는 이번 변경에 사용하지 않았다. Server2 CPU unittest **141 PASS**.
새 profile 4개 검사는 20-commit/final checkpoint CPU fixture, 생성 호출0, cap3 DAG와
truthful deferred config를 확인한다. 기존 qualification-input의 AST/import 보존 검사도
완화하지 않고 통과했다. CPU 테스트는 실제 GPU 증거가 아니다.
