# MEMIT history SH3 인계 검토

사용자 지정 review 전체, 원 BASE_MEMIT local audit의 method/runtime/설정표,
고정 BLUE311b076의 공개 MEMIT_seq writer, fixed10k 및 최신 PROTOCOL을 확인했다.
Git main389d1003 기반 clean GH worktree에서 기존 user root dirty는 보존했다.

## 승인 경계

하나의 blue=false history baseline, L4–L8, freshW0/H0, fixed10k B100×100,
server3 cap1, noCP. 추가 arm·history 계수 sweep·z-hook 도입0.
실험적 변화는 history이며 hardware 비교 한계와 numerical certification을
구분한다. 과거 MEMIT의 no-history를 구현버그/잘못된 결과로 소급하지 않는다.
검토문을 원 SHA 그대로 게시했다. 문서의 4-arm 제안은 이번 승인범위가 아니다.

## History 구현상 주의

cache_c wrapper 전달/반환, prior-H solve, 5층 전체 temporary native write 이후
post-key 재계산 누적을 고정했다. Solve에 쓰인 pre-layer K와 history용 post-key는
동일하다고 가정하지 않는다. Static C0 또는 target cache를 history로 세지 않는다.
CPU/실제 첫batch state/counter는 구조검사이며 큰 numerical GPU gate는 추가하지 않는다.

## Local audit 사본 provenance

`baseline-source/` 세 파일은 GH local review의 해당 파일을 byte-identical 복사한
읽기 전용 참조다. 원격 frozen runtime을 이번 GH가 새로 회수한 것은 아니다.
특히 이 사본 runtime SHA065dccfe…는 과거 compatibility CSV의 실행 runtime
SHAcef9e07b…와 **다르다**. 따라서 이 사본을 original execution bytes라고 인증하지
않는다. SH3는 새 실행에 연결할 원 frozen source/config/import를 직접 결속하고,
이 사본/표/새 adapter의 차이를 남겨야 한다. 임의로 SHA를 일치 처리하지 않는다.
별도 새 GPU/model/readiness 재검산·scheduler/Slurm write는 GH가 하지 않았다.

## 자산과 상태

SH3 등록세션 direct inspect는 idle/CWD 일치, app read의 최신 사용자는 이전 GSS
두 job 취소 및 C4 generated reference 삭제였다. 이번 baseline은 C4 teacher가
필요 없으므로 그 삭제를 되돌리지 않는다. 현재 자원은 SH3 제출 직전 확인 대상.
기존 readiness의 model/data/C0를 우선 사용하고 noCP와 cap1을 다시 잠근다.
원격 모델/CP 대용량 재전송 및 기존 파일 삭제 승인은 없다.

## 검토 수준

GH owner source/계약 검토이며 독립 red agent 미사용. 실제 GPU/성능 PASS 아님.
경로/JSON/원문 SHA/사본 SHA/단일arm/resource/noCP 규칙을 게시 전 CPU 검사한다.
런타임 preflight와 actual counters는 SH3 책임이다. Generic path helper 제한은
명시 scope와 별도로 기록하며 허위 PASS 또는 shared policy 변경으로 우회하지 않는다.

게시 CPU 검사에서 review SHA, 세 local audit 사본 byte identity, JSON cap/arm/
분모/noCP 정합성은 확인했다. `git diff --cached --check`는 method.py 및
observation.py의 원래 EOF 추가 빈 줄 두 건을 표시했다. 원 source 사본의 SHA를
유지하기 위한 한정 형식 예외로 보존했으며 전체 whitespace PASS로 기록하지 않는다.
Agent access 검사는 PASS. Authority0ab6c79f nonforce main/remote 일치 확인.
