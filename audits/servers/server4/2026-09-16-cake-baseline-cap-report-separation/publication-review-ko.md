# CAKE/baseline 및 alpha-cap 독립 정본 v2 — publication 검토

Instruction `ODEEDIT-S06-CAKE-BASELINE-CAP-REPORT-SEPARATION-SH4-V1`.
문서 기준 main `4d9bd91006df5b6966f36e080e89b0e7ba1b4da7`, CPU publication source
`78c3097aa4da6079ee1891678b64edad97f4f204`. 실제 실험 source는 각 v1 provenance를
재사용하며 이 publication source를 실행 source로 부르지 않는다.

## 실제 구성

- CAKE v2는 W0, BASE_ALPHAEDIT_NATIVE, BASE_MEMIT_NATIVE, 각 family의 BLUE
  L4+L8 및 physical L4/L5/L6/L7/L8-only와 CAKE만 포함한다. 두 family에 반복
  표시한 CAKE는 같은 실행이며 분모/비용을 두 번 합산하지 않는다.
- Alpha-cap v2는 CAP1 reuse/CAP10/CAP100/NORM_ONLY의 first1000만 중심으로
  정리하고 같은1k baseline 참고는 유지한다. CAKE 수치/그림은 포함하지 않는다.
- 새 종합 v2 README는 두 정본과 역사 v1로 가는 링크 목록이다. 혼합 과학표,
  그림, 합산비용은 없다. Server4 README에는 두 정본을 기본 링크로 두고
  다른 기존 보고 디렉터리 링크를 보존했다.

## 확인 결과와 수준

- 집중 CPU publication 테스트 9개 PASS. Scientific model/GPU 검사 아님.
- 74개 원출처→새 view/copy mapping의 SHA/bytes를 확인했다. Exact copy는
  원 CRLF 포함 byte 그대로이며 label-only CSV는 수치 문자열/분모/빈값 불변이다.
- Alpha-cap 기존 표의 실제 HTML cell 전부 동일. CAKE 기존 표는 명시적
  native/BLUE-style label 변환 외 동일하고 W0 기존 관측표만 추가했다.
- markdown-it-py 3.0.0 GFM HTML DOM 검사: CAKE 9표/92행/22링크,
  cap 19표/170행/26링크, 새 목록 0표/3링크, Server4 목록 0표/50링크.
  HTML cell/행열/링크 검사이며 browser pixel render 검증으로 부르지 않는다.
- CAKE/baseline 비교와 누적곡선 PNG 2개를 직접 코드 생성하고 별도 output에서
  byte 재현했다. 실제 이미지에서도 family label과 범례를 확인했다.
  기존 CAKE 그림 2개, cap 그림 6개는 scope가 일치해 exact bytes를 재사용했다.
- 세 새 package의 source/member/check/receipt 121개 참조 SHA/size 일치.
  총 88개 package 파일은 Markdown/CSV/JSON/PNG뿐이다. 기존 raw tensor,
  checkpoint, prompt, teacher, full stdout payload를 추가하지 않았다.
- 5개 보호된 v1/baseline package Git tree는 정본 main307ba7ae와 동일하다.
  새 package 외 원 보고/runtime/source/raw 수정은 없다.

수치 validation은 변경하지 않았다. Alpha-cap의 SKIPPED_USER_DIRECTED /
NOT_ESTABLISHED와 CAKE no-W/M-checkpoint·continuation 미검증을 그대로 유지했다.
새 scheduler query/GPU/model/forward/evaluator/teacher/FD/ULP/rsync/delete는 0이다.

## 제한과 provenance

이번에는 완료리뷰 v1의 검산 결과를 재사용했다. 새 raw NLL 재집계·full tensor
rehash·모델 재평가를 하지 않았고 기존 검증을 신규 독립 raw 감사로 표시하지 않았다.
이 검토는 SH4 자체 CPU publication 검사이며 별도 red/GH 중복 감사가 아니다.
원래 report hash/경로와 v2 mapping, 새 그림 입력/코드/출력 hash는 각 manifest에 있다.
초기 build의 범위 문자열 검사 실패와 보존된 새 partial 출력은 source README에
기록했다. 기존 scientific 실패나 metric 변경이 아니다.

Family rooted receipt의 `PENDING_MAIN_PUBLICATION`은 seal 당시 상태로 보존한다.
최종 non-force main 게시 HEAD/tree는 task status와 별도 publication receipt에 기록한다.
