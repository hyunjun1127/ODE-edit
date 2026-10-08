# server4 실행 연결

이 폴더는 server4가 소유하는 EasyEdit 자산 연결과 runner 경로다. `assignment.json`의 모델·방법을 담당한다.

- `prepare.py`는 기존 자산의 존재/선택적 SHA를 확인한다. GPU runner 완료를 뜻하지 않는다.
- 담당자가 `run.py`, 제출 스크립트와 재개 검증을 이 폴더에 구현한다. 공통 알고리즘·평가기는 `official/` 내부 모듈만 import한다.
- `assets.example.json`을 참고해 `assets.local.json` 또는 ignored local manifest를 만든다. EasyEdit는 자산 경로로만 사용한다.
- 데이터, C0, P, weights, raw 출력은 원래 자산 경로/ignored local에 두고 복사하거나 Git에 넣지 않는다.
- 실행 source는 `main`의 commit과 `official/` tree SHA로 고정한다. 기존 실행을 취소하거나 코드를 바꾸지 않는다.
- 공유 코드 수정이 필요하면 원인을 보고하고 공통 수정으로 통합한다. 서버별 수치 구현 fork를 만들지 않는다.
