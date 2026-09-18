# 2026-09-18 작업 공간 전체 main 통합 기록

현재 대화의 사용자 지시 “전부 push해봐”에 따라 연구 설계·분석·검증·GH 지시문을 최신 main 기반의 별도 integration worktree에서 통합했다. 이 기록은 통합 내용과 검증 범위를 설명하며 Git remote 갱신 성공 여부는 실제 push 이후 별도 확인한다.

- 비교 기준 main: `13fae514d476a1c5a15b34f3cf3964c105ba4eb9`.
- 확인한 작업 공간 파일: 294개, 원본 총22,320,841 bytes. 데이터셋·checkpoint·full log는 포함하지 않는다.
- 94개는 main의 기존 파일과 동일했다. 185개는 main에 없었고, 15개는 main과 내용이 달랐다.
- 최신 single-layer-zflow runtime README는 main의 내용을 유지했다. 더 오래된 로컬 CPU README는 `historical/single-layer-zflow-cpu-reference-readme.md`에 원본 bytes로 보존했다.
- Server4 소유 디렉터리에 있던 미추적 PNG2개는 같은 bytes로 `experiment-reports/global/layer-realization-debt-figures-2026-09-18-v1/`에 넣었다. `check-agent-access.sh`가 지적한 경로 소유권을 파일 이동으로 해결했으며 guard를 변경하지 않았다.
- 루트 및 proposal README는 기존 연구 이력을 보존하며 최신 BPCW-v2와 GH 지시문을 안내하도록 통합했다.
- 본래 작업 공간 파일을 삭제하거나 원격 이력을 강제로 덮어쓰지 않았다.

## 검증

BPCW CPU algebra12개 및 L4 repair 계약/CPU scale41개가 통과했다. JSON77개·JSONL2개·CSV88개·Python AST43개를 검사했고 형식 오류가 없었다. 일반적인 credential signature 검출은0건이었다. 이 검증은 전체 과거 연구의 재실험이나 실제 LLM 수치/성능 검증이 아니다. 신규 모델/GPU 실행은0건이다.

모든 원본 source path/SHA256과 실제 반영 경로/SHA256을 [manifest.json](manifest.json)에 보존했다. 현재 연구자는 [BPCW 방법](../../../plans/global/2026-09-18-base-choice-constrained-write-v2.md), [계약](../../../plans/global/2026-09-18-base-choice-constrained-write-contract-v2.json), [GH 지시문](../../../project/proposals/2026-09-18-base-choice-constrained-write-gh-instruction-v2.md)을 우선 읽는다.
