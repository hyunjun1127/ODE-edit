# Native v3.1 GH 독립 검토 — main 게시 인덱스

사용자 요청에 따라 기존 코드·결과·분석 package에 GH 독립 검토를 추가한다. 기존 실험 package는 수정하거나 재생성하지 않는다. 검토서의 Git commit/push=0은 검토 당시의 작업 경계를 뜻하며, 이 게시 작업은 이후의 별도 사용자 승인이다.

- `gh-independent-review-ko.md`: 독립 코드·수학·산출물 검토와 남은 계측 한계.
- `verification.json`: 원자료 495개 SHA, primary/audit 지표, source·receipt·예산 재검산.
- `refinement-verification.json`: 실제 CPU tensor endpoint 거리 8쌍 재검산.
- `verify.py`: 검토 당시의 source/worktree와 raw 경로를 고정한 read-only 검증 코드. Python 표준 라이브러리와 Git/Slurm 조회만 사용하며 모델·GPU를 실행하지 않는다. 범용 실행기가 아니므로 원본 review worktree와 raw를 보존한 server1에서 사용한다.
- `publication.json`: local 검토 산출물과 게시본의 byte-exact SHA 결속.

기존 코드 경로는 `project/run_scripts/native_response_ode_v31/`, 기존 결과·분석 package는 `experiment-reports/servers/server1/native-response-v31-b10-warm-pilot-2026-09-06-v1/`다. 둘 다 기준 main `347a892a910317b606443cf53d9155ae6a460831`에 이미 포함되어 있다. 새로운 게시본에 raw prompt/logit/weight/model/cache/checkpoint/log는 포함하지 않는다.

결론: 핵심 구현과 기록된 결과는 유효하며, 모든 측정 계약 완료 또는 retention 우월성을 승인한 것은 아니다. Scientific promotion=false. 보완 지적 자체를 source 수정이나 추가 GPU 실행 승인으로 해석하지 않는다.
