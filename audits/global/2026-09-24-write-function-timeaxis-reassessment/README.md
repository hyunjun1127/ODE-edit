**과거 update의 시간축에 초점을 맞춘 재검토 — 2026-09-24**

사용자의 질문은 sequential edit이 누적될 때 **과거 사실의 유지**와 **그 사실의 기록에 참여한 과거 update의 기능 유지**가 같은지다. 최신 지시에 따라 allocation 및 단층·다층 비교는 실험 제안에서 제외했다. 대상은 첨부 설계의 기존 `BASE_ALPHAEDIT`와 `BASE_MEMIT` 실행본이다.

**읽을 문서**

2026-09-24 게시 추가: [review §1.1](review-ko.md#11-완료-e3에서-현재-실험의-동기로-남길-부분)에 완료 E3의 배경 의존적 효과, 평균 상쇄, 반대 결과와 현재 실험의 연결을 추가했다. [봉인 설계 및 게시 범위](../../../plans/global/2026-09-24-historical-update-timeaxis-v1/PUBLICATION.md)를 함께 읽는다. 아래 원문 독해·실행 상태 설명은 최초 검토 당시 이력이며, 이번 추가는 원 저장 보고/CSV의 GH 해석이다.

| 파일 | 내용 |
|---|---|
| [연구 방향 재검토](review-ko.md) | 이전 프레이밍의 문제, 제공 리뷰의 타당한 점과 과도한 해석, 신규성과 연구 가치 |
| [수정 측정안](measurement-proposal-ko.md) | 전체 historical update의 제거 효과, 사실과 기여의 분해, cohort×time, 후속 구간 귀속, 단계 의존성 |
| [추가 논문별 원문 검토](paper-notes-ko.md) | 6편의 설정·실험·부록·한계와 이번 연구의 차별점 |
| [다운로드 manifest](download-manifest.json) | 원문 URL, 실제 PDF, 버전, SHA256 |
| [독해 범위 기록](reading-coverage.json) | 전체 읽은 페이지, 이미지 대조 페이지, 검토 범위와 제한 |

핵심 제안은 같은 전체 update U에 대해 `M_t=m(θ_t)`, `B_t=m(θ_t−U)`, `C_t=M_t−B_t`를 반복 측정하고, `M_t−M_b=(B_t−B_b)+(C_t−C_b)`로 사실 score의 변화와 U의 조건부 기여 변화를 구분하는 것이다. C는 현재 배경에 조건부인 제거 효과이며, 독립적인 지식 trace의 보존이나 U가 없었던 편집 이력의 총효과를 뜻하지 않는다.

**원문 검토 범위**

이번에는 Spurious Forgetting(66쪽), Suppressed, Not Erased(7쪽), Forgetting Is Not a Fix(8쪽), RLEdit(23쪽), AI Engram(26쪽), Superficial Editing(25쪽)의 원문 PDF를 직접 다운로드하고 본문·수식·실험·참고문헌·부록의 전체 지문을 읽었다. 합계 6편, 실제 PDF 155쪽이며 선택한 9쪽은 이미지로도 대조했다. 추출 텍스트와 원문은 `papers/`, 대조 이미지는 `figure-checks/`에 있다. `*-pdf-pages/`가 물리적 PDF 페이지에 대응하는 추출본이다. 앞서 생성한 form-feed 분할 디렉터리는 실제 페이지 번호로 사용하지 않는다.

이전 18편 검토(로컬 이력 `audits/global/2026-09-24-delayed-write-lifelong-review/README.md`)를 함께 참조했다. 해당 과거 원문은 이번 게시 범위 밖이다. 이전 18편을 이번에 모두 다시 읽었다는 뜻은 아니다. 실제 PDF 페이지 수를 다시 확인한 결과 이전 총합은 484가 아니라 **483쪽**이었다. PRUNE의 text segmentation 28개를 물리적 28쪽으로 잘못 셌으며 실제로는 27쪽이다. 읽기 누락이 아닌 페이지 계수 오류다. 이 정정을 포함하면 전체 검토 대상은 24편, 638쪽이다. 봉인된 기존 파일은 변경하지 않고 이번 기록에 정정했다.

PDF, 논문 전문 추출본과 대조 이미지는 원 local 폴더에 보존하고 Git에는 넣지 않았다. manifest/coverage의 local 경로는 보존 위치 기록이며 저장소에 payload가 포함됐다는 뜻이 아니다. 논문 원문은 [논문별 노트](paper-notes-ko.md)의 출처 URL로 접근한다.

원문 독해는 코드 재현이나 모든 표 수치의 독립 재계산을 뜻하지 않는다. 이번 검색 범위에서 동일한 설계를 찾지 못했다는 판단이며, 문헌 전체를 망라한 최초성 인증이 아니다. 일부 preprint의 날짜 표기 불일치는 논문별 노트에 기록했다.

**실행 상태**

이번 산출물은 검토와 측정 제안이다. 모델 forward, GPU 실험, CPU gate 재실행, GH/SH4 상태 조회·모니터링, 지시 변경·전송은 하지 않았다. 기존 지시문·DAG·dispatch 관련 파일은 보존했다. 첨부 문서와 사용자가 인용한 리뷰는 분석 대상 자료로 취급했으며 그 안의 실행 제안을 별도 명령으로 승계하지 않았다.
