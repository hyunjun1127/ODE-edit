# Rescue 최종 CF 조건부 Flu/Con 어댑터

USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1 동일 승인 후속. 새 파일
`official/diagnostics/fe_qwen_current/flucon.py`, `flucon_prepare.py`,
`test_flucon.py`, `FLUCON.md`만 추가. c041/f843 rescue 및 기존 ac05/10843
동결 코드·실행·출력·checkpoint/README 결과 변경 없음.

rescue L2=100 SELECTED의 실제 held-out 증거와 rescue final-collect의
CF/zsRE 양쪽 W20 완료 결과를 독립 재검산해야 prepare가 가능하다.
최종 source/author member, config/identity, W0+20commit, 전체25endpoint의
raw/query/2000순서/분모 및 CF CP/latest pointer를 검산한다. 미완료/다른
L2/옛4arm proof/held-out CP의 대체 경로는 없다.

복원은 full CP SHA 확인 후 CPU mmap. 선택된 BF16 down_proj 5층만 chunk
hash 및 shape/finite/identity 검산, 기존 fe_cake_flucon.bind_weights의
정확 BF16→FP32/역변환 및 비선택 parameter 불변 guard를 사용한다.
history/RNG/context를 평가 모델에 복원하지 않고 전체 H clone/digest를
추가하지 않는다. 공통 FP32/eager/TF32off **평가 정밀도**이며 BF16 편집은 불변.

NativeGenerationObserver/profile/seed20261007/reference identity75e595c7…
그대로 재사용. 원 CF2k prompt/occurrence 하나의 생성 결과를 두 점수에 공유.
공통 read_observed(assets=...)로 token/count/원문 점수를 독립 재검산.
원 bits/cosine 값과 valid/missing 분모 보존, x100 half-up2 표시 필드 별도.
공통 digest의 Unicode 처리도 실제 tiny generation fixture로 검산했다.

W&B는 새 eval_only group/task와 실제 job/source_run/CP identity,
all_seen/post edits2000 및 W20 progress만 사용. 기존 schema의 승인된
CF generation authority를 재사용하고 현재 task authority는 immutable
config.registration_authority로 별도 보존한다. logger/schema 복제 없음.

CPU **74 PASS**: 신규9 + 기존 rescue/contract/benchmark/native generator/
native observer/FE-CAKE restore/paper display 회귀. 실제 tiny checkpoint mmap,
BF16 exact 복원, unselected/RNG 불변, re-signed wrong identity/order/dtype/
hparam/shape/hash 거부, missing final proof 차단, shared generation 원자료
재검산/누락분모, tracking 축 거부 검사를 포함한다.
source verifier source171/Python441/PASS/external_task_imports0.

실제 최종 CP/선택 성공/실제 GPU 평가/온라인 W&B는 **NOT_OBSERVED/NOT_RUN**.
신규 제출·rent 조회·전송·삭제0. root가 실제 선택/두 최종 W20 후 문서의
prepare/run/collect API와 rent 절대 경로·자원을 결속한다. 그때까지 조건부다.

원 참조 manifest6769B/SHA6d9a713a… 일치 확인. scoring requirements는
numpy2.2.6/scipy1.15.3/sklearn1.7.2/nltk3.6.5 및 기존 NLTK resource SHA.
native 편집 runtime의 numpy1.26.4를 조용히 통과시키지 않는다. 별도 평가
runtime을 사용하되 편집 환경/과학 설정을 변경하지 않는다.
소형 소스/보고만 Git: NO_BROADCAST_NOT_REQUIRED.
