# AlphaEdit key causal E0–E4 — 준비/입력/실행 경계

최신 override nonce: `ODEEDIT-GH-SH4-ALPHA-KEY-CAP2-PENDING-20260923-R1`.
현재 단계는 구현·CPU 검산 및 upfront queue 등록 준비다. 실제 Llama gate, native100, SHAM은 아직 실행하지 않았다.

## 입력 및 소스

- GH small input 23개 / 5,051,920 bytes exact pull 및 SHA PASS.
- Server2 BASE_ALPHAEDIT 12 checkpoint / 63,418,321,276 bytes exact pull 및 SHA PASS. 원본 KEEP.
- 별도 CPU weights_only/mmap 검산으로 12개의 W4–W8/M tensor finite·내용 SHA·원 commit·요청 순서·context 연결 PASS(192.93초). GPU continuation 검증은 아니다.
- 원 실행 의존 파일 1,819개 중 실행/model/data/P 등 1,817개는 현재 전량 SHA로 확인했다. 삭제된 비실행 정책 문서 2개는 Git `6fd7f1482c395b9ea7271c15c94967120dffca7e`에서 정확한 원 SHA/size를 발견했고 immutable supplemental binding으로 회수한다.
- Server2 sidecar 지정 위치는 36개 모두 부재였다. 필요한 원 commit/native-observation은 동일 Server4 자료 또는 exact 원 bindings의 raw로 재사용하며 파일명으로 대체하지 않는다.
- Native source는 frozen non-BLUE AlphaEdit를 직접 호출한다. 별도 optimized-z, BLUE, L4-only, L2 변경은 없다.
- 현재 registry/thread/host/origin은 사용자 지정 SH4와 일치한다. generic session helper의 과거 session env는 stale이며 NOT_PASS로 기록하고 공용 설정을 수정하지 않았다.

## 구현과 검증 수준

E1 28 / E2 18 / E3 4 / E4-H 20 / E4-W 16 / E4-KR 8 = 94 family를 유한 실행기로 연결했다. SEQ/ORDER/FUTURE 7개 설계 행은 FOLLOWUP_NOT_SUBMITTED다.

원 apply 함수의 module-local wrapper는 실제 원 target/key/residual/solve/write/history를 호출한다. Same-entry z만 공유하며 downstream K/R/solve와 current H512는 branch별 새로 계산한다. Timestamp bank 차감은 실제 원 key SHA parity 뒤에만 가능하다.

세 bounded worker가 geometry/components, native/diagnostics, observer/technical을 소유권 분리하여 구현했다. 별도 CPU integration red가 schema/endpoint/token/panel/RNG 오류를 찾아 수정 후 fixture로 재검산했다. CPU 성공은 actual 모델 검증이 아니다. 최종 CPU 개수와 source hash는 audit JSON이 정본이다.

SHAM은 실제 subtract/re-add하며 차이를 숨기지 않는다. 사전에 확립된 다른 수치 envelope가 없으므로 동일-control exact 비교에서 차이가 나면 `NUMERICAL_CONTROL_NOT_ESTABLISHED`로 기록하고 후속 E4를 차단한다. 낮은 scientific score에 의한 선별은 아니다. Full-delta hook은 같은 FP32(U+Δ) GEMM을 사용해 physical write와 exact 비교하며, Ux+Δx sum-route 오차는 별도 관측한다.

원 P를 직교화하여 writer에 대입하지 않는다. 실제 P의 대칭/idempotence 오차, raw 비대칭 H와 조건수, actual Δ response를 보존한다. sym(H)의 PSD mode는 명시적 근사 진단이며 ideal orthogonal P identity PASS가 아니다.

## 자원·저장

Preflight 관측 free 285,940,449,280 bytes. 입력 CP12는 이미 저장됐다. 이후 232,700,000,000 bytes 계획에는 E1 key/mean 56.2GB, E2 36.5GB, writer factors32GB, stage/H keys38GB, component/observer12GB, source/atomic/log8GB, 공유볼륨 안전여유50GB가 포함된다. 독점 reserve나 향후 여유 보장은 아니다.

각 GPU job 1GPU/8CPU/60416MiB, exportNONE/Requeue0. Gate1GPU 뒤 geometry1 + writer1 = project/task 최대2GPU이며 다른 owner admission이 있으면 이 graph를 그대로 추가하지 않는다. CPU reducer는 GPU0이다. Host peak 계획49GiB는 추정이고 실제값은 program receipt로 구별한다.

7일 wall은 scheduler 내 운영 reservation이지 과학 GPUh 예산이나 실측 완료 예상이 아니다. 실제 prefix/full microbatch, 첫 native target/solve/I/O 및 observer 시간·peak는 runner가 기록한다. 사용자 GPUh hardcap은 미지정이다.

새 W/M/RNG/optimizer/full endpoint resume checkpoint는 저장하지 않는다. 승인된 K/R/Δ/native target/current-bank/기하 diagnostic만 저장하며 branch rollback은 RAM과 입력 CP로 한다. 새로운 branch exact crash-resume은 보장하지 않는다.

## 제출 및 종료

`gate → afterok:{geometry,writers} → afterany:CPU reducer`를 전량 held 등록·검사 후 release한다. GPU 부족이 실제 확인되면 gate를 기다리지 않고 job mapping을 인계하고 모든 agent/worker monitoring을 멈춘다. 실행 중이면 actual G0–G3 초기 경계를 확인한다. 제출/PENDING을 gate PASS라고 하지 않는다.

프로그램은 완료/기술실패/조건부 정의불가/미승인 후속을 구별하여 local raw와 자동 compact report를 기록한다. 능동 후기 결과 회수는 사용자 recall 전 하지 않는다. Own branch만 게시하며 main 통합은 GH clean integration 검토 소유다.

NO_BROADCAST_NOT_REQUIRED. Input의 narrow pull 외에 대형 자료 재방송·원본 삭제·타 job 변경은 없다.
