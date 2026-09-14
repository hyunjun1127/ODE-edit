# REFIT4 write-refresh 구현 preflight

Instruction: ODEEDIT-S06-REFIT4-WRITE-REFRESH-SEQ1000-SH4-V1.
Base cc348e012707f2507dc2d66cfdb5af937a44b4fc; 실행 SHA는 freeze receipt에 별도 기록한다.

원문/설계/cells/contract/review/checks/envelope/PROTOCOL을 정독하고 지정 SHA를 확인했다. 원첨부 SHA와 Git 게시 SHA는 다르다. Contract 내부의 과거 attachment_sha256은 원문 그대로 보존하며 최신 전달문의 attachment identity와 동일하다고 주장하지 않는다.

## 재사용과 새 실행

N4/REFIT4는 실행5e96dcb의 완료10batch, 같은 원 W50/M50/context/RNG·model/config/evaluator/order를 확인하여 재사용한다. 기존 Adam 내부 state 부재는 NOT_RECORDED이고 새 state로 복원하지 않는다. 신규 FROZEN2/I2/FROZEN4/I4는40batch, 전체표는6정책60logicalbatch다. 이전M8준비·reference할당시간은 새비용에 가산하지 않는다.

## Source-semantic 검사

기존 `NativeSingletonFitter.fit` 및 BLUE compute_z bytes 변경0. ExternalTargetSingletonFitter는 guard된 native AST를 그대로 실행하고 private function globals의 compute_z 공급만 own absoluteZ로 치환한다. 매 chunk fresh K/Y/residual/directsolve, 마지막 native finalizer만 history append1이다.

TargetStepper는 동일 request의 leaf u와 Adam 객체·m/v/t를 유지한다. hook은 u+(a0-aj), a0/teacher/clamp는 batch-entry 기준, aj는 chunk 첫 loss의 unhooked 값이다. nonsquared norm regularizer, reverse KL 인수순서, initial/postloss·24update budget과 .05 chunk-local stop을 유지한다. quota 재분배0. Frozen 후속chunk는 targetforward0, 현재Y solve는 수행한다.

모델 requires_grad를 stepper 생성 전 False로 설정하여 cached head view의 불필요한 backward 경로를 막는다. 원래 flag는 process 종료시 복원한다. 이는 target/math 변경이 아니라 원native가 첫 compute_z에서 수행하는 model freeze를 명시화한 것이다. 원native 첫request와 계측비용 차이는 기술 ledger로 분리한다.

## CPU 검증과 실제 gate 구분

CPU25 tests PASS: stepper7, externalwriter6, 기존fitting5, runtime/evaluator7. shell syntax/compile/explicit60416M memory audit PASS. 실제 모델 GPU parity는 아직 미실행이다.

두 기술process는 동일prepared에서 native B100와 I1 B100을 독립 실행한다. 실행 전 등록 기준은 torch.equal targets/actual nativeweight/.75stored materialization/history 및 unrounded native loss sequence다. Native loss 관찰은 원 함수 code를 같은 private globals로 실행하며 print 시점의 scalar locals만 읽는다. fixedW pause/resume u/m/v/t/teacher/a0/마지막loss도 exact 비교한다. 불일치 evidence를 먼저 저장하고 HOLD하며 사후 tolerance 완화는 없다. 기술batch·비용은 main 분모에서 제외한다.

## 운영

신규array%2, 독립1GPU/8CPU/60416M/exportNONE. 제출직전 전체 project node+pending QOS admission을 확인하고 기존admission이 있으면 해당 종료 afterany 뒤에 신규array%2를 배치한다. 타job mutation0. 기술2h/정책12h wall request는 보수적 scheduler reserve이며 user GPUh hardcap은 null이다.

저장계획120GiB+50GiB여유, 관측가용약305GB. 이는 초기추정이다. selected W4/M4 CP12, subwrite120, request/chunk evidence를 보존한다. 원본/공유자산 삭제0.

이번 task만 초기gate 이후 Middle1000요청·검산·보고까지 계속한다. Late는 GH policy lock 전 미제출. scientific_promotion=false, 타paused task/중지ORBODE변경0.
