# 최대 B10 dependency 실행기 준비 — actual 결과 미관측

최신 사용자 지시는 “실행기 구현 마치고 pending 까지만 걸어놔”, “checkpoint 저장은 하지말고 진행해”이다. 공통 미저장 정책 `4d871dc6aeb328675d4912f4525ed23434bc0ad8`을 함께 적용한다. 기존 50974를 취소·수정하거나 결과를 모니터링하지 않는다.

## 프로그램 범위

신규 단일 GPU 프로그램은 `afterok:50974`로 등록한다. 내부 순서는 기존 hook 기술 산출물 결속 → 나머지 한정 T0 → cold B1 4개 정책 → 원 B1 gate → N4/STEP/CUM B2–B3 → 원 S3 gate → 같은 세 chain B4–B10이다. 같은 프로그램이 각 chain의 RAM state를 보존하며, 프로그램 안에서 Slurm 호출·후속 제출·callback은 없다. B1 STEP=CUM은 독립 RAM clone이지 중복 fit/실험이 아니다. B2 N4 own-entry STEP/CUM shadow는 원 설계의 별도 진단이며 N4 state에 commit하지 않는다.

최대10 batch를 등록하는 것이 모든 gate 통과나 B10 완주를 의미하지 않는다. 기술 실패·과학 gate 실패·자원 부족을 분리하여 terminal에 기록한다. 기존 50974의 현재 PASS/FAIL/진행 상태는 이 준비에서 확인하지 않았다. afterok 의존성이 충족되지 않으면 신규 프로그램은 시작하지 않는다.

## 저장 정책

`save_checkpoints=false`, `checkpoint_saved=false`, `exact_resume=NOT_AVAILABLE`; 저장 예외 없음. 새 W/M/RNG/optimizer resume bundle, native/selected 전체 weight, 전체 solve update, 후보 복원용 dense basis 및 전체 gradient의 디스크 저장을 하지 않는다. RAM rollback/history/chain continuation은 유지한다. 실제 solve update는 source-order 기전 진단에서 RAM으로만 소비한다.

남기는 자료는 평가·NLL·ID, 선택/commit/history hash, source/config, 비용·gate·작은 J/JH, native target/key와 설계상 작은 기전 factor/activation derivative 증거다. target/key 보존을 exact endpoint reconstruction이나 GPU crash-resume PASS로 확대하지 않는다. 과거 제출된 50974의 원 source/lock 및 기존 파일은 불변이다.

## 구현 및 CPU 검증 수준

새 controller/science/program, 전체 active-history canonical cache, observer, 기전 진단, RAM transaction, 제출 검사 경로를 연결했다. 독립 read-only red에서 발견한 receipt property 호출, T0 4-request parity의 B100 오기, held actual argv 미확인, 압축 array capacity 과소계수 가능성을 수정했다. 서로 다른 helper가 구현과 다른 파일을 검토한 범위만 독립 red로 기록하며 전체 actual Llama 검증을 의미하지 않는다.

CPU 모의 program은 native/model 없이 실제 stage 제어와 RAM transaction을 검사한다. 상세 테스트 결과는 동 task audit의 CPU receipt로 결속한다. T0/실제 모델/효능/새 job 초기 gate는 모두 아직 `NOT_OBSERVED`다. 원 T0 수치 ceiling, native hparams, full512 후보 검사와 stage gate는 변경하지 않았다.

## 자원 계획과 제한

신규 job GPU1/CPU8/60416MiB/exportNONE/Requeue0, wall7일. project cap2의 다른 admission은 제출 직전에 resource-only로 한정 확인한다. 50974와 새 job은 afterok로 비중첩이다. wall7일은 실측 예측이나 사용자 GPUh hardcap이 아닌 prospectively configured scheduler limit이다.

신규 공간 추정은 T0/B1 24GiB, S3 추가32GiB, S10 추가112GiB이다. 이는 보수적 계획이며 실측/전용 예약이 아니다. 준비 중 관측 free 약55.5GiB만으로 모든 후속 단계를 보장하지 않는다. checkpoint 미저장 이후에도 전체512 derivative 증거와 history cache의 저장 비용은 남는다. 프로그램은 stage별 실제 여유를 검사하며 원 storage waiver·파일 삭제·평가 축소로 우회하지 않는다. 저장 오류와 부족 범위는 사실로 남긴다.

등록·held inspection·release 및 compact 인계 후 agent monitoring은 중지한다. 모든 job의 완료·실제 초기 gate를 기다리지 않는다. 상세 결과 검산은 사용자 recall 이후다. `NO_BROADCAST_NOT_REQUIRED`: 같은 서버의 기존 immutable input을 재사용하며 새 대량 전송은 없다.
