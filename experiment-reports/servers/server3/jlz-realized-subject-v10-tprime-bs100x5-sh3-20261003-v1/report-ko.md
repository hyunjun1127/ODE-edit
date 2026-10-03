# JLZ v10 T′ SH3 초기 인계

**MAIN_INITIAL_PASS**: 실제 main A B1의 same-weight commit/H once → R/P/N observer → B2 own-entry 연결을 검산했다.
A500/B500 및 전체 terminal 완료는 아직 관측하지 않았다. 여기서 능동 모니터링을 중단한다.

Execution source: c2d5fb107a0435491d8b4705b43b75f6177bbb5c
Execution lock SHA256: cf03a94ccf873d93f36bdfe3c0a378998bc2340ffbb4cda4d5b18914c31cbe23
Analysis source: 5bbfdd6245e194f6aa80fd97e912bc86b1d62814

| 단계 | job | 상태/경계 |
|---|---:|---|
| Q1 | 57698 | COMPLETED/0:0, actual GPU qualification |
| A | 57699 | B1 완료/observer 완료/B2 entry; A500 미관측 |
| B | 57700 | 사전등록·release, Q1 afterok + A afterany; B500 미관측 |
| CPU collector 원본 | 57701 | 원 source 보존; W0 row-order 검증 오류 기록 |
| CPU collector 수리 | 57704 | GPU 0; 세 GPU job 및 원 collector afterany |

## A B1의 실제 Current 관측

| Family | preference 성공/분모 | TF token correct/valid | TF micro | TF prompt macro | TF strict | true NLL | new NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| R | 100/100 | 101/101 | 1.000000 | 1.000000 | 100/100 | 15.580228 | 0.003485 |
| P | 195/200 | 157/202 | 0.777228 | 0.775000 | 155/200 | 12.074287 | 1.059060 |
| N | 856/1000 | 186/1030 | 0.180583 | 0.171500 | 157/1000 | 5.502299 | 11.086255 |

Preference는 R/P new<true, N true<new이고 ties failure다. TF는 teacher-forced이며 자유생성 정확도가 아니다. N의 desired target은 true다. 위 표는 first100 W1이며 기존 W5/first500과 분모를 혼합하지 않는다.

B1 프로그램 시간 1175.397s, 그 안의 후보 시간 합 1135.940s; observer 23.452s. 중첩 timer를 더하지 않는다. Q1 parent425GPU-sec는 별도 준비비용이다. Main 전체 GPU allocation/ETA/terminal peak는 아직 NOT_MEASURED다.

Q1은 A/B cold BS2×2의100후보96update를 완료했다. 고정 B1/B3 shape/full native hook/dense/whole-B microbatch gradient 검사는 [Q1-ko.md](Q1-ko.md)에 한계를 함께 기록했다. Main Q2는 A B1 자체로25후보24update이며 별도 B100fit0이다.

첫100 input·prompt/target/token identity 및 R100/P200/N1000 분모를 별도 CPU reducer로 다시 검산했다. 평가 전후 W/H 및 context/RNG/ledger identity가 같고 B2 entry가 B1 commit과 일치한다. 과학 품질은 실행 gate가 아니다.

## 출처·제약

[실제 S3 source/runtime/입력 검토](implementation-audit-ko.md), [DAG 제출](submission-ko.md), [기존 W5 역사 참고](historical/README.md). Pinned source/config/입력/archive는 audit과 local attempt-v2에 보존한다.

CPU 집계기의 W0 family-major 순서와 actual case-major 순서 비교 오류를 발견해 reference canonicalization만 수리했다. 기존 job/source/partial을 보존하고 별도 immutable CPU collector를 등록했다. 실제 GPU fitting source와 과학 trajectory는 변경하지 않았으며 추가 GPU0이다.

cap1/각1GPU8CPU59GiB, FP32/eager/TF32off, noCP. Edited W/H/optimizer/동등복원 bundle 저장0, exact resume NOT_AVAILABLE. 다른 중단 task 재개/타 job 변경/추가 baseline fit0.

등록된 두 cold chain은 각W5까지 자연 진행하며 CPU collector가 기존 raw를 집계한다. 이 인계 이후 polling/heartbeat/callback/자동재시작은 하지 않는다. 상세 완료 검토는 사용자 recall에서 수행한다. Owner audit이며 별도 독립 reviewer는 사용하지 않았다.
