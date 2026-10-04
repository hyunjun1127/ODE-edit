# V13 B1 구현 preflight (SH4 owner audit)

권한: ODEEDIT-USER-GH-SH4-JLZ-V13-B1-REALIZATION-20261005-R1.
Authority 37019ce3ea3b4ae24f8fed29a236b31d102911a8. 독립 reviewer는 사용하지 않았다.
정본 method/experiment/contract/telemetry/reference/audit와 B1 override를 정독했다.
정본 원본은 수정하지 않았다. FULL_READ/SHA와 실행 source/GPU 상태는 별도 receipt다.

## Red 체크리스트

- 범위: cold first100, fit 호출 1개, RT/RD/MT/MD/CD 순서. B2/pilot chain/추가 fit/baseline/sweep 코드 경로 없음.
- Planner: 1d27a830의 entry/optimize/optimizer/telemetry exact bytes 확인. source/실제 runtime/입력은 새로 결속. v12 과거 task 재개 아님.
- 목표: RT/MT는 native FP32 z−현재 h 후 FP64, RD/MD는 D, CD는 rewrite 및 KL owner D. 0·중복 열 보존. D=0에 따른 tracking 생략 없음.
- Metric: FP64 symmetric part와 skew 기록, Cholesky. jitter/목표 축소/자동 ridge fallback 없음.
- Equality: whitened X의 QR-SVD, strict eps64*maxshape*sigma cutoff, right triangular solve. range residual과 수치 residual 분리.
- 적용: FP32 old + cast32(U64), ideal/cast/effective/actual local action 분리. elementwise 고정 tolerance. 모든 branch effective Q 계산.
- Causal capture: 각 branch 각 층에서 전체 native 행을 다시 측정. physical capture에는 delta hook 없음. 다른 branch key 공유 없음.
- History: 모든 write 및 local parity 후 final rewrite-only nested mean CPUFP32 Gram 층별 1회. KL은 H에서 제외.
- 복원: branch마다 RAM W/H/RNG, plan/virtual rows/cache/context/ledger/nonselected/hooks identity 확인. branch 관측 후 항상 원복. 실패 후 부분 commit 성공 표기 없음.
- 평가: W0와 5 endpoint R100/P200/N1000, TF/NLL/paired; P/N 학습·선택 입력 없음. 누락은 NOT_MEASURED. 5 branch를 unique500으로 합치지 않음.
- 저장: torch.save/np.save production 경로 없음. W/H/D/P/factor/optimizer 복원 payload 영속 저장 없음. raw는 ignored local.
- CPU: reference16, production14(작은 random CPU Llama 포함), 기존 planner7. 실제 8B GPU qualification 아님.
- 실제 qualification: B1 first2 입력의 고정 후보 비교, native key 및 RAM 복원, 새 writer별 실제 cast/local parity. 추가 optimizer fit 없음.
- 자원: GPU1/CPU8/59392MiB/24h, collector CPU8/24576MiB/4h. task cap1, 최신 local/tracked project cap의 최소값. fresh held 검사 후 release; 기존 job 변경 없음.
- Artifact: NO_BROADCAST_NOT_REQUIRED. 동일 서버 B1 진단이며 compact source/receipt/report만 Git 공유, 대형 raw/model 무전송.

실제 모델 검사는 제출 이후 runner가 수행하며 기술 불일치 시 허용오차 변경 없이 중단한다.
낮은 성능·range mismatch·배분 집중·예산 소진은 gate가 아니다.

## 자원 사전 계획

현재 /data 여유 약64GiB, inode 여유 약2.23억; 12GiB 출력/atomic/여유 reserve.
Host: load peak 계획34GiB, 실행 단계 계획28GiB, 요청58GiB.
GPU: FP32 model30GiB + entry weights1.1 + activation/head20 + metric/factor/solve14 + branch transient5 + margin8.
한 층 metric/factor만 resident, 5 endpoint weight snapshot 동시 보관 없음.
24h는 유한 wall 상한이며 실측 ETA가 아니다. actual peak와 cost는 별도 보고한다.

초기 unittest discover 호출은 package context가 없어 relative import 오류였다.
module-qualified 호출로 7개 planner 회귀를 확인했으며 원 실패 사실을 CPU receipt에 보존한다.
