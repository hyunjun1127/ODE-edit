# GH EN 실행 경로 변경 기록

## 변경 결과

사용자 첨부 리뷰 SHA256 `30ecf5fb56646e021f8791e65e2c1103ec52e5eaf3752c3282c98882f4258e30` 및 후속 GPU gradient 누적/검증 삭제 지시를 반영했다. 근거 main `d5d93cdf3ae737ac55f0a829bc9aac419752c814`에서 독립 worktree로 수정했다. 기존 shared dirty/실험 source/archive/raw/teacher/checkpoint 및 다른 서버 작업은 변경하지 않았다.

새 정책은 [override](../../plans/global/2026-09-19-en-runtime-diagnostic-skip-gpu-gradient-override.md)에 기록했다. 수정 대상은 `project/run_scripts/en_execution_reuse/`의 runtime·보고·해당 CPU tests다. 원 EN optimizer/geometry/native/evaluator 수치 코드는 그대로다.

- MATCHED_B1의 별도 physical AD/FD/parity, selected parity, matched exactness gate, nonselected 전체 해시, checkpoint 재로드 진단을 호출하지 않는다.
- Teacher 초기/반복 payload SHA·finite·argmax·정규화 전수 검사와 prefix/endpoint 반복 byte 검증을 생략한다. 대형 teacher는 문서 단위 read-only mmap이고, objective는 logp만 열어 이미 상주한 key/residual를 재접근하지 않는다.
- GPU FP64 accumulator에 원 문서 순서대로 누적해 평균을 낸 뒤 최종 합을 CPU로 한 번 보낸다. FP32→FP64 변환 순서·문서별 gradient 생성·전체512 참여·Ti 문서평균은 그대로다.
- 별도 진단을 생략한 receipt/보고는 `SKIPPED_USER_DIRECTED`, `NOT_ESTABLISHED`다. 새 policy lock을 요구해 옛 CPU 누적 lock과 새 GPU 경로를 섞지 않는다. 과거 준비 lock은 read-only lineage로만 별도 허용한다.
- 소유권/version/shape/dtype·source/sample/권한 결속, nonfinite 계산 오류, EN 수학적 수용조건, rollback/history1/atomic 저장 및 새 산출물 provenance는 유지한다. NumPy/.data mutation 또는 sealed 파일 변경에 대한 완전한 byte 검출은 주장하지 않는다.

## 개발 확인

GPU를 비활성화한 CPU 단위/작은 합성 모델 회귀 테스트 **157개, 59.141초, OK**. 실제 Llama/GPU numerical parity·속도·peak memory·새 실험 결과는 NOT_RUN이다. 사용자 지시로 제거한 진단을 실험 선행조건으로 다시 넣지 않았다.

확인 범위:

- Runtime teacher reader에서 payload hash/finite/argmax/normalization 함수가 호출되면 실패하는 regression; logp만 요청 시 key/residual 미로딩.
- Runtime skip 전후 작은 CPU 모델의 전체512 loss/rows/gradient 일치 및 full coverage.
- Gradient loop 내부 CPU transfer 없음, 최종 accumulator transfer site 하나, accumulator device=`self.device`.
- 같은 endpoint 반복 borrow의 새 byte hash0; 신규 endpoint provenance hash1; version mutation 처리 유지.
- 진단 호출 제거 AST regression, atomic 저장의 no-reload/no-overwrite 및 history1.
- 新/과거 policy 경계, 원 EN 실제 optimizer CPU 연결 테스트 및 skipped 보고서가 PASS 문구를 출력하지 않는 경계.

첫 시도는 기본 transformers4.57.1로 fixture의 기존 pinned4.44.2 import 조건에서 setup 오류가 났다. 검사를 완화하지 않고 기존 로컬 `deps-py312`4.44.2를 읽기 전용 PYTHONPATH overlay로 사용해 전체 suite를 통과했다. 공용 환경 변경/패키지 설치0.

재현명령(해당 overlay가 존재하는 GH 환경):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/deps-py312:. \
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest discover \
  -s project/run_scripts/en_execution_reuse -t . -p 'test_*.py'
```

대형 gradient 전송량 512회/240,518,168,576B → 1회/469,762,048B는 source 차원 산술이며 GPU 실측이 아니다. GPU accumulator448MiB와 FP64 변환 temporary 최대448MiB를 고려해야 한다. CPU와 GPU의 FP64 연산이 bitwise 같다는 주장0. Step 수 증가·목적함수 변경·reference 축소·KV 최적화·추가 batch/sequential은 수행하지 않았다.

## 실행 상태

새 GPU/model load/실물 forward/Slurm 조회·변경/새 job 제출0. 기존 SH4 완료 실험 및 리뷰는 frozen source 그대로다. 다음 실험은 이 변경을 반영한 새 execution lock과 별도 사용자 실행 요청이 필요하다.
