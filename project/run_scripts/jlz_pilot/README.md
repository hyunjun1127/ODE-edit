# JLZ native pilot

공동 local-z의 실제 모델 경량 실행 경로다. KL은 native `compute_z`와 같은
`KL(current || entry)`로 고정한다. `jlz_ref/`는 첨부 원본의 감사용 보존본이며
그 KL 방향을 이 실행에서 사용하지 않는다.

- `prompts.py`: native target·subject lookup, 별도 key prompt, 요청/context별 loss 축약.
- `solver.py`: FP32 block proximal solver, 반환 평가, 수렴·상한·nonfinite 및 signed KKT.
- `run.py`: 로컬 Llama-3-8B-Instruct/C0 캐시 사용, L4–L8 공동 fit, commit·history 검증.
- `pilot.sbatch`: server1/devbox GPU1, RAM64000M, wall60분.

설계는 `plans/global/2026-10-01-jlz-native-pilot-v2/design-ko.md`에 있다.
실행 데이터는 고정 순서 앞 4개이며 BS2 두 batch가 상한이다. 두 번째 batch는
첫 batch의 수렴·commit 검증 통과 후에만 수행한다. solver cap120과 별도 parity
probe2를 구분한다. tol1e-4는 이 pilot의 임시 기준이며 production 교정값이 아니다.

저장소 root에서 CPU 검사:

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest \
  project.run_scripts.jlz_pilot.test_prompts \
  project.run_scripts.jlz_pilot.test_solver \
  project.run_scripts.jlz_pilot.test_integration -v
```

GPU 실행은 resource cap 확인 후 Slurm을 통해 배정받는다. `JLZ_PILOT_OUTPUT`은
새로운 실행별 결과 디렉터리여야 한다. receipt에는 source SHA, 환경, case IDs,
solver/commit 상태와 실제 수치가 기록된다. 모델 파일·checkpoint는 쓰지 않는다.

`PILOT_COMPLETED`는 설정한 두 batch 실행 완료다. `STOPPED_*`는 미수렴 등의
원인으로 후속 batch를 실행하지 않았다는 뜻이다. 둘 다 성공률이나 장기 성능
비교 결과로 해석하지 않는다. `TECHNICAL_FAILURE`에는 exception과 traceback을
남긴다. 중단 후 임의로 tol·clamp·decay를 완화하거나 미수렴 결과를 commit하지 않는다.
