# B010 NATIVE_WEAK_NLL1

권한: `ODEEDIT-GH-SH2-NATIVE-WEAK-B010-NLL1-20260930-R1`.
BASE_ALPHAEDIT B010 전체 W/M/context/RNG에서 시작하는 BS1×100 한 경로다.
Original 3arm은 읽기 전용 비교이며 재실행하지 않는다.

`binding.patch_source`는 봉인 native compute_z 사본의 `loss < 5e-2`를
`nll_loss <= 1.0`으로 한 번 치환하고 AST의 나머지가 동일한지 확인한다.
같은 forward의 raw NLL을 backward 이전에 검사한다. Finite 검사와 native
trace 연결만 추가한다. 25 forward/24 Adam, 원 objective/optimizer/clamp/5층
writer/history는 유지한다. 최종 forward의 crossing도 threshold 도달로 기록한다.

원 `Runtime.fit_batch`를 그대로 호출하며 target1/key10/5개 post-write Gram
exact append를 재사용한다. `WeakTrace`는 five-layer solve 경로와 fit NLL/KL/
decay/total/초기·최종 norm/Adam/stop reason을 기록한다. 실제 write 후 동일
native contexts의 입력/target/position을 대조하고 별도로 NLL을 측정한다.
Latent NLL과 actual-write NLL을 합치지 않는다.

기존 관측 시점1/5/10/25/50/75/100, R/P와 fixed observer, greedy 규칙을
유지한다. 마지막 RAM state에서 R100/P200/N1000을 평가한 뒤 CPU reducer를
수행한다. `save_checkpoints=false`, `exact_resume=NOT_AVAILABLE`이며 failure
경로도 weight/history를 저장하지 않는다. 분석 NLL/token·작은 상태 hash는 local.

```bash
python3 -B -m unittest project.run_scripts.native_weak_b010.test_protocol project.run_scripts.joint_multilayer_bs10.test_review_b010 -v
python3 -B -m project.run_scripts.native_weak_b010.launch prepare --repo CLEAN_WORKTREE
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B -m project.run_scripts.native_weak_b010.launch tokens
# source commit 후
python3 -B -m project.run_scripts.native_weak_b010.launch freeze --repo CLEAN_WORKTREE
python3 -B -m project.run_scripts.native_weak_b010.launch submit
```

Create-once root `local/native-weak-b010/20260930-v1/`; 기존 실행/출력 덮어쓰기0.
Original pinned closure는 새 source archive에 byte-copy하고 원본은 변경하지 않는다.
Project cap3/task1, 60416MiB/exportNONE/Requeue0/8h. Initial B1 write→B2 state
또는 실제 released resource-pending 인계 후 agent pause; 자동 terminal monitoring0.
기술 model PASS와 CPU tests를 구별하며 성능을 gate로 삼지 않는다.
