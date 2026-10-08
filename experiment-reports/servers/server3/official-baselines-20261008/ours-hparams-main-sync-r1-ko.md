# server3 official ours hparam 동기화 적용

nonce `GH-SH3-OURS-HPARAMS-MAIN-SYNC-20261009-R1`을 실제 적용했다. 서버 `ubuntu`, 준비 checkout `/data/janghj/ODE-edit/local/official-baselines-20261008/worktree`, branch `codex/server3-official-baselines-20261008`에서 clean 상태를 확인한 뒤 origin/main fetch와 ff-only로 동기화했다.

적용 HEAD와 관측 fetched main은 모두 `67d00d4b166a9a186c898dec49f369ad081a875a`다. 필수 PR merge `cc2d7190e4860ae2934be86f615fef37e84ca4ba` 및 fetched main의 HEAD ancestor 검사 모두 PASS다. 문서 README, 세 모델 writer/price JSON, schema와 `official.ours.config`를 읽고 SHA를 audit에 결속했다. 공통 source 자체 수정은 없다.

`python3 -m official.tools.verify`: source157 SHA PASS, Python233, external task imports0. 기존 과학 venv에서 `CUDA_VISIBLE_DEVICES=''`로 `unittest discover -s official/tests -q`를 실행해 **70 tests PASS**다. 환경 설치·변경은 없으며 실제 GPU qualification은 NOT_RUN이다.

이 기록은 준비 checkout 적용 완료이며 새 실험 제출이나 frozen 실행 수정이 아니다. root dirty와 기존 RUNNING/PENDING source/config/archive/runtime는 작업 대상에서 제외했다. native 자산·raw·checkpoint·model·Slurm·W&B 온라인 검사/변경0이다. 이후 source freeze는 실제 새 source와 resolved config SHA를 결속한다. README main table은 GH 소유로 유지한다. own receipt 게시 commit과 위 동기화 HEAD는 구분한다.
