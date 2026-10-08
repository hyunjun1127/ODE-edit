# SH4 OURS hparam main 동기화

Nonce: `GH-SH4-OURS-HPARAMS-MAIN-SYNC-20261009-R1`.
관측: 2026-10-09 02:54:40 KST. 상태: **APPLIED_CPU_VERIFIED**.

실제 server4의 `/data/janghj/ODE-edit/local/official-baselines-20261008/worktree`,
branch `codex/server4-official-baselines-20261008`에 적용했다.
기존 own HEAD `aa5e59a7d2fd504f833d17a20b259bf1e14e07e1`를 보존해
fetched main `67d00d4b166a9a186c898dec49f369ad081a875a`를 충돌 없이 merge했다.
검증한 적용 HEAD는 `12750e3e92ad5231678c8e8f0590ea922c8bc3c0`,
official tree는 `b65c00a2c7ffc91f7982944382766a2243176ac3`이다.
이후 receipt commit은 위 코드 검증 HEAD와 구분한다.

- `cc2d7190e4860ae2934be86f615fef37e84ca4ba` ancestor: PASS.
- fetched main 및 이전 own HEAD ancestor: 각각 PASS.
- 모델 3종 writer/price, schema, resolver 및 지정 README 전체를 읽었다. GPU smoke 문서는 계획이며 실행하지 않았다.
- `python3 -m official.tools.verify`: source 157 SHA / Python 250 / external imports 0, PASS.
- 기존 EasyEdit venv, CUDA 비활성·OMP/MKL 각 2: official CPU 70 PASS, own runner CPU 29 PASS.
- 기존 runner bytes는 merge 전과 동일하다. 환경 설치·공통 수학 임의 수정 없음.

root dirty, 타 작업자 변경, 기존 frozen 실행 source/config/archive/runtime, 자산 및 CP는
변경하지 않았다. 이번 sync에서 Slurm/GPU/W&B 온라인 작업은 0이다. README 직접 편집 및
SH4의 main push도 하지 않았다. 적용과 own branch receipt 게시는 main 통합과 별개다.

이 CPU 결과는 실제 pretrained/GPU/W&B PASS가 아니다. own 테스트가 재현하는 기존
display-score/logger 불일치와 shared W0 input 대기는 해제되지 않았다.
다음 준비 source freeze에서 새 source/config SHA를 기록하며 기존 frozen run에 주입하지 않는다.

상세 근거: `audits/servers/server4/official-baselines-20261008/ours-hparams-main-sync.json`.

## 후속 공통 deferred FLU/CON 입력 적용

Nonce: `GH-SH4-DEFERRED-FLUCON-READY-20261009-R1`.
관측: 2026-10-09 02:57:39 KST.

같은 clean 준비 WT에 fetched main `6c7152fae8fa2b6d49af6e5593e6113c64c5ee1f`를
안전 merge했다. 적용 코드 HEAD `cd7ad87512497faab9bdfbda1f67bbd38db4191b`,
official tree `e46c7d5cd125676b3d12a276a373337d36d4f77a`.
요구 commit `6d35c65de19ec378a30749683762eba87fee50e3`와 fetched main의 ancestor 검사는
모두 PASS다. 이전 own commit과 runner bytes를 보존했으며 충돌은 없었다.

공통 보고서/API를 전체 읽고 source verify 157 SHA / Python 250 / imports 0 PASS,
지정 tracking + server2 checkpoint profile CPU **32 PASS**, SH4 caller CPU **9 PASS**를 확인했다.
server2 전체 suite 또는 실제 GPU/온라인 검증으로 확대 해석하지 않는다.

새 API는 명시적 CF `DEFERRED_CHECKPOINT_EVALUATION`을 허용하며 schedule 외
generation metadata와 reference assets SHA를 생략한다. FLU/CON 점수·count·progress·phase는
0 placeholder도 거부한다. deferred 소비자는 실제 최종 W20 checkpoint와 source/config/sample/RNG
identity를 보존하고 후속 평가 완료 전 삭제·이관 완료를 가정해서는 안 된다.

**SH4는 이번 코드 수신만으로 기존 평가 일정을 변경하지 않았다.** own caller 검산은 기존
enabled CF / zsRE 경로의 호환 검사다. GPU 평가·smoke·sweep·Slurm 제출·W&B 온라인 호출,
기존 frozen 경로 변경 및 checkpoint 전송·삭제 모두 0. main 직접 게시/README 편집도 없다.
공통 코드 적용과 실제 과학 gate/등록 완료는 별개이며 기존 미완료 사항을 PASS로 바꾸지 않는다.
