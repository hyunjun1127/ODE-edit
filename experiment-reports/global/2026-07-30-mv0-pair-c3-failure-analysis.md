# MV-0 paired c3 job 15508 조기 실패 분석

- 작성일: 2026-07-30
- 분석 역할: 실행에 참여하지 않은 실패 분석 agent
- 대상 job: `15508` (`odeedit_mv0_pair_c3`)
- 분석 범위: 지정된 stdout/stderr, paired execution preflight, 두 Slurm
  wrapper, `mv0_fidelity.py`의 공통 실행 시작부와 `_git_runtime_state()`,
  `gpu_runtime.py`의 model loader 시작부
- 결론: **실험 실패가 아니라 launcher/runtime 환경의 공통 조기 실패**

## 판정

job 15508의 두 child는 model allocation이나 fixed artifact 검증에
도달하기 전에, 실행 당시 `_git_runtime_state()`가 상대 실행명
`"git"`을 찾지 못해 `FileNotFoundError`로 종료한 것으로 판단한다.
원인 확도는 높다.

가장 작은 수정은 `_git_runtime_state()`가 검증된 절대 경로
`/usr/bin/git`을 사용하게 하는 것이다. 이 수정은 현재 worktree에 이미
반영되어 있으며, `--export=NONE`의 격리와 credential firewall을
약화하지 않는다.

## 확인한 사실

1. stdout에는 두 child 각각 다음 sanitized event만 있다.

   ```json
   {"error_type": "FileNotFoundError", "raw_exception_persisted": false, "status": "aborted"}
   ```

2. stderr에서 두 step `.0`, `.1` 모두 exit code `2`였고 parent는
   `paired MV-0 child failed`로 종료했다. 제공된 scheduler 사실은
   job `FAILED 1:0`, elapsed 약 5초, step MaxRSS `264K/312K`,
   filesystem/disk failure `0`이다.
3. 두 model output directory, manifest, summary는 생성되지 않았고 pair
   submission marker만 남았다.
4. pair wrapper는 allocation 자체를 `#SBATCH --export=NONE`으로
   시작하고, 두 child를 nested `srun --exclusive --exact`로 실행한다.
5. child runtime wrapper는 `/usr/local/bin/uv`를 절대 경로로 실행하지만
   `PATH`를 따로 export하지 않는다.
6. job 실행 시점의 committed `mv0_fidelity.py`에서
   `_git_runtime_state()`의 두 subprocess argv[0]은 상대 실행명
   `"git"`이었다. 로그 시각 뒤 현재 worktree에서는
   `GIT_BIN = Path("/usr/bin/git")`와 존재성 검사를 추가하고 두 호출을
   `str(GIT_BIN)`으로 교체했다.
7. `run_mv0()`의 공통 순서는 `easyedit_root` resolve, model spec 선택,
   `_git_runtime_state()`, Slurm identity 검사, fixed artifact preflight,
   model load 순이다. 따라서 Git subprocess는 model/GPU allocation보다
   앞선다.
8. `/usr/bin/git`과 `/usr/local/bin/uv`는 현재 executable이다.

## 원인 재현

무환경에서 `uv run --offline --project /mnt/raid5/janghj/EasyEdit
--frozen --no-sync python`을 실행해 Python이 받은 환경을 확인했다.

```text
PATH=/mnt/raid5/janghj/EasyEdit/.venv/bin
subprocess.run(("git", "--version"), ...): FileNotFoundError
```

같은 환경에서 argv[0]만 `/usr/bin/git`으로 바꾸면 return code `0`이다.
또한 `PATH=""`인 Python에서도 상대 `"git"`은 `FileNotFoundError`,
절대 `/usr/bin/git`은 성공했다.

단순히 `PATH` 변수가 완전히 없다는 것만으로는 충분한 설명이 아니다.
Python은 PATH가 unset이면 기본 search path를 사용할 수 있다. 이번
재현에서 결정적인 연결고리는 `uv run`이 child Python의 PATH를
EasyEdit venv 하나로 구성했고 그 디렉터리에 `git`이 없다는 점이다.

## 대안 평가

### 채택: Git 절대 경로 고정

- 변경: `GIT_BIN = Path("/usr/bin/git")`, 실행 전 `is_file()` 검사,
  두 subprocess에서 절대 경로 사용
- 장점: 변경 범위가 가장 작고 deterministic하다. Slurm environment를
  넓히지 않으며 두 model에 동일하게 적용된다.
- 잔여 조건: 재시도 전 `/usr/bin/git` 존재, 수정 code의 test, clean
  tracked commit을 확인해야 한다.

### 비권장: `SLURM_EXPORT_ENV=ALL` 또는 `srun --export=ALL`

- 이 방식도 system PATH를 child에 전달할 수 있다.
- 그러나 parent/user environment 전파 범위를 넓혀 `--export=NONE`의
  격리 의도를 훼손하고 credential/config contamination 면적을 키운다.
  한 개 실행 파일 lookup 문제에 비해 수정 범위가 크다.

### 차선: 고정된 최소 PATH를 명시적으로 export

- 예: runtime wrapper에서 venv, `/usr/local/bin`, `/usr/bin`, `/bin`만
  포함하는 PATH를 구성한다.
- 향후 다른 외부 executable이 필요하면 유용하지만, 현재 실패를 고치는
  데는 절대 Git 경로보다 범위가 넓다. 별도 필요가 확인되기 전에는
  채택하지 않는다.

### 비권장: Git 정보를 환경변수로 주입하거나 `.git`을 직접 파싱

- launcher와 runner 사이에 새로운 trust boundary가 생기거나 Git
  worktree semantics를 불완전하게 재구현하게 된다.
- 현재 clean-commit gate를 유지하는 데 불필요하게 복잡하다.

## 대안 원인 배제

- GPU OOM/resource exhaustion: model load 전 공통 경로에서 종료했고
  제공된 MaxRSS와 fs/disk failure가 이를 지지하지 않는다.
- model별 snapshot 또는 tokenizer 문제: 서로 다른 두 model이 같은
  error type으로 거의 동시에 조기 종료했으며 model loader보다 Git
  검사가 앞선다.
- output collision: 두 run ID가 다르고 output directory 자체가
  생성되지 않았다.
- EasyEdit root 부재: exact EasyEdit project를 대상으로 `uv run`이
  Python을 시작했으므로 주원인과 맞지 않는다.

이 배제는 원시 traceback을 보존하지 않은 상태의 분석이다. 다만
실행 당시 상대 Git 호출, exact 무환경 `uv run` 재현, 두 child의 동일
증상이 함께 성립하므로 다른 가설보다 설명력이 충분히 높다.

## 최소 재시도 조건

다음을 모두 만족한 뒤 Llama/Qwen pair를 다시 한 allocation에서
동시 제출한다.

1. `/usr/bin/git` 절대 경로 수정이 tracked commit에 포함되어 있다.
2. `_git_runtime_state()` 단위 검증에서 PATH가 venv-only 또는 빈
   환경이어도 `rev-parse HEAD`와 tracked-status 확인이 성공한다.
3. runner의 기존 필수 CPU test와 shell syntax check가 통과한다.
4. worktree가 clean이고 `main == origin/main`이다.
5. 실패한 `*_c3_v1` output은 없다는 현재 사실을 유지하되, durable
   marker 재사용을 피하도록 새 run ID/job identity를 사용한다.
6. 재시도도 GPU 2개, model당 GPU 1개, `130000M`, nested concurrent
   child라는 기존 audited resource envelope를 유지한다.

재시도 성공 판정은 model별 `planned/attempted/pass = 3/3/3`, exact
rollback 및 fidelity 기준 충족으로만 내린다. job 15508은 Motivation
signal이나 ODE-Edit 기대 개선폭에 대한 음성 결과로 집계하지 않는다.
