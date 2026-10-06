# GPT2-XL stock EasyEdit baseline

사용자 승인 `USER-GH-SH1-GPT2XL-EASYEDIT-BASELINES-2K-20261007-R1`의 두 독립 cold first2000/BS100×20 경로입니다.

실제 로컬 EasyEdit의 `apply_memit_to_model` / `apply_AlphaEdit_to_model`을 직접 호출합니다. package shell은 무관 top-level optional import만 제외하며 17개 실제 imported source를 SHA/size로 결속합니다. fitting/writer 수식은 복사·대체하지 않습니다. MEMIT에는 edit history H가 없습니다. AlphaEdit의 module-global history는 첫 실제 apply에서만 초기화되고 이후 native 실행이 층마다 한 번 append합니다.

`prepare`는 기존 asset/stat/context/W0의 identity와 실제 YAML parser를 CPU에서 확인합니다. `run`은 편집 전 current 관측→native apply 1회→편집 후 관측 순서입니다. `collect`는 저장 per-case row를 독립 reducer로 집계하며 collector 성공과 20-batch scientific 완료를 구분합니다.

```bash
python -m project.run_scripts.gpt2xl_native_baselines.prepare --out <new-preparation> --attempt <new-attempt>
python -m project.run_scripts.gpt2xl_native_baselines.preflight --config <preparation>/config.json --out <new-cpu-receipt>
python -m project.run_scripts.gpt2xl_native_baselines.submit --config <cpu-receipt>/config-sealed.json --attempt <new-attempt>
```

실제 Slurm 등록은 SH1 owner만 수행합니다. 제출 전 config/source commit과 동결 archive를 생성하고 모든 job을 held 검사한 뒤 release합니다. 기존 writer-lane 말단의 afterany는 자원 순서이며 성능 gate가 아닙니다. 자동 retry/추가 fit/모니터는 없습니다.

W&B `fit/global_candidate`는 이미 끝난 native z-call의 누적 관측축입니다. batch의 `candidate`는 해당 호출의 request ordinal이며 native 내부 Adam candidate 횟수를 뜻하지 않습니다. 실제 내부 backward/update 횟수가 없으면 NOT_RECORDED입니다. 평가 지표는 edits축과 canonical scalar schema를 사용합니다.

W0는 동일 runtime/model/token/row identity의 기존 first2k 관측만 read-only reference로 재사용합니다. source method H0는 raw 그대로 보존하고 baseline 실제 H 상태를 별도의 model-weight projection receipt에 둡니다. 이는 edited-state resume가 아닙니다.

NoCP, Z disk cache 없음, `exact_resume=NOT_AVAILABLE`. RAM rollback만 허용됩니다. raw/credential/model/tensor/fullstdout은 Git/W&B 업로드하지 않습니다. source와 compact receipt만 Git에 게시합니다.
