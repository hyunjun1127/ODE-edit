# PRE_EDIT full10k / BLUE lifelong v3 CPU 통합 검산

instruction_id: ODEEDIT-S06-FIXED10K-PREEDIT-BLUE-LIFELONG-REPORT-INTEGRATION-SH2-V1

작성 Server2/SH2, 2026-09-09 KST. 판정 `PASS_WITH_DOCUMENTED_WARN`, scientific_promotion=false.
별도 모델/evaluator/Slurm 실행 없이 독립 reducer와 fixture/산술 회귀로 검사했다. 독립 실행 agent의 검토라고 주장하지 않는다.

## 읽기·ownership·source 경계

- 최초/최종 pre-push fetched origin/main: `4d70b8b8c7ce2fcfdc1513618b0ce25386dc0aaf`, tree `fcb59adfffba93fddf745051e52ed6cfc55df625`.
- isolated `codex/server2-fixed10k-preedit-blue-report-v1`, worktree `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-fixed10k-preedit-blue-report-v1`.
- PROTOCOL full read 1269 lines /60382B/SHA `af806a449be800251393bfcd81b2dfa5689ee34305f3bf1323e0fae82f16c87b`. Session boundary `01a0493a-074c-7f91-9a13-769116326fef` PASS. Git agent `head-server2-sh2`/`server2`.
- 원문 create-once local `local/fixed10k-preedit-blue-report/20260909-v1/authoritative-instruction.txt`, mode0600, SHA `654760711a769aeff741acd551798f6d43b8222e36bfce130b3b83e0b01643d0`. UTF-8 텍스트와 terminal newline을 저장; 별도 byte-transfer identity가 지정된 입력은 아니며 새로운 과학 gate로 취급하지 않는다.
- 기존 root의 untracked `agents/server2/` 보존. Shared EasyEdit/env/source/cache 수정0. 기존 source12ad4cbe 및 raw output 불변. 이번 main 범위는 신규 CPU 분석 코드와 명시 승인된 Server4 v3 통합 report package이며 downstream branch를 합치지 않는다.

## 완료·raw·denominator 검사

42673 한 번의 bounded accounting 확인: owner janghj, name odeedit_fixed10k_preedit_s2, COMPLETED0:0, elapsed01:14:00, 1GPU/8CPU/59G. `scontrol` live record는 purged(Invalid job id specified); 이를 terminal failure로 해석하지 않고 sacct와 exact runtime/lock/source binding을 사용했다. 신규/다른 job 조회 없음.

- 실행 source `12ad4cbe417d9935b1e7e55600214d14ba0d6360` / tree `be4ee8bede7e4e962fd13064814bd2fbd500b56a`.
- execution.lock SHA `7ea4991018fb18b1cb4dc520acf3e6adb386ad39112440fa2f57af9e33bdf167`.
- 입력1777 source/environment/model members 16123553429B, output1404 members 228945537B full SHA 재검산. terminal 자체 SHA 별도. 공개 CSV에는 source1777, output+terminal1405 path/size/SHA만 보존.
- fixed dataset10000 unique, prompt130000, new/true raw target rows260000. Dataset 문장/target/case/index 순서와 identity exact, raw→part100개→full endpoint 모든 reducer행/bitorder/numerator/denominator 일치.
- RS791/10000, PS1997/20000, NS89212/100000. ties/nonfinite/missing/duplicate/imputation0. NLL을 logits로 다시 계산한 것이 아니라 source-defined 저장 mean-token NLL의 independent preference reduction.
- secondary token bit는 token_predictions==target_token_ids로 독립 재검산; strict=all token bits. Pair-strict와 구분.
- selected5 W0 bytes/pointer/version 및 전체 parameter pointer/version/requiresgrad100parts guard. 비선택 전체 model bytehash NOT_CLAIMED. edit/compute_z/optimizer/backward/key/solve/history append0.

## BLUE와의 교차 검산

- v2 report SHA `65645e52fb19f6aa4ac30ee00ce36334c5c3a9e22ed1ce92c142ed9c251239b8`와 manifest/member root를 재검산.
- SH4가 허용·재확인한 완료6개의 B100/seen-full.json만 원격 read-only CPU pairing. 총399737324B SHA/size를 읽기 전후 재검사. model/evaluator/Slurm/remote file-write0.
- W0 scalar identity/NLL/bit 15433954B를 메모리로 전달, aggregate response1258221B. Prompt/token/tensor 및 대형 raw는 전송하지 않았다. raw 자동전파 불필요 예외를 paired receipt에 명시.
- case/index/prompt-target identity/order exact780000 pair observations. 전체18행 및 cohort1800행의 retained/lost/gained/both_failed 분할, before−lost+gained=after, denominator/pp 검산 PASS.
- 1757 common source/model assets SHA 일치, safetensors4shards/config/tokenizer/index/evaluator 포함. 6 BLUE entry selected W0 key SHA와 PRE_EDIT5keys의 해당부분 일치.
- **WARN:** A6000 vs Blackwell forward numerical parity 미검증; kernel microbatch16은 같지만 W0 100×B100, BLUE final past9900+current100 호출묶음은 다르다. evaluator add_bos runtime NOT_EXPOSED이며 source/tokenizer identity 수준만 확인. source 또는 데이터 mismatch로 은폐하지 않고 report에 명시.

## 회귀·출판 검사

`/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest -q project.run_scripts.fixed10k_preedit_blue_report.test_focused project.run_scripts.blue_lifelong_analysis.test_revision project.run_scripts.blue_lifelong_analysis.test_focused`: **31 tests PASS**(신규15+기존16). System python3에는 matplotlib가 없어 기존plot관련2module import가 실패했으며 기존 설치 venv로 재실행해 PASS; env 설치/변경0, scientific code 수정0. System python3 신규15 및 package verify도 PASS.

v2 파일77/CSV47/PNG10 byte-identical, 6chain 원래 모든 column/value/order exact. W0 availability 문장만 명시된 5종7곳 갱신; 기존 본문전체와 원본 자체도 보존. W0한 번을 두 family표에 참조, 전역7state표에는 중복W0 없음. BASE full10k NOT_AVAILABLE 유지.

새 PNG는 source Python matplotlib 실행2회 SHA exact 및 시각검토 PASS. 기존 PNG 재생성0. imagegen/visualize0. 새 output105파일/16944802B, 원래v2 evidence 포함, raw-free CSV61개 및 PNG11개. Full report v3 source/member rooted rehash와 relative image access PASS. Binary weights/log/prompt/tensor/cache Git 추가0. 신규 model/GPU/evaluator/edit/Slurm mutation0, downstream42706/기존baseline/L567/중지ORBODE 조회·변경0.

CSV는 Python csv writer 및 원래 v2의 CRLF bytes를 보존했다. 기본 diff checker가 CR을 trailing whitespace로 표시하므로 봉인 CSV를 LF로 변환하지 않았다. 명령 단위 `git -c core.whitespace=trailing-space,space-before-tab,cr-at-eol diff --cached --check`에서 **0 issues/PASS**; 영구 Git 설정 변경0, 내용/수치/manifest 변경0. 이 처리는 CRLF line ending의 문법적 허용이지 일반 whitespace gate 해제가 아니다.

## Canonical v3

`experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v3/`

- factual-report-ko.md SHA `839a903839743ad6de7c8cb08d255e7ad2e2c43099f6b81b27a3c91483fb991f`
- analysis-manifest.json SHA `c1cb564175c606681f266a0ec53c4b2f77ebcf28cf49583b50c8ffcb0604ac26`
- rooted-receipt.json SHA `3f27b486859e183d381a28211500ab069d39ddeb4187db2a31ce7721bfb3c16d`

Main非force publication 허용. 충돌 시중단/force0. 이 감사는 scientific promotion 또는 다른 task 재개 권한이 아니다.
