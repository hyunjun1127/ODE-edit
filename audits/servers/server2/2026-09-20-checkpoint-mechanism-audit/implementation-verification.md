# 구현·CPU 검증 인계 (GPU 미검증)

Instruction `ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1`.
2026-09-20 최신 사용자 “gpu 자리가 없으니 구현까지만 완료하고 user의 호출 기다려”를 적용한다.
Status: `IMPLEMENTATION_COMPLETE_AWAITING_USER`; 전체 과학 task 완료가 아니다.

## 구현 범위

정본 read/identity, create-once 선택 수신, pinned original dependency/source import,
원 기록 독립 reducer, actual W/M geometry, native capture·C00/C01 gate,
key bank/F00, history/operator/native-demand/counterfactual lane, all-token
activation lane, raw-free Korean report/CSV/PDF/PNG builder, source lock/archive,
held inspection/resource admission 코드가 존재한다. 원 writer/evaluator/shared env 수정0.

새 model checkpoint·W/M/resume/delta 저장0, 새 z/편집/history append0.
소스/CPU 검사만으로 실제 Llama precision/성능/복원 parity를 통과했다고 하지 않는다.

## 최종 CPU checks

```bash
PYTHONPATH=/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1/deps-transformers-4.44.2 \
PYTHONDONTWRITEBYTECODE=1 /mnt/raid5/janghj/EasyEdit/.venv/bin/python \
  -m unittest discover -s project/run_scripts/checkpoint_mechanism_audit -t . -p 'test_*.py' -q
```

108 tests PASS (2.699s). 이 수는 synthetic/CPU/source-import fixtures이며 model load0.
원 context[1,5]의 .5/.1 key 평균, tokenizer/import/version, source hash,
FP64 actual-delta/열별·요청별 최대오차, zero-reference, repeat10%, prefix early-stop,
nonselected pointer/version, create-once/pause admission guard, nonsymmetric LU,
thin-SVD energy, counterfactual20, full-token signed VJP/packing, report dependency를 포함한다.
초기 unittest discover에서 `-t .`를 빠뜨려 relative import 4건이 실패했으며
명령을 수정한 뒤 같은 source tests가 통과했다. 이를 actual gate 실패로 세지 않는다.

두 신규 `run.sbatch`/`analysis.sbatch`: memory audit checked2/failures0,
각1GPU/8CPU/60416M/exportNONE/no-requeue. `bash -n`, compile 및 diff checks PASS.
전체 repository audit의 이전 server4 launcher 6개 현 정책 초과는 unrelated historical source로
남겨 두었다. 본 task를 위해 타 source의 자원 요청을 수정하지 않았다.

## 사용자 override 이전 실제 완료한 독립 CPU 작업

- Input source-map:2304 members; 2084 reuse,220 new/450763288B; W/M transfer0.
  source-map SHA202a1bf65ddce6fb031294254bab7c69f0761a2fbfed3e2e9c0130fedc1213b0.
- A01: current100/seen12,837200 dedup observations,130000 at-write anchors,
  duplicate15600 exact. 2000 cluster resamples(각request와subject-relation) 완료.
  receipt SHA8756cea928cc159c9eb9c32f8cf148ac77952a57b5cabaef5d6b05119ca63ac3,
  CPU wall517.221661s. B100 원 기록 RS9939/10000,PS19136/20000,NS65348/100000.
- B00: W0 expected tensor hash, 기존12CP full SHA/size/W/M·commit identity PASS,
  total12683997116B. 13상태/11구간 공통256-vector sketch. CPU128.346s,
  hash15.292s, peakRSS9580608KiB. receipt SHAdce0e872d38144b43dc4ac4447c323f598500ce322cfdc2592630064cdb2e6df.

위 기존 CPU 산출물은 불변으로 보존했다. 사용자 override 후 새로운 과학 분석/그림/최종 보고를
추가 실행하지 않았다. Implementation tests의 tiny synthetic arithmetic와 구분한다.

## 실행·publication identity 분리

Gate source HEAD3f65d1705ff69fe1d53c8b5e30313265889d5c2a,
tree9587d5673e927cd1ec94d9d66d16c43f71e838ac.
Execution lock SHA f06da6977695e69a90655db0315788d472369691524ccc04f1626d79540936de.
source archive: local/checkpoint-mechanism-audit/20260920-v1/attempt-v1/execution-gate-r1/.
후속 구현 source의 token provenance/asset guard/runner는 그 frozen archive와 다르며
held job을 hot-patch하지 않았다. 당시 CPU hash function의 header는 원 helper와 일치한다.

Job51071은 held inspection 후 release되었으나 node GPU8/8·host admission 부족으로
PENDING 상태였다. 최신 USER 지시에 따라 정확 own job을 다시 hold했다.
최종 단발 관측은 PENDING/Reason=JobHeldUser, actual GPU gate 미실행이다.
그 뒤 scheduler polling·release·새 submit0, 다른job 변경0.
receipt: local/checkpoint-mechanism-audit/20260920-v1/attempt-v1/receipts/user-implementation-only-20260920.json.

## 남은 실제 검증

C00/C01 physical Llama/key/writer/evaluator parity, full model asset read verification,
C02 actual key bank, O/N/E/F/G GPU 계산, 비용 계측, 최종 H1–H4 판단 및 report/plot 재현,
전체 scope main 통합은 미완료다. GPU 허용오차는 고정하며 CPU PASS를 대신 쓰지 않는다.
소스 내 `verify_model_assets`는 동일 stat-bound full-hash receipt를 재사용하거나 실제 로딩 전에
hash를 수행하도록 구현했으며 이번 pause 뒤 real model bytes를 새로 읽었다고 주장하지 않는다.

Recall 후 shared B1 prerequisite, 두 pilot lanes(0/10 vs1/100), extension histories의
single-writer partition 및 두 suffix 구간을 cap2 이내에서 실행한다. 실제 비용을 먼저 측정하며
8h reservation을 추정 완료시간으로 말하지 않는다. 기존 held job 및 source 처리도 USER recall
범위에서만 한다. 자동 daemon/heartbeat/cascade 없음. 현재는 source branch만 게시하고 대기한다.

독립 분담은 archival/geometry/numerical gate·activation 구현에 한정했다. 별도 red agent는
CPU 수식·source 검토만 수행했으며 실제 model red PASS는 없다. 최종 owner는 SH2다.
Raw/prompt/checkpoint/log는 local-only, NO_BROADCAST_NOT_REQUIRED.
