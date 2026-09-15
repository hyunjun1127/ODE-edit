# Server4 종료 실험 종합 사실 리뷰

Instruction `ODEEDIT-S06-SERVER4-COMPLETED-CAKE-CAP-DETAILED-REVIEW-SH4-V1`. 최신main7ef4fc05에서독립worktree를만들었으며sharedroot와sweep a7bfd2f의dirty pause2파일을보존했다.

CAKE48101과CAP10/100/NORM_ONLY48148/48149/48150은한정scheduler조회에서모두COMPLETED0:0이며각actualterminal도확인했다. 이번에NOT_COMPLETED또는신규실행실패로남은대상은없다. 과거EP기술실패는그대로실패이며CAP1수치미검증상태도유지한다.

| 대상 | job | 범위 | 상태 |
| --- | --- | --- | --- |
| CAKE | 48101 | W0_B100x100_10k | COMPLETED_0_0_NEW_CPU_REVIEW |
| CAP10 | 48148 | W0_B100x10_1k | COMPLETED_0_0_NEW_CPU_REVIEW |
| CAP100 | 48149 | W0_B100x10_1k | COMPLETED_0_0_NEW_CPU_REVIEW |
| NORM_ONLY | 48150 | W0_B100x10_1k | COMPLETED_0_0_NEW_CPU_REVIEW |
| CAP1 EP-TW-1 | 47962 | PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT | COMPLETED_REUSED_NUMERICAL_NOT_ESTABLISHED |
| BLUE/native14chains + W0 | 39307/39283_1..5/40441..46/42657/42658/42673 | PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT | SEALED_COMPLETE_REUSE |
| BLUE/L4/L8 canonical6 | 39307/39283_1..5 | PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT | SEALED_COMPLETE_REUSE |
| BLUE/JVP/O fivearm1k | 38940/38997/38988 | PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT | SEALED_COMPLETE_REUSE |
| JVP L8 takeover | 38433_4/5 | PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT | SEALED_COMPLETE_REUSE |
| lowcost static core6 | 46451 | PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT | SEALED_COMPLETE_REUSE |
| lowcost seq10 sixarm | 46475_0..5 | PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT | SEALED_COMPLETE_REUSE |
| write-refresh Middle | 47020_0..3 + reusedN4REFIT4 | PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT | SEALED_COMPLETE_REUSE |
| BG1 | 47592teacher only | PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT | CALIBRATION_MISSING_SCIENCE_NOT_SUBMITTED; NO_REAUDIT |
| EP original/repair technical | 47884/47942 | PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT | FAILED_PRIOR_RCA_REUSED; NO_REDIAGNOSIS |
| Checkpoint preservation | 183CP | PRIOR_SEALED_SCOPE_NOT_NEW_REAUDIT | STORAGE_TASK_COMPLETE_REUSE_NO_RAW_ACCESS |
| ORBODE cumulative | 37649 | LINK_ONLY | STOPPED_USER_REVIEW_RAW_DELETED_USER_AUTHORITY_NO_REAUDIT |

[CAKE 10k 상세 보고](../cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v1/diagnostic-report-ko.md)

[EP alpha cap sweep 1k 상세 보고](../ep-tw1-alpha-cap-sweep-2026-09-15-v1/completed-review-v1/diagnostic-report-ko.md)

CAKE W100:9840/10000,17755/20000,62935/100000. CAP1/10/100/NORM_ONLY W10:first1000의독립표는각보고서에있다. 서로다른규모/시점/W50warm을같은분모로합산하지않는다. CAKE B10의동일first1000는별도참고표다.

신규실행allocation CAKE32194초 + cap22933초 =55127GPU초(15.313056GPUh). CAP1재사용7694초/teacher98초/기존실패473+69초는신규지출이아니다. 네job start/endinterval상최대2GPU할당이며utilization은미측정이다. 리뷰새GPU0.

새30개capCP fullSHA/CPU shape/dtype/finite/headerbridge·27links·3000target/30native solve/30final history를검산했다. CAKE는100commit/99metadata links/10000target/500solve/500layerhistory이며W/M tensor미저장useroverride에따라복원가능성을주장하지않는다.

과거정본은위report/hash를재사용했다. 14chain whole10k 비교의Original은native/BLUE를명시적으로구분했다. source-backed설계동작·모든후보·반대문항전이·NLLtail·비용은각family보고와CSV에보존했다. 과학우열/인과기여/승격·후속선택은GH소유다.

CPU reviewer 수리: CAKE weight_state/cache_sha256/정수0 schema, NORM_ONLY strict-loss ID의문자열정렬vs숫자정렬(동일set3개),실행closure의cake_native_lifelong 단축namespace경로를반영했다. 실패한CPU분석namespace는보존했으며runtime·수치threshold·raw수정0. 현재cap lock의구형 numerical_rationale alpha1설명은실제numerical_policy와구분했다.

새Slurm/model/forward/evaluator/FD/ULP/teacher/baseline/rsync/delete0. 사용자pause를그대로보존한완료리뷰recall이며task완료후STOP/automatic_resume=false. 미측정Report/Audit/MMLU/FutureN·GPUreplay는이번에채우지않았다.
