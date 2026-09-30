# B010 final full evaluation report update

사용자가 `odeedit_joint_finaleval_s2`를 기존 B010 report에 합치고 JOINT_CUM 종료까지 확인하도록 명시 recall했다. 정확한 55331_0/1/2는 모두 COMPLETED/0:0, 각각 173/168/166 allocated GPU-seconds다. 다른 task 재개·새 제출·취소 없이 CPU 검산/보고만 수행했다.

|Arm|R100|P200|N1000|
|---|---:|---:|---:|
|NATIVE|99|190|744|
|JOINT_STEP|100|142|785|
|JOINT_CUM|100|143|786|

NATIVE 대비 NS lost/gained는 STEP 3/44, CUM 2/44. Same W0/entry NS1000이 없으므로 시간순 보존/회복으로 표현하지 않는다. 원 R/P NLL/strict/token은 재평가와 정확히 같았다. 300 request files/7800 candidate rows 및 terminal/restore/SHA 결속을 CPU 확인했고 source/runtime/raw는 변경하지 않았다. Source `62005204cf2b0a5abf5ef2830849bc9103199515`, lock `95d898d31c1f8f4fa41a7b8af74dbfbaf0771b5b19dc8fde266d80a8a1b338ef`.

갱신 report: `experiment-reports/servers/server2/joint-multilayer-bs1-20260929-v1/b010-review-20260930-v1/report-ko.md`

SHA256: `65a734727e45e7aca28ded62590649879053812506ed654b5d91d7b615af86b1`.

새 raw-free CSV/verification은 같은 report package의 `final-eval-update-r1/`, 새 publication/owner audit는 `audits/servers/server2/joint-multilayer-bs1-20260929-v1/b010-review-20260930-v1/final-eval-update-r1/`. 기존 CSV/PNG/manifest는 SHA 불변 확인했다. 초기 report SHA는 역사 manifest와 Git에 유지한다. CPU reducer tests 6 PASS; independent red agent 사용 주장은 없다. `NO_BROADCAST_NOT_REQUIRED`, `scientific_promotion=false`, 보고 게시 후 STOP.
