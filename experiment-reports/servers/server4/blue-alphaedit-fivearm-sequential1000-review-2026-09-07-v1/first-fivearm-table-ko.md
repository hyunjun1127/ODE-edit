# Llama 동일 JVP 표본 — 최종 W10 전체1,000 첫 비교표

|arm|state|RS|PS|NS|rewrite_acc|rephrase_acc|
|---|---|---|---|---|---|---|
|BLUE|FINAL_W10_ALL1000|997/1000 (99.70%)|1939/2000 (96.95%)|8057/10000 (80.57%)|997/1000|1347/2000|
|BLUE_L4_ONLY|FINAL_W10_ALL1000|998/1000 (99.80%)|1943/2000 (97.15%)|8072/10000 (80.72%)|995/1000|1347/2000|
|BLUE_L8_ONLY|FINAL_W10_ALL1000|996/1000 (99.60%)|1837/2000 (91.85%)|6936/10000 (69.36%)|991/1000|1291/2000|
|JVP|FINAL_W10_ALL1000|923/1000 (92.30%)|1690/2000 (84.50%)|7259/10000 (72.59%)|899/1000|1121/2000|
|JVP_L8|FINAL_W10_ALL1000|918/1000 (91.80%)|1702/2000 (85.10%)|7214/10000 (72.14%)|897/1000|1148/2000|
|O_NATIVE|FINAL_W10_ALL1000|1000/1000 (100.00%)|1910/2000 (95.50%)|7584/10000 (75.84%)|996/1000|1436/2000|

RS/PS=new NLL<true NLL; NS=true NLL<new NLL; tie failure. Acc는 teacher-forced all-target-token prompt accuracy. Current pooling 아님. O/JVP는 sealed Git publication 재해시, 나머지4개는 local NLL pair 독립 재계산. Source/config/context가 달라 method-only 인과비교 아님. L4 사전 gate는 SKIPPED_USER_DIRECTED. L4 observer P index metadata=4이나 runtime allp[[0]]; 별도 CPU projector 검산 예정.
