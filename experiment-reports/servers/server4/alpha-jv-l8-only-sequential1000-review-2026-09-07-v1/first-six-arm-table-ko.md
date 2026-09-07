# Final W10 전체 1,000 요청: O / JV / L8-only

각 arm의 최종 materialized W10에서 동일 sealed 1,000 requests 전체를 평가했다. Current B100/online 합계가 아니다. RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL; tie 실패. Cross-host end-to-end descriptive 비교이며 동일 W/M/z 대조가 아니다.

|Model|Arm|RS|PS|strict PS|NS|
|---|---|---:|---:|---:|---:|
|llama3-8b-inst|O_NATIVE|1000/1000 (100.00%)|1910/2000 (95.50%)|927/1000|7584/10000 (75.84%)|
|llama3-8b-inst|JV_NATIVE|923/1000 (92.30%)|1690/2000 (84.50%)|786/1000|7259/10000 (72.59%)|
|llama3-8b-inst|L8_ONLY_NATIVE|918/1000 (91.80%)|1702/2000 (85.10%)|797/1000|7214/10000 (72.14%)|
|qwen2.5-7b-inst|O_NATIVE|992/1000 (99.20%)|1887/2000 (94.35%)|900/1000|6978/10000 (69.78%)|
|qwen2.5-7b-inst|JV_NATIVE|997/1000 (99.70%)|1894/2000 (94.70%)|910/1000|7377/10000 (73.77%)|
|qwen2.5-7b-inst|L8_ONLY_NATIVE|995/1000 (99.50%)|1934/2000 (96.70%)|939/1000|7335/10000 (73.35%)|

O/JV: sealed Server2 Git publication 재해시. L8: completed Server4 public/raw scalar를 재계산. Remote Server2 raw/checkpoint 재해시와 paired prompt transition은 미수행. 세부 무결성 결과는 별도 integrity receipt에 결속한다.
