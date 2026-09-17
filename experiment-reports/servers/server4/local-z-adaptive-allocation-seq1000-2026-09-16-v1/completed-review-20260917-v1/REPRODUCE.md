# CPU 재현과 evidence 범위

Python `/data/janghj/EasyEdit/.venv/bin/python`; 실행 모듈이 아닌 아래 analysis만 호출한다. 원 output은 read-only다.

```bash
python -B -m unittest project.run_scripts.local_z_adaptive_allocation.analysis.test_analysis -v
python -B -m project.run_scripts.local_z_adaptive_allocation.analysis.reduce --attempt analysis-new-recall
python -B -m project.run_scripts.local_z_adaptive_allocation.analysis.tensors --attempt tensor-new-recall --reuse-inventory /data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1/review-20260917-v1/tensor-audit/raw-inventory.json
python -B -m project.run_scripts.local_z_adaptive_allocation.analysis.plot --data REPORT_DIRECTORY --out NEW_FIGURE_DIRECTORY
```

실제 이번 metric 출력은 analysis-r2, tensor는 tensor-audit, supplemental은 supplement-v1이다. bootstrap의 scheduler 조회는 이미1회 완료했으므로 재현시 실행하지 않는다. `audit`/`publish`는 create-once 경로를 사용하므로 원폴더가 있으면 종료한다. 새허용reviewnamespace에서만 경로를 정해 재실행한다. 보고 구성은 publish.py이며 기존CSV를복사/집계하고raw수치를변경하지않는다.

Raw inventory 재사용은 fullSHA가 같다는 새확인이 아니라 과거이검토의fullSHA+현재size 재사용이다. 독립 fullSHA를 다시 원하는 경우 reuse-inventory를빼되불필요중복해시를기본요구하지않는다. 모델·teacher기존immutable자산은priorfullSHA+stat재결속이며새모델검증이아니다. figures는matplotlib코드로PNG재생성후byteSHA대조했다. 분석코드수정전실패analysis-v1은보존되며 runtime/science 재실행0.
