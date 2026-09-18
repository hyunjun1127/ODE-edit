# EN-F sequential 완료 리뷰 재현

정본: [diagnostic-report-ko.md](diagnostic-report-ko.md). Exact job50071_1만 CPU로 분석한다.
필수 local source/raw를 소유한 server4에서 실행한다. 다른 arm raw나 GPU를 읽지 않는다.
새 output은 본 package와 review scratch뿐이며 원 raw를 수정하지 않는다.

```bash
cd /data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/S/completed-en-f-review-r1/worktree
export CUDA_VISIBLE_DEVICES=''
REVIEW_SCRATCH=/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/S/completed-en-f-review-r1
PY=/data/janghj/EasyEdit/.venv/bin/python
PKG=project.run_scripts.single_layer_edit_preserving_correction.analysis
$PY -B -m "$PKG.enf_sequential_review" --scratch "$REVIEW_SCRATCH" --phase first
$PY -B -m "$PKG.enf_sequential_review" --scratch "$REVIEW_SCRATCH" --phase metrics
$PY -B -m "$PKG.enf_tensor_audit" --scratch "$REVIEW_SCRATCH"
$PY -B -m "$PKG.enf_review_package" --scratch "$REVIEW_SCRATCH" --phase provenance
$PY -B -m "$PKG.enf_review_package" --scratch "$REVIEW_SCRATCH" --phase supplement
$PY -B -m "$PKG.enf_review_package" --scratch "$REVIEW_SCRATCH" --phase plots
$PY -B -m "$PKG.enf_write_report" --scratch "$REVIEW_SCRATCH"
$PY -B -m unittest "$PKG.test_enf_review" -v
```

tensor audit는 CPU weights_only/mmap으로 현재 저장된 tensor를 확인한다. GPU continuation이 아니다.
전체 tensor/teacher/model/prompt/원 log는 local-only다. Full K_E 및 finalizer selected K 미보존 한계는 본문/coverage에 명시했다.
그림은 metrics-summary.json/tensor-summary.json에서 코드로 생성한다. 소수점 본문 표시는 반올림이며 CSV가 full-precision 정본이다.
publication 검사/manifest 생성은 enf_publication 모듈을 사용한다. HTML 렌더는 system python3의 markdown_it을 사용했다.
원 envelope 및 이전 FULL_READ exact identity는 input-manifest에 결속한다.
