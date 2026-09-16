"""Attach generated figures and seal only new publication namespaces."""
import argparse
import json
from pathlib import Path
from .common import *

def figures(source):
    names=['cake-family-comparison.png','family-cumulative.png','generated-plot-manifest.json']
    for name in names:
        target=CAKE/'figures'/name
        with target.open('xb') as f:f.write((source/name).read_bytes())

def seal(checks,tests):
    validation=json.loads(checks.read_text());assert validation['status']=='CPU_PUBLICATION_SCOPE_NUMERIC_AND_GFM_HTML_PASS'
    test=json.loads(tests.read_text());assert test['returncode']==0
    save(AUDIT/'checks.json',validation);save(AUDIT/'tests.json',test)
    sources=[ref(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.md')]
    outputs=[]
    for root in (CAKE,CAP,INDEX):
        report=root/('README.md' if root==INDEX else 'diagnostic-report-ko.md')
        prior=CAKE_OLD if root==CAKE else CAP_OLD if root==CAP else BASE/'completed-experiments-review-2026-09-16-v1'
        manifest=dict(instruction_id=TASK,source_head=git('rev-parse','HEAD'),source_tree=git('rev-parse','HEAD^{tree}'),
            publication_type='REPORT_SEPARATION_ONLY_NO_SCIENCE_RECOMPUTATION',source=sources,
            prior_package=str(prior.relative_to(WT)),prior_receipt=ref(prior/'rooted-receipt.json'),
            mapping=ref(root/'source-to-v2.json') if root!=INDEX else 'LINKS_ONLY_NO_MIXED_SCIENTIFIC_TABLE',
            members=[ref(p) for p in sorted(root.rglob('*')) if p.is_file()],
            checks=ref(AUDIT/'checks.json'),tests=ref(AUDIT/'tests.json'),new_GPU_seconds=0,new_scheduler_queries=0,
            v1_unchanged=True,numerical_validation=(
                'GPU_CONTINUATION_NOT_TESTED_NO_NEW_NUMERICAL_CLAIM' if root==CAKE else
                'NOT_ESTABLISHED' if root==CAP else 'NOT_APPLICABLE_LINK_INDEX'),automatic_resume=False)
        save(root/'analysis-manifest.json',manifest)
        receipt=dict(instruction_id=TASK,status='REPORT_RESTRUCTURING_COMPLETE_PENDING_MAIN_PUBLICATION',
             report=ref(report),manifest=ref(root/'analysis-manifest.json'),v1_unchanged=True,
             new_GPU_seconds=0,new_model_forwards=0,scheduler_queries=0,automatic_resume=False)
        save(root/'rooted-receipt.json',receipt);outputs.append(ref(root/'rooted-receipt.json'))
    common=dict(instruction_id=TASK,status='PUBLICATION_READY',reports=outputs,new_GPU_seconds=0,
                monitoring_active=False,automatic_resume=False,numerical_validation_change=False)
    save(WT/'tasks/status/odeedit_cake_baseline_cap_report_separation_s4_20260916_v1/server4.json',common)
    save(WT/'runs/odeedit_cake_baseline_cap_report_separation_s4_20260916_v1/receipt.json',common)
    write(WT/'messages/server-heads/server4/2026-09-16-cake-baseline-cap-report-separation.md',
        '# CAKE/baseline 및 alpha-cap 독립 정본 v2\n\n'
        '두 family 보고와 링크 전용 새 목록을 실제 생성했다. CAKE는 W0/native/BLUE pair·physical-layer singleton만, alpha-cap은 네 정책과 동일1k 참고 baseline만 포함한다. 원 수치/분모와 v1 bytes 불변. GFM HTML의 실제 표 cell/링크를 검사했고 새 명칭의 baseline 그림은 코드 생성했다.\n\n'
        '새 scheduler/GPU/model/evaluator/raw tensor/Slurm/rsync/delete0. 검증된 source/report-only publication을 non-force main에 게시 후 TASK_COMPLETE_STOP한다. Rooted receipts는 각 completed-review-v2 및 목록에 있다.')
    return common

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('figures');f.add_argument('--source',type=Path,required=True)
    s=sub.add_parser('seal');s.add_argument('--checks',type=Path,required=True);s.add_argument('--tests',type=Path,required=True)
    a=p.parse_args();print(figures(a.source) if a.action=='figures' else seal(a.checks,a.tests))
