"""Review-package validation, not experimental acceptance gates."""
from .review import *
from .package import plot
import ast
import importlib.util
import re
import subprocess
import sys
import unittest
from io import StringIO
from .test_review import ReviewTest

def run():
    # CSV newline formatting is analysis-owned, not a change to original raw.
    for p in PACKAGE.glob('*.csv'):
        assert b'\r\n' not in p.read_bytes(),('REVIEW_CSV_NOT_LF',p)
    first=read(PACKAGE/'first-table-receipt.json');first['table_sha256']=sha(PACKAGE/'first-actual-table.csv');first['joint_sha256']=sha(PACKAGE/'first-joint-table.csv')
    write_json(PACKAGE/'first-table-receipt.json',first)
    inv_receipt=read(PACKAGE/'output-inventory-receipt.json');inv_receipt['sha256']=sha(PACKAGE/'output-inventory.csv');write_json(PACKAGE/'output-inventory-receipt.json',inv_receipt)
    stream=StringIO();tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReviewTest))
    assert tests.wasSuccessful()
    (SCRATCH/'cpu-tests.txt').write_text(stream.getvalue())
    report=PACKAGE/'report-ko.md';text=report.read_text();tables=[];lines=text.splitlines();i=0
    while i<len(lines):
        if lines[i].startswith('|'):
            block=[]
            while i<len(lines) and lines[i].startswith('|'):block.append(lines[i]);i+=1
            widths=[len(x.strip().strip('|').split('|')) for x in block]
            assert len(set(widths))==1,('GFM_COLUMN_MISMATCH',widths)
            tables.append(dict(rows=len(block)-2,columns=widths[0]))
        else:i+=1
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if '://' in target:continue
        p=report.parent/target.split('#')[0];assert p.exists(),('MISSING_LINK',target)
        links.append(dict(target=target,exists=True))
    image=PACKAGE/'actual-endpoints-and-geometry.png';before=sha(image);plot();assert sha(image)==before
    files=list(Path(__file__).parent.glob('*.py'))
    for p in files:ast.parse(p.read_text())
    inv=list(csv.DictReader((PACKAGE/'output-inventory.csv').open()))
    for r in inv:
        s=Path(r['realpath']).stat();assert s.st_size==int(r['bytes']) and s.st_mtime_ns==int(r['mtime_ns'])
    source=list(csv.DictReader((PACKAGE/'frozen-source-members.csv').open()));assert all(r['matches']=='True' for r in source)
    cov=list(csv.DictReader((PACKAGE/'family-coverage.csv').open()));assert sum(r['status']=='COMPLETED' for r in cov)==94
    assert sum(r['status']=='FOLLOWUP_NOT_SUBMITTED' for r in cov)==7
    metric=list(csv.DictReader((PACKAGE/'all-endpoint-metrics.csv').open()));assert len(metric)==1608
    for r in metric:
        assert int(r['numerator'])<=int(r['denominator'])
        assert abs(float(r['percent'])-100*int(r['numerator'])/int(r['denominator']))<1e-10
    for p in PACKAGE.iterdir():
        assert p.suffix in ('.md','.csv','.json','.png'),('UNEXPECTED_GIT_ARTIFACT',p)
    result=dict(status='PASS_SCOPED_CPU_REVIEW_VALIDATION',tests=tests.testsRun,failures=0,
        tests_output_sha256=sha(SCRATCH/'cpu-tests.txt'),source_AST_parse=len(files),source_member_SHA_matches=len(source),
        raw_output_stat_unchanged_members=len(inv),metric_rows=len(metric),coverage_completed=94,followups_not_submitted=7,
        GFM_table_structure=tables,relative_links=links,PNG_reproduction_SHA256=before,PNG_visual_inspection='OWNER_TOOL_VIEW_PERFORMED',
        HTML_GFM_render='NOT_RUN_RENDERER_NOT_INSTALLED',browser_screenshot='NOT_RUN',
        independent_agent_red=False,review_type='owner source audit plus independent reducer implementation',
        GPU_model_evaluator_calls=0,Slurm_mutations=0,original_runtime_modifications=0,
        analysis_only_repairs=['raw-schema path mapping calibration512/KR','event duplicate-key parser','test fixture list alias'],
        raw_free='no prompt/tokens/weights/teacher/fullstdout in Git package; compact hashes/case aggregates only')
    write_json(PACKAGE/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(list,dict))},indent=2))

if __name__=='__main__':run()
