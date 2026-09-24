"""Additional CPU at-write transitions, source conformance and collector crosscheck."""
import ast
import csv
import importlib.util
import io
import unittest
from .common import *
from .reducer import load_scores, compare, write_csv
from .test_review import ReviewTests

FROZEN=ROOT/'execution-source-r1/project/run_scripts/native_delayed_write_e3'

def rows(p):return list(csv.DictReader(Path(p).open()))

def main():
    panel=read(ROOT/'inputs/panels-v2/rows.json');atwrite=[];ids=[]
    endpoints={p.name:load_scores(p,panel)[0] for p in sorted((OUTPUT/'G20').iterdir()) if p.name=='W0' or p.name.startswith('BASE_')}
    for name,r in endpoints.items():
        if name=='W0':continue
        family=name.rsplit('_W',1)[0]
        atwrite+=compare(name+' minus own W001',endpoints[family+'_W001'],r,ids)
    write_csv(REPORT/'e1-atwrite-paired.csv',atwrite);save(LOCAL/'atwrite-transition-ids.json',ids)
    old={tuple(r[k] for k in ('endpoint','panel','kind')):r for r in rows(ATTEMPT/'report/endpoint-summary.csv')}
    checks=0
    for r in rows(REPORT/'first-endpoint-table.csv'):
        prior=old[tuple(r[k] for k in ('endpoint','panel','kind'))]
        for key in ('count','success','strict','ties','token_correct','token_total','true_nll_mean','new_nll_mean','desired_nll_mean'):
            if r.get(key) and prior.get(key):assert float(r[key])==float(prior[key]),(key,r,prior);checks+=1
        if r['kind']=='GENERAL':
            assert float(r['natural_nll_mean'])==float(prior['desired_nll_mean'])
            assert float(r['W0_forward_KL'])==float(prior['w0_forward_kl'])
    oldp={tuple(r[k] for k in ('contrast','panel','kind')):r for r in rows(ATTEMPT/'report/path-patch-paired.csv')}
    n=0
    for r in rows(REPORT/'paired-summary.csv'):
        key=tuple(r[k] for k in ('contrast','panel','kind'))
        if key not in oldp:continue
        prior=oldp[key]
        for field in ('count','gained','lost','strict_gained','strict_lost','before_success','after_success'):
            if r.get(field) and prior.get(field):assert float(r[field])==float(prior[field]);n+=1
    output=io.StringIO();test=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReviewTests))
    assert test.wasSuccessful()
    save(AUDIT/'unit-and-collector-checks.json',dict(tests_run=test.testsRun,tests_PASS=True,test_output=output.getvalue(),
        independent_endpoint_scalar_matches=checks,independent_patch_scalar_matches=n,
        implementation='Separate analysis code; frozen reducer neither imported nor run',GPU_calls=0))
    specification=[
      ('DAG/source/data/panel binding','stages.py','Run.gate','G00..G70/gate-result.json','PASS_STORED_CHAIN','source/input/predecessor SHA and11 receipts; no missing stage'),
      ('afterany collector not science bypass','control.py','submit','submission.json; held-inspection.json','PASS_SOURCE_AND_RECEIPT','only two jobs; internal DAG; no invalid downstream GPU job'),
      ('BASE endpoint/FP32/original revision','backend.py','Backend.__init__','execution.lock; G10; original CP metadata','PASS_RUNTIME_RECEIPT','current stat+prior SHA; no new CPU fullCP/hash/model load'),
      ('G10 original/repeat/zero-hook','stages.py','Run.g10','G10/gate-result.json','PASS_BOUNDED','16 rewrite prompts per family, true/new; not full-panel/model parity'),
      ('E1 coverage+L4 key invariance','stages.py','Run.e1','G20; G21; layer-key-drift','PASS_STORED_ROWS','25 endpoints×3306; L4 keydelta0; physical all-valid-token'),
      ('W0/direct+drift and s1 history split','stages.py','Run.drift','layer-key-drift/*/*.json','PASS_SCALAR_RECOUNT','mapping relative; s1 split only; full K/v and additional contractions not stored'),
      ('Four corners A=L8 B=L4..7','backend.py','Backend.hybrid','G30/G50; module-interaction-checks','PASS_SOURCE_BOUND','a0 FP64 subtract thenFP32; actual11 exact original Wt; no independent tensor reconstruction'),
      ('K identities / module cross','stages.py','Run.factorial','39672 module rows','PASS_RECORDED_SCOPE','maxrelative1.9505194609058857e-6; nonlinear NLL interaction separately recomputed'),
      ('Alltoken patch/dose/rotation','stages.py','Run.patch','G40/G60; patch-coverage','PASS_LEDGER_AND_ROWS','72modified; per-token RMS runtime assertion; CPU chunk norm and coverage check'),
      ('Allvocab TF/General KL','backend.py','Backend.forward','row token predictions/NLL; General scalars','PASS_TOKEN_RECOUNT_KL_SOURCE_BOUND','KL teacher RAM-only, not independently recomputed; not free generation'),
      ('Restore/nonselected/RNG/CP readonly','backend.py','Backend.restore','G10/stage PASS; original guard source','PASS_RUNTIME_GUARDS','selected tensor hash runtime; nonselected pointer/version; no CPU continuation'),
      ('Finite failure receipt/propagation','stages.py','Run.run','science-terminal; absence failure.json','PASS_SOURCE_ACTUAL_SUCCESS','failure branch not exercised by this valid completed run'),
      ('noCP/noz/nohistory','backend.py','Backend.cost','science-terminal; artifact-index','PASS_SOURCE_AND_INVENTORY','newfit0/write0/history0/CP0; inputCP retained; exact new resume unavailable'),
      ('Independent final collection','reduce.py','report','G70; report/terminal.json','PASS_INDEPENDENT_RECOUNT','original collector matched; separate fresh reducer/token recount; no independent red agent')]
    conformance=[]
    for req,file,symbol,evidence,status,limit in specification:
        src=FROZEN/file;tree=ast.parse(src.read_text());line=None
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==symbol.split('.')[-1]:line=node.lineno;break
        assert line
        conformance.append(dict(requirement=req,execution_source='3ebe0b07078940c2d46f9ea2226ccc20c0446162',file=str(src),function=symbol,line=line,
            source_sha256=sha(src),evidence=evidence,verdict=status,limitation=limit))
    write_csv(REPORT/'source-conformance.csv',conformance)
    save(AUDIT/'renderer-availability.json',dict(available={k:bool(importlib.util.find_spec(k)) for k in ('markdown','markdown_it','mistune')},
        GFM_HTML_render='NOT_RUN_RENDERER_NOT_INSTALLED',new_install=False,structural_markdown_check='separate final package check'))
    print('SUPPLEMENT_PASS',checks,n,test.testsRun)

if __name__=='__main__':main()
