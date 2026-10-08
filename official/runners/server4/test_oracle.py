"""CPU wiring only; mocks never constitute actual oracle/resume evidence."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from official.evaluation import cf_native_reference as reference
from official.evaluation.reduce import counterfact
from official.experiments.prepare import write_new, read
from official.runners.server4 import oracle, tracking
from official.runners.server4.observe import scores
from official.runners.server4.qualify import plan
from official.runners.server4.resume_check import compare as compare_resume
from official.tracking.schema import metrics


class BindingTests(unittest.TestCase):
    def test_exact_shared_source_not_GPU_PASS(self):
        value=oracle.source_binding()
        self.assertEqual(value['original_bytes'],7941)
        self.assertEqual(value['actual_GPU'],'NOT_OBSERVED')

    def test_unchanged_canonical_identity_and_separate_actual_state_binding(self):
        canonical={'identity':{'external_identity':{'original':'do-not-relabel'}}}
        old=copy.deepcopy(canonical)
        state=SimpleNamespace(model=object(),tokenizer=object(),signature=lambda:{'weights':'locked'})
        success={'status':'PASS','evidence':{reference.SMOKE_SCOPE:'PASS'}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(oracle,'locks',return_value=({'model':'locked'},{'tokenizer':'locked'})), patch.object(reference,'compare_native_counterfact',return_value=success) as call:
            path=Path(tmp)/'raw.json'
            oracle.compare(state,{}, {},[],canonical,[],path,reference.SMOKE_SCOPE)
            self.assertEqual(canonical,old)
            kwargs=call.call_args.kwargs
            self.assertIs(kwargs['identity'],canonical['identity']['external_identity'])
            self.assertEqual(kwargs['state_identity'],{'weights':'locked'})
            self.assertEqual(kwargs['state_callback'](),{'weights':'locked'})
            self.assertNotIn('test_only_cpu',kwargs)

    def test_mismatch_is_preserved_then_blocks(self):
        state=SimpleNamespace(model=object(),tokenizer=object(),signature=lambda:{'W':'x'})
        failure={'status':'MISMATCH','evidence':{reference.SMOKE_SCOPE:'FAIL'}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(oracle,'locks',return_value=({},{})), patch.object(reference,'compare_native_counterfact',return_value=failure):
            path=Path(tmp)/'raw.json'
            with self.assertRaisesRegex(ValueError,'CF_NATIVE_ORACLE_NOT_PASS:MISMATCH'):
                oracle.compare(state,{}, {},[],{'identity':{'external_identity':{}}},[],path,reference.SMOKE_SCOPE)
            self.assertEqual(read(path),failure)

    def test_cpu_scope_cannot_enter_production_wrapper(self):
        with self.assertRaisesRegex(ValueError,'SCOPE_NOT_PLANNED'):
            oracle.compare(None,{}, {},[],{},[],None,reference.CPU_SCOPE)

    def test_qualification_budget_and_original_smoke_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for method in ('ALPHAEDIT','ALPHAEDIT_BLUE','SPHERE'):
                cfg=root/(method+'.json');write_new(cfg,dict(method=method,dataset='cf'))
                steps=plan('/python',cfg,root/'assets',root/'ready',root/'output')
                self.assertEqual(len(steps),3)
                self.assertEqual([s['argv'][s['argv'].index('--stop-after')+1] for s in steps],['3','2','3'])
                self.assertEqual(sum('--resume' in s['argv'] for s in steps),1)
                self.assertEqual(sum('--oracle-smoke' in s['argv'] for s in steps),int(method=='ALPHAEDIT'))

    def test_resume_requires_actual_restore_link_before_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,b=root/'a',root/'b'
            write_new(b/'resume-qualification-resume-b3.json',dict(start_batch=1))
            write_new(b/'batch-02-commit.json',{})
            with self.assertRaisesRegex(ValueError,'ACTUAL_B2_RESTORE_LINK_REQUIRED'):
                compare_resume(a,b)
            with self.assertRaisesRegex(ValueError,'INDEPENDENT_COLD_CHAINS_REQUIRED'):
                compare_resume(a,a)

    def test_shared_display_reducer_logger_incompatibility_is_reproduced(self):
        # Exact source's already-published 2000-row rounding regression fixture.
        # A passing reproduction test documents a BLOCKER, not logging readiness.
        rng=np.random.default_rng(20261009)
        for _ in range(58):counts=rng.integers(0,11,size=2000)
        good,bad={'target_new':1.,'target_true':2.},{'target_new':2.,'target_true':1.}
        cases=[dict(rewrite_prompts_probs=[good],paraphrase_prompts_probs=[good],
                    neighborhood_prompts_probs=[bad]*int(n)+[good]*(10-int(n))) for n in counts]
        summary=counterfact(cases)
        cfg=tracking.config(dict(run_id='llama3-cf-alphaedit',method='ALPHAEDIT',dataset='cf'),
            dict(generation_identity_sha256='a'*64),dict(code_commit='b'*40,config_sha256='c'*64),'cpu-fixture')
        with self.assertRaisesRegex(ValueError,'OFFICIAL_DISPLAY_SCORE_MISMATCH'):
            metrics(scores({'summary':summary},'all_seen/post',20),scientific=True,config_values=cfg)


if __name__=='__main__':unittest.main()
