"""Independent reducer/accounting fixture tests; no live science or scheduler."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import collect
from .common import TASK,NONCE,digest,write
from project.run_scripts.jlz_realized_writer_sequential.review_completed import Reader,reduce_rows


def raw(case,kind,index,endpoint,new=1.,true=2.,new_correct=1,true_correct=2):
    identity=digest([case,kind,index])
    return dict(case_id=case,kind=kind,prompt_index=index,identity=identity,endpoint=endpoint,
        active_at_endpoint=True,new_nll=new,true_nll=true,new_token_count=2,true_token_count=3,
        new_token_correct=new_correct,true_token_correct=true_correct,new_strict=new_correct==2,
        true_strict=true_correct==3,new_token_identity='n'+identity,true_token_identity='t'+identity,
        margin_true_minus_new=true-new,margin_new_minus_true=new-true)


class CollectTests(unittest.TestCase):
    def test_counts_distinguish_native_CAKE_and_AlphaBLUE(self):
        self.assertEqual(collect.EXPECTED['CAKE']['native_z'],100)
        self.assertEqual(collect.EXPECTED['ALPHAEDIT_BLUE']['native_z'],200)
        self.assertEqual(collect.EXPECTED['ALPHAEDIT_BLUE']['history_appends'],2)
        self.assertEqual(collect.EXPECTED['ALPHAEDIT_BLUE']['history_keys'],2)

    def test_independent_n_desired_true_tie_failure_token_micro(self):
        values=[raw(1,'N',0,'W1',new=1,true=1,new_correct=2,true_correct=1),
                raw(2,'N',0,'W1',new=3,true=1,new_correct=2,true_correct=3)]
        reduced=reduce_rows(values)['N']
        self.assertEqual(reduced['numerator'],1)
        self.assertEqual(reduced['desired_token_correct'],4)
        self.assertEqual(reduced['desired_token_count'],6)
        self.assertEqual(reduced['strict_numerator'],1)

    def test_endpoint_reduction_exact_order_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);state=dict(W={'13':'w'},H={'13':'h','17':'h'})
            values=[raw(7,'R',0,'W1')]+[raw(7,'P',i,'W1') for i in range(2)]\
                +[raw(7,'N',i,'W1') for i in range(10)]
            write(folder/'chunk-0000.json',dict(state=state,rows=values,optimizer_feedback=False))
            write(folder/'summary.json',dict(endpoint='W1',state=state,requests=1,row_count=13,
                row_order=digest([r['identity'] for r in values]),summary=reduce_rows(values),
                seconds=1.,no_mutation=True,optimizer_feedback=False))
            result=collect.endpoint(Reader(),folder,values,[7],'W1',state)
            self.assertEqual(result['summary']['N']['denominator'],10)
            with self.assertRaisesRegex(RuntimeError,'STATE|SCOPE'):
                collect.endpoint(Reader(),folder,values,[7],'W1',dict(W={},H={}))

    def test_accounting_only_exact_two_parents_separates_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp);lock=dict(source_commit='a'*40,owner='owned')
            write(attempt/'submission.json',dict(instruction_id=NONCE,task_id=TASK,
                source_commit='a'*40,jobs=dict(CAKE='123',ALPHAEDIT_BLUE='124')))
            calls=[]
            def runner(argv,**kwargs):
                calls.append(argv)
                return SimpleNamespace(returncode=0,stdout=
                    '123|'+TASK+'-CAKE|owned|FAILED|1:0|9|gres/gpu=1|\n'
                    '124|'+TASK+'-ALPHAEDIT_BLUE|owned|COMPLETED|0:0|10|gres/gpu=1|\n')
            result=collect.allocation_once(Reader(),attempt,lock,runner=runner,owner='owned')
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][calls[0].index('-j')+1],'123,124')
            self.assertEqual(result['status'],'RECORDED')
            self.assertEqual([r['allocated_GPU_seconds'] for r in result['records']],[9,10])
            self.assertEqual(result['records'][0]['scheduler_state'],'FAILED')

    def test_missing_runtime_remains_partial_without_zero_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=dict(packs=[dict(ids=list(range(i*100,(i+1)*100))) for i in range(20)])
            result=collect.review_arm(Reader(),Path(tmp),c,{},'CAKE',[],[])
            self.assertEqual(result['scientific_status'],'PARTIAL_OR_NOT_VERIFIED')
            self.assertEqual(result['metric_rows'],[])
            self.assertEqual(result['missing'],['RUNTIME_NOT_RECORDED'])


if __name__ == '__main__': unittest.main()
