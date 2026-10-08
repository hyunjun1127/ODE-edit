"""CPU schedule/commit fixtures; no native fitting or GPU certification."""
import ast
import contextlib
import copy
import inspect
import io
import random
import unittest

import torch

from . import generation_final_run as final
from . import generation_run as native
from .test_generation_run import config, fixture


def final_config():
    value=config();value['generation']={'evaluation_schedule':final.SCHEDULE}
    return value


def execute(arm,**kwargs):
    parts=fixture(arm,**kwargs)
    model,engine,view,bench,records,generation,ops,*_=parts
    observed=[];committed=[];stages=[]
    original=ops.observe
    def observe(*args,**kw):
        observed.append((args[5],len(args[3]),model.number,model.pruned))
        return original(*args,**kw)
    ops.observe=observe
    with contextlib.redirect_stdout(io.StringIO()):
        terminal=native.execute_chain(final_config(),{'source_commit':'fixture-source'},
            '/mock/'+arm,arm,model,None,view,engine,bench,records,generation,object(),
            ops=ops,on_commit=lambda r:committed.append(copy.deepcopy(r)),on_stage=stages.append)
    return terminal,parts,observed,committed,stages


class FinalScheduleTests(unittest.TestCase):
    def test_six_arms_twenty_native_writes_rpn_unchanged_one_final_generation(self):
        for arm in native.ARMS:
            with self.subTest(arm=arm):
                terminal,parts,rpn,commits,stages=execute(arm)
                model,engine,view,bench,records,generation,ops,written,logged,txs=parts
                self.assertEqual(engine.applies,list(range(1,21)))
                self.assertEqual(engine.counts,{k:v*20 for k,v in native.expected_counts(arm).items()})
                self.assertEqual(len(rpn),40)
                self.assertEqual([r[1] for r in rpn[::2]],[100]*20)
                self.assertEqual([r[1] for r in rpn[1::2]],
                    [100 if i not in native.MILESTONES else i*100 for i in range(1,21)])
                self.assertEqual(generation.loads,0)
                self.assertEqual(generation.subsets,[])
                self.assertEqual(len(generation.endpoints),1)
                self.assertEqual(generation.endpoints[0]['endpoint'],'W20')
                self.assertEqual(generation.endpoints[0]['count'],2000)
                self.assertEqual(generation.endpoints[0]['number'],20)
                self.assertEqual([p[0] for p in logged],['all_seen/post'])
                self.assertEqual(logged[0][2:],(2000,1900,2000))
                self.assertEqual(terminal['generation_endpoints'],1)
                self.assertEqual(terminal['generation_W0_endpoints'],0)
                self.assertEqual(terminal['generation_intermediate_endpoints'],0)
                self.assertEqual(len(commits),20)
                for receipt in commits:
                    self.assertEqual(receipt['generation_schedule'],final.SCHEDULE)
                    self.assertFalse(receipt['generation_available'])
                    self.assertFalse(receipt['checkpoint_saved'])
                    self.assertFalse(any(k.startswith('gen_') for k in receipt))
                self.assertTrue(all(tx.done and not tx.rollback_verified for tx in txs))
                actual=written['/mock/'+arm+'/generation-final.json']
                self.assertEqual(actual['requests'],2000)
                self.assertEqual(actual['model_state'],commits[-1]['after'])
                self.assertEqual(actual['summary']['planned_count'],2000)
                self.assertEqual(stages[-2:],['FINAL_W20_GENERATION','FINAL_W20_GENERATION_COMPLETE'])

    def test_prune_terminal_transform_precedes_final_rpn_and_generation(self):
        terminal,parts,rpn,commits,stages=execute('PRUNE')
        model,engine,view,bench,records,generation,*_=parts
        self.assertEqual(engine.terminal_calls,1)
        self.assertEqual(rpn[-1],('W20',2000,20,True))
        self.assertTrue(generation.endpoints[0]['pruned'])
        self.assertLess(stages.index('TERMINAL_PRUNE_BASE_FIX'),stages.index('B20_POST_RPN'))
        self.assertLess(stages.index('B20_POST_RPN'),stages.index('FINAL_W20_GENERATION'))

    def test_final_failure_preserves_all_twenty_commits_no_native_rollback(self):
        for mode in ('exception','rng','state'):
            parts=fixture('BASE_MEMIT',mutation={'rng':'rng:W20','state':'W20'}.get(mode))
            model,engine,view,bench,records,generation,ops,written,logged,txs=parts
            saved=random.getstate();committed=[]
            if mode=='exception':
                def fail(*args,**kw):raise OSError('FINAL_GENERATION_FAILURE_FIXTURE')
                generation.endpoint=fail
            try:
                with contextlib.redirect_stdout(io.StringIO()),self.assertRaises((OSError,RuntimeError)):
                    native.execute_chain(final_config(),{'source_commit':'fixture-source'},
                        '/mock/BASE_MEMIT','BASE_MEMIT',model,None,view,engine,bench,records,
                        generation,object(),ops=ops,on_commit=committed.append)
                self.assertEqual(len(committed),20)
                self.assertEqual(len(engine.applies),20)
                self.assertTrue(all(tx.done and not tx.rollback_verified for tx in txs))
                self.assertIn('/mock/BASE_MEMIT/batch-20/commit.json',written)
                self.assertNotIn('/mock/BASE_MEMIT/generation-final.json',written)
                self.assertEqual(logged,[])
            finally:random.setstate(saved)

    def test_commit_io_failure_still_rolls_back_native_and_never_generates(self):
        parts=fixture('BASE_ALPHAEDIT',fail_commit=True)
        model,engine,view,bench,records,generation,ops,written,logged,txs=parts
        with self.assertRaises(OSError):
            native.execute_chain(final_config(),{'source_commit':'fixture-source'},
                '/mock/BASE_ALPHAEDIT','BASE_ALPHAEDIT',model,None,view,engine,bench,records,
                generation,object(),ops=ops)
        self.assertEqual(engine.applies,[1]);self.assertEqual(model.number,0)
        self.assertTrue(txs[-1].rollback_verified)
        self.assertFalse(generation.endpoints)

    def test_schedule_explicit_no_scoring_duplication_or_checkpoint_calls(self):
        source=inspect.getsource(final);tree=ast.parse(source)
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls&{'save','savez','savez_compressed','save_pretrained','dump'})
        self.assertEqual(source.count('generation.endpoint('),1)
        self.assertNotIn('generation.load_W0(',source)
        self.assertNotIn('generation.subset',source)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
