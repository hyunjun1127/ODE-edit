"""CPU control-flow checks only; no generation or scientific fixture fitting."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from .generation_schedule import generation_due,first_generation_batch
from . import run


class ScheduleTests(unittest.TestCase):
    def test_explicit_schedule_legacy_and_rejection(self):
        self.assertEqual([b for b in range(1,21) if generation_due({'generation':{'schedule':'W20_ONLY'}},b)],[20])
        self.assertEqual(first_generation_batch({'generation':{'schedule':'W20_ONLY'}}),20)
        self.assertTrue(all(generation_due({'generation':{}},b) for b in range(1,21)))
        with self.assertRaises(ValueError):generation_due({'generation':{'schedule':'WRONG'}},1)

    def test_twenty_batches_keep_rpn_only_final_generation(self):
        from project.run_scripts.jlz_price_gptj import tracking
        from . import generation_adapter
        from project.run_scripts.experiment_generation_eval import kv_qualification
        from project.run_scripts import experiment_generation_eval
        from project.run_scripts.llama3_native_baselines import generation,producer
        from project.run_scripts.jlz_interference_l1 import cap_tracking
        records=[{'case_id':i} for i in range(2000)]
        identity=dict(model='gptj',revision='test-revision',observation_identity='test-observer')
        state={'W':'unchanged'};writes=[];logged=[]
        def require(ok,reason):
            if not ok:raise RuntimeError(reason)
        original=Mock(return_value={'RPN':'preserved'})
        parent=SimpleNamespace(observer=original,require=require,state=lambda *a:state,
            rng_snapshot=lambda:0,rng_equal=lambda r:True,digest=lambda x:'digest',
            write=lambda path,value:writes.append((path.name,value)),guard=lambda *a:None,
            verify=lambda member:Path(member['path']))
        gen=Mock()
        gen.observe.return_value={'summary':{'count':2000}}
        gen.subset.side_effect=lambda observed,rows,name:{'summary':{'count':len(rows)}}
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            path=Path(tmp)/'plan.json';path.write_text(json.dumps({'model_identity':identity}))
            cfg=dict(model_alias='gptj',model_revision='test-revision',observation_identity='test-observer',
                generation=dict(schedule='W20_ONLY',reference_manifest={},asset_paths={},assets_sha256='assets',
                    profile='profile',qualification_plan={'path':str(path)}),
                storage={'next_batch_bytes':0},generation_reserve_bytes=0)
            stack.enter_context(patch.object(experiment_generation_eval,'load_assets',return_value=SimpleNamespace(sha='assets')))
            stack.enter_context(patch.object(generation_adapter,'GenerationObserver',return_value=gen))
            qualify=stack.enter_context(patch.object(generation_adapter,'run_qualification',return_value={'member':{}}))
            stack.enter_context(patch.object(kv_qualification,'verify_actual_receipt',return_value={
                'selected_route':'EQUAL_LENGTH_KV_BATCH','fixed_microbatch':4}))
            stack.enter_context(patch.object(generation,'normalize_summary',side_effect=lambda x:x))
            stack.enter_context(patch.object(generation,'endpoint_ref',side_effect=lambda x:x))
            stack.enter_context(patch.object(producer,'generation_values',side_effect=lambda prefix,s,n:{prefix+'/count':n}))
            stack.enter_context(patch.object(cap_tracking,'safe_log',side_effect=lambda t,f,label:logged.append(f())))
            original_log=stack.enter_context(patch.object(tracking,'log_batch'))
            a=SimpleNamespace(model=None,hook_signature=lambda:())
            bench=SimpleNamespace(tokenizer=None,contexts=[])
            run.attach_post_generation(parent,a,bench,records,{},cfg,Path(tmp),None,{'source_commit':'test'})
            for b in range(1,21):
                current=records[(b-1)*100:b*100];seen=records[:b*100]
                selected=seen if b in (5,10,15,20) else current
                args=(a,bench,seen,selected,{},'B'+str(b)+'_PRE',Path(tmp),cfg,[],[r['case_id'] for r in current])
                self.assertEqual(parent.observer(*args),{'RPN':'preserved'})
                args=list(args);args[5]='W'+str(b)
                self.assertEqual(parent.observer(*args),{'RPN':'preserved'})
                tracking.log_batch(None,{'batch':b})
                if b<20:
                    qualify.assert_not_called();gen.observe.assert_not_called();self.assertFalse(logged)
            self.assertEqual(original.call_count,40)
            self.assertEqual(original_log.call_count,20)
            qualify.assert_called_once();gen.observe.assert_called_once()
            self.assertEqual(len(gen.observe.call_args.args[0]),2000)
            self.assertEqual(gen.subset.call_count,2)  # W20 current100 / first500, CPU subsets only
            self.assertEqual(len(logged),1)
            self.assertEqual(logged[0]['edits'],2000)
            self.assertEqual(logged[0]['all_seen/post/count'],2000)
            self.assertEqual(sum(name=='generation-skipped.json' for name,_ in writes),19)

if __name__=='__main__':unittest.main()
