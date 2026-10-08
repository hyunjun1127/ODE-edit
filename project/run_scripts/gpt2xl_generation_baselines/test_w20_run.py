"""Task-private production CPU-loop fixtures; no model/GPU/native fitting."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import w20_run
from .test_run import View, Engine, Generation, summary


def records():
    return [dict(case_id=i,generation_prompts=['text'],requested_rewrite=dict(
        prompt='{} works',subject='s',target_new={'str':'new','id':'n'})) for i in range(2000)]


class Tests(unittest.TestCase):
    def test_locked_verifies_new_session_policy_transition_and_source_reference_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=root/'plan.json';plan.write_text('{}')
            keys=('session_boundary_member','transition_receipt','source_reference','source_reference_lock')
            config=dict(instruction_id=w20_run.NONCE,task_id=w20_run.TASK,
                source_configs={arm:{} for arm in w20_run.ARMS},assets=[],runtime={},evaluator_sources=[],
                model='model',model_revision='revision',policy_members=[{'path':'policy'}],
                generation=dict(assets_manifest_member={'path':'assets'},
                    qualification_plan_member={'path':str(plan)},qualification_plan_sha256=w20_run.digest({})))
            config.update({key:{'path':key} for key in keys})
            lock=dict(instruction_id=w20_run.NONCE,task_id=w20_run.TASK,
                source_commit='s',config_sha256='c',native_closure=[{'path':'native'}])
            (root/'config.json').write_text(w20_run.json.dumps(config))
            (root/'execution.lock.json').write_text(w20_run.json.dumps(lock))
            verified=[]
            def verify(item):
                verified.append(item['path']);return Path(item['path'])
            with patch.object(w20_run,'validate_config'),patch.object(w20_run,'sha',return_value='c'),\
                patch.dict(w20_run.os.environ,{w20_run.SOURCE_ENV:'s'}),\
                patch.object(w20_run,'verify',side_effect=verify),\
                patch.object(w20_run.subprocess,'check_output',return_value='revision\n'):
                w20_run.locked(root)
            self.assertTrue(set(keys)|{'policy'} <= set(verified))

    def test_three_actual_production_loops_only_generation_after_terminal_transform(self):
        rows=records()
        chunks=lambda rs:((index+1,rs[index*100:(index+1)*100],rs[:(index+1)*100]) for index in range(20))
        for arm in w20_run.ARMS:
            view=View();engine=Engine(view,arm);bench=SimpleNamespace(contexts=[['{}']])
            rpn=[];generation_calls=[]
            def observe(view,bench,all_records,selected,H,endpoint,out,current_ids):
                rpn.append((endpoint,len(selected),len(current_ids),engine.pruned))
                return dict(summary=dict(requests=len(selected)),current=dict(requests=len(current_ids)))
            with tempfile.TemporaryDirectory() as tmp,patch.object(w20_run,'batches',chunks),\
                patch.object(w20_run,'rows',return_value=[]),patch.object(w20_run,'observe',observe),\
                patch.object(w20_run,'log_batch',return_value=True),patch.object(w20_run,'log_generation',return_value=True),\
                patch.object(w20_run.gc,'collect'),patch('builtins.print'):
                generation=Generation(Path(tmp)/'raw')
                generation.qualification_member={'path':'CPU fixture only','bytes':1,'sha256':'a'*64}
                original=generation.observe
                def capture(*args,**kwargs):
                    generation_calls.append(dict(edits=100*(engine.next_batch-1),pruned=engine.pruned,
                        endpoint=args[1],record_count=len(args[0]),state=copy.deepcopy(kwargs['state_identity'])))
                    return original(*args,**kwargs)
                generation.observe=capture
                config=dict(packs=[dict(ids=list(range(index*100,(index+1)*100))) for index in range(20)])
                commits=[]
                final=w20_run.native_loop(config,dict(source_commit='a'*40),Path(tmp),arm,rows,
                    view.model,view,engine,bench,SimpleNamespace(),commits,generation)
                self.assertEqual(engine.calls,list(range(1,21)))
                self.assertEqual(len(commits),20)
                self.assertEqual(generation.calls,[('W20',2000,False)])
                self.assertEqual(generation_calls[0]['edits'],2000)
                self.assertEqual(generation_calls[0]['pruned'],arm == 'PRUNE')
                self.assertEqual(generation_calls[0]['state'],w20_run.generation_state(final))
                self.assertEqual([value['batch'] for value in commits if value['generation_measured']],[20])
                self.assertTrue(all(value['generation'] is None for value in commits[:19]))
                self.assertEqual(commits[-1]['generation']['summary']['planned_count'],2000)
                self.assertEqual([count for name,count,_,_ in rpn if name in ('W5','W10','W15','W20')],
                                 [500,1000,1500,2000])
                self.assertTrue(all(current == 100 for _,_,current,_ in rpn))
                self.assertFalse((Path(tmp)/'generation-W0.json').exists())
                self.assertTrue((Path(tmp)/'generation-W20.json').exists())
                self.assertEqual([path.name for path in (Path(tmp)/'generation-work').glob('*.json')],['W20_POST.json'])

    def test_generation_failure_preserves19commits_and_rolls_back_only_final_uncommitted_state(self):
        rows=records();view=View();engine=Engine(view,'PRUNE');bench=SimpleNamespace(contexts=[['{}']])
        chunks=lambda rs:((index+1,rs[index*100:(index+1)*100],rs[:(index+1)*100]) for index in range(20))
        commits=[]
        with tempfile.TemporaryDirectory() as tmp,patch.object(w20_run,'batches',chunks),\
            patch.object(w20_run,'rows',return_value=[]),\
            patch.object(w20_run,'observe',return_value=dict(summary={},current={})),\
            patch.object(w20_run,'log_batch',return_value=True),patch.object(w20_run.gc,'collect'),patch('builtins.print'):
            generation=SimpleNamespace(observe=lambda *args,**kwargs:(_ for _ in ()).throw(ValueError('W20-generation-original')))
            config=dict(packs=[dict(ids=list(range(index*100,(index+1)*100))) for index in range(20)])
            with self.assertRaisesRegex(ValueError,'W20-generation-original'):
                w20_run.native_loop(config,dict(source_commit='a'*40),Path(tmp),'PRUNE',rows,
                    view.model,view,engine,bench,SimpleNamespace(),commits,generation)
            self.assertEqual(len(commits),19)
            self.assertEqual(engine.calls,list(range(1,21)))
            self.assertEqual(w20_run.state(view,engine.history()),commits[-1]['after'])
            self.assertFalse(engine.pruned)
            self.assertEqual(engine.next_batch,20)
            self.assertFalse((Path(tmp)/'batch-20/commit.json').exists())
            self.assertFalse((Path(tmp)/'generation-W20.json').exists())

    def test_w20_progress_axis_and_explicit_sdk_rejection_not_remote_success(self):
        seen=[];tracker=SimpleNamespace(log=lambda payload:seen.append(payload) or False)
        payload={'phase':'generation_evaluation','generation_progress/step':1,
            'generation_progress/completed_cases':4,'generation_progress/total_cases':2000}
        with tempfile.TemporaryDirectory() as tmp:
            accepted=w20_run.log_generation_progress(tracker,payload,Path(tmp))
            self.assertFalse(accepted)
            self.assertEqual(seen[0]['phase'],'W20_generation')
            self.assertNotIn('edits',seen[0]);self.assertNotIn('fit/global_candidate',seen[0])
            receipt=w20_run.json.loads((Path(tmp)/'generation-progress-transport/step-1.json').read_text())
            self.assertEqual(receipt['status'],'LOGGING_DEGRADED')
            self.assertFalse(receipt['accepted'])
        with self.assertRaisesRegex(RuntimeError,'PROGRESS_ONLY'):
            w20_run.log_generation_progress(tracker,dict(payload,edits=2000))

    def test_generation_final_summary_requires2000_and_only_all_seen_post(self):
        values=[];tracker=SimpleNamespace(log=lambda payload:values.append(payload) or True)
        observed=dict(summary=summary(2000),identity=dict(endpoint='W20',cohort='ALL_SEEN',
            ordered_occurrences=list(range(1,2001))),RNG_restored=True,observer_no_mutation=True)
        self.assertTrue(w20_run.log_generation(tracker,observed))
        self.assertEqual(values[0]['edits'],2000)
        self.assertEqual(values[0]['pre_state_edits'],1900)
        self.assertEqual(values[0]['post_state_edits'],2000)
        self.assertTrue(all(key.startswith('all_seen/post/') or key in
            {'edits','pre_state_edits','post_state_edits','batch'} for key in values[0]))
        observed['summary']=summary(1999)
        with self.assertRaisesRegex(RuntimeError,'COMPLETE_BEFORE_LOG'):
            w20_run.log_generation(tracker,observed)

    def test_fresh_per_arm_runtime_qualification_no_shared_W0_reader(self):
        generation=dict(schedule=w20_run.SCHEDULE,qualification_plan_member={},qualification_plan_sha256='p',
            qualification_receipts={},source_identity='source',model_identity='fixed')
        config=dict(generation=generation)
        for arm in w20_run.ARMS:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/arm;generation['qualification_receipts'][arm]=str(out/'qualification/qualification-actual.json')
                actual=dict(member=dict(path=generation['qualification_receipts'][arm],bytes=1,sha256='a'*64),
                    selected_route='UNPADDED_KV_SINGLETON',fixed_microbatch=1,route_results={'fixed':'not scientific quality'})
                with patch.object(w20_run,'verify',return_value=Path(tmp)/'plan.json'),\
                    patch.object(w20_run.json,'loads',return_value={}),patch.object(Path,'read_text',return_value='{}'),\
                    patch.object(w20_run,'digest',return_value='p'),patch.object(w20_run,'run_qualification',return_value=actual) as qualify,\
                    patch.object(w20_run,'verify_actual_receipt',return_value=actual),patch.object(w20_run,'write'):
                    result=w20_run.bind_runtime_generation(config,arm,None,None,None,None,None,None,out)
                    qualify.assert_called_once()
                    self.assertEqual(qualify.call_args.kwargs['out'],out/'qualification')
                    self.assertEqual(result['generation']['generation_route'],'UNPADDED_KV_SINGLETON')
                    self.assertNotIn('shared_W0_root',result['generation'])
        generation['shared_W0_root']='forbidden'
        with self.assertRaisesRegex(RuntimeError,'NO_W0_GENERATION_DEPENDENCY'):
            w20_run.bind_runtime_generation(config,'PRUNE',None,None,None,None,None,None,Path('/fixture'))


if __name__=='__main__':unittest.main()
