"""Native20/RPN schedule unchanged; one shared-text generation only at W20.

Generation is deliberately outside native RAM rollback transactions. A final
generation failure preserves all20 actual commits and their R/P/N receipts;
it does not reinterpret those committed writes as rolled back or completed
generation. No W0/current/milestone generation or old W0 reuse is performed.
"""
import time
from pathlib import Path

import torch

from . import generation_run as native
from .generation_common import digest, expected_counts, require, writer_identity


SCHEDULE='FINAL_W20_ONLY'


def execute_chain(config,lock,out,arm,model,tokenizer,view,engine,bench,records,
                  generation,tracker,*,ops=None,on_stage=None,on_commit=None):
    require(config['generation'].get('evaluation_schedule')==SCHEDULE,
            'FINAL_GENERATION_EXPLICIT_SCHEDULE')
    ops=native.production_ops() if ops is None else ops
    native_config=native.arm_configuration(config,arm)
    _,normalize=native.family_api(arm)
    out=Path(out);commits=[];cursor=[]
    def stage(value):
        if on_stage is not None:on_stage(value)
    stage('W0_RPN_REUSE')
    w0_state=ops.state(view,engine.history())
    w0=ops.install_W0(native_config,view,engine.history(),out,bench,records)
    w0raw=ops.rows(out/'W0',w0_state)
    ops.log_w0(tracker,w0['summary'])
    previous=ops.state(view,engine.history())
    engine.progress=native.make_progress(engine,tracker,arm,ops.safe_log)
    for number,current,seen in ops.batches(records):
        require(1<=number<=20 and len(current)==100 and len(seen)==number*100,
                'GENERATION_NATIVE_EXACT_20_PACKS')
        folder=out/f'batch-{number:02d}'
        ids=[record['case_id'] for record in current];started=time.monotonic()
        require(ops.state(view,engine.history())==previous,'GENERATION_BATCH_ENTRY_LINK')
        ops.reserve(out,config)
        normalized=normalize(current);pack_identity=digest(normalized)
        stage(f'B{number}_PRE_RPN')
        pre=ops.observe(view,bench,seen,current,engine.history(),f'B{number}_PRE',
                        folder/'pre',current_ids=ids)
        stage(f'B{number}_NATIVE_APPLY')
        with ops.transaction(view,engine,bench) as tx:
            returned,receipt_native=engine.apply(normalized,number)
            require(returned is model,'SAME_ACCUMULATED_NATIVE_MODEL')
            require(all(bool(torch.isfinite(weight).all()) for weight in view.weights.values())
                    and all(bool(torch.isfinite(value).all()) for value in engine.history().values()),
                    'NATIVE_NONFINITE_COMMIT')
            expected=expected_counts(arm)
            counts={field:receipt_native['delta'][field] for field in expected}
            require(counts==expected,'NATIVE_CALL_HISTORY_COUNTS')
            require(set(engine.history())==(set(view.sites) if arm in native.HISTORY_ARMS else set()),
                    'GENERATION_NATIVE_HISTORY_BOUNDARY')
            if number==20 and arm=='PRUNE':
                stage('TERMINAL_PRUNE_BASE_FIX')
                terminal=engine.terminal_prune()
                require(terminal.get('explicit_repair')==native.BASE_FIX
                        and terminal.get('prune_applied') is True,'PRUNE_EXPLICIT_TERMINAL_BASE_FIX')
                receipt_native.update(terminal_prune=terminal,prune_applied=True,
                                      explicit_repair=native.BASE_FIX)
            selected=seen if number in native.MILESTONES else current
            stage(f'B{number}_POST_RPN')
            post=ops.observe(view,bench,seen,selected,engine.history(),f'W{number}',
                             folder/'post',current_ids=ids)
            after=ops.state(view,engine.history())
            contexts=native.native_contexts(engine,arm)
            require(isinstance(contexts,list) and len(contexts)==2 and contexts[0]==['{}']
                    and len(contexts[1])==5,'GENERATION_NATIVE_CONTEXT_CARRIED_1_PLUS5')
            proposed_cursor=cursor+ids
            receipt=dict(task=config['task_id'],arm=arm,writer=writer_identity(arm),batch=number,
                case_ids=ids,source=lock['source_commit'],config=digest(config),
                before=previous,after=after,native=receipt_native,native_counts=counts,
                pre=pre['summary'],post=post['summary'],post_current=post['current'],
                post_scope='ALL_SEEN' if number in native.MILESTONES else 'CURRENT',
                seen_requests=len(seen),native_pack=pack_identity,ledger=digest(proposed_cursor),
                generation_schedule=SCHEDULE,generation_available=False,
                generation_unavailable_reason='FINAL_W20_ONLY_SCHEDULE',
                generation_observer_status='NOT_SCHEDULED_INTERMEDIATE',observer_no_mutation=True,
                seconds=time.monotonic()-started,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
            ops.write(folder/'native-context-identity.json',dict(context_sha256=digest(contexts),
                groups=[len(group) for group in contexts],generated_inside_native=True))
            tx.finish()
            try:ops.write(folder/'commit.json',receipt)
            except BaseException:
                tx.done=False
                raise
        commits.append(receipt);cursor,previous=proposed_cursor,after
        if on_commit is not None:on_commit(receipt)
        ops.log_batch(tracker,receipt,w0raw,ids,[r['case_id'] for r in seen])
        print(native.json.dumps(dict(event='native_final_generation_batch_committed',arm=arm,
            batch=number,edits=len(cursor),native_counts=counts,generation_schedule=SCHEDULE)),flush=True)
        ops.cleanup()
    require(len(commits)==20 and len(cursor)==2000,'GENERATION_FULL_20_BATCH_COMPLETION')
    # All native writes, PRUNE terminal transform and W20 R/P/N have committed.
    # The single generation endpoint may fail, but cannot roll back these writes.
    stage('FINAL_W20_GENERATION')
    raw_out=out/'generation-final'
    generated=native.guarded_generation(lambda:generation.endpoint(records,out=raw_out,
        endpoint='W20',model_state=previous,cohort_label='ALL_SEEN'),
        view,engine,arm,bench,ops,2000)
    final=native.generation_receipt(generated,records,'W20',previous,raw_out)
    final.update(task=config['task_id'],arm=arm,source=lock['source_commit'],config=digest(config),
        generation_schedule=SCHEDULE,edits=2000,actual_model_edits=2000,
        generation_observer_no_mutation=True,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    ops.write(out/'generation-final.json',final)
    ops.log_generation(tracker,'all_seen/post',generated['summary'],2000,1900,2000)
    stage('FINAL_W20_GENERATION_COMPLETE')
    return dict(status='COMPLETED',completed_batches=20,commits=20,edits=2000,
        state=previous,native_counts=engine.counts,source=lock['source_commit'],config=digest(config),
        generation_schedule=SCHEDULE,generation_endpoints=1,generation_W0_endpoints=0,
        generation_intermediate_endpoints=0,generation_final_requests=2000,
        final_generation_identity=generated['identity'],
        final_generation_identity_sha256=generated.get('identity_sha256'),
        native_scope_completed=True,final_generation_completed=True,
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
