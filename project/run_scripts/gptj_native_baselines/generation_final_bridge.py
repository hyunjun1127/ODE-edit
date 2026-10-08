"""Explicit final-W20 observer: unchanged qualified generation/scoring APIs.

The first cold owner qualifies the same fixed <=8 prompts. Qualification is
technical route evidence, never a cold-W0 scientific generation endpoint.
No W0 generation/compatibility/cache READY or intermediate endpoint is used.
"""
from pathlib import Path
from project.run_scripts.experiment_generation_eval.common import immutable_write
from .generation_cache_bridge import GenerationObserver as CacheObserver
from .generation_common import digest, require


SCHEDULE='FINAL_W20_ONLY'


class GenerationObserver(CacheObserver):
    def __init__(self,config,lock,view,engine,tokenizer,records,out,arm,tracker=None):
        require(config['generation'].get('evaluation_schedule')==SCHEDULE,
                'FINAL_GENERATION_EXPLICIT_SCHEDULE')
        self._final_attempted=False
        super().__init__(config,lock,view,engine,tokenizer,records,out,arm,tracker=tracker)
        require(not hasattr(self,'private_compatibility_member'),
                'FINAL_GENERATION_NO_W0_COMPATIBILITY')

    def load_W0(self):
        raise RuntimeError('FINAL_ONLY_W0_GENERATION_NOT_SCHEDULED')

    def _progress(self,payload):
        require(payload.get('phase')=='generation_evaluation'
            and payload.get('generation_progress/total_cases')==2000,
            'FINAL_GENERATION_PROGRESS_ENDPOINT')
        return super()._progress(dict(payload,phase='W20_generation'))

    def _wrap(self,receipt,state_identity,out=None):
        # Keep the actual selected-route identity through the production
        # wrap -> compact runner receipt -> independent reducer seam.
        result=super()._wrap(receipt,state_identity,out=None)
        result.update(selected_route=self.qualification_link['selected_route'],
                      fixed_microbatch=self.qualification_link['fixed_microbatch'])
        if out is not None:
            path=Path(out)/'receipt.json'
            immutable_write(path,result)
            result['receipt_path']=str(path.resolve())
        return result

    def endpoint(self,records,*,out,endpoint,model_state,cohort_label):
        selected=self._selected(records)
        require(not self._final_attempted and endpoint=='W20' and cohort_label=='ALL_SEEN'
                and len(selected)==2000
                and [r['occurrence_index'] for r in selected]==list(range(1,2001)),
                'FINAL_GENERATION_ONE_W20_FULL_FIRST2000')
        require(self.engine.next_batch==21,'FINAL_GENERATION_AFTER_TWENTY_NATIVE_COMMITS')
        # Each history site must be the caller's final live state, not cold RAM.
        from project.run_scripts.gptj_cake_blue_prune_rect.metrics import state
        require(state(self.view,self.engine.history())==model_state,
                'FINAL_GENERATION_ACTUAL_FINAL_STATE')
        self._final_attempted=True
        receipt=self.shared.observe(selected,'W20',cohort='ALL_SEEN',state_identity=model_state)
        require(receipt['identity']['ordered_occurrences']==list(range(1,2001))
                and receipt['identity']['state_sha256']==digest(model_state),
                'FINAL_GENERATION_FINAL_COHORT_STATE')
        return self._wrap(receipt,model_state,out)

    def subset_receipt(self,*args,**kwargs):
        raise RuntimeError('FINAL_ONLY_INTERMEDIATE_GENERATION_NOT_SCHEDULED')

    def subset(self,*args,**kwargs):
        raise RuntimeError('FINAL_ONLY_INTERMEDIATE_GENERATION_NOT_SCHEDULED')
