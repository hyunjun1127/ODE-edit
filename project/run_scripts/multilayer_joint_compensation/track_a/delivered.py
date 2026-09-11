"""Observation-only module-coordinate delivery, not block-z subtraction."""
import importlib
import torch
from project.run_scripts.single_layer_cumulative_risk.microbatch import bounded_reader

def capture(model,tok,native,hp,contexts,records,ledger):
    reader=importlib.import_module('rome.repr_tools');original=reader.get_reprs_at_idxs
    reader.get_reprs_at_idxs=bounded_reader(original,ledger,8)
    templates=[c.format(r['requested_rewrite']['prompt']) for r in records for kind in contexts for c in kind]
    subjects=[r['requested_rewrite']['subject'] for r in records for kind in contexts for _ in kind]
    try:
        answer={}
        for layer in (4,8):
            with torch.no_grad(),ledger.time('delivered_module_capture'):
                incoming,outgoing=native.get_module_input_output_at_words(model,tok,layer,
                  context_templates=templates,words=subjects,module_template=hp.rewrite_module_tmp,
                  fact_token_strategy=hp.fact_token)
            if incoming.shape[0]!=len(templates) or outgoing.shape[0]!=len(templates):
                raise ValueError('DELIVERED_CONTEXT_ORDER')
            answer[layer]=dict(input=incoming.detach().cpu(),output=outgoing.detach().cpu())
            ledger.add('delivered_module_contexts',len(templates))
        return answer
    finally:reader.get_reprs_at_idxs=original
