"""One actual first-input check for the new observer path, not a full panel."""
import time
import torch
from .contracts import ContractBoundary, digest
from .evaluation import diagnostic_layout
from .fixtures import capture_rng, restore_rng, tensor_sha
from .instrumentation import ReadOnlyCapture
from .observation_panels import temporary_weight, per_position_energy, evaluate_general


def validate(model,tok,weight,module_name,entry,record,general,K,F,direction):
    start=time.monotonic();rng=capture_rng();before=tensor_sha(weight)
    pointer,version=weight.data_ptr(),weight._version
    try:
        with temporary_weight(weight,entry),torch.no_grad():
            b=diagnostic_layout(tok,record,device=weight.device)
            with ReadOnlyCapture(model.get_submodule(module_name),destination=weight.device) as c:
                model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False)
            if len(c.records)!=1:raise ContractBoundary('INITIAL_QUERY_MODULE_COUNT')
            energy=per_position_energy(c.records[0]['input'],K,F,direction)
            text=evaluate_general(model,general['rows'][:1],state_label='INITIAL_GATE_ENTRY')
        if tensor_sha(weight)!=before or weight.data_ptr()!=pointer or weight._version!=version:
            raise ContractBoundary('INITIAL_QUERY_RESTORE')
        consumed=capture_rng()!=rng
        if consumed:raise ContractBoundary('INITIAL_QUERY_RNG_CONSUMPTION')
    finally:
        restore_rng(rng)
    return dict(status='FIRST_QUERY_GENERAL_VALID',query_pairs=1,general_sequences=1,
        input_identity=b['identity'],general_row_identity=digest(general['rows'][0]),
        query_energy_fields=sorted(energy),position_count=sum(len(v) for v in next(iter(energy.values()))),
        General_NLL=text[0]['nll'],scalar='GENERAL_NLL_NOT_RPN_MARGIN',
        pointer_version_bytes_rng_restored=True,whole_panel_complete=False,
        native_calls=0,diagnostic_forward_calls=2,seconds=time.monotonic()-start)
