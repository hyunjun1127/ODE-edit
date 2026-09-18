"""CPU-only RCA of first M endpoint metadata; no runtime edits or submissions."""
import io
import json
from pathlib import Path
import torch
from .common import ROOT,member,write,tensor_sha


def main():
    torch.set_num_threads(8)
    parent=ROOT/'M/attempt-skip-t-v1';episode=parent/'episodes/b001/attempt-v1'
    failure=json.loads((episode/'failure.json').read_text())
    if failure['stage']!='controllers' or 'TorchVersion' not in failure['error']:
        raise ValueError('NOT_EXPECTED_METADATA_FAILURE')
    outcomes=[]
    for label,version in [('TorchVersion',torch.__version__),('plain_str',str(torch.__version__))]:
        buffer=io.BytesIO();torch.save(dict(weight=torch.ones(2,2),identity=dict(torch=version)),buffer);buffer.seek(0)
        try:
            value=torch.load(buffer,weights_only=True,map_location='cpu')
            outcomes.append(dict(input_type=label,loaded=True,metadata_type=type(value['identity']['torch']).__name__))
        except Exception as exc:outcomes.append(dict(input_type=label,loaded=False,error_type=type(exc).__name__,error=str(exc)))
    if outcomes[0]['loaded'] or not outcomes[1]['loaded']:raise ValueError('CPU_REPRODUCTION_MISMATCH')
    endpoint=episode/'arms/N4/final-L4.pt';before=member(endpoint)
    globals_found=torch.serialization.get_unsafe_globals_in_checkpoint(endpoint)
    if globals_found!=['torch.torch_version.TorchVersion']:raise ValueError('UNEXPECTED_CHECKPOINT_GLOBALS')
    # One exact known class from the installed torch package; never weights_only=False.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved=torch.load(endpoint,weights_only=True,map_location='cpu',mmap=True)
    weight=saved['weight'];native=json.loads((episode/'native/native-binding.json').read_text())
    raw_hash=tensor_sha(weight)
    checks=dict(shape=list(weight.shape),dtype=str(weight.dtype),finite=bool(torch.isfinite(weight).all()),
        raw_weight_sha256=raw_hash,matches_bound_native=raw_hash==native['endpoint'],
        actual_torch_metadata_type=type(saved['tokenizer_identity']['torch']).__module__+'.'+type(saved['tokenizer_identity']['torch']).__name__,
        actual_metadata_string=str(saved['tokenizer_identity']['torch']),history=saved['history'],
        safe_load_scope='single known TorchVersion class allowed in local CPU context, weights_only still true',
        selection_seal_exists=(episode/'arms/N4/selection-seal.json').exists(),GPU_continuation='NOT_TESTED')
    if not checks['finite'] or not checks['matches_bound_native']:raise ValueError('N4_STORED_WEIGHT_INVALID')
    if member(endpoint)!=before:raise ValueError('RAW_CHANGED_DURING_CPU_RCA')
    result=dict(status='METADATA_SERIALIZATION_RCA_CONFIRMED_NOT_METHOD_FAILURE',
        failure=member(episode/'failure.json'),native_binding=member(episode/'native/native-binding.json'),
        actual_file=before,actual_saved_checks=checks,toy_CPU_reproduction=outcomes,
        immediate_cause='TorchVersion metadata serialized by runtime identity; stock weights_only endpoint reload rejects it',
        proposed_minimum_fix='serialize version metadata as plain builtin str; preserve weights_only loader; new immutable source only',
        proposal_applied_to_runtime=False,new_GPU=0,new_native_fit=0,new_submit=0,new_cancel=0,
        source_raw_modified=False,normal_M_initial='NOT_REACHED',EN_F_actual_science='NOT_RUN',
        endpoint_completion='N4 payload exists but mandatory original reload/selection seal failed; not completed M result')
    print(json.dumps(write(ROOT/'receipts/storage-waiver-r1/serialization-rca.json',result)))


if __name__=='__main__':main()
