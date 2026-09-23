"""Exact native tokenizer witness reuse; never changes native tokenization."""
import hashlib
import json
from pathlib import Path
from .common import file_sha,digest

REFERENCE_SHA='1b6273a0ffeb8cf9ed49933adc9a682a1826efc78d31507ee1cce74356341efd'


def reference_path(root):
    return Path(root)/'failure-diagnosis-r1/bos-cpu-reproduction.json'


def load_reference(root):
    path=reference_path(root)
    if file_sha(path)!=REFERENCE_SHA:raise RuntimeError('NATIVE_TOKEN_REFERENCE_DRIFT')
    reference=json.loads(path.read_text())
    assert reference['status']=='FIRST_EXCEPTION_REPRODUCED_CPU_ONLY'
    assert reference['original_current_token_pack_exact']=={'input_ids':True,'attention_mask':True}
    assert reference['actual_sequences']==len(reference['rows'])==12
    assert not reference['method_modified'] and reference['model_loads']==reference['GPU_allocations']==0
    return reference


def compare_reference(tok,rows,contexts,reference):
    """Exact original native recipe/IDs/masks/lookups, not blanket BOS removal."""
    assert type(tok).__name__==reference['tokenizer_class'],'WRITER_TOKENIZER_CLASS_DRIFT'
    assert tok.add_bos_token is reference['assigned_add_bos_token'],'WRITER_DECLARED_BOS_DRIFT'
    assert tok.bos_token_id==reference['bos_token_id'],'WRITER_BOS_ID_DRIFT'
    assert hashlib.sha256(tok.backend_tokenizer.post_processor.__getstate__()).hexdigest()==reference['postprocessor_sha256'],'WRITER_BACKEND_POSTPROCESSOR_DRIFT'
    context_sha=hashlib.sha256(json.dumps(contexts,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert context_sha==reference['context_semantic_sha256'],'WRITER_NATIVE_CONTEXT_DRIFT'
    assert len(rows)==reference['actual_sequences'],'WRITER_NATIVE_SEQUENCE_COUNT'
    for row,expected in zip(rows,reference['rows'],strict=True):
        for key in ('case_id','context_index','subject_last','valid_tokens'):
            assert row[key]==expected[key],('WRITER_NATIVE_POSITION_DRIFT',key)
        ids=row['input_token_ids']
        assert digest(ids)==expected['input_token_ids_sha256'],'WRITER_NATIVE_INPUT_IDS_DRIFT'
        assert ids[0]==expected['first_token_id'],'WRITER_NATIVE_FIRST_TOKEN_DRIFT'
        assert (ids[0]==tok.bos_token_id)==expected['bos_present'],'WRITER_NATIVE_ACTUAL_BOS_DRIFT'
        assert expected['right_padding'] and expected['lookup_in_range'],'INVALID_NATIVE_TOKEN_REFERENCE'
    return dict(status='PASS',reference_sha256=REFERENCE_SHA,sequences=len(rows),
                token_ids_masks_lookup='EXACT_NATIVE_REFERENCE',
                configured_add_bos_token=tok.add_bos_token,
                actual_bos_sequences=sum(r['input_token_ids'][0]==tok.bos_token_id for r in rows),
                native_tokenization_modified=False)
