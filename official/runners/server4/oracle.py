"""SH4 caller for the single shared, independently executed original CF oracle."""
from pathlib import Path
from official.evaluation import cf_native_reference as reference
from official.experiments.prepare import ROOT, file_sha, write_new

MODULE_SHA='0473673abf92e483b19d534743278ee33177ce22e66c059e9788e40ba7e12216'
LOCK_SHA='e8f540ee60db8f2343bca99307f1ebaca51c11cf31cfe39c05bf3d808c5199ae'


def source_binding():
    lock=ROOT/'hparams/cf-native-reference.lock.json'
    if file_sha(reference.__file__)!=MODULE_SHA or file_sha(lock)!=LOCK_SHA:
        raise ValueError('CF_ORIGINAL_ORACLE_LOCK_CHANGED')
    reference.load_original_counterfact()  # Verify/compile unchanged original AST, no forward.
    return dict(module_sha256=MODULE_SHA,lock_sha256=LOCK_SHA,
        original_sha256=reference.SOURCE_SHA256,original_bytes=reference.SOURCE_BYTES,
        actual_GPU='NOT_OBSERVED')


def locks(config,assets,tokenizer):
    snapshot=Path(assets['model_snapshot'])
    consumed={Path(row['path']).name:row['sha256'] for row in assets['members']
              if Path(row['path']).parent==snapshot}
    weights={name:sha for name,sha in consumed.items() if name.endswith('.safetensors')}
    if not weights or 'config.json' not in consumed:
        raise ValueError('CF_ORACLE_CONSUMED_MODEL_LOCK_REQUIRED')
    model=dict(model='llama3',method=config['method'],revision=assets['model_revision'],
               model_id=config['model_identity']['model_id'],
               config_sha256=consumed['config.json'],weight_sha256=weights)
    token=dict(files=assets['tokenizer_files'],padding_side=tokenizer.padding_side,
               bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,
               pad_token_id=tokenizer.pad_token_id)
    return model,token


def compare(state,config,assets,records,canonical,locked_stream,output,scope):
    source_binding()
    if scope not in (reference.SMOKE_SCOPE,reference.FULL_SCOPE):
        raise ValueError('SH4_ORACLE_SCOPE_NOT_PLANNED')
    model,token=locks(config,assets,state.tokenizer)
    signature=state.signature()
    result=reference.compare_native_counterfact(state.model,state.tokenizer,records,canonical,
        identity=canonical['identity']['external_identity'],evidence_scope=scope,
        locked_cohort=[dict(case_id=r['case_id'],occurrence_index=r['occurrence_index'])
                       for r in locked_stream],
        model_identity=model,tokenizer_identity=token,state_identity=signature,
        state_callback=state.signature)
    write_new(output,result)  # Full native rows/identities stay in ignored local.
    if result['status']!='PASS' or result['evidence'][scope]!='PASS':
        raise ValueError('CF_NATIVE_ORACLE_NOT_PASS:'+result['status'])
    return result
