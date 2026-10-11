"""Fresh native checkout only. Writer AST remains byte-identical to author."""
import hashlib
from official.diagnostics.fe_native.patch import patched_source, function_text

def native_source(original):
    text=patched_source(original)
    text=text.replace('    from .native_checkpoint_observer import snapshot',
        '    from official.diagnostics.fe_qwen_current.observer import snapshot, observed_write, observed_targets')
    text=text.replace('zs_all = target_supplier(model, tok, requests_chunks, cfg, context_templates,',
                      'zs_all = observed_targets(target_supplier, model, tok, requests_chunks, cfg, context_templates,')
    text=text.replace('snapshot(cfg, model, cache_c, context_templates,',
                      'snapshot(cfg, model, tok, cache_c, context_templates,')
    text=text.replace('        batch_edit(cfg,model,tok,requests_chunks,device,cache_c,zs_all)',
        '        observed_write(batch_edit,cfg,model,tok,requests_chunks,device,cache_c,zs_all,batch_index)')
    for name in ('batch_edit','load_cov','get_context_templates','upd_matrix_match_shape'):
        if function_text(original,name)!=function_text(text,name):raise ValueError('NATIVE_AST_CHANGED')
    return text

def main_source(original):
    old='            save_model(edited_model,cfg)'
    if original.count(old)!=1:raise ValueError('UNEXPECTED_NATIVE_EXPORT_CALL')
    return original.replace(old, '            # Storage only: selected-state latest already saved after W20 evaluation.\n'
        '            from official.diagnostics.fe_qwen_current.observer import finish\n'
        '            finish(cfg)')
