"""Separate query tensor work from attention-visible cached context."""
def count_tokens(ledger,ids,mask):
    ledger.add('actual_model_forward_input_token_elements',ids.numel())
    queries=int(mask[:,-ids.shape[1]:].sum()) if mask is not None else ids.numel()
    visible=int(mask.sum()) if mask is not None else ids.numel()
    ledger.add('actual_model_forward_nonpadding_query_tokens',queries)
    ledger.add('actual_model_forward_attention_visible_tokens',visible)
    # Preserve the R1/R2 field's bytes/meaning; reports label this legacy name.
    ledger.add('actual_model_forward_input_tokens',visible)
