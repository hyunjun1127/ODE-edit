"""Canonical scalar events, separated from scientific differentiation."""
from .common import digest,tensor_sha

def metadata(events,a,c,pilot,source):
    events.emit('run_metadata',dict(method_revision='jlz-v12-method-contract-20261004-v1',source_identity=source,
        adapter_id='JLZ_V12_ALPHAEDIT_LLAMA_FP32',model_identity=digest(c['model']),dataset_identity=digest(c['stream']),
        request_order_identity=digest(c['packing']),native_rows_identity=c['native_input_alignment']['sha256'],
        tokenizer_identity=digest(c['model']),qualification_status='NOT_RUN' if pilot else 'QUALIFIED',
        intended_workload=dict(request_count=4 if pilot else 2000,nominal_batch_size=2 if pilot else 100),
        execution_status='QUALIFICATION' if pilot else 'RUNNING',reference_code_is_production=False,
        native_profile=dict(lambda_kl=.0625,lambda_norm=.5,lambda_covariance=0.,budget_radius=.75,
            native_learning_rate=.1,native_adam_epsilon=1e-8,adam_beta1=.9,adam_beta2=.999,stop_threshold=.05,
            max_logical_evaluations=25,max_backward_updates=24,native_anchor_layer=str(c['profile']['anchor_layer']),
            nll_readout='31',kl_readout='final',kl_direction='CURRENT_TO_ENTRY',vocabulary='FULL',norm_zero_subgradient='ZERO',
            nll_reduction='target_token_mean_then_equal_context_mean_per_request',kl_reduction='full_vocab_sum_one_context',
            writer_key_reduction='native_nested_group_mean_FP32',dtype_model_plan_moments='float32',
            dtype_geometry_projection='float64',dtype_history='cpu_float32',cast_add_order='entry32+native_FP32_solve(Pi@(KKt+H)+10I,Pi@K@Rt).T'),
        eligible_layers=[dict(layer_id=str(l),block_output_dimension=a.dims[l][0],writer_key_dimension=a.dims[l][1],
            injection_module=f'model.layers.{l}',canonical_output_module=f'model.layers.{l}',
            writer_module=f'model.layers.{l}.mlp.down_proj',weight_orientation='out_in',
            local_additivity_mapping='down_proj_output_adds_to_fullblock_residual',local_additivity_status='NOT_QUALIFIED') for l in a.sites],
        tolerances=dict(kkt_active_norm_abs=1e-10,kkt_boundary_relative=1e-6,kkt_boundary_scale='max(1,c)',
            projection_castback_abs=0.,projection_castback_relative=1e-6,native_parity_contract_id='v12-fixed-technical-tolerances')))

def batch_entry(events,entry,before):
    pack=entry['pack'];rows=[r for g in entry['groups'] for r in g['rows']]
    events.emit('batch_entry',dict(actual_request_count=pack['n_requests'],nominal_batch_size=pack['n_requests'],
        ordered_request_ids=list(map(str,pack['record_ids'])),ordered_request_identity=digest(pack['record_ids']),
        entry_model_state_id=digest(before['W']),entry_history_state_id=digest(before['H']),
        teacher_identity=digest(entry['teacher_hash']),anchor_identity=digest({l:tensor_sha(v) for l,v in entry['anchors'].items()}),
        native_row_counts=[dict(request_id=str(case),rewrite_context_count=pack['n_rw'],kl_context_count=1,
            target_token_counts=[int((r['target']!=-100).sum()) for r in rows if r['request']==i and r['kind']=='rewrite'])
            for i,case in enumerate(pack['record_ids'])],rollback_snapshot_id=digest(before),fit_model_weights_frozen=True,fit_history_frozen=True))

def evaluation(events,summary,state_id,evaluator,ids,population):
    for kind,s in summary.items():
        n=s['denominator'];desired='true' if kind=='N' else 'new'
        metrics={dict(R='RS',P='PS',N='NS')[kind]:(s['numerator'],n,'PROMPT'),
            'ACC_TF_STRICT':(s['strict_numerator'],n,'PROMPT'),
            'ACC_TOKEN_MICRO':(s['desired_token_correct'],s['desired_token_count'],'VALID_TARGET_TOKEN'),
            'ACC_PROMPT_MACRO':(s['prompt_macro']*n,n,'PROMPT'),
            'NLL_TRUE':(s['true_nll_mean']*n,n,'PROMPT'),'NLL_NEW':(s['new_nll_mean']*n,n,'PROMPT'),
            'NLL_DESIRED':(s[desired+'_nll_mean']*n,n,'PROMPT'),
            'NLL_MARGIN_TRUE_MINUS_NEW':((s['true_nll_mean']-s['new_nll_mean'])*n,n,'PROMPT')}
        for metric,(num,den,unit) in metrics.items():
            events.emit('evaluation_aggregate',dict(model_state_id=state_id,evaluator_identity=evaluator,
                population_kind=population,population_identity=digest(ids),request_count=len(ids),prompt_count=n,
                valid_target_token_count=s['desired_token_count'],family=kind,metric=metric,desired_target=desired.upper(),
                status='MEASURED',value=num/den,numerator_or_sum=num,denominator=den,denominator_unit=unit,
                definition_id='v12:'+metric,not_applicable_reason=None,teacher_forced=True))
