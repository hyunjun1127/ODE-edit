"""Pre-outcome selection contract. No fitting, generation, or job submission."""
import math
from official.experiments.prepare import digest

NONCE = 'USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1'
SPEC = dict(instruction=NONCE, model='qwen25', dataset='cf', seed=0,
    paper_commit='74e36c38ce498c07f9b438096be53660316d412d',
    author_commit='478134dfb24b43f4e18b47e8500893ce3f9cc50f',
    z_reference='batch_entry', first_target_layer=4, later_targets='firstforward_same_entry',
    layers=[4,5,6,7,8], clamp=1, steps=35, lr=0.5, weight_decay=0.001,
    C0_weight=15000, dtype='bfloat16', attention='author_default', add_old_keys=True,
    L2_arms=[0,0.1,1,10], promotable_L2=[0.1,1,10],
    selection_slice=[4000,6000], main_slice=[0,2000], prior_alt_slice=[2000,4000],
    batches=20, batch_size=100, milestones=[5,10,15,20],
    gate=dict(Efficacy=95,Generalization=85,Specificity=70),
    ranking='maximum_unrounded_CF_harmonic_Score_then_smaller_L2',
    veto='all_three_tuned_scores_below_control_or_no_tuned_gate_pass',
    failure_policy='explicit_verified_numeric_failure_equals_minus_infinity; all_other_incomplete_states_block_selection',
    control='L2_0_heldout_only_never_promote',
    checkpoint='latest_only_after_evaluation_every_500_no_W0',
    generation='DEFERRED_CHECKPOINT_EVALUATION')

def split_rows(rows, main_rows, alt_rows):
    if len(rows) < 6000: raise ValueError('FIXED10K_TOO_SHORT')
    ids=lambda r:[x['case_id'] for x in r]
    if ids(main_rows)!=ids(rows[:2000]) or ids(alt_rows)!=ids(rows[2000:4000]):
        raise ValueError('PRIOR_STREAM_NOT_EXPECTED_FIXED10K_ORDER')
    groups=[ids(rows[a:b]) for a,b in ([0,2000],[2000,4000],[4000,6000])]
    if any(len(set(g))!=2000 for g in groups) or len(set(sum(groups,[])))!=6000:
        raise ValueError('SELECTION_CASE_ID_OVERLAP')
    selected=rows[4000:6000]
    return selected,dict(selection_order_sha256=digest(groups[2]),
        main_order_sha256=digest(groups[0]),alt_order_sha256=digest(groups[1]),
        selection_content_sha256=digest(selected),pairwise_case_id_disjoint=True)

def select(results):
    """Only complete independently validated raw endpoints may reach this reducer."""
    if set(results)!=set(SPEC['L2_arms']):raise ValueError('FOUR_ARM_COVERAGE_REQUIRED')
    finite={};scores={}
    for l2, r in results.items():
        if r.get('status')=='NUMERICAL_FAILURE':
            if r.get('verified_failure') is not True or not r.get('failure_receipt_sha256'):
                raise ValueError('VERIFIED_NUMERICAL_FAILURE_RECEIPT_REQUIRED')
            finite[l2]=False;scores[l2]=-math.inf
            continue
        if r.get('status','W20_COMPLETE')!='W20_COMPLETE':
            raise ValueError('UNRESOLVED_IMPLEMENTATION_OR_RESOURCE_FAILURE')
        if r.get('requests')!=2000 or r.get('committed_batches')!=20:
            raise ValueError('W20_COVERAGE_REQUIRED')
        vals=[r.get(k) for k in ('Efficacy','Generalization','Specificity','Score')]
        finite[l2]=all(type(x) in (int,float) and math.isfinite(x) for x in vals)
        if not finite[l2]:raise ValueError('NONFINITE_W20_REQUIRES_EXPLICIT_FAILURE_RECEIPT')
        if finite[l2]:
            expected=0.0 if min(vals[:3])==0 else 3/sum(1/x for x in vals[:3])
            if any(not 0<=v<=100 for v in vals) or abs(expected-vals[3])>1e-9:
                raise ValueError('INVALID_REQUEST_MACRO_SCORE')
            scores[l2]=vals[3]
    eligible=[l for l in SPEC['promotable_L2'] if finite[l] and
              all(results[l][k]>=v for k,v in SPEC['gate'].items())]
    all_worse=finite[0] and all(scores[l]<scores[0] for l in SPEC['promotable_L2'])
    winner=None if all_worse or not eligible else min(eligible,key=lambda l:(-results[l]['Score'],l))
    return dict(status='SELECTED' if winner is not None else 'NO_PROMOTION',selected_L2=winner,
        eligible=eligible,all_tuned_below_control=all_worse,contract_sha256=digest(SPEC),
        control_promoted=False,scientific_recovery_claim=False)

def tracking_config(config, source):
    # Existing transport's task_id is the SDK group. No shared logger fork.
    l2=str(config['L2']).replace('.','p')
    values=dict(server=config['server'],task_id='fe-qwen-current-z-heldout-selection-20261012',
        model='qwen25',model_family='qwen2',writer='fe_native_batch_entry_z',
        baseline='FE-heldout-control' if config['L2']==0 else 'FE-heldout-tuned-L2',
        role='scientific',arm=f'heldout4000-6000-L2-{l2}-clamp1-steps35-lr0p5-wd0p001',
        attempt=config['attempt'],source_sha=source,config_sha=config['config_sha256'],
        observation_identity=config['selection_contract_sha256'],
        stream_sha256=config['stream']['sha256'],evaluation_profile='heldout-selection-not-main',
        dataset='cf',metric_schema='official-baselines-scalar-v1',
        instruction_id='USER-OFFICIAL-BASELINES-20261008-R1',
        generation_schedule='DEFERRED_CHECKPOINT_EVALUATION')
    if config.get('stage')=='final_benchmark':
        values.update(task_id='fe-qwen-current-z-final-benchmark-20261012',
            baseline='FE-selected-tuned-L2',dataset=config['dataset'],
            arm=f"final-{config['dataset']}-L2-{l2}-cold-first2000",
            evaluation_profile='selected-native-current-z-final-first2000',
            observation_identity=config['selected_result']['sha256'])
        if config['dataset']=='zsre':values.pop('generation_schedule')
    return values
