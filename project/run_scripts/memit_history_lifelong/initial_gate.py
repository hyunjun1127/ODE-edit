"""One-shot CPU inspection of B1 commit -> B2 entry; never polls or loads model."""
import argparse,json,math
from pathlib import Path
from .io import digest,file_sha,save,content

def check(attempt):
    attempt=Path(attempt);lock=json.loads((attempt/'execution.lock.json').read_text());out=Path(lock['output'])
    names=['runtime.json','B001/entry.json','B001/native-observation.json','B001/current.json','B001/commit.json','B002/entry.json','actual-import-closure.json']
    missing=[n for n in names if not (out/n).is_file()]
    if missing:return dict(status='WAITING_FOR_INITIAL_GATE',missing=missing)
    docs={n:json.loads((out/n).read_text()) for n in names}
    runtime=docs['runtime.json'];e1=docs['B001/entry.json'];native=docs['B001/native-observation.json'];current=docs['B001/current.json'];commit=docs['B001/commit.json'];e2=docs['B002/entry.json']
    records=json.loads((Path(lock['dataset_root'])/'counterfact.json').read_text())[:200]
    expected={tag:[] for tag in ('RS','PS','NS')}
    for r in records[:100]:
        x=r['requested_rewrite']
        for tag,prompts in [('RS',[x['prompt'].format(x['subject'])]),('PS',r['paraphrase_prompts']),('NS',r['neighborhood_prompts'])]:
            for j,p in enumerate(prompts):expected[tag].append(digest([r['case_id'],j,p,x['target_new']['str'],x['target_true']['str']]))
    checks=dict(
      runtime_lock=runtime['lock_sha256']==file_sha(attempt/'execution.lock.json'),
      entrypoint=runtime['native_entrypoint']['function']=='apply_memit_seq_to_model' and runtime['native_entrypoint']['file_sha256']==lock['entrypoint_sha256'],
      config=runtime['hparams']['blue'] is False and runtime['hparams']['layers']==[4,5,6,7,8] and runtime['seed']==20260907,
      first100=e1['request_ids']==[r['case_id'] for r in records[:100]] and e1['request_hashes']==[digest(r['requested_rewrite']) for r in records[:100]],
      second100=e2['request_ids']==[r['case_id'] for r in records[100:200]],
      B1_H0=e1['history_norms']==[0.]*5,
      commit=commit['status']=='BATCH_COMMITTED' and commit['batch']==1 and commit['requests']==100,
      counts=(commit['compute_z'],commit['solve_calls'],commit['history_append_layers'])==(100,5,5),
      post_keys=[x['layer'] for x in native['keys']]==[4,5,6,7,8]*2 and all(x['phase']=='post_all_layers' for x in native['keys'][5:]),
      finite=commit['finite_W_H'] and commit['nonfinite']==0 and all(math.isfinite(x) and x>0 for x in commit['history_norms']),
      cache_return=commit['cache_c_returned_same_object'],
      W_H_link=content(e2['signature'])==commit['endpoint'] and e2['history_norms']==commit['history_norms'],
      auxiliary_link=e2['auxiliary']==commit['auxiliary_endpoint'],
      observer_unchanged=commit['observer_context_rng_ledger_unchanged'] and commit['observer_mutation']==0 and commit['observer_auxiliary_before']==commit['observer_auxiliary_after'],
      ledger=e2['seen_before']==100 and e2['auxiliary']['ledger']['requests']==100 and e2['auxiliary']['ledger']['case_order_hash']==digest(e1['request_ids']),
      noCP=commit['save_checkpoints'] is False and commit['exact_resume']=='NOT_AVAILABLE',
      imports=any(x['path'].endswith('/memit/memit_seq_main.py') and x['sha256']==lock['entrypoint_sha256'] for x in docs['actual-import-closure.json']['files']))
    for tag,n in [('RS',100),('PS',200),('NS',1000)]:
        m=current['metrics'][tag]
        checks[tag+'_denominator_identity']=m['denominator']==n and [x['identity'] for x in m['rows']]==expected[tag]
        checks[tag+'_finite_TF_NLL']=m['tf_token_count']>0 and all(math.isfinite(m[k]) for k in ('new_nll','true_nll','desired_nll','tf_token_micro','tf_prompt_macro','tf_strict'))
    result=dict(status='INITIAL_GATE_PASS' if all(checks.values()) else 'INITIAL_GATE_FAIL',checks=checks,source_commit=lock['source_commit'],lock_sha256=file_sha(attempt/'execution.lock.json'),job_id=runtime['slurm_job'],B1_metrics={k:{a:b for a,b in v.items() if a!='rows'} for k,v in current['metrics'].items()},B1_edit_seconds=commit['edit_seconds'],B1_eval_seconds=commit['evaluation_seconds'],B1_history_norms=commit['history_norms'],evidence=[dict(path=str(out/n),bytes=(out/n).stat().st_size,sha256=file_sha(out/n)) for n in names],numerical_cross_host_certification='NOT_ESTABLISHED',B2_completion='NOT_OBSERVED',terminal='NOT_OBSERVED')
    save(attempt/'initial-gate.json',result)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);r=check(p.parse_args().attempt);print(json.dumps(r));raise SystemExit(1 if r['status']=='INITIAL_GATE_FAIL' else 0)
