"""Read-only W20 raw reducer for the exact approved completed GPTJ cells."""
import json
from pathlib import Path
from official.evaluation import reduce
from official.experiments.prepare import digest
from official.runners.server2.collect import member, verify_member

ROOT = Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1')
CELLS = [('registration-cf-checkpoint-r1', 'FT/chain', 61650),
         ('registration-no-gpu-qual-r1', 'CF_MEMIT', 61725)]
CELLS += [('registration-no-gpu-qual-r1', 'ZSRE_' + m, j) for m,j in
          [('FT',61726),('MEMIT',61728),('ALPHAEDIT',61730),('ALPHAEDIT_BLUE',61732),('MEMIT_FE',61734),('SPHERE',61735)]]
CELLS += [('registration-cf-display-r1', 'CF_' + m, j) for m,j in
          [('ALPHAEDIT',61778),('ALPHAEDIT_BLUE',61779),('MEMIT_FE',61780),('SPHERE',61781)]]

def main():
    output = []
    cohort = {}
    for attempt, role, job in CELLS:
        folder = ROOT / attempt / role
        result = json.loads((folder/'result.json').read_text())
        assert result['status'] == 'SCIENTIFIC_COMPLETE' and result['batches'] == 20
        assert len(result['commits']) == 20 and result['ownstate_links'] == 19
        for batch, ref in enumerate(result['commits'], 1):
            commit = json.loads(verify_member(ref).read_text())
            assert commit['batch'] == batch and commit['requests'] == 100
            assert commit['method'] == result['method']
            assert commit['checkpoint_identity'] == result['checkpoint_identity']
        endpoint = result['factual_endpoints']['W20']
        raw = json.loads(verify_member(endpoint).read_text())
        dataset = result['dataset']
        assert raw['endpoint'] == 'W20' and raw['requests'] == 2000
        assert raw['identity_sha256'] == digest(raw['identity'])
        assert raw['identity']['ordered_occurrences'] == list(range(1,2001))
        assert [c['occurrence_index'] for c in raw['cases']] == list(range(1,2001))
        assert raw['identity']['dataset'] == dataset and len(raw['cases']) == 2000
        key = digest([(c['occurrence_index'], c['case_id']) for c in raw['cases']])
        assert cohort.setdefault(dataset, key) == key
        summary = getattr(reduce, 'counterfact' if dataset == 'cf' else 'zsre')(raw['cases'])
        for k,v in summary.items():
            assert abs(v - raw['summary'][k]) <= 1e-10, (role,k)
        if dataset == 'zsre':
            summary['Score'] = reduce.harmonic([summary[k] for k in ['Efficacy','Generalization','Specificity']])
        counts = ({kind: sum(len(c[kind+'_prompts_probs']) for c in raw['cases'])
                   for kind in ['rewrite','paraphrase','neighborhood']} if dataset == 'cf' else
                  {kind: sum(len(c[kind]) for c in raw['cases']) for kind in
                   ['rewrite_prompts_correct','paraphrase_prompts_correct','neighborhood_W0_agreement','neighborhood_prompts_correct']})
        output.append(dict(model='gptj', dataset=dataset, method=result['method'], job_id=str(job),
            status='W20_RAW_CPU_REDUCED_2000', summary=summary, denominators=counts,
            counts_semantics='prompt_pairs' if dataset=='cf' else 'teacher_forced_token_decisions',
            source=result['code_commit'], config_sha256=result['checkpoint_identity']['config_sha256'],
            stream_sha256=result['checkpoint_identity']['stream_sha256'], ordered_cohort_sha256=key,
            W20_raw=endpoint, result=member(folder/'result.json'), committed_batches=20,
            generation='DEFERRED' if dataset=='cf' else 'NOT_APPLICABLE',
            checkpoint_consumer_pending=result['checkpoint_evaluation_consumer_pending'],
            CP_payload_loaded=False, online_delivery='NOT_REQUERIED'))
    print(json.dumps(dict(schema=1, rows=output, GPU=0, model_loads=0,
        raw_reduction='exact SHA/bytes, 20 commit receipts, 2000 ordered cases, independent reducer',
        source_policy='original source/config/raw identities preserved'), indent=2))

if __name__ == '__main__':
    main()
