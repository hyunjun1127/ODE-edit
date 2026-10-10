"""CPU-only raw reduction; source observations and transport history are read-only."""
import csv
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from official.experiments.prepare import read, file_sha, digest, write_new
from official.evaluation.reduce import counterfact
from official.evaluation.generation.paper_display import paper_cell

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
ROOT = Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/registration-r1')

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def member(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path))

def verified(ref):
    require(file_sha(ref['path']) == ref['sha256'], 'MEMBER_SHA')
    require(Path(ref['path']).stat().st_size == ref['bytes'], 'MEMBER_BYTES')
    return read(ref['path'])

def display(value, unit):
    require(isinstance(value, (float, int)) and math.isfinite(value), 'RAW_FINITE')
    own = str((Decimal(str(value))*Decimal(100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    shared = paper_cell(value,metric='Flu' if unit=='bits' else 'Con',raw_unit=unit)
    require(own == shared, 'SHARED_DISPLAY_PARITY')
    return dict(raw_value=value, raw_unit=unit,
                paper_display_x100=shared,
                display_unit='bits_x100' if unit == 'bits' else 'cosine_x100',
                rounding='DECIMAL_HALF_UP_2_AFTER_RAW_MEAN_TIMES_100', is_percentage=False)

def generation(path, stream):
    gen = read(path)
    require(gen['observer_no_mutation'] and gen['RNG_restored'], 'GENERATION_GUARD')
    rows = gen['rows']
    require(len(rows) == 2000, 'FULL_2K')
    require([(r['occurrence'], r['case_id']) for r in rows] ==
            [(r['occurrence_index'], r['case_id']) for r in stream], 'ORDER')
    execution = verified(gen['native_execution_member'])
    result = {}
    for field, valid, label, unit in [('ngram_entropy','fluency_valid','Flu','bits'),
                                      ('reference_score','consistency_valid','Con','cosine_0_to_1')]:
        values = [r['metrics'][field] for r in rows if r['metrics'][valid]]
        require(len(values) == 2000 and all(math.isfinite(v) for v in values), 'VALID_COUNTS')
        mean = math.fsum(values)/len(values)
        require(abs(mean-gen['summary'][field]) < 1e-12, 'STORED_MEAN')
        result[label] = dict(**display(mean,unit), denominator=len(values))
    result.update(raw=member(path), endpoint_identity_sha256=gen['identity_sha256'],
        native_execution=gen['native_execution_member'], requests=2000,
        prompts=sum(r['metrics']['generation_prompt_count'] for r in rows),
        generated_tokens=sum(r['metrics']['generated_token_count'] for r in rows),
        runtime_sha256=gen['identity']['runtime'], endpoint=gen['identity']['endpoint'],
        sampling_stream_sha256=gen['identity']['sampling_stream_sha256'],
        state_sha256=gen['identity']['state_sha256'])
    return result

def main():
    # Boundaries distinguish multiplying the raw mean from multiplying a rounded display.
    assert display(6.252105796227186,'bits')['paper_display_x100'] == '625.21'
    assert display(0.2591242773267912,'cosine_0_to_1')['paper_display_x100'] == '25.91'
    assert display(0.00005,'bits')['paper_display_x100'] == '0.01'
    assert display(1.234449,'bits')['paper_display_x100'] == '123.44'
    data = read(OUT/'table-rows.json')
    stream = read(ROOT/'streams/cf-stream.json')
    for row in data['rows']:
        if row.get('generation_raw'):
            row['generation'] = generation(row['generation_raw']['path'], stream)
        else:
            status = ('NOT_APPLICABLE' if row['dataset']=='zsre' else
                      'DEFERRED' if row.get('generation_schedule')=='DEFERRED_CHECKPOINT_EVALUATION' else 'NOT_MEASURED')
            row['generation'] = dict(Flu=status, Con=status)
    previous = read(REPO/'audits/servers/server2/main-table-refresh-20261009/table-rows.json')
    for old in previous['rows']:
        if old['model']!='gptj' or old['dataset']!='cf' or not old['numeric_eligible']:
            continue
        raw = verified(old['raw']); terminal = verified(old['terminal'])
        cases = raw['cases']
        require(len(cases)==2000 and [(c['occurrence_index'],c['case_id']) for c in cases] ==
                [(r['occurrence_index'],r['case_id']) for r in stream], 'GPTJ_2K_ORDER')
        values = counterfact(cases)
        for k,v in values.items():
            require(abs(v-old['metrics'][k]) < 1e-10, 'GPTJ_REDUCE')
        row = dict(old, metrics=values, raw_rechecked_at=data['at'],
                   generation=dict(Flu='DEFERRED',Con='DEFERRED'),
                   scheduler_snapshot_reused=True)
        data['rows'].append(row)
    w0path = ROOT/'shared-w0/qwen25-cf/w0-receipt.json'
    w0 = read(w0path)
    g = generation(w0['generation_rows_path'],stream)
    obs = read(ROOT/'shared-w0/qwen25-cf/generation/observer-identity.json')
    require(g['runtime_sha256']==obs['identity_sha256'], 'W0_RUNTIME')
    require(g['endpoint_identity_sha256']==w0['generation_identity_sha256'], 'W0_RECEIPT')
    require(w0['stream_sha256']==file_sha(ROOT/'streams/cf-stream.json'), 'W0_STREAM')
    w0row = dict(server='server2',model='qwen25',dataset='cf',method='W0',endpoint='W0',
        job_id='61898',job_name='s2-qwen25-cf-ft-gpu',source_commit=w0['code_commit'],
        receipt=member(w0path),generation=g,generation_conditions=obs['identity'],
        stream=member(ROOT/'streams/cf-stream.json'),tokenizer_sha256=w0['tokenizer_sha256'],
        ordered_sample_sha256=read(ROOT/'streams/cf-stream.lock.json')['ordered_case_ids_sha256'],
        factual_cells='KEEP_SH3_SELECTED_W0_61813',
        selected_table_generation_status='DEFERRED_COMPATIBILITY_NOT_YET_CONFIRMED',
        independent_SH2_generation_raw_verified=True)
    sh3path = REPO/'audits/servers/server3/flucon-paper-scale-20261010/qwen-w0-provenance.json'
    sh3 = read(sh3path)
    compatibility = dict(SH3_provenance=member(sh3path), SH2_receipt=member(w0path),
        revision_match=obs['identity']['model_identity'].split('@')[1] == Path(sh3['identity']['model_snapshot']).name,
        stream_match=w0['stream_sha256']==sh3['identity']['stream_sha256'],
        ordered_cases_match=w0row['ordered_sample_sha256']==sh3['ordered_case_ids_sha256'],
        requests_match=sh3['requests']==g['requests']==2000,
        SH3_cold_verified=sh3['cold_state_verified'],
        SH3_generation=sh3['generation'],SH2_generation_conditions=obs['identity'],
        status='PARTIAL_IDENTITY_MATCH_NOT_FULL_COMPATIBILITY',
        blockers=['SH3 tokenizer/model payload historical SHA lists are empty; same revision/path alone is not complete payload identity proof',
                  'SH3 has no generation observation/seed/protocol/reference; no cross-server generation equivalence claim'],
        decision='KEEP_SELECTED_W0_GENERATION_DEFERRED; SH2 independently verified values retained with separate provenance',
        factual_cells='KEEP_SH3_61813',GPU=0)
    require(all(compatibility[k] for k in ['revision_match','stream_match','ordered_cases_match','requests_match','SH3_cold_verified']), 'W0_BASIC_BINDING')
    write_new(OUT/'w0-compatibility.json',compatibility)
    data.update(nonce='USER-GH-FLUCON-PAPER-SCALE-TABLE-REFRESH-20261010-R1',
        accepted_turn='01a121bf-dc12-7420-9754-c50086b87646',W0_generation=w0row,
        display_reducer=member(__file__),CPU_display_controls=4,
        broadcast='NO_BROADCAST_NOT_REQUIRED',raw_and_WB_unchanged=True,
        model_forward_calls=0,job_mutations=0,checkpoint_deletions=0,
        shared_display_source=member(REPO/'official/evaluation/generation/paper_display.py'),
        compatibility=member(OUT/'w0-compatibility.json'))
    write_new(OUT/'paper-table-ready.json',data)
    with (OUT/'paper-table-ready.csv').open('x',newline='') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['model','dataset','method','job_id','status','Flu_raw_bits','Flu_x100','Con_raw_cosine','Con_x100'])
        for r in data['rows']+[w0row]:
            a=r['generation']['Flu'];b=r['generation']['Con']
            w.writerow([r['model'],r['dataset'],r['method'],r['job_id'],r.get('status',r.get('selected_table_generation_status')),
                        a['raw_value'] if isinstance(a,dict) else a,a['paper_display_x100'] if isinstance(a,dict) else a,
                        b['raw_value'] if isinstance(b,dict) else b,b['paper_display_x100'] if isinstance(b,dict) else b])
    print(json.dumps(dict(rows=len(data['rows']),W0_Flu=g['Flu'],W0_Con=g['Con'])))

if __name__=='__main__':
    main()
