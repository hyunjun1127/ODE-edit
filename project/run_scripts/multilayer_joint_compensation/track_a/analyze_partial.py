"""CPU-only immutable Middle A0 partial publication; never invokes a model.

Endpoint JSONs supply all performance observations. Tensor files are read on
CPU only for already-saved physical endpoint norms/identities. No inference,
resampling, imputation or retrospective performance gate is introduced.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import stat
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA = Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')
DATA_SHA = '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
PANELS = ('Current100', 'Fixed100', 'Past100')
METRICS = ('RS', 'PS', 'NS')


class AnalysisBoundary(RuntimeError):
    pass


def require(ok, text):
    if not ok: raise AnalysisBoundary(text)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def member(path):
    p = Path(path).absolute()
    for part in [p, *p.parents]:
        require(not part.is_symlink(), f'SYMLINK_PATH: {part}')
    s = p.lstat()
    require(stat.S_ISREG(s.st_mode), f'NONREGULAR_FILE: {p}')
    return dict(path=str(p), bytes=s.st_size, mode=oct(stat.S_IMODE(s.st_mode)), sha256=sha(p))


def read(path):
    member(path)
    return json.loads(Path(path).read_text())


def save_json(path, value):
    with Path(path).open('xb') as f: f.write(canonical(value) + b'\n')


def write_csv(path, rows):
    require(bool(rows), 'EMPTY_PUBLICATION_TABLE')
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def verify_package(path):
    doc = read(path)
    require(digest(doc['members']) == doc['members_root'], 'MEMBERS_ROOT_MISMATCH')
    seen = set()
    for expected in doc['members']:
        p = Path(expected['path'])
        require(p.is_relative_to(Path(path).parent), 'RAW_MEMBER_ESCAPES_PACKAGE')
        require(str(p) not in seen, 'DUPLICATE_RAW_MEMBER')
        seen.add(str(p)); require(member(p) == expected, f'RAW_MEMBER_MISMATCH: {p}')
    return doc, dict(seal=member(path), member_count=len(seen),
                     member_bytes=sum(x['bytes'] for x in doc['members']),
                     members_root=doc['members_root'], verified_members=doc['members'])


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 1 and len(x) and np.isfinite(x).all(), 'NONFINITE_OR_EMPTY_METRIC')
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.quantile(x, .9, method='linear')), max=float(x.max()))


def endpoint_rows(doc, panel, records, *, current_only=False):
    expected = []
    for name, ordinals in panel['panels'].items():
        if current_only and not name.startswith('Current'): continue
        for metric, field in [('RS', None), ('PS', 'paraphrase_prompts'), ('NS', 'neighborhood_prompts')]:
            if current_only and metric == 'NS': continue
            for ordinal in ordinals:
                r = records[ordinal]; rw = r['requested_rewrite']
                prompts = [rw['prompt'].format(rw['subject'])] if field is None else r[field]
                for i, prompt in enumerate(prompts):
                    identity = digest([r['case_id'], i, prompt, rw['target_new']['str'], rw['target_true']['str']])
                    expected.append((name, metric, r['case_id'], i, identity))
    actual = [(r['panel'], r['metric'], r['case_id'], r['prompt_index'], r['identity']) for r in doc['rows']]
    require(actual == expected and len(set(actual)) == len(actual), 'ENDPOINT_SAMPLE_ORDER_OR_DUPLICATE')
    require(doc['pairs'] == len(expected) and doc['panel_identity'] == digest(panel), 'ENDPOINT_DENOMINATOR_PANEL')
    require(doc['controller_influence'] == 0, 'EVALUATOR_INFLUENCE')
    for r in doc['rows']:
        require(all(math.isfinite(r[k]) for k in ('new_nll', 'true_nll', 'margin')), 'NONFINITE_ENDPOINT')
        success = r['true_nll'] < r['new_nll'] if r['metric'] == 'NS' else r['new_nll'] < r['true_nll']
        require(r['success'] == success, 'WRONG_SUCCESS_DIRECTION_OR_TIE')
        require(r['margin'] == r['true_nll'] - r['new_nll'], 'NLL_MARGIN_IDENTITY')
        for prefix in ('new', 'true'):
            require(0 <= r[prefix + '_token_correct'] <= r[prefix + '_token_count'] and r[prefix + '_token_count'] > 0,
                    'TOKEN_DENOMINATOR')
            require(r[prefix + '_strict'] == (r[prefix + '_token_correct'] == r[prefix + '_token_count']), 'STRICT_IDENTITY')
    return doc['rows']


def summarize(rows, state):
    result = []
    for p in PANELS:
        for m in METRICS:
            rr = [r for r in rows if r['panel'] == p and r['metric'] == m]
            if not rr: continue  # Missing observations are not fabricated zero rows.
            row = dict(state=state, panel=p, metric=m, success_n=sum(r['success'] for r in rr), denominator=len(rr),
                       success_rate=sum(r['success'] for r in rr) / len(rr),
                       new_strict_n=sum(r['new_strict'] for r in rr), true_strict_n=sum(r['true_strict'] for r in rr),
                       new_token_correct=sum(r['new_token_correct'] for r in rr),
                       new_token_denominator=sum(r['new_token_count'] for r in rr),
                       true_token_correct=sum(r['true_token_correct'] for r in rr),
                       true_token_denominator=sum(r['true_token_count'] for r in rr))
            for field in ('new_nll', 'true_nll', 'margin'):
                row.update({field + '_' + k: v for k, v in stats([r[field] for r in rr]).items()})
            result.append(row)
    return result


def attribution(states):
    """Signed NLL improvement, joint interaction; no fabricated Shapley effect."""
    indexed = {name: {(r['metric'], r['identity']): r for r in rows if r['panel'] == 'Current100' and r['metric'] != 'NS'}
               for name, rows in states.items()}
    reference = indexed['We']
    require(all(set(x) == set(reference) for x in indexed.values()), 'ATTRIBUTION_UNPAIRED')
    rows = []
    for key, base in reference.items():
        w4, w8, both = (indexed[n][key] for n in ('We_plus_D4', 'We_plus_D8', 'A0'))
        e4, e8, e48 = (base['new_nll'] - r['new_nll'] for r in (w4, w8, both))
        rows.append(dict(metric=key[0], case_id=base['case_id'], prompt_index=base['prompt_index'],
                         identity=key[1], e4=e4, e8=e8, e48=e48, interaction=e48-e4-e8,
                         marginal_L4_given_L8=e48-e8, marginal_L8_given_L4=e48-e4,
                         entry_success=base['success'], joint_success=both['success'],
                         entry_success_to_failure=base['success'] and not both['success'],
                         entry_failure_to_success=not base['success'] and both['success']))
    aggregate = []
    for m in ('RS', 'PS'):
        rr = [r for r in rows if r['metric'] == m]
        for field in ('e4', 'e8', 'e48', 'interaction', 'marginal_L4_given_L8', 'marginal_L8_given_L4'):
            vv = [r[field] for r in rr]
            aggregate.append(dict(metric=m, quantity=field, denominator=len(rr), **stats(vv),
                                  positive_n=sum(x > 0 for x in vv), zero_n=sum(x == 0 for x in vv), negative_n=sum(x < 0 for x in vv)))
    return rows, aggregate


def functional_summary(doc, bank, calibration):
    result = []; flat = []; checks = []
    for role, data in doc.items():
        values, nlls, weights = [], [], []
        for row in data['context_rows']:
            require(len(row['values']) == len(row['nll']) == len(row['context_weights']), 'FUNCTIONAL_CONTEXT_LENGTH')
            ids = row['identity'].split('|')
            require(len(ids) == len(row['values']), 'FUNCTIONAL_ID_LENGTH')
            for ident, v, nll, w in zip(ids, row['values'], row['nll'], row['context_weights']):
                require(all(math.isfinite(x) for x in (v, nll, w)), 'FUNCTIONAL_NONFINITE')
                flat.append(dict(role=role, identity=ident, raw_value=v, context_nll=nll, context_weight=w))
                values.append(v); nlls.append(nll); weights.append(w)
        require(len(values) == bank['context_counts'][role] and len(set(r['identity'] for r in flat if r['role'] == role)) == len(values),
                'FUNCTIONAL_DENOMINATOR')
        require(abs(sum(weights)-1.) < 1e-12, 'FUNCTIONAL_WEIGHT_SUM')
        value = float(np.dot(values, weights)); nll = float(np.dot(nlls, weights))
        for name, exact, recorded, scale in [('risk', value, data['value'], sum(abs(v*w) for v,w in zip(values,weights))),
                                             ('nll', nll, data['mean_nll'], sum(abs(v*w) for v,w in zip(nlls,weights)))]:
            bound = 8 * np.finfo(np.float32).eps * max(scale, 1.)
            require(abs(exact-recorded) <= bound, 'FUNCTIONAL_WEIGHTED_REDUCTION')
            checks.append(dict(role=role, field=name, residual=abs(exact-recorded), bound=bound,
                               rule='8*eps_FP32*max(1,sum_abs_weighted_terms)'))
        nr = calibration['raw_native_risk'].get(role)
        result.append(dict(role=role, contexts=len(values), requests=len(bank['inventory']['current_effective']) if role=='Current'
                          else len(bank['inventory']['bank'][role]), weighted_value=value, recorded_weighted_value=data['value'],
                          mean_nll=nll, negative_raw_value_n=sum(x<0 for x in values),
                          **{'value_'+k:v for k,v in stats(values).items()},
                          native_same_bank_value=nr if nr is not None else 'NOT_RECORDED',
                          A0_to_native_ratio=value/nr if nr is not None and nr>0 else 'N/A'))
    return result, flat, checks


def scheduler_rows():
    text = subprocess.check_output(['sacct','-j','44970,44952','--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,Submit,Start,End',
                                    '--parsable2','--noheader'], text=True)
    result = []
    for line in text.splitlines():
        cols = line.split('|')
        if cols[0] not in ('44952','44970'): continue
        require(cols[2:4] == ['COMPLETED','0:0'] and 'gres/gpu=1' in cols[5], 'SCHEDULER_TERMINAL')
        result.append(dict(job_id=cols[0], phase='common_setup' if cols[0]=='44952' else 'A0',
                           job_name=cols[1], state=cols[2], exit_code=cols[3], GPU_seconds=int(cols[4]),
                           alloc_tres=cols[5], submit=cols[6], start=cols[7], end=cols[8],
                           accounting='allocation elapsed * one GPU; batch/extern not added'))
    require({r['job_id'] for r in result} == {'44952','44970'}, 'MISSING_SCHEDULER_JOB')
    return result


def plots(summary, attr, trajectory, physical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':120,
                         'savefig.dpi':120,'axes.spines.top':False,'axes.spines.right':False})
    result = {}
    def emit(name, fig):
        buff=io.BytesIO(); fig.savefig(buff,format='png',metadata={'Software':'repository-python-analyzer-v1'})
        plt.close(fig); result[name]=buff.getvalue()
    fig, axes = plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for ax, p in zip(axes,PANELS):
        rr=[next(r for r in summary if r['state']=='A0' and r['panel']==p and r['metric']==m) for m in METRICS]
        ax.bar(METRICS,[100*r['success_rate'] for r in rr],color=['#215a9b','#d68b2d','#3c8562'])
        ax.set_ylim(0,109);ax.set_title(p);ax.set_ylabel('NLL-pair success (%)')
        for i,r in enumerate(rr):ax.text(i,100*r['success_rate']+2,f"{r['success_n']}/{r['denominator']}",ha='center',fontsize=9)
    emit('endpoint-rates.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    names=['We','We_plus_D4','We_plus_D8','A0']
    for m in ('RS','PS'):
        rr=[next(r for r in summary if r['state']==n and r['panel']=='Current100' and r['metric']==m) for n in names]
        axes[0].plot(range(4),[r['new_nll_mean'] for r in rr],marker='o',label=m)
    axes[0].set_xticks(range(4),['We','We+D4','We+D8','Joint']);axes[0].set_ylabel('Target-new NLL');axes[0].legend()
    fields=['e4','e8','e48','interaction']
    for i,m in enumerate(('RS','PS')):
        axes[1].bar(np.arange(4)+(i-.5)*.35,[next(r['mean'] for r in attr if r['metric']==m and r['quantity']==f) for f in fields],width=.35,label=m)
    axes[1].set_xticks(range(4),fields);axes[1].axhline(0,color='black',linewidth=.6);axes[1].legend();axes[1].set_ylabel('Signed NLL improvement / interaction')
    emit('current-signed-attribution.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for field,label in [('current_nll','Current six-context NLL'),('objective','Total A0 objective')]:
        axes[0].plot([r['completed_updates'] for r in trajectory],[r[field] for r in trajectory],label=label)
    axes[0].legend();axes[0].set_xlabel('Completed Adam updates');axes[0].set_ylabel('Objective value')
    for layer in (4,8):axes[1].plot([r['completed_updates'] for r in trajectory],[r[f'writer_metric_energy_L{layer}'] for r in trajectory],label=f'L{layer}')
    axes[1].set_xlabel('Completed Adam updates');axes[1].set_ylabel('Native writer metric energy (not Frobenius)');axes[1].legend()
    emit('a0-optimization-trajectory.png',fig)
    fig,ax=plt.subplots(figsize=(5.2,3.8),layout='constrained')
    ax.bar([f"L{r['layer']}" for r in physical],[r['actual_FP32_delta_frobenius'] for r in physical],color=['#215a9b','#d68b2d'])
    ax.set_title('Layer-wise Update Magnitude');ax.set_ylabel('Actual endpoint delta Frobenius norm')
    emit('layer-wise-update-magnitude.png',fig)
    return result


def build(args):
    import torch
    torch.set_num_threads(4)
    a, c, out = (Path(x).absolute() for x in (args.a0,args.common,args.output))
    require(not out.exists(), 'PUBLICATION_CREATE_ONCE')
    terminal, tverify = verify_package(a/'terminal.json'); common,cverify=verify_package(c/'READY.json')
    require(terminal['status']=='TERMINAL_VALID' and terminal['arm']=='A0' and not terminal['full_campaign_completed'], 'WRONG_PARTIAL_TERMINAL')
    require(common['entry']==terminal['entry']=='Middle', 'ENTRY_MISMATCH')
    binding=read(a/'common-binding.json');require(binding['ready']==member(c/'READY.json') and binding['members_root']==common['members_root'], 'COMMON_BINDING')
    require(binding['common_source_head']==common['source_head'], 'COMMON_SOURCE')
    trajectory=read(a/'A0-trajectory.json');require(trajectory['common_ready_sha']==sha(c/'READY.json') and trajectory['entry_identity']==common['entry_identity'], 'TRAJECTORY_ENTRY')
    run=read(a/'run.lock.json');require(run['source_head']==terminal['source_head']==terminal['endpoint_identity']['source_head'] and run['job']=='44970','EXECUTION_SOURCE')
    input_doc=read(c/'input.lock.json');panel=input_doc['binding']['panel'];bank=read(c/'bank-manifest.json')
    require(bank['entry']==common['entry_identity'], 'BANK_ENTRY')
    require(sha(DATA)==DATA_SHA, 'SEALED_DATASET_SHA');records=read(DATA)
    require(len(records)==10000, 'DATASET_COUNT')
    for idx,h in panel['record_hashes'].items():require(digest(records[int(idx)])==h, 'PANEL_RECORD_IDENTITY')
    require(panel['panels']['Current100']==bank['inventory']['current_effective']==list(range(5000,5100)), 'CURRENT_ORDINALS')
    states={}
    for name,file in [('A0','endpoint-metrics.json'),('We','attribution-We.json'),('We_plus_D4','attribution-We_plus_D4.json'),('We_plus_D8','attribution-We_plus_D8.json')]:
        states[name]=endpoint_rows(read(a/file),panel,records,current_only=name!='A0')
    summary=[row for name,rows in states.items() for row in summarize(rows,name)]
    paired,attr=attribution(states)
    functional,contexts,weighted_checks=functional_summary(read(a/'functional-endpoint.json'),bank,common['calibration'])
    hfinal=read(a/'history-finalization/complete.json');hist=read(c/'history-L8/complete.json')
    require(hfinal==terminal['history'] and hfinal['endpoint_identity']==terminal['endpoint_identity'], 'TERMINAL_HISTORY_BINDING')
    require(hfinal['terminal_batch_finalizations']==1 and hfinal['terminal_layer_appends']==2 and hfinal['inner_history_appends']==0,'HISTORY_COUNTS')
    require(hist['completed_requests']==5000 and hist['completed_batches']==50 and hist['original_M4_unchanged'], 'HISTORY_RECONSTRUCTION')
    require(digest({k:v for k,v in hist.items() if k!='identity'})==hist['identity'], 'HISTORY_ROOT_IDENTITY')
    require(digest(hist['chunks'])==hist['chunks_root'], 'HISTORY_CHUNKS_ROOT')
    checks=read(a/'gpu-initial-checks.json')
    require(checks['status']=='INITIAL_CORRECTNESS_PASS' and not checks['failures']
            and checks['full_live_pointer_version_bytes_restored'] and not checks['changed_parameters'], 'RECORDED_INITIAL_GATE')
    require(checks['full_parameter_entry_root']==checks['full_parameter_exit_root'], 'FULL_PARAMETER_RESTORE')
    counts=terminal['compute']['counts']
    require(counts['authoritative_endpoint_materializations']==counts['terminal_restores']==1 and counts['terminal_version_increments']==0, 'TERMINAL_TRANSACTION')
    receipt=trajectory['receipt']; require(receipt['counts']['optimizer_updates']==25 and receipt['counts']['native_z_calls']==receipt['counts']['history_appends']==0, 'A0_COUNTS')
    rows=[]
    for x in trajectory['trajectory']+[dict(receipt['final'],step=26)]:
        row=dict(completed_updates=x['step']-1,position=x['position'],current_nll=x['current_nll'],balance=x['balance'],objective=x['objective'])
        require(abs(row['objective']-(row['current_nll']+.1*row['balance']))<1e-10,'A0_OBJECTIVE_IDENTITY')
        for i,l in enumerate((4,8)):row[f'writer_metric_energy_L{l}']=x['writer_metric_energy'][i]
        rows.append(row)
    require(len(rows)==26 and [x['completed_updates'] for x in rows]==list(range(26)), 'A0_TRAJECTORY_COUNT')
    # mmap CPU storage; never instantiate/evaluate a model.
    entry=torch.load(c/'entry.pt',map_location='cpu',weights_only=True,mmap=True)
    endpoint=torch.load(a/'endpoint.pt',map_location='cpu',weights_only=True,mmap=True)
    plan=torch.load(a/'A0-plan.pt',map_location='cpu',weights_only=True,mmap=True)
    require(entry['entry_identity']==endpoint['entry_identity']==plan['entry_identity']==common['entry_identity'],'TENSOR_ENTRY_BINDING')
    physical=[]
    for i,l in enumerate((4,8)):
        w0,we,w=entry['W0'][i],entry['We'][i],endpoint['weights'][i]
        actual=w.double()-we.double(); proposed=plan['deltas'][i].double()
        require(w.dtype==we.dtype==torch.float32 and bool(torch.isfinite(w).all()),'PHYSICAL_DTYPE_FINITE')
        require(hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest()==terminal['endpoint_identity']['weights'][entry['names'][i]], 'ENDPOINT_TENSOR_SHA')
        geo=read(c/f'native-geometry-L{l}.json')
        physical.append(dict(layer=l,actual_FP32_delta_frobenius=float(actual.norm()),actual_FP32_delta_frobenius_sq=float(actual.square().sum()),
          proposed_delta_frobenius=float(proposed.norm()),materialization_roundoff_frobenius=float((actual-proposed).norm()),
          W0_net_frobenius=float((w.double()-w0.double()).norm()),RHS_frobenius=float(plan['rhs'][i].double().norm()),
          q_raw=receipt['q_raw'][i],q_balanced=receipt['q_balanced'][i],native_writer_metric_energy=receipt['final']['writer_metric_energy'][i],
          projector_rank=geo['projector_rank'],input_dimension=geo['dimension'],native_metric_mean_eigenvalue=geo['mean_eigenvalue']))
        del actual,proposed
    total=sum(x['actual_FP32_delta_frobenius_sq'] for x in physical)
    for x in physical:x['physical_energy_share']=x['actual_FP32_delta_frobenius_sq']/total if total else None
    generation=read(a/'generation.json');require(len(generation['rows'])==60 and generation['greedy'] and generation['max_new_tokens']==32, 'GENERATION_COMPLETENESS')
    generated=dict(rows=60,literal_prefix_n=sum(r['literal_prefix'] for r in generation['rows']),semantic_accuracy='NOT_MEASURED',
                   raw_outputs_committed=False)
    schedule=scheduler_rows();require({r['job_id']:r['GPU_seconds'] for r in schedule}=={'44952':916,'44970':1561},'ALLOCATION_COST_CHANGED')
    compute=[]
    for phase,raw in [('common_setup',common),('A0',terminal)]:
        for unit,group in [('seconds',raw['compute']['seconds']),('count',raw['compute']['counts'])]:
            for name,v in group.items():compute.append(dict(phase=phase,component=name,unit=unit,value=v,
                interpretation='measured scoped counter/time; timings can be nested, do not sum as total'))
    out.mkdir(parents=True)
    write_csv(out/'endpoint-summary.csv',summary);write_csv(out/'signed-attribution.csv',paired)
    write_csv(out/'signed-attribution-summary.csv',attr);write_csv(out/'functional-summary.csv',functional)
    write_csv(out/'functional-context-metrics.csv',contexts);write_csv(out/'a0-trajectory.csv',rows)
    write_csv(out/'physical-layer-update.csv',physical);write_csv(out/'compute-accounting.csv',compute)
    write_csv(out/'scheduler-accounting.csv',schedule)
    save_json(out/'generation-summary.json',generated)
    # Re-execute deterministic rendering twice from the identical derived input.
    figs=plots(summary,attr,rows,physical);again=plots(summary,attr,rows,physical)
    require(figs==again, 'PNG_BYTE_REPRODUCTION')
    for name,blob in figs.items():
        with (out/name).open('xb') as f:f.write(blob)
    import matplotlib
    command=f'{sys.executable} -m project.run_scripts.multilayer_joint_compensation.track_a.analyze_partial --a0 {a} --common {c} --output <new-create-once-output>'
    save_json(out/'plot-reproduction.json',dict(command=command,render_count=2,byte_identical=True,
      environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,torch=torch.__version__),
      input_files={n:sha(out/n) for n in ['endpoint-summary.csv','signed-attribution-summary.csv','a0-trajectory.csv','physical-layer-update.csv']},
      figures={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in figs.items()},
      mechanism='same Python plotting function re-executed with the identical derived objects; no image-generation tools/manual edits'))
    verified=[]
    for check in (tverify,cverify):
        for m in check['verified_members']:
            require(member(m['path'])==m,'INPUT_CHANGED_DURING_ANALYSIS')
            verified.append(m)
        require(member(check['seal']['path'])==check['seal'],'SEAL_CHANGED_DURING_ANALYSIS')
    require(sha(DATA)==DATA_SHA,'DATASET_CHANGED_DURING_ANALYSIS')
    source=[]
    for head,names in [(terminal['source_head'],['track_a/run_a0.py','track_a/planner.py','functional.py','evaluation.py','history.py']),
                       (common['source_head'],['common_reference/prepare.py','history.py','track_a/native_geometry.py'])]:
        for name in names:
            path='project/run_scripts/multilayer_joint_compensation/'+name
            blob=subprocess.check_output(['git','show',f'{head}:{path}'],cwd=ROOT)
            source.append(dict(execution_head=head,path=path,git_blob=subprocess.check_output(['git','rev-parse',f'{head}:{path}'],cwd=ROOT,text=True).strip(),sha256=hashlib.sha256(blob).hexdigest()))
    analysis_source=[member(Path(__file__)),member(ROOT/'project/run_scripts/multilayer_joint_compensation/tests/test_a_partial_analysis.py')]
    verification=dict(status='PASS',raw_member_count=len(verified),raw_member_bytes=sum(m['bytes'] for m in verified),
      before_after_sha_bytes_mode_unchanged=True,terminal=tverify,common=cverify,dataset=member(DATA),
      endpoint_pairs=3900,attribution_observations=900,logical_current_requests=100,current_contexts=600,
      history_reconstructed_requests=5000,history_terminal_batch_finalizations=1,history_terminal_layer_appends=2,
      history_inner_appends=0,optimizer_updates=25,PCG='NOT_APPLICABLE_A0_ADAM',weighted_reduction_checks=weighted_checks,
      recorded_initial_gate_status=checks['status'],actual_new_model_actions=0,new_gpu_actions=0,new_submissions=0,new_endpoint_evaluations=0,
      verification_scope='full sealed file hashes/metadata + CPU sample/denominator/identity and saved tensor checks; no new model test',
      scientific_promotion=False)
    save_json(out/'input-verification.json',verification)
    coverage=[dict(scope='Middle A0',status='TERMINAL_VALID',new_endpoint_count=1),
      dict(scope='Middle N4',status='EXACT_WEIGHT_REFERENCE_REUSED; full same-panel metrics NOT_RECORDED_HERE',new_endpoint_count=0),
      dict(scope='Middle AOS',status='NO_LOCAL_TERMINAL_INPUT_IN_THIS_PACKAGE',new_endpoint_count=0),
      dict(scope='Middle BOS',status='PEER_REPORTED_COMPLETED_UNVERIFIED_NOT_INCLUDED_IN_MEASURED_TABLES',new_endpoint_count=0),
      dict(scope='Other A/B arms, Early/Late, nested comparisons',status='NOT_RUN_IN_THIS_PARTIAL_PUBLICATION',new_endpoint_count=0),
      dict(scope='Four 10-batch short chains',status='NOT_RUN_IN_THIS_PARTIAL_PUBLICATION',new_endpoint_count=0),
      dict(scope='We NS and Fixed/Past before-after loss/recovery',status='NOT_RECORDED_SCHEMA_GAP',new_endpoint_count=0)]
    write_csv(out/'coverage.csv',coverage)
    text=report(summary,attr,functional,rows,physical,schedule,common,terminal,generated,verification,command)
    with (out/'diagnostic-report-ko.md').open('x') as f:f.write(text)
    members=[dict(name=p.name,bytes=p.stat().st_size,mode=oct(stat.S_IMODE(p.stat().st_mode)),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]
    manifest=dict(schema='multilayer-middle-A0-partial-analysis-v1',scientific_promotion=False,
      execution_source_head=terminal['source_head'],execution_source_tree=subprocess.check_output(['git','rev-parse',terminal['source_head']+'^{tree}'],cwd=ROOT,text=True).strip(),
      common_source_head=common['source_head'],common_source_tree=subprocess.check_output(['git','rev-parse',common['source_head']+'^{tree}'],cwd=ROOT,text=True).strip(),
      analysis_source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),analysis_source_files=analysis_source,
      execution_source_files=source,inputs=[tverify['seal'],cverify['seal']],source_and_result_immutability=True,
      members=members,members_root=digest(members),reproduction_command=command)
    save_json(out/'manifest.json',manifest)
    receipt_out=dict(status='PARTIAL_MIDDLE_A0_REVIEW_READY',manifest_sha256=sha(out/'manifest.json'),
      members_root=manifest['members_root'],report_sha256=sha(out/'diagnostic-report-ko.md'),
      raw_inputs_sha256=[tverify['seal']['sha256'],cverify['seal']['sha256']],
      completed_new_endpoint_count=1,full_campaign_complete=False,GPU_seconds=sum(r['GPU_seconds'] for r in schedule),
      new_model_GPU_evaluation_submission_actions=0,scientific_promotion=False)
    receipt_out['identity']=digest(receipt_out);save_json(out/'rooted-receipt.json',receipt_out)
    for m in members:require(sha(out/m['name'])==m['sha256'],'PUBLICATION_REHASH')
    print(json.dumps(dict(output=str(out),report_sha256=receipt_out['report_sha256'],manifest_sha256=receipt_out['manifest_sha256'],
                         receipt_sha256=sha(out/'rooted-receipt.json'),receipt_identity=receipt_out['identity']),ensure_ascii=False))


def report(summary,attr,functional,traj,physical,schedule,common,terminal,generation,verify,command):
    def fmt(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x)
    def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(fmt(x) for x in row)+' |\n' for row in rows)
    rate_rows=[[r['state'],r['panel'],r['metric'],f"{r['success_n']}/{r['denominator']}",100*r['success_rate'],r['new_nll_mean'],r['true_nll_mean'],r['new_strict_n']] for r in summary]
    frows=[[r['role'],r['requests'],r['contexts'],r['weighted_value'],r['value_p90'],r['value_max'],r['native_same_bank_value'],r['A0_to_native_ratio']] for r in functional]
    arows=[[r['metric'],r['quantity'],r['mean'],r['median'],r['p90'],r['positive_n'],r['negative_n']] for r in attr]
    return f'''# Middle B100 A0 공동 편집 — 부분 사실 보고서

상태: PARTIAL_MIDDLE_A0_REVIEW_READY. **전체 A/B campaign 완료 보고가 아니다.** Llama3-8B-Instruct의 같은 Middle W50 entry에서 A0 한 경로만 새 endpoint로 완료했다. AOS/BOS의 local terminal과 나머지 arm·4개 short chain은 이 패키지에 없다. 이 보고는 저장된 결과에 대한 CPU 분석이며 새 모델/평가/GPU/제출은 모두 0이다. scientific_promotion=false.

## 1. 지표와 관측 범위

RS는 rewrite, PS는 두 paraphrase prompt, NS는 열 neighborhood prompt에 대한 NLL-pair strict 비교다. RS/PS는 target-new NLL < target-true NLL, NS는 target-true NLL < target-new NLL이다. Tie는 실패다. NLL은 target token 평균이며 낮을수록 해당 target을 더 선호한다. Margin=true−new는 RS/PS에서 양수가 유리하고 NS에서는 음수가 유리하다. Teacher-forced strict accuracy는 모든 token argmax 일치이며 preference success와 다른 지표다.

각 Current100/Fixed100/Past100에서 분모는 RS100·PS200·NS1000이고 전체 endpoint는 3900 prompt-pair다. Current는 frozen10k ordinal5000:5100, B51의 유효100개다. 세 패널은 서로 다른 역할이며 한 pooled 성공률을 대표값으로 만들지 않았다. PS prompt200개를 request200개로 부르지 않는다. We와 D4-only/D8-only의 저장 관측은 Current RS/PS 각300pair뿐이며 **We NS 및 Fixed/Past entry-before 값은 미기록**이다. 따라서 NS 개선/악화, Fixed/Past forgetting 또는 recovery를 A0 endpoint 수준만으로 판정하지 않는다.

## 2. Endpoint 결과

{table(['state','panel','metric','success n/d','rate %','new NLL mean','true NLL mean','new strict n'],rate_rows)}

new strict의 분모는 각 success 분모와 같다. NS에서는 true strict가 보존-target strict이므로 표의 new strict를 locality accuracy로 해석하지 않는다. 모든 new/true NLL mean/median/p90/max, margin, strict 및 token-correct n/d는 [endpoint-summary.csv](endpoint-summary.csv)에 있다. **높은 Current RS와 동시에 측정된 Base/Past harm은 그대로 남긴다.**

![Endpoint rates](endpoint-rates.png)

## 3. 네 실제 state의 signed attribution

L(W)를 Current target-new NLL이라 할 때 e4=L(We)−L(We+D4), e8=L(We)−L(We+D8), e48=L(We)−L(We+D4+D8), interaction=e48−e4−e8이다. 아래 값은 실제 저장된 네 state의 request/prompt-matched 관측이며 단순 weight norm share가 아니다. Conditional marginal은 e48−e8 및 e48−e4다. 음의 interaction은 개별 효과의 중복/비선형성까지 포함하며 곧바로 방해 또는 실패를 뜻하지 않는다. 합의 NLL 개선이 양수인 사실만으로 50:50 부담 분산이나 A0-L4 대비 capacity 우위를 입증하지 않는다.

{table(['metric','quantity','mean','median','p90','positive n','negative n'],arows)}

개별 signed 값/entry success→failure 및 failure→success flags는 [signed-attribution.csv](signed-attribution.csv)에 있다. 다른 패널에는 entry 관측이 없어 해당 paired 값을 만들지 않았다. A0-L4, A0-bal0, AOS/BF 및 같은-entry native 전체 성능 비교가 아직 없으므로 공동 target 자체·추가 L8·balance 효과는 분리되지 않았다.

![Signed attribution](current-signed-attribution.png)

## 4. Functional Base/Past와 독립 Audit

Base는 고정 We full-vocabulary teacher의 KL(p_We∥p_endpoint), Past는 context별 token-mean NLL 증가에 ψτ(τ=.1)을 적용한 뒤 context/request 평균이다. BaseAudit/PastAudit는 controller bank와 분리된 동일 정의의 관측이다. Current functional은 **We-relative KL/.1 + NLL-profile 제곱/(2·.1²)** 관측이며 A0가 최소화한 edit NLL 또는 향후 BF의 시간별 reference와 동일하지 않다. 여기서는 이를 새로운 성능 gate로 삼지 않는다.

{table(['role','requests','contexts','weighted value','raw context p90','raw max','native same-bank value','A0/native'],frows)}

Native Base/Past 값은 common calibration에 봉인된 같은 We의 N4 risk이며 N4의 새 full3900 endpoint 평가를 대신하지 않는다. Audit native risk와 We NS는 미기록이다. 모든 raw negative KL count와 weighted reduction residual은 [functional-summary.csv](functional-summary.csv), [input-verification.json](input-verification.json)에 남겼다. Microbatch 평균을 다시 평균내지 않고 저장된 context weight로 독립 재집계했다. Controller bank fitting과 audit 일반화 또는 locality 우월성은 아직 비교군이 없으므로 주장하지 않는다.

## 5. A0 최적화와 layer magnitude

Adam25 updates, lr=.1, β=(.9,.999), epsilon1e−8, balance λ=.1이다. q는 entry에서 한 번 캡처하고 두 RHS는 zero에서 공동 시작했다. 25번 update와 최종 observation을 합해 26개 callback gradient set이 기록됐으며 7800 physical backward microbatch=26×300이다. **PCG는 A0에 적용되지 않으며 N/A**다. 초기 derivative/transaction 검사의 JVP9·FD54는 진단 비용으로 분리한다.

Train six-context Current NLL은 {fmt(traj[0]['current_nll'])}→{fmt(traj[-1]['current_nll'])}, 총 objective는 {fmt(traj[0]['objective'])}→{fmt(traj[-1]['objective'])}다. 마지막 unweighted balance={fmt(traj[-1]['balance'])}, weighted contribution={fmt(.1*traj[-1]['balance'])}이다. 편집 NLL이 낮아졌어도 balance를 포함한 총 목적값은 초기보다 높다. 이는 실제 finite optimization 결과이며 convergence 또는 objective 최적화 성공으로 포장하지 않는다. Pre-update step25는 completed24이며 final row는 post-update25다. [a0-trajectory.csv](a0-trajectory.csv)는 이 시간축을 명시한다.

{table(['layer','actual FP32 ΔW ||·||F','actual squared energy','energy share','planned Δ norm','cast/add difference norm','W0-net norm','native writer energy','q'],[[r['layer'],r['actual_FP32_delta_frobenius'],r['actual_FP32_delta_frobenius_sq'],r['physical_energy_share'],r['proposed_delta_frobenius'],r['materialization_roundoff_frobenius'],r['W0_net_frobenius'],r['native_writer_metric_energy'],r['q_raw']] for r in physical])}

Actual ΔW는 저장된 FP32 endpoint−We를 FP64로 차분했다. Planned Δ와 실제 덧셈 뒤 차이는 따로 남겼다. W0-net은 기존5000 edit의 누적 L4 상태를 포함하므로 이번 batch update와 섞지 않는다. Native metric energy는 Frobenius 제곱과 다른 단위다. 중간 step별 실제 dense delta는 저장되지 않았으므로 native energy trajectory를 물리 net displacement로 대체하지 않았다.

![Optimization](a0-optimization-trajectory.png)

![Layer magnitude](layer-wise-update-magnitude.png)

## 6. 비용·공통 준비·검증

{table(['job','phase','state','GPU seconds','GPU hours'],[[r['job_id'],r['phase'],r['state'],r['GPU_seconds'],r['GPU_seconds']/3600] for r in schedule])}

두 1GPU job의 실제 allocation 합은 {sum(r['GPU_seconds'] for r in schedule)} GPU-sec ({sum(r['GPU_seconds'] for r in schedule)/3600:.6f} GPUh)다. Slurm batch/extern 하위행을 중복 합산하지 않았다. common setup은 별도 재사용 비용이며 A0에 포함시킨 cold subtotal은 위 합계지만 원래 N4 z/이전 checkpoint 생성비용은 포함하지 않는다.

A0 joint target optimization {terminal['compute']['seconds']['A0_joint_target_optimization']:.3f}s, 그 안의 Current forward/backward {terminal['compute']['seconds']['current_forward_backward']:.3f}s, full endpoint evaluator {terminal['compute']['seconds']['full_endpoint_evaluation']:.3f}s, generation {terminal['compute']['seconds']['generation']:.3f}s다. Common native key capture {common['compute']['seconds']['native_key_capture']:.3f}s에는 현재 key와5000 raw-history key가 포함된다. Nested timer는 서로 더해 총 wall로 만들지 않는다. Actual model forward 호출은 common {common['compute']['counts']['actual_model_forward_invocations']}, A0 {terminal['compute']['counts']['actual_model_forward_invocations']}이고 기능별 wrapper count와 구분했다. A0 peak allocated GPU={terminal['peak_GPU_bytes']/2**30:.3f}GiB, common peak allocated={common['peak_allocated_GPU_bytes']/2**30:.3f}GiB다. Native z를 새로 최적화하지 않은 캐시 재사용과 A0 신규 RHS 최적화를 혼동하지 않는다.

L8 history는 같은 We에서 모든 과거 raw5000 requests를 B100×50 chronological FP32 Gram으로 재구성했다. M4는 원본을 유지하며 이것이 옛 L8 trajectory history와 동일하다는 주장은 없다. Endpoint 후 두 layer key를 한 번씩 finalization, terminal batch finalization1, inner append0가 봉인됐다. Full P* rank는 L4={physical[0]['projector_rank']}, L8={physical[1]['projector_rank']}; 100/256 rank truncate가 아니다.

초기 actual derivative/transaction receipt의 PASS와 pointer/version/bytes restore를 확인했다. 이는 A0 initial probe 범위이며 아직 실행하지 않은 full protection PCG의 PASS가 아니다. Raw terminal/common 총 {verify['raw_member_count']}개 member, {verify['raw_member_bytes']}bytes의 SHA/bytes/mode/root를 전후 두 번 검증했다. 표본/order·panel/row identity·NLL 비교방향·finite·strict token분모·history·source binding도 독립 확인했다. Raw tensor의 file SHA 검증과 model inference 재검증은 다른 수준이며 후자는 새로 하지 않았다. 외부SH2 검증 여부는 이 local sealed READY의 false/ACK_REQUIRED를 임의 갱신하지 않았다.

Generation은 고정20 requests×3prompts=60행, greedy≤32tokens; literal-prefix {generation['literal_prefix_n']}/60이다. 이는 semantic accuracy가 아니며 raw 생성문은 Git에서 제외했다.

## 7. 한계와 다음 해석

관측: A0의 Current 편집성은 높고 두 layer 단독 intervention과 joint intervention이 다른 결과를 냈다. 동시에 Base/Past functional harm은 같은-bank native calibration보다 컸고 총 A0 objective도 entry보다 높았다. 가능한 설명은 공동 writer의 nonlinear 전달 및 balance/고정 Adam 경로 사이 tradeoff지만, A0-L4·balance0·OS/BF가 없어 그 원인을 분리 확정할 수 없다. 현재 결과만으로 편집 부담 분산 성공, locality 개선, BF 효용, 후속10batch 보존 또는 lifelong 우위를 주장하지 않는다.

미실행/미기록 범위는 [coverage.csv](coverage.csv)에 분리했다. 낮은 성능이나 큰 harm 때문에 제외한 endpoint는 없다. 가장 중요한 남은 질문은 **동일 Current 품질에서 기능적 OS/BF가 독립 Base/Past audit 손상을 실제로 낮추는가**다. 이 부분 보고서가 후속 승인 범위를 늘리거나 자동 promotion하지 않는다.

## 8. 재현과 provenance

실행 A0 source `{terminal['source_head']}`, common source `{common['source_head']}`. Analysis source file SHA와 실행 Git blob은 [manifest.json](manifest.json), raw input closure는 [input-verification.json](input-verification.json), package root는 [rooted-receipt.json](rooted-receipt.json)에 있다. Model revision은 `{common['entry_identity']['model_revision']}`다.

```bash
{command}
```

PNG는 repository Python 코드로 동일 derived 입력에서 두 번 실제 rendering하여 byte equality를 확인했다. [plot-reproduction.json](plot-reproduction.json)에 입력/코드 실행명령·환경·출력 SHA를 결속했다. Endpoint 원문 prompt, teacher, tensor, model, full log는 Git에 포함하지 않는다. 본 보고서의 분석은 설명적 비교이며 과학적 promotion은 false다.
'''


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--a0',required=True);parser.add_argument('--common',required=True);parser.add_argument('--output',required=True)
    build(parser.parse_args())
