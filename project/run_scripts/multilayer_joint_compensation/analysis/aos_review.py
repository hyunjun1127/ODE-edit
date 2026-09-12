"""Immutable, CPU-only A-OS review. No model/evaluator/scheduler invocation.

Reuses the pinned, independently published CPU reduction helper, not a scientific
runtime. All inferential observations must already exist in sealed JSON files.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import types

import numpy as np

HELPER_COMMIT = 'ba91f274ddf452be4124aedfc1ff37c1fb92fdb8'
HELPER_PATH = 'project/run_scripts/multilayer_joint_compensation/track_a/analyze_partial.py'
HELPER_SHA = 'a3db0b15028252b43d4880fec570252842f5ac48a293c426c06375321ab5b5bb'
EXECUTION = '815f933e3bb6e6694cb3c0ad8bb6cd6ff63234d0'
DEFAULT_RAW = '/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-multilayer-joint-edit-a-v1'
REL = 'local/multilayer-joint-compensation/20260911-v1'
REPORT = 'experiment-reports/servers/server1/multilayer-joint-edit-a-2026-09-11-v1/middle-aos-review-recall-v1'


def helper(repo):
    source = subprocess.check_output(['git', '-C', str(repo), 'show', f'{HELPER_COMMIT}:{HELPER_PATH}'])
    if hashlib.sha256(source).hexdigest() != HELPER_SHA:
        raise RuntimeError('PINNED_CPU_HELPER_MISMATCH')
    module = types.ModuleType('pinned_cpu_reduction')
    module.__file__ = str(Path(repo) / HELPER_PATH)
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def pair_rows(h, before, after, comparison):
    key = lambda r: (r['panel'], r['metric'], r['identity'])
    b = {key(r): r for r in before}; a = {key(r): r for r in after}
    h.require(len(b) == len(before) and len(a) == len(after), 'DUPLICATE_PAIR_KEY')
    h.require(set(b) == set(a), 'PAIR_SUPPORT_MISMATCH')
    result = []
    for k, old in b.items():
        new = a[k]
        sign = -1 if old['metric'] == 'NS' else 1
        result.append(dict(comparison=comparison, panel=k[0], metric=k[1], identity=k[2],
                           case_id=old['case_id'], prompt_index=old['prompt_index'],
                           before_success=old['success'], after_success=new['success'],
                           loss=bool(old['success'] and not new['success']),
                           recovery=bool(not old['success'] and new['success']),
                           new_nll_delta=new['new_nll']-old['new_nll'],
                           true_nll_delta=new['true_nll']-old['true_nll'],
                           desired_margin_delta=sign*(new['margin']-old['margin'])))
    return result


def pair_summary(h, rows):
    result = []
    for comparison, panel, metric in sorted({(r['comparison'], r['panel'], r['metric']) for r in rows}):
        rr = [r for r in rows if (r['comparison'], r['panel'], r['metric']) == (comparison, panel, metric)]
        base = dict(comparison=comparison, panel=panel, metric=metric, denominator=len(rr),
                    before_n=sum(r['before_success'] for r in rr), after_n=sum(r['after_success'] for r in rr),
                    loss_n=sum(r['loss'] for r in rr), recovery_n=sum(r['recovery'] for r in rr))
        base['delta_pp'] = 100*(base['after_n']-base['before_n'])/len(rr)
        for field in ('new_nll_delta', 'true_nll_delta', 'desired_margin_delta'):
            vv = [r[field] for r in rr]
            result.append(dict(**base, quantity=field, **h.stats(vv), positive_n=sum(x>0 for x in vv),
                               equal_n=sum(x==0 for x in vv), negative_n=sum(x<0 for x in vv)))
    return result


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def physical_checks(h, common, run, parent, terminal, trajectory):
    import torch
    h.require(not torch.cuda.is_initialized(), 'CPU_REVIEW_CUDA_INITIALIZED')
    torch.set_num_threads(8)
    load = lambda p: torch.load(p, map_location='cpu', weights_only=True, mmap=True)
    entry, endpoint, anchor = load(common/'entry.pt'), load(run/'endpoint.pt'), load(parent/'endpoint.pt')
    final_m = load(run/'final-history.pt')
    rows, checks = [], []
    for j, layer in enumerate((4, 8)):
        w, we, wa = endpoint['weights'][j], entry['We'][j], anchor['weights'][j]
        h.require(w.dtype == torch.float32 and w.shape == we.shape == wa.shape == (4096, 14336), 'PHYSICAL_SHAPE_DTYPE')
        name = endpoint['names'][j]
        h.require(tensor_sha(w) == terminal['endpoint_identity']['weights'][name], 'ENDPOINT_TENSOR_HASH')
        delta = w.double()-we.double(); correction = w.double()-wa.double()
        actual = float(torch.linalg.vector_norm(correction)); cumulative = float(torch.linalg.vector_norm(delta))
        recorded = trajectory['per_layer'][j]
        # Independent FP64 reductions; report differences, no new performance/solver gate.
        rows.append(dict(layer=layer, actual_correction_frobenius=actual, actual_cumulative_frobenius=cumulative,
                         correction_reduction_abs_difference=abs(actual-recorded['actual_correction_frobenius']),
                         cumulative_reduction_abs_difference=abs(cumulative-recorded['actual_cumulative_frobenius']),
                         actual_correction_nonzero_n=int(torch.count_nonzero(correction)),
                         actual_correction_numel=correction.numel(), selected_sha256=tensor_sha(w),
                         intended_correction_frobenius=recorded['correction_frobenius'],
                         raw_native_energy_before=recorded['raw_native_energy_before'],
                         raw_native_energy_after=recorded['raw_native_energy_after'],
                         signed_current_linear_change=recorded['signed_current_linear_change']))
        del delta, correction
        mr = terminal['history']['members'][j]
        key = load(run/'history-finalization'/mr['key']['name'])
        if isinstance(key, dict): key = key['keys']
        hist = final_m[layer]
        h.require(tensor_sha(key) == mr['key']['tensor_sha256'], 'KEY_TENSOR_HASH')
        h.require(tensor_sha(hist) == mr['final_history_sha256'], 'HISTORY_TENSOR_HASH')
        base = entry['M4' if layer == 4 else 'M8']
        if base.ndim == 3: base = base[0]
        if hist.ndim == 3: hist = hist[0]
        # Recompute the source FP32 append in row blocks, not a second writer.
        ss = maxabs = refsq = 0.
        for start in range(0, len(hist), 256):
            expected = base[start:start+256] + key[start:start+256] @ key.T
            err = hist[start:start+256].double()-expected.double()
            ss += float(err.square().sum()); maxabs = max(maxabs, float(err.abs().max()))
            refsq += float(hist[start:start+256].double().square().sum())
        checks.append(dict(layer=layer, history_replay_maxabs=maxabs, history_replay_relative_l2=(ss/refsq)**.5,
                           replay='CPU FP32 M_entry + K K.T in 256-row blocks; reduction order may differ',
                           final_sha256=mr['final_history_sha256'], finite=bool(torch.isfinite(hist).all()),
                           source_recorded_terminal_appends=1, source_recorded_inner_appends=0))
        h.require(checks[-1]['finite'], 'NONFINITE_HISTORY')
    h.require(not torch.cuda.is_initialized(), 'CPU_REVIEW_CUDA_INITIALIZED')
    return rows, checks


def plot_bytes(summary, paired, functional, physical, trajectory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':120,'savefig.dpi':120,
                         'axes.spines.top':False,'axes.spines.right':False})
    figures = {}
    def emit(name, fig):
        buf = io.BytesIO(); fig.savefig(buf, format='png', metadata={'Software':'aos-cpu-review-v1'})
        plt.close(fig); figures[name] = buf.getvalue()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    names = ['N4','A0','A-OS','B-OS']
    for ax, panel in zip(axes, ('Current100','Fixed100','Past100')):
        for i, name in enumerate(names):
            rr = [next(r for r in summary if r['state']==name and r['panel']==panel and r['metric']==m) for m in ('RS','PS','NS')]
            ax.bar(np.arange(3)+(i-1.5)*.18, [100*float(r['success_rate']) for r in rr], width=.18, label=name)
        ax.set_xticks(range(3), ['RS','PS','NS']); ax.set_ylim(0,105); ax.set_title(panel)
    axes[0].set_ylabel('NLL-pair success (%)'); axes[-1].legend()
    emit('endpoint-rates.png',fig)
    fig, axes = plt.subplots(1, 2, figsize=(10,3.8), layout='constrained')
    for name, d in trajectory['solver']['pcg'].items():
        axes[0].semilogy(range(1,len(d['recursive_residuals'])+1), d['recursive_residuals'], label=name)
        axes[0].scatter([d['iterations']], [d['relative_residual']], marker='x', s=80)
    axes[0].axhline(1e-4, color='grey', linestyle='--', label='predeclared rtol')
    axes[0].set(xlabel='PCG iteration', ylabel='Relative residual'); axes[0].legend()
    for i, state in enumerate(('A0','A-OS')):
        axes[1].bar(np.arange(4)+(i-.5)*.35,[next(r['weighted_value'] for r in functional if r['state']==state and r['role']==x)
                                          for x in ('Base','Past','BaseAudit','PastAudit')], width=.35,label=state)
    axes[1].set_xticks(range(4), ['Base','Past','BaseAudit','PastAudit']); axes[1].set_ylabel('Raw functional risk');axes[1].legend()
    emit('solver-and-risk.png',fig)
    fig, axes = plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for ax, field, title in zip(axes, ('actual_correction_frobenius','actual_cumulative_frobenius'), ('A-OS minus A0','A-OS minus We')):
        ax.bar(['L4','L8'],[r[field] for r in physical]);ax.set_title(title);ax.set_ylabel('Physical Frobenius norm')
    fig.suptitle('Layer-wise Update Magnitude');emit('layer-update-magnitude.png',fig)
    fig, ax = plt.subplots(figsize=(8,3.8),layout='constrained')
    rr = [r for r in paired if r['comparison']=='A0_to_A-OS' and r['quantity']=='new_nll_delta']
    ax.bar(range(len(rr)),[r['mean'] for r in rr]);ax.set_xticks(range(len(rr)),[r['panel'].replace('100','')+' '+r['metric'] for r in rr],rotation=35,ha='right')
    ax.axhline(0,color='black',linewidth=.6);ax.set_ylabel('Paired target-new NLL delta (A-OS - A0)')
    emit('paired-nll-deltas.png',fig)
    return figures


def build(args):
    repo=Path(args.repo).resolve(); raw=Path(args.raw_root).resolve(); out=Path(args.output).absolute()
    out.mkdir(parents=True,exist_ok=True); h=helper(repo)
    run=raw/REL/'track-a/os-r1-Middle'; parent=raw/REL/'track-a/a0-r1-Middle'; common=raw/REL/'common/common-r1-Middle'
    terminal, rawcheck=h.verify_package(run/'terminal.json')
    h.require(terminal['status']=='TERMINAL_VALID' and terminal['source_head']==EXECUTION,'WRONG_EXECUTION_TERMINAL')
    tr=h.read(run/'trajectory.json'); ready=h.read(common/'READY.json'); bank=h.read(common/'bank-manifest.json')
    inp=h.read(common/'input.lock.json'); panel=inp['binding']['panel']; calibration=h.read(common/'calibration.json')
    h.require(h.sha(common/'READY.json')==tr['source_fixture']['common'],'COMMON_READY_IDENTITY')
    h.require(h.digest(ready['members'])==ready['members_root'],'COMMON_MEMBER_ROOT')
    used=[common/x for x in ('READY.json','input.lock.json','bank-manifest.json','calibration.json','entry.pt')]
    used += [parent/x for x in ('terminal.json','endpoint.pt','endpoint-metrics.json','functional-endpoint.json','attribution-We.json')]
    input_members=[h.member(p) for p in used]
    expected={m['path']:m for m in ready['members']}
    for m in input_members:
        if m['path'] in expected: h.require(m==expected[m['path']],'USED_COMMON_MEMBER_MISMATCH')
    h.require(h.sha(parent/'terminal.json')==tr['source_fixture']['parent'],'A0_PARENT_IDENTITY')
    h.require(h.sha(h.DATA)==h.DATA_SHA,'FIXED_DATASET_SHA')
    data=json.loads(h.DATA.read_text()); h.require(len(data)==10000,'FIXED_DATASET_SIZE')
    states={'A-OS':h.endpoint_rows(h.read(run/'endpoint-metrics.json'),panel,data),
            'A0':h.endpoint_rows(h.read(parent/'endpoint-metrics.json'),panel,data),
            'We':h.endpoint_rows(h.read(parent/'attribution-We.json'),panel,data,current_only=True)}
    for x in ('We_plus_D4','We_plus_D8'):
        states[x]=h.endpoint_rows(h.read(run/f'attribution-{x}.json'),panel,data,current_only=True)
    summary=[r for name, rows in states.items() for r in h.summarize(rows,name)]
    peer=raw/'experiment-reports/servers/server1/multilayer-joint-edit-a-2026-09-11-v1/peer-comparison-r2'
    pm=h.read(peer/'manifest.json');h.require(pm['common_ready_sha']==h.sha(common/'READY.json') and pm['input_panel_sha']==h.digest(panel),'PEER_FIXTURE_MISMATCH')
    h.require(h.digest(pm['members'])==pm['members_root'],'PEER_ROOT_MISMATCH')
    for m in pm['members']:
        got=h.member(peer/m['path']);h.require(got['sha256']==m['sha256'] and got['bytes']==m['bytes'],'PEER_MEMBER_MISMATCH')
        input_members.append(got)
    input_members.append(h.member(peer/'manifest.json'))
    for r in csv.DictReader((peer/'same-entry-summary.csv').open()):
        name=r.get('state',r.get('arm'))
        if name in ('N4','B-OS'):
            r=dict(r);r['state']=name;r['success_n']=int(r['numerator']);r['denominator']=int(r['denominator']);r['success_rate']=r['success_n']/r['denominator']
            r['new_nll_mean']=float(r['new_nll']);r['true_nll_mean']=float(r['true_nll'])
            r['provenance']='sealed peer aggregate; no new remote raw rehash or paired inference';summary.append(r)
    pairs=pair_rows(h,states['A0'],states['A-OS'],'A0_to_A-OS')
    current=[r for r in states['A-OS'] if r['panel']=='Current100' and r['metric']!='NS']
    pairs+=pair_rows(h,states['We'],current,'We_to_A-OS_Current_only')
    paired=pair_summary(h,pairs)
    attr, attr_summary=h.attribution(dict(We=states['We'],We_plus_D4=states['We_plus_D4'],We_plus_D8=states['We_plus_D8'],A0=states['A-OS']))
    for r in attr+attr_summary:r['joint_endpoint']='A-OS';r['delta_reference']='We (cumulative batch delta, not A-OS minus A0)'
    functional=[];contexts=[];reduction_checks=[]
    for state, path in [('A0',parent/'functional-endpoint.json'),('A-OS',run/'functional-endpoint.json')]:
        d=h.read(path); d=d.get('observations',d)
        fs, flat, cc=h.functional_summary(d,bank,calibration)
        for r in fs:r['state_to_native_ratio']=r.pop('A0_to_native_ratio');r['state']=state
        for r in flat+cc:r['state']=state
        functional+=fs;contexts+=flat;reduction_checks+=cc
    physical,history=physical_checks(h,common,run,parent,terminal,tr)
    pcg=[dict(rhs=k,**{f:v for f,v in d.items() if f!='recursive_residuals'}) for k,d in tr['solver']['pcg'].items()]
    pcg_nodes=[dict(rhs=k,iteration=i+1,recursive_relative_residual=v) for k,d in tr['solver']['pcg'].items() for i,v in enumerate(d['recursive_residuals'])]
    costs=[dict(category='allocation',name='job45633',value=23657,unit='GPU-sec',counting='sacct allocation elapsed; batch/extern not added')]
    for section in ('seconds','counts'):
        costs += [dict(category=section,name=k,value=v,unit='seconds' if section=='seconds' else 'recorded_count',counting='nested/subset counters; not additive total') for k,v in terminal['compute'][section].items()]
    for role, counts in tr['counts'].items():
        costs += [dict(category='functional_'+role,name=k,value=v,unit='recorded_count',counting='subset of model totals') for k,v in counts.items()]
    tables={'endpoint-summary.csv':summary,'paired-summary.csv':paired,'paired-case-deltas.csv':pairs,
            'functional-risk.csv':functional,'functional-context-values.csv':contexts,'functional-reduction-checks.csv':reduction_checks,
            'signed-attribution.csv':attr,'signed-attribution-summary.csv':attr_summary,'physical-update.csv':physical,
            'history-checks.csv':history,'pcg.csv':pcg,'pcg-iterations.csv':pcg_nodes,'compute-ledger.csv':costs}
    for name, rows in tables.items():h.write_csv(out/name,rows)
    h.save_json(out/'solver-and-transaction.json',dict(solver=tr['solver'],actual_delta_norm=tr['actual_delta_norm'],
        fp32_addition_residual_norm=tr['fp32_addition_residual_norm'],predicted_current_change=tr['predicted_current_change'],
        actual_current_change=tr['actual_current_change'],predicted_risk_change=tr['predicted_risk_change'],actual_risk_change=tr['actual_risk_change'],
        permitted_range_relative_residual=tr['permitted_range_relative_residual'],terminal_forward_parity=h.read(run/'terminal-forward-parity.json'),
        final_restore_reference='We; not W0',full_model_bytes_restore='NOT_RECORDED',
        live_guard='all parameter pointer/version/shape/dtype/device + selected bytes; no full buffer/parameter byte manifest',
        inner_append=tr['history_append'],terminal_batch_finalizations=terminal['history']['terminal_batch_finalizations'],
        terminal_layer_appends=terminal['history']['terminal_layer_appends'],compute_counts=terminal['compute']['counts']))
    figures=plot_bytes(summary,paired,functional,physical,tr)
    second=plot_bytes(summary,paired,functional,physical,tr)
    h.require(figures==second,'PNG_BYTE_REPRODUCTION_FAILURE')
    for name, content in figures.items():h.write_identical_or_create(out/name,content)
    import matplotlib
    command=f'{args.python} project/run_scripts/multilayer_joint_compensation/analysis/aos_review.py --repo . --raw-root {raw} --output {REPORT}'
    sourcefiles=[h.member(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    manifest=dict(status='PARTIAL_MIDDLE_AOS_CPU_REVIEW',scientific_promotion=False,execution_source=EXECUTION,
                  execution_tree='4b35bec2bf6f29a8d45dbaee5bb0c9a99dcc877e',analysis_source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
                  helper_commit=HELPER_COMMIT,helper_path=HELPER_PATH,helper_sha256=HELPER_SHA,source_members=sourcefiles,
                  common_ready_sha256=h.sha(common/'READY.json'),common_members_root=ready['members_root'],common_full_rehash='prior184-member receipt reused; this invocation used subset only',
                  input_members=input_members,raw_terminal=rawcheck,fixed_dataset_sha256=h.DATA_SHA,panel_sha256=h.digest(panel),
                  remote_raw_rehash=False,imputation=0,new_gpu=0,new_model=0,new_forward_evaluator=0,new_slurm=0,
                  python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,reproduce_command=command,
                  scheduler=dict(job_id=45633,owner='janghj',name='odeedit_multilayer_AOS_s1',state='COMPLETED',exit='0:0',allocation_GPU_seconds=23657,
                                 start='2026-09-12T12:24:06',end='2026-09-12T18:58:23',source='bounded sacct -X observation, not analyzer polling'))
    h.save_json(out/'input-manifest.json',manifest)
    h.save_json(out/'plot-reproduction.json',dict(status='PASS_BYTE_IDENTICAL_TWO_RENDER_CALLS',command=command,
        input_manifest_sha256=h.sha(out/'input-manifest.json'),plot_source_sha256=h.sha(Path(__file__)),
        environment={k:manifest[k] for k in ('python','numpy','matplotlib')},
        outputs=[h.member(out/name) for name in figures]))
    # Full new raw-member verification again after all CPU analysis; input immutability.
    _, raw_after=h.verify_package(run/'terminal.json');h.require(raw_after==rawcheck,'RAW_CHANGED_DURING_REVIEW')
    h.require(all(h.member(m['path'])==m for m in input_members),'INPUT_CHANGED_DURING_REVIEW')
    from aos_report import render
    h.write_identical_or_create(out/'factual-report-ko.md',render(summary,paired,functional,physical,tr,attr_summary,manifest).encode())
    members=[dict(path=p.name,bytes=p.stat().st_size,sha256=h.sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('analysis-manifest.json','rooted-receipt.json')]
    h.save_json(out/'analysis-manifest.json',dict(members=members,members_root=h.digest(members),source_sha256=h.sha(Path(__file__)),scientific_promotion=False))
    h.save_json(out/'rooted-receipt.json',dict(status='PASS_CPU_FACTUAL_PACKAGE_APPROXIMATE_ENDPOINT',campaign_complete=False,
        report_sha256=h.sha(out/'factual-report-ko.md'),manifest_sha256=h.sha(out/'analysis-manifest.json'),members_root=h.digest(members),
        raw_members_root=rawcheck['members_root'],raw_member_count=rawcheck['member_count'],input_before_after_unchanged=True,
        endpoint_pairs=3900,paired_A0_AOS=3900,attribution_pairs=300,plot_byte_reproduction=True,
        new_model=0,new_forward=0,new_GPU=0,new_Slurm=0,scientific_promotion=False))
    print(json.dumps(dict(report=str(out/'factual-report-ko.md'),sha256=h.sha(out/'factual-report-ko.md'),members_root=h.digest(members))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--raw-root',default=DEFAULT_RAW)
    p.add_argument('--output',default=REPORT);p.add_argument('--python',default='/mnt/raid5/janghj/EasyEdit/.venv/bin/python')
    build(p.parse_args())
