"""Terminal T0 report packager; CPU only, immutable failed attempt inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from .technical_decision import classify_fd


def member(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def write_json(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def run(attempt,destination):
    attempt=Path(attempt);out=attempt/'output';td=out/'technical-decision'
    failure=json.loads((out/'failure.json').read_text())
    checks=json.loads((td/'checks.json').read_text())
    if failure['error']!="RuntimeError('T0_FULL_CHECKS_NOT_ESTABLISHED')":raise ValueError('EXACT_T0_FAILURE')
    if checks['blocking']!=['pair_factor_AD_FD']:raise ValueError('ADDITIONAL_BLOCKER')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    pairs=[];fdrows=[]
    for path in sorted(td.glob('reference-*/result.json')):
        v=json.loads(path.read_text());fd=v['FD']
        replay=classify_fd(fd['full_grid'],AD=fd['AD'],noise=fd['noise'],direction_norm=v['direction_norm'])
        if replay!=fd:raise ValueError('FD_SELECTION_REPLAY_MISMATCH')
        errors=[r['relative_error'] for r in fd['full_grid'] if r['relative_error'] is not None]
        pairs.append(dict(reference_index=v['index'],status=v['status'],
            direct_cached_gradient_relative=v['gradient_relative'],
            coefficient_relative=v['coefficient_relative_gap'],AD=fd['AD'],
            min_FD_relative=min(errors),matching_scales=','.join(str(r['k']) for r in fd['full_grid'] if r['match']),
            FD_status=fd['status'],selected_window=str(fd['selected_window']),noise=fd['noise']))
        for row in fd['full_grid']:
            fdrows.append(dict(reference_index=v['index'],**{k:row[k] for k in
                ('k','h','AD','FD','relative_error','signal','noise','resolved','match','rounded_linear_FD')}))
    for name,rows in [('T0-pairs.csv',pairs),('FD-grid.csv',fdrows)]:
        with (destination/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    statuses={name:dict(pass_=bool(row['pass_']),scope=row.get('scope',row.get('status','SEE_RAW')))
              for name,row in checks['results'].items()}
    inventory=[member(path) for path in sorted(out.rglob('*')) if path.is_file()]
    write_json(destination/'input-manifest.json',dict(members=inventory,
        execution_lock=member(attempt/'execution.lock.json'),original_raw_modified=False))
    write_json(destination/'T0-summary.json',dict(checks=statuses,blocking=checks['blocking'],
        FD_stored_scores_reselection='EXACT_CPU_REPLAY_4_OF_4',
        raw_new_GPU=0,whole_T0_status='NOT_ESTABLISHED',B1='NOT_RUN',
        failure_member=member(out/'failure.json'),checks_member=member(td/'checks.json'),
        runtime_source=failure['source'],program_seconds=failure['seconds'],
        T0_seconds=checks['seconds'],checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
        input_member_count=len(inventory),input_logical_bytes=sum(r['bytes'] for r in inventory)))
    write_json(destination/'rooted-receipt.json',dict(members=[member(p) for p in sorted(destination.iterdir())],
        code=member(Path(__file__)),independent_agent_red=False,
        level='CPU_STORED_VALUE_RESELECTION_AND_SHA; NO_GPU_PARITY_OR_B1_EFFICACY'))
    return pairs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();print(json.dumps(run(a.attempt,a.destination)))
