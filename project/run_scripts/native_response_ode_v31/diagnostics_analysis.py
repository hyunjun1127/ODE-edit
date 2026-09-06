"""Frozen post-primary refinement and audit reduction; CPU tensors only."""
import json
import math
from pathlib import Path
import torch
from .analysis import summarize,old_loss,csv_once,sha,md_table
from .provenance import save,ARM_ORDER
from project.run_scripts.ordered_response_barrier_ode.preflight import canonical_hash


def endpoint_distances(first,second):
    if set(first['delta'])!=set(second['delta']):raise ValueError('REFINEMENT_WEIGHT_INVENTORY')
    distance=scale=0.
    for name in first['delta']:
        a=first['delta'][name].double().cpu();b=second['delta'][name].double().cpu()
        if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('REFINEMENT_TENSOR_BOUNDARY')
        distance+=float((a-b).square().sum());scale+=float(b.square().sum())
    a=first['activation'].double().cpu();b=second['activation'].double().cpu()
    ad=float(torch.linalg.vector_norm(a-b));ab=float(torch.linalg.vector_norm(b))
    return dict(delta_W_distance=math.sqrt(distance),relative_to_second_update=math.sqrt(distance/scale) if scale else None,
        activation_distance=ad,relative_activation_distance=ad/ab if ab else None,
        residual_difference_distance=ad,residual_difference_identity='same fixed target: residual difference = negative activation difference',
        hash_only_convergence_claim=False)


def build(primary,followup,output):
    primary,followup,output=map(Path,(primary,followup,output))
    if output.exists():raise ValueError('CREATE_ONCE_DIAGNOSTIC_PACKAGE')
    sample=json.loads((primary/'sample.lock.json').read_text())
    meta=[r for r in sample['records'] if r['fixture']=='H10']
    inputs=[]
    for path in sorted(followup.rglob('*')):
        if path.is_symlink():raise ValueError('FOLLOWUP_MEMBER_SYMLINK')
        if path.is_file():inputs.append(dict(path=str(path.absolute()),sha256=sha(path),bytes=path.stat().st_size))
    output.mkdir(parents=True,mode=0o700)
    refine=[];distances=[];audit=[];request=[];registry=[];node_rows=[]
    for cell in range(4):
        root=followup/f'cell-{cell}';done=root/'terminal.json';failure=root/'failure-boundary.json'
        terminal=json.loads(done.read_text()) if done.exists() else json.loads(failure.read_text()) if failure.exists() else {'status':'NOT_RUN_BUDGET'}
        registry.append(dict(cell=cell,status=terminal.get('status','FAILURE'),stage=terminal.get('stage'),allocated_seconds=terminal.get('allocated_seconds')))
        valid=[]
        for n in (2,4,8):
            path=root/f'D2-N{n}.json'
            if not path.exists():continue
            raw=json.loads(path.read_text())
            if raw['status']!='TERMINAL_VALID' or raw['w0_restore'] is not True:raise ValueError('REFINEMENT_TERMINAL_BOUNDARY')
            valid.append(n)
            refine.append(dict(cell=cell,fixture='D2_COLD_DEV',N=n,T=2.,V_ratio=raw['V_ratio'],
                native_net_raw=raw['native_net_raw'],native_net_normalized=raw['native_net_normalized'],
                E_T=raw['E_T'],L_N=raw['L_N'],edit_core_seconds=raw['edit_core_seconds'],case_seconds=raw['case_seconds'],
                main_jvp_count=raw['main_jvp_count'],model_forward_calls=raw['forward_count']))
            previous_barrier=0.
            for node in raw['nodes']:
                increment=node['barrier']-previous_barrier
                affine=node['h']*(1-node['h']/2)*node['response_sq']
                node_rows.append(dict(cell=cell,N=n,node=node['node'],V=node['V'],V_exit=node['V_exit'],barrier=node['barrier'],
                    finite_barrier_increment=increment,affine_reference_increment=affine,affine_reference_defect=increment-affine,
                    model_error=node['model_error'],finite_defect_is_exclusion=False))
                previous_barrier=node['barrier']
        for a,b in ((2,4),(4,8)):
            if a not in valid or b not in valid:continue
            first=torch.load(root/'D2'/f'N{a}'/'endpoint-state.pt',map_location='cpu',weights_only=True)
            second=torch.load(root/'D2'/f'N{b}'/'endpoint-state.pt',map_location='cpu',weights_only=True)
            distances.append(dict(cell=cell,first_N=a,second_N=b,**endpoint_distances(first,second)))
            del first,second
        entry_path=root/'H10-entry.json'
        if not entry_path.exists():
            why=json.loads((root/'audit-status.json').read_text()) if (root/'audit-status.json').exists() else {'status':'NOT_COMPLETED'}
            audit.append(dict(cell=cell,arm='ALL',status=why['status']));continue
        entry=json.loads(entry_path.read_text())
        for arm in ARM_ORDER:
            path=root/f'H10-{arm}.json'
            if not path.exists():continue
            raw=json.loads(path.read_text());value=raw['result']
            endpoint=value if 'evaluation' in value else value['endpoint']
            if raw['w0_restore'] is not True or raw['source_entry_sha']!=entry['entry_sha'] or raw['fixed_z_sha']!=entry['fixed_z_sha']:
                raise ValueError('AUDIT_ENTRY_TARGET_RESTORE_BOUNDARY')
            facts,rows,_=summarize(endpoint['evaluation'],meta,entry['evaluation'])
            old,_=old_loss(raw['old_before'],raw['old_after'])
            audit.append(dict(cell=cell,arm=arm,status='TERMINAL_VALID',**facts,**old,V_ratio=raw['V_ratio'],
                native_net_normalized=raw['actual_physical_action']['native_net_normalized'],
                total_seconds=raw['total_seconds'],evaluation_seconds=raw['evaluation_seconds'],forward_count=raw['forward_count']))
            request.extend(dict(cell=cell,arm=arm,**r) for r in rows)
    for name,rows in [('refinement.csv',refine),('refinement_distances.csv',distances),('refinement_nodes.csv',node_rows),
                      ('audit_main_table.csv',audit),('audit_endpoint_metrics.csv',request),('run_registry.csv',registry)]:csv_once(output/name,rows)
    save(output/'external-inputs.json',inputs)
    save(output/'followup.lock.json',json.loads((followup/'followup.lock.json').read_text()))
    report='''# B10 short-history pilot 후속 refinement/audit

D2는 사전 hash-rank DEV B1 cold fixture이며 warm D10B를 이어 편집한 상태가 아니다. T2 고정, N2/4/8의 actual block delta와 activation을 비교한다. 유한 barrier defect/품질 저하는 제외 사유가 아니며 최적 N을 선택하지 않는다.

H10은 D10A 이후 봉인한 동일 warm W/cache에서 독립 시작한다. D10B endpoint carry=0, D10A replay=0. 늦은 H10 평가로 primary 설정을 바꾸지 않는다. 예산으로 미실행된 항목과 유효 완료 endpoint를 구분한다.

## Refinement

'''+md_table(refine,['cell','N','V_ratio','native_net_normalized','E_T','L_N','main_jvp_count','case_seconds'])+'\n\n'+md_table(distances,['cell','first_N','second_N','delta_W_distance','relative_to_second_update','activation_distance'])+'\n\n## H10 audit\n\n'+md_table(audit,['cell','arm','status','RS_n','RS_d','PS_n','PS_d','NS_n','NS_d','new_failure_n','entry_success_d','V_ratio'])+'\n\nscientific_promotion=false. 다른 server lifelong/retention 결과와 합산하지 않는다.\n'
    path=output/'diagnostic-factual-ko.md'
    with path.open('x') as handle:handle.write(report)
    path.chmod(0o600)
    members=[dict(path=str(p.relative_to(output)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()]
    save(output/'manifest.json',dict(members=members,members_root=canonical_hash(members),input_root=canonical_hash(inputs)))
    receipt=dict(status='FACTUAL_DIAGNOSTIC_PACKAGE',members_root=canonical_hash(members),report_sha256=sha(path),
        manifest_sha256=sha(output/'manifest.json'),scientific_promotion=False,imputation_count=0,input_mutation_count=0)
    receipt['identity']=canonical_hash(receipt);save(output/'rooted-receipt.json',receipt)
    for member in inputs:
        if sha(member['path'])!=member['sha256']:raise ValueError('FOLLOWUP_RAW_CHANGED')
    return receipt
