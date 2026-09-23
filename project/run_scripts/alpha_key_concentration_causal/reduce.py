"""Autonomous CPU factual reducer; reads completed local rows, never forwards."""
import csv
import json
import math
from pathlib import Path
from .common import save,file_sha,digest

def independent_pairs(raw):
    families={'RS':('rewrite_target_new','rewrite_target_true',False),
              'PS':('rephrase_target_new','rephrase_target_true',False),
              'NS':('locality_target_new','locality_target_true',True)}
    output={}
    for metric,(newkey,truekey,reverse) in families.items():
        if newkey not in raw and truekey not in raw:continue
        new,true=raw[newkey],raw[truekey];assert len(new)==len(true)
        rows=[];seen=set()
        for a,b in zip(new,true):
            identity=(a['case_id'],a['prompt_index'],a['prompt'])
            assert identity==(b['case_id'],b['prompt_index'],b['prompt']) and identity not in seen
            assert a.get('endpoint_id')==b.get('endpoint_id'),'PAIRED_ENDPOINT_MISMATCH'
            assert a.get('full_vocab',{}).get('prompt_token_ids_sha256')==b.get('full_vocab',{}).get('prompt_token_ids_sha256'),'PAIRED_TOKEN_MISMATCH'
            seen.add(identity)
            an,bn=float(a['nll']),float(b['nll']);assert math.isfinite(an) and math.isfinite(bn)
            rows.append(dict(identity=digest([*identity,a['target'],b['target']]),case_id=a['case_id'],
                prompt_index=a['prompt_index'],new_nll=an,true_nll=bn,
                success=(bn<an if reverse else an<bn),desired_margin=(an-bn if reverse else bn-an)))
        output[metric]=dict(numerator=sum(r['success'] for r in rows),denominator=len(rows),rows=rows)
    return output

def csv_file(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or sorted({k for r in rows for k in r}) or ['status']
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def reduce_package(root,out,repo,lock):
    from .control import attempt_paths
    expected=attempt_paths(root,lock['attempt'])
    subdir=lock.get('report_subdir',expected['report_subdir'])
    assert subdir==expected['report_subdir'],'REDUCER_ATTEMPT_COLLISION'
    package=Path(repo)/'experiment-reports/servers/server4/alpha-key-causal-20260923-r1'/subdir
    package.mkdir(parents=True,exist_ok=False)
    inventory=[]
    seen=set()
    for base in [Path(out),*[Path(x) for x in lock.get('inherited_output_roots',[])]]:
        for p in sorted(base.rglob('*')):
            if p.is_file():
                real=p.resolve();assert real.is_relative_to(Path(root).resolve()),'REUSE_OUTSIDE_TASK_ROOT'
                if real in seen:continue
                seen.add(real)
                inventory.append(dict(path=str(p.relative_to(root)),resolved_path=str(real),
                    reused_input=p.is_symlink() or base!=Path(out),bytes=p.stat().st_size,sha256=file_sha(p)))
    save(Path(root)/'execution'/lock['attempt']/'raw-inventory.json',inventory)
    tables=[];transitions=[];observed=[]
    for p in sorted((Path(out)/'writers').rglob('current.json')):
        x=json.loads(p.read_text());reduced=independent_pairs(x['raw_local_only'])
        for metric,value in reduced.items():
            old=x['compact']['metrics'][metric]
            assert (value['numerator'],value['denominator'])==(old['numerator'],old['denominator'])
            tables.append(dict(endpoint=x['compact']['endpoint_id'],metric=metric,numerator=value['numerator'],denominator=value['denominator'],
                percent=100*value['numerator']/value['denominator'],raw_relative=str(p.relative_to(root)),raw_sha256=file_sha(p)))
    for p in sorted((Path(out)/'writers').rglob('observation.json')):
        x=json.loads(p.read_text());observed.append(dict(endpoint=x['endpoint_id'],N512=x['N']['denominator'],
            N512_true_correct=x['N']['true_argmax_correct'],N512_true_nll=x['N']['true_nll_mean'],seconds=x['seconds'],nonmutation=x['observer_nonmutation']))
    from .observer import protection_transitions
    for entry in (50,70,80,90):
        for branch in ('NATIVE','SHAM','H5','H6','H56','MASS56'):
            a=Path(out)/'writers'/f'W{entry:03d}'/branch/'stages/entry/N512.json'
            b=a.parent.parent/'history/N512.json'
            if a.exists() and b.exists():
                pair=protection_transitions(json.loads(a.read_text())['compact']['rows'],json.loads(b.read_text())['compact']['rows'])
                transitions.append(dict(entry=entry,branch=branch,denominator=pair['denominator'],entry_successes=pair['entry_successes'],lost=pair['lost'],recovered=pair['recovered']))
    csv_file(package/'same_entry_effects.csv',tables)
    csv_file(package/'operator_stage.csv',observed)
    csv_file(package/'protection_transitions.csv',transitions)
    if observed:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4))
        for entry in (50,70,80,90):
            prefix=f'W{entry:03d}/NATIVE/'
            series=[r for r in observed if r['endpoint'].startswith(prefix)]
            order={'entry':0,'z':1,'W4':2,'W5':3,'W6':4,'W7':5,'W8':6,'history':7}
            series=sorted(series,key=lambda r:order[r['endpoint'].split('/')[-1]])
            if series:ax.plot([order[r['endpoint'].split('/')[-1]] for r in series],[r['N512_true_correct'] for r in series],marker='o',label=f'W{entry} -> B{entry+1}')
        ax.set_xticks(range(8),['entry','z','W4','W5','W6','W7','W8','history']);ax.set_ylabel('N512 true argmax count / 512');ax.legend();fig.tight_layout()
        fig.savefig(package/'native-stage-N512.png',dpi=140,metadata={'Software':'alpha-key-causal CPU reducer'})
        plt.close(fig)
    from shutil import copyfile
    g=Path(out)/'geometry/state_cohort_geometry.csv'
    if g.exists():copyfile(g,package/g.name)
    failures=[];programs=[]
    for phase in ('gate','geometry','writers'):
        for suffix,dest in (('failure',failures),('program',programs)):
            p=Path(out)/f'{phase}-{suffix}.json'
            if p.exists():
                value=json.loads(p.read_text())
                dest.append({k:v for k,v in value.items() if k not in ('traceback','message')})
    modes=[];history_rows=[];components=[];kr=[];family_count=0
    gp=Path(out)/'geometry/terminal.json'
    if gp.exists():
        gx=json.loads(gp.read_text())
        if gx['status']=='COMPLETED':family_count+=int(gx['E1_families'])+int(gx['E2_families'])
    for entry in (50,70,80,90):
        for branch in ('NATIVE','SHAM','H5','H6','H56','MASS56'):
            d=Path(out)/'writers'/f'W{entry:03d}'/branch
            p=d/'write/terminal.json'
            if p.exists() and json.loads(p.read_text())['status']=='COMPLETED':family_count+=1
            p=d/'writer-modes.json'
            if p.exists():
                for layer,row in json.loads(p.read_text()).items():
                    flat={k:v for k,v in row.items() if v is None or isinstance(v,(str,bool,int,float))}
                    modes.append(dict(entry=entry,branch=branch,layer=layer,**flat))
                    coverage=row.get('history_coverage',{})
                    history_rows.append(dict(entry=entry,branch=branch,layer=layer,**{k:v for k,v in coverage.items() if v is None or isinstance(v,(str,bool,int,float))}))
        cp=Path(out)/'writers'/f'W{entry:03d}'/'components/terminal.json'
        if cp.exists():
            value=json.loads(cp.read_text())
            if value['status']=='COMPLETED':family_count+=int(value['families'])
            for row in value['endpoints']:
                target=kr if row['family']=='KR' else components
                target.append(dict(entry=entry,**row))
    csv_file(package/'writer_modes.csv',modes)
    csv_file(package/'history_penalty_mismatch.csv',history_rows)
    csv_file(package/'component_interchange.csv',components)
    csv_file(package/'kr_operand_effects.csv',kr)
    terminal='COMPLETED' if len(programs)==3 and not failures and family_count==94 else 'TECHNICAL_FAILED_OR_PARTIAL'
    save(package/'coverage.json',dict(completed_families=family_count,expected_families=94,
        per_request_key_geometry_parquet=str(Path(out)/'geometry/per_request_key_geometry.parquet'),
        raw_parquet_local_only=True,whitened_modes_levels=sorted(set(str(r.get('history_whitened_modes_status','NOT_MEASURED')) for r in modes))))
    save(package/'cost_breakdown.json',dict(programs=programs,failures=failures,
        parent_failed_execution=lock.get('parent_failed_execution'),
        reused_writer_cost='52565 1494 allocated GPU seconds; original NATIVE/SHAM costs counted once, not new forwards' if lock.get('reuse_manifest') else None,
        allocation='Slurm accounting requires later explicit user recall; not inferred from wall',
        nested_timers_additive=False,observer_and_geometry='inclusive boundaries in source receipts',
        user_gpu_hour_hardcap=None))
    save(package/'source-state-receipt.json',dict(status=terminal,source_commit=lock['source_commit'],source_tree=lock['source_tree'],
        execution_lock_sha256=file_sha(lock.get('execution_lock_path',expected['control']/'execution.lock.json')),raw_inventory_count=len(inventory),
        raw_inventory_root=digest(inventory),new_full_state_checkpoints=False,followups='FOLLOWUP_NOT_SUBMITTED',
        independent_raw_NLL_rows=sum(r['denominator'] for r in tables),GPU_continuation='NOT_TESTED'))
    for name in ('sequential_event_order.csv','order_replication.csv'):
        csv_file(package/name,[dict(status='FOLLOWUP_NOT_SUBMITTED',reason='Outside current user authority')])
    text=['# AlphaEdit key E0–E4 factual report', '',f'실제 상태: `{terminal}`. 실행 source `{lock["source_commit"]}`.',
          '', '본 문서는 등록된 CPU reducer가 저장된 실제 관측만 집계했다. 미실행 dependent 단계는 완료로 세지 않는다.',
          '원본 N4가 아니라 BASE_ALPHAEDIT L4–L8/L2=10 대조이며 N512/P는 observer-only다.',
          'Allocation은 program wall과 다르며 아직 회수하지 않은 scheduler 완료 정보는 추정하지 않았다.',
          '', '| Endpoint | Metric | Count | Denominator | Percent |', '|---|---|---:|---:|---:|']
    text += [f'| {r["endpoint"]} | {r["metric"]} | {r["numerator"]} | {r["denominator"]} | {r["percent"]:.6f} |' for r in tables if r['endpoint'].endswith('/history')]
    text+=['','N512 기존 실패와 신규 loss/recovery는 [전이표](protection_transitions.csv)에 분리했다.',
           '[Stage](operator_stage.csv), [same-entry](same_entry_effects.csv), [비용](cost_breakdown.json), [출처](source-state-receipt.json).',
           '', '성능 우열·인과 귀속·새 방법 채택은 이 reducer의 판정 범위가 아니다. SEQ/ORDER/FUTURE 미제출.',
           '원 input CP12는 보존; 새 W/M/optimizer resume checkpoint는 생성하지 않았다.',
           'NO_BROADCAST_NOT_REQUIRED. 실제 detailed review 및 scheduler 후속 회수는 사용자 recall 후에만 수행한다.']
    if lock.get('numerical_comparison_policy'):
        comparisons=[]
        for name in ('sham-control.json','full-hook-physical-parity.json'):
            for p in sorted((Path(out)/'writers').rglob(name)):
                x=json.loads(p.read_text());comparisons.append(dict(path=str(p.relative_to(root)),status=x['status'],
                    comparison_verdict=x.get('comparison_verdict'),blocks_execution=x.get('blocks_execution'),sha256=file_sha(p)))
        save(package/'numerical-comparison-coverage.json',dict(policy=lock['numerical_comparison_policy'],
            authority=lock['user_numerical_override'],rows=comparisons,full_numerical_equivalence='NOT_ESTABLISHED',
            original_failed_attempt_preserved=True,threshold_relaxed_to_claim_PASS=False))
        text+=['','## 최신 사용자 수치 비교 관찰 정책','',
            'SHAM 및 hook/physical 수치 차이는 `OBSERVATION_ONLY_USER_DIRECTED`로 보존한다. 미일치를 PASS로 바꾸지 않았고, 원 FP32 연산식/관측집합은 변경하지 않았다.',
            '실행 완료와 수치 동등성은 별개이며 `full_numerical_equivalence=NOT_ESTABLISHED`이다. NaN/shape/입력/상태/저장 무결성 검사는 계속 차단 조건이다.',
            '[수치 비교 검산 범위](numerical-comparison-coverage.json).']
    text += ['',f'완료 family `{family_count}/94`; 필수 표의 누락은 성공으로 대체하지 않는다.',
             '[Writer](writer_modes.csv), [H penalty](history_penalty_mismatch.csv), [component](component_interchange.csv), [K/R](kr_operand_effects.csv).']
    if observed:text+=['','![Actual N512 stage counts](native-stage-N512.png)','자동 코드 생성 그림이며 agent 육안 검토는 아직 하지 않았다.']
    rendered='NOT_RUN_TOOL_UNAVAILABLE'
    try:
        import markdown
        html=markdown.markdown('\n'.join(text),extensions=['tables'])
        (package/'report-ko.html').write_text('<!doctype html><meta charset="utf-8">'+html)
        rendered='GENERATED_HTML; browser visual review NOT_RUN'
    except ImportError:pass
    save(package/'render-check.json',dict(status=rendered,links_exist=all((package/name).exists() for name in ('same_entry_effects.csv','writer_modes.csv','coverage.json'))))
    (package/'report-ko.md').write_text('\n'.join(text)+'\n')
    save(package/'terminal.json',dict(status=terminal,monitoring_active=False,automatic_resume=False))
    members=[dict(path=str(p.relative_to(package)),bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(package.iterdir()) if p.is_file()]
    save(package/'rooted-receipt.json',dict(members=members,root=digest(members)))
