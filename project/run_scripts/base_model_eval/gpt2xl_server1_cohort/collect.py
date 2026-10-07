"""Independent CPU re-reduction and coverage; no evaluator or model loaded."""
import csv
import io
from .common import *
from .curves import build_curves,curve_coverage,reduce_rows,validate_rows,REFERENCE_SCHEMA
from project.run_scripts.jlz_realized_writer_sequential.review_completed import Reader

def collect(config_path,lock_path):
    c,lock=verify_lock(config_path,lock_path);root=Path(c['attempt']);output=root/'collector'
    require(not output.exists(),'COHORT_COLLECTOR_CREATE_ONCE');output.mkdir();reader=Reader()
    try:
        terminal=reader.json(root/'W0/terminal.json') if (root/'W0/terminal.json').exists() else dict(status='NOT_RECORDED')
        expected=reader.bound(c['observer_identity'])['rows'];raw=[]
        chunks=sorted((root/'W0').glob('chunk-*.json'))
        for index,path in enumerate(chunks):
            require(path.name==f'chunk-{index*50:04d}.json','COHORT_CHUNK_SEQUENCE')
            chunk=reader.json(path)
            require(chunk['fresh_W0'] and chunk['reference_only'] and not chunk['optimizer_feedback'] and
                chunk['selected_W']==c['cold_selected_W'] and chunk['actual_model_edits']==chunk['actual_applied_edits']==
                chunk['pre_state_edits']==chunk['post_state_edits']==0 and not chunk['history_present'],'COHORT_RAW_SCOPE')
            raw.extend(chunk['rows'])
        validate_rows(raw,expected[:len(raw)],require_full=terminal['status']=='COMPLETED')
        complete=terminal['status']=='COMPLETED';payloads=[];coverage=dict(status='NOT_MEASURED',missing='incomplete fresh raw')
        config=dict(c['reference_config'],metric_schema=SCHEMA)
        if complete:
            require(len(chunks)==40 and terminal['source']==lock['source_commit'] and terminal['config']==lock['config_sha256'],'COHORT_COMPLETE_SOURCE')
            summary=reader.json(root/'W0/summary.json')
            require(summary['fresh_W0'] and summary['no_mutation'] and summary['selected_W']==c['cold_selected_W'] and
                summary['ordered_occurrence_rows']==digest([[row['occurrence_ordinal'],row['identity']] for row in raw]),'COHORT_COMPLETE_IDENTITY')
            require(summary['physical_candidate_rows']==52000 and summary['prompt_pairs']==26000,'COHORT_PHYSICAL_COUNT')
            payloads=build_curves(raw,expected);coverage=curve_coverage(payloads,config)
            stored=reader.json(root/'W0/curve-payloads.json')
            require(stored['payloads']==payloads and stored['coverage']==coverage and stored['new_model_forward_calls']==0,'COHORT_INDEPENDENT_REDUCTION')
        delivery=reader.json(root/'W0/curve-acceptance.json') if (root/'W0/curve-acceptance.json').exists() else dict(status='NOT_RECORDED')
        finish=reader.json(root/'W0/tracking-finish.json') if (root/'W0/tracking-finish.json').exists() else dict(status='NOT_RECORDED')
        value=dict(status='COMPLETED' if complete else 'PARTIAL_OR_FAILED',source=lock['source_commit'],config=lock['config_sha256'],
            runner=terminal,prompt_pairs=len(raw),candidate_rows=2*len(raw),independently_reduced=reduce_rows(raw),
            curve_coverage=coverage,tracking_acceptance=delivery,tracking_finish=finish,
            actual_model_edits=0,reference_only=True,evaluation_model_state='W0',
            new_forward_calls_by_reducer=0,parent_allocation='NOT_QUERIED_BY_COLLECTOR',no_checkpoint=True)
        write(output/'curve-summary.json',dict(schema=REFERENCE_SCHEMA,payloads=payloads,coverage=coverage,config=config))
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['reference_cohort_edits','actual_model_edits','metric','value']);writer.writeheader()
        for payload in payloads:
            for key,item in payload.items():
                if '/' in key:writer.writerow(dict(reference_cohort_edits=payload['edits'],actual_model_edits=0,metric=key,value=item))
        write_bytes(output/'curves.csv',stream.getvalue().encode())
        report='# GPT2-XL W0 cohort 곡선 사실 보고\n\n'+f"상태: {value['status']}. "
        report+=('신규 cold actual W0 1회 관측과 saved-row 곡선 집계 완료.' if complete else
            f"신규 관측 미완료/미관측: 저장 prompt-pair {len(raw)}, runner stage {terminal.get('stage','NOT_RECORDED')}.")
        report+=' 기존 raw를 이번 측정으로 재사용하지 않았다.\n\n'
        report+='edits는 reference cohort 비교 진행축, model/applied/pre/post edits는 모두0이다. 실제 편집된 W1..W20 관측이 아니다.\n\n'
        report+='| 비교 x | current R/P/N | all-seen R/P/N | actual edits |\n|---:|---|---|---:|\n'
        for payload in payloads:
            x=payload['edits'];current='/'.join(str(payload.get(f'current/post/{kind}/count','NA')) for kind in 'RPN')
            seen='/'.join(str(payload.get(f'all_seen/post/{kind}/count','NA')) for kind in 'RPN')
            report+=f'| {x} | {current} | {seen} | 0 |\n'
        report+='\nR/P desired=new, N desired=true; tie=failure; pct/nats와 token-micro 실분모를 유지한다. raw는 occurrence ordinal로 선택하며 숫자 case-ID나 dedup을 쓰지 않는다.\n'
        report+='\nSDK 접수와 실제 remote coverage/value/identity readback 결과를 분리한다. 미관측/누락은0으로 채우지 않는다. NoCP/exact_resume NOT_AVAILABLE, raw local KEEP.\n'
        write_bytes(output/'report-ko.md',report.encode())
        write(output/'manifest.json',dict(files=list(reader.files.values()),source=lock['source_commit'],saved_rows_only=True,no_new_forward=True))
        write(output/'result.json',value)
        write(output/'terminal.json',dict(status=value['status'],report=member(output/'report-ko.md'),curves=member(output/'curves.csv'),
            manifest=member(output/'manifest.json'),result=member(output/'result.json'),actual_CPU_reduction=True,model_loaded=False,GPU=False))
    except BaseException as error:
        value=dict(status='FAILED',error_type=type(error).__name__,error=str(error)[:1200],new_forward_calls=0,model_loaded=False,GPU=False)
        write(output/'failure.json',value);write(output/'terminal.json',value)
    return value
