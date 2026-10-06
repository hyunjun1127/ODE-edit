"""Independent stdlib W0 reduction; never torch/model/Slurm/GPU evaluation."""
import csv
import io
import math
from pathlib import Path

from .gpt2xl_server1_common import *
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    validate_rows, reduce_rows, compare_summary, paired, Reader)


def collect(config_path, lock_path):
    config, lock = verify_config_lock(config_path, lock_path)
    root = Path(config['attempt'])
    output = root / 'collector'
    require(not output.exists(), 'W0_COLLECTOR_CREATE_ONCE')
    output.mkdir()
    reader, result = Reader(), {}
    try:
        terminal_path = root / 'W0' / 'terminal.json'
        terminal = reader.json(terminal_path) if terminal_path.exists() else dict(status='NOT_RECORDED')
        expected = reader.bound(config['observer_identity'])['rows']
        chunks = sorted((root / 'W0').glob('chunk-*.json'))
        actual = []
        for index, path in enumerate(chunks):
            require(path.name == f'chunk-{index*50:04d}.json', 'W0_CHUNK_SEQUENCE')
            chunk = reader.json(path)
            require(chunk['fresh_W0'] is True and chunk['optimizer_feedback'] is False
                    and chunk['history_present'] is False and chunk['edits'] == 0
                    and chunk['selected_W']==config['cold_selected_W'], 'W0_RAW_OBSERVATION_SCOPE')
            actual.extend(chunk['rows'])
        validate_rows(actual, expected[:len(actual)], 'W0')
        summary = reduce_rows(actual)
        complete = terminal['status'] == 'COMPLETED'
        if complete:
            require(terminal['source']==lock['source_commit'] and terminal['config']==lock['config_sha256'],'W0_COMPLETE_SOURCE_CONFIG')
            require(len(actual) == 26000 and len(chunks) == 40
                    and {kind: value['denominator'] for kind, value in summary.items()} == DENOMINATORS,
                    'W0_COMPLETE_DENOMINATORS')
            stored = reader.json(root / 'W0' / 'summary.json')
            require(stored['fresh_W0'] and stored['no_mutation'] and not stored['reference_only'], 'W0_FRESH_NONMUTATION')
            require(stored['selected_W']==config['cold_selected_W'] and stored['row_order']==digest([row['identity'] for row in actual]),'W0_RAW_WEIGHT_ROW_ORDER')
            compare_summary(summary, stored['summary'])
        old = config.get('old_W0_CPU_crosscheck')
        crosscheck = dict(status='NOT_AVAILABLE', fresh_measurement_replaced=False)
        if complete and old:
            require(old['status'] == 'QUALIFIED_EXACT_REUSE'
                    and old['cold_state']['W'] == config['cold_selected_W'], 'OLD_W0_COLD_WEIGHTS')
            before = []
            for row in old['chunks']:
                before.extend(reader.bound(row)['rows'])
            validate_rows(before, expected, 'W0')
            prior_summary = reader.bound(old['summary'])
            require(prior_summary['requests'] == 2000 and prior_summary['row_count'] == 26000
                    and prior_summary['endpoint'] == 'W0', 'OLD_W0_COMPLETE_SCOPE')
            compare_summary(reduce_rows(before), prior_summary['summary'])
            prior_runtime = reader.bound(old['runtime'])
            contrasts = paired(before, actual)
            crosscheck = dict(status='STORED_OLD_RAW_CPU_PAIRED_ONLY', fresh_measurement_replaced=False,
                paired=contrasts, max_abs_NLL_difference={label:max(abs(a[label+'_nll']-b[label+'_nll'])
                    for a,b in zip(actual,before)) for label in ('new','true')},
                prior_runtime={key:prior_runtime[key] for key in ('device','torch','transformers','model',
                    'FP32','eager','TF32','autocast')},
                prior_runtime_identity='SOURCE_BOUND; no cross-hardware equivalence inferred')
        result = dict(status='COMPLETED' if complete else 'PARTIAL_OR_FAILED', runner=terminal,
            prompt_pairs=len(actual), candidate_rows=2*len(actual), independently_reduced=summary,
            crosscheck=crosscheck, no_new_forward=True, no_checkpoint=True,
            source=lock['source_commit'], config=lock['config_sha256'], input_members=list(reader.files.values()),
            cost=dict(program_seconds=terminal.get('program_seconds'),
                parent_allocation='NOT_QUERIED_BY_COLLECTOR', peak_GPU_allocated_bytes=terminal.get('peak_GPU_allocated_bytes'),
                peak_host_RSS_bytes=terminal.get('peak_host_RSS_bytes')))
        write(output / 'manifest.json', dict(files=list(reader.files.values()), source=lock['source_commit'],
            fresh_W0=True, scope='saved scalar/token identity rows only'))
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=['kind','numerator','denominator','success_pct',
            'token_acc_pct','prompt_acc_pct','strict_acc_pct','true_nll','new_nll','margin_true_minus_new'])
        writer.writeheader()
        for kind, value in summary.items():
            writer.writerow(dict(kind=kind, numerator=value['numerator'], denominator=value['denominator'],
                success_pct=100*value['rate'], token_acc_pct=100*value['token_micro'],
                prompt_acc_pct=100*value['prompt_macro'],
                strict_acc_pct=100*value['strict_numerator']/value['strict_denominator'],
                true_nll=value['true_nll_mean'],new_nll=value['new_nll_mean'],
                margin_true_minus_new=value['true_nll_mean']-value['new_nll_mean']))
        write_bytes(output / 'metrics.csv', stream.getvalue().encode())
        report = '# GPT2-XL cold W0 사실 보고\n\n'
        report += f"상태: {result['status']}. 신규 cold W0 관측이며 기존 raw는 CPU 대비에만 사용했다.\n\n"
        report += '편집/fit/solve/history/C0/P/새 checkpoint: 모두 0. TF는 teacher-forced 정확도이며 자유생성이 아니다.\n\n'
        report += '| 종류 | 성공/분모 | 선호 % | token ACC % | strict ACC % |\n|---|---:|---:|---:|---:|\n'
        for kind, value in summary.items():
            report += f"| {kind} | {value['numerator']}/{value['denominator']} | {100*value['rate']:.6f} | {100*value['token_micro']:.6f} | {100*value['strict_numerator']/value['strict_denominator']:.6f} |\n"
        report += '\nR/P desired=new, N desired=true; ties=failure. 전체 26000 prompt-pair/52000 candidate-row 분모는 완결시만 주장한다.\n'
        report += '\n기존 W0와의 비교는 row/token identity를 고정한 저장값 대비이며 동일 hardware/수치적 동등성을 증명하지 않는다.\n'
        write_bytes(output / 'report-ko.md', report.encode())
        write(output / 'result.json', result)
        write(output / 'terminal.json', dict(status=result['status'], report=member(output/'report-ko.md'),
            metrics=member(output/'metrics.csv'), manifest=member(output/'manifest.json'), result=member(output/'result.json'),
            actual_CPU_reduction=True, model_loaded=False, GPU=False))
    except BaseException as error:
        result = dict(status='FAILED', error_type=type(error).__name__, error=str(error)[:1200],
            no_new_forward=True, model_loaded=False, GPU=False)
        write(output / 'failure.json', result)
        write(output / 'terminal.json', result)
    return result
