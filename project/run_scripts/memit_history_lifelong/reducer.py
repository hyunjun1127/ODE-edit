"""CPU postrun reduction from recorded metrics; no model/evaluator imports used."""
import json
from pathlib import Path
from .io import save
from .metrics import summarize,paired,strata,FULL_BATCHES

def reduce_run(output,records):
    output=Path(output)
    final=json.loads((output/'B100/seen-full.json').read_text())
    early=json.loads((output/'B005/seen-full.json').read_text())
    at={tag:[] for tag in ('RS','PS','NS')}
    for i in range(1,101):
        x=json.loads((output/f'B{i:03d}/current.json').read_text())
        for tag in at:at[tag].extend(x['metrics'][tag]['rows'])
    report=dict(status='CPU_REDUCED_FROM_RECORDED_ROWS',baseline_job=42658,baseline_final={'RS':[6453,10000],'PS':[11407,20000],'NS':[49838,100000]},
        baseline_TF='PUBLISHED_TABLE_AVAILABLE; NO_LOCAL_PAIRED_RAW',baseline_paired='NOT_AVAILABLE_WITHOUT_COMPATIBLE_BASELINE_RAW',full_eval_batches=FULL_BATCHES,
        final={tag:summarize(final['metrics'][tag]['rows'],tag) for tag in at},strata=strata(final,records),at_write={},first500_W5_to_W100={},save_checkpoints=False,exact_resume='NOT_AVAILABLE')
    for tag in at:
        f=final['metrics'][tag]['rows'];e=early['metrics'][tag]['rows'];ids={r['identity'] for r in e}
        report['at_write'][tag]=paired(at[tag],f)
        report['at_write'][tag]['tf_strict']=paired([dict(r,success=r[('true' if tag=='NS' else 'new')+'_strict']) for r in at[tag]],[dict(r,success=r[('true' if tag=='NS' else 'new')+'_strict']) for r in f])
        report['first500_W5_to_W100'][tag]=paired(e,[r for r in f if r['identity'] in ids])
        b=report['baseline_final'][tag];assert len(f)==b[1]
        report['final'][tag]['delta_BASE_MEMIT_percentage_points']=100*(report['final'][tag]['rate']-b[0]/b[1])
    save(output/'summary.json',report)
    # Compact report contains no prompt/tensor/weights.
    lines=['# MEMIT history fixed10k 기록','', '프로그램 CPU reducer 생성. GPU 완료 상태는 terminal.json 별도 확인.','',
      '|지표|성공/분모|TF token micro|TF prompt macro|TF strict|','|---|---|---|---|---|']
    for k,v in report['final'].items():lines.append(f"|{k}|{v['numerator']}/{v['denominator']}|{v['tf_token_micro']:.6f}|{v['tf_prompt_macro']:.6f}|{v['tf_strict']:.6f}|")
    with (output/'report-ko.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    return report
