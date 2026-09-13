"""Observed same-cohort native baseline/replay comparison, CPU only."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .contracts import ContractBoundary, member, save, digest
from .cold_analysis import table, text_once


def compare_rows(original, replay, category):
    def keyed(rows):
        out = {}
        for r in rows:
            k = (r['case_id'], r['prompt_index'], r['identity'])
            if k in out: raise ContractBoundary('DUPLICATE_PROMPT')
            if not all(math.isfinite(r[f]) for f in ('new_nll','true_nll')):
                raise ContractBoundary('NONFINITE_NLL')
            desired = r['new_nll']-r['true_nll'] if category=='NS' else r['true_nll']-r['new_nll']
            if bool(r['success']) != (desired>0): raise ContractBoundary('STRICT_SUCCESS_DIRECTION')
            out[k] = dict(r, desired_margin=desired)
        return out
    a,b=keyed(original),keyed(replay)
    if a.keys()!=b.keys(): raise ContractBoundary('NONIDENTICAL_PROMPT_TARGET_SUPPORT')
    n=len(a)
    if not n: raise ContractBoundary('EMPTY_COMPARISON')
    an=sum(r['success'] for r in a.values());bn=sum(r['success'] for r in b.values())
    result=dict(metric=category,denominator=n,original_numerator=an,replay_numerator=bn,
        delta_pp=100*(bn-an)/n,loss=sum(a[k]['success'] and not b[k]['success'] for k in a),
        recovery=sum(not a[k]['success'] and b[k]['success'] for k in a),
        paired_identity_root=digest(sorted(a)))
    for f in ('new_nll','true_nll','desired_margin'):
        x=np.array([a[k][f] for k in a]);y=np.array([b[k][f] for k in a]);d=y-x
        for label,values in [('original',x),('replay',y),('delta',d)]:
            for stat,value in [('mean',values.mean()),('median',np.median(values)),('p90',np.quantile(values,.9)),('max',values.max())]:
                result[f'{f}_{label}_{stat}']=float(value)
    return result


def run(plan_path, output):
    plan=json.loads(Path(plan_path).read_text());dest=Path(output)
    if dest.exists(): raise ContractBoundary('CREATE_ONCE_OUTPUT')
    bindings=[member(plan_path)];rows=[]
    for cell in plan['comparisons']:
        paths=[cell['original'],cell['replay']];bindings.extend(member(p) for p in paths)
        a,b=[json.loads(Path(p).read_text()) for p in paths]
        if a['requests']!=b['requests'] or a['request_order']!=b['request_order']:
            raise ContractBoundary('REQUEST_ORDER_MISMATCH')
        for category in ('RS','PS','NS'):
            row=compare_rows(a['metrics'][category]['rows'],b['metrics'][category]['rows'],category)
            for obj,field in [(a,'original_numerator'),(b,'replay_numerator')]:
                if obj['metrics'][category]['denominator']!=row['denominator'] or obj['metrics'][category]['numerator']!=row[field]:
                    raise ContractBoundary('SOURCE_AGGREGATE_MISMATCH')
            rows.append(dict(endpoint=cell['endpoint'],panel='Current-B100',**row))
    dest.mkdir(parents=True)
    table(dest/'paired-performance.csv',rows)
    save(dest/'coverage.json',dict(comparisons=plan['comparisons'],missing=plan['missing'],extra_forward=0,
        trajectory_equivalence=False,performance_is_not_tensor_fidelity=True,scientific_promotion=False))
    lines=['# E01 원본 대비 replay 성능 차이 — 기존 관측 보충','',
      '동일 case/prompt/target identity와 request order가 일치하는 Current-B100 관측만 paired 비교했다. RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL이며 tie는 실패다. Δpp는 replay−원본이다.','',
      '| Endpoint | 지표 | 원본 n/d | Replay n/d | Δpp | 성공→실패 | 실패→성공 |',
      '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows: lines.append(f"| {r['endpoint']} | {r['metric']} | {r['original_numerator']}/{r['denominator']} | {r['replay_numerator']}/{r['denominator']} | {r['delta_pp']:+.3f} | {r['loss']} | {r['recovery']} |")
    lines+=['','true/new NLL 및 desired-margin의 원본/replay/paired delta mean·median·p90·max는 CSV에 기록했다. 원래 raw margin은 수정하지 않았다. W/M 차이는 checkpoint-comparison-recall-r2의 별도 tensor 비교이며 성능 유사성이 trajectory fidelity PASS를 뜻하지 않는다.','',
      'B011은 B020 endpoint가 아니다. B020 replay의 같은-state 성능은 NOT_MEASURED다. B011 Historical128에 해당하는 원본 B011 seen-full 관측은 수신 inventory에 없으므로 다른 checkpoint의 full-prefix 총점으로 대체하지 않았다. Cold B001은 원본 B001과 직접 비교했다. B060/B100은 아직 replay terminal 관측이 없으며 기존 runner는 최초 B051/B091 성능을 저장한다. Terminal 성능 보충 forward는 다음 사용자 recall의 별도 관측으로 남긴다.','',
      '원본 source BLUE311b076a 및 singleton L4/L2=1, 같은 fixed10k order·Llama revision·evaluator 규약을 사용한다. 실행 host와 replay source 및 W/context/target/RNG 차이는 원 cold/warm report와 input locks에 별도로 결속되어 있으며 완전 수치동일성으로 간주하지 않는다. 신규 model/GPU/forward=0, scientific_promotion=false.','',
      f'재현: `python -m project.run_scripts.baseline_mechanism_first.performance_compare --plan {plan_path} --output <new-directory>`']
    text_once(dest/'factual-report-ko.md','\n'.join(lines)+'\n')
    outputs=[member(p) for p in sorted(dest.iterdir())]
    manifest=save(dest/'manifest.json',dict(inputs=bindings,source=member(__file__),outputs=outputs,output_root=digest(outputs),imputation=0))
    return save(dest/'rooted-receipt.json',dict(manifest=manifest,identity=digest(manifest),status='OBSERVED_PAIRED_PERFORMANCE_ONLY'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.plan,a.output)))
