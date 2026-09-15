"""Korean factual publication from CPU-verified aggregates; no model calls."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess

from .analysis import read_json, member, require, digest, _write_json


def table(rows, columns):
    def value(v):
        if v is None or v == '': return 'NOT_RECORDED'
        if isinstance(v, float): return f'{v:.6g}'
        return str(v).replace('|', '\\|')
    return '\n'.join(['| ' + ' | '.join(label for _, label in columns) + ' |',
                      '| ' + ' | '.join('---' for _ in columns) + ' |'] +
                     ['| ' + ' | '.join(value(row.get(key)) for key, _ in columns) + ' |' for row in rows])


def load_csv(path):
    with path.open() as stream: return list(csv.DictReader(stream))


def publish(aggregates, figures, output, technical, n4_reuse, allocation):
    aggregates, figures, output, technical = map(lambda p: Path(p).absolute(), (aggregates, figures, output, technical))
    report_path = output / 'diagnostic-report-ko.md'
    require(not report_path.exists(), 'report create-once')
    verification = read_json(aggregates / 'verification.json')
    require(verification['status'] == 'CPU_VERIFIED_SEQ1000' and
            verification['checkpoint_verification'] == 'FULL_FILE_TENSOR_RNG_SHA256' and
            verification.get('saved_state_arithmetic') == 'CPU_FP32_ONE_GRAM_AND_FP64_STORED_DELTA_COST',
            'verified full ten required')
    manifest = read_json(aggregates / 'manifest.json')
    for item in manifest['members']: member(Path(item['path']), item)
    plot = read_json(figures / 'plot-receipt.json')
    for item in plot['inputs'] + plot['outputs']: member(Path(item['path']), item)
    metrics, batches, pairs = [load_csv(aggregates / f) for f in ('metrics.csv', 'batch.csv', 'paired.csv')]
    gate = read_json(technical / 'TECHNICAL_VALID.json')
    require(gate['status'] == 'ACTUAL_LLAMA_TECHNICAL_VALID', 'actual technical gate absent')
    reuse = read_json(Path(n4_reuse)); require(reuse['decision'] == 'REUSE', 'N4 reuse receipt absent')
    jobs = read_json(Path(allocation))
    require(jobs['status'] == 'BOUNDED_TERMINAL_VERIFIED' and jobs['main_exit_code'] == '0:0', 'terminal allocation receipt')
    measured = []
    n4 = {r['metric']: r for r in metrics if r['scope'] == 'historical-n4'}
    final = {r['metric']: r for r in metrics if r['scope'] == 'seen-full' and r['batch_index'] == '10'}
    for tag in ('RS', 'PS', 'NS'):
        a, b = n4[tag], final[tag]
        measured.append(dict(metric=tag, n4=f"{a['numerator']}/{a['denominator']} ({float(a['percent']):.2f}%)",
            main=f"{b['numerator']}/{b['denominator']} ({float(b['percent']):.2f}%)",
            delta=float(b['percent'])-float(a['percent'])))
    source = verification['execution_source']; counts = verification['terminal_counts']
    sections = ['# SL-ZFlow W0→SEQ1000 사실 보고', '',
        '작성: SH2/server2. scientific_promotion=false. 이 문서는 관측·산술·기술 검증만 보고한다. 품질·보존·비용의 과학적 해석은 별도 GH review 범위다.', '',
        '## 1. 실제 완료 범위와 같은 분모의 비교', '',
        f"신규 MAIN 1 chain, B100×10, unique/attempted 1,000개가 완료됐다. Accepted {counts['accepted']}, rejected {counts['rejected']}, oracle {counts['oracle_calls']}이며 각 batch의 `1+accepted+rejected`를 검산했다. History append {counts['history_appends']}, no-update batch {counts['no_update_batches']}. 기술 실행은 과학 분모 밖이며 기존 N4는 재사용했다. 신규 N4/Adam/barrier/다른 order 실행은 0이다.", '',
        table(measured, [('metric','지표'),('n4','기존 N4 W10'),('main','SL-ZFlow W10'),('delta','SL−N4 (pp)')]), '',
        'RS/PS는 target-new mean-token NLL < target-true, NS는 반대이며 tie는 실패다. TF strict는 모든 target token의 teacher-forced top-1 일치이고 token 지표는 correct/total token으로, 자유 생성 정확도가 아닌 별도 보조 지표다. 위 N4는 동일 case/prompt/target/order의 기존 관측이며 새 측정이 아니다. N4 cudnn TF32=True와 MAIN False 차이, host 차이가 있어 bitwise-equivalent execution을 주장하지 않는다.', '',
        '고정 MAIN: Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, 실제 모델/write FP32/eager, seed20260907, 물리 weight `model.layers.4.mlp.down_proj.weight` 하나다. 매 batch 자기 W_entry/M_entry에서 X=0 및 W=W_entry+XB, native nonsymmetric LU로 B/S를 한 번 준비한다. lambda_write=1, lambda_flow=1, beta=.0625, barrier=off/budget=null, eta_initial=max=1, max_oracle_calls=25다. lambda_flow=1은 미튜닝 초기값이며 최적 trade-off 설정으로 주장하지 않는다. Native key clean1/2·generated각1/10과 edit loss 6contexts각1/6은 서로 다르다. 완성 native z/endpoint warm start나 normalization은 사용하지 않았다.', '',
        '## 2. batch별 actual 상태와 current 관측', '',
        table(batches, [('batch','Batch'),('solver_status','종료'),('accepted','Accept'),('rejected','Reject'),('oracle_calls','Oracle'),('history_append','Append'),('terminal_C','C'),('terminal_F','F'),('actual_delta_cost_fp64','실제 저장 Δ cost')]), '',
        table([r for r in metrics if r['scope']=='current'], [('batch','Batch'),('metric','지표'),('numerator','n'),('denominator','d'),('percent','%'),('new_tf_strict_num','new strict n'),('new_tf_strict_den','strict d')]), '',
        'RESOURCE_STOP은 계산 한도 종료이며 최적성·성과 인증이 아니다. FIRST_ORDER_STATIONARY는 reduced-space 1차 조건만 뜻한다. 유효 finite/no-update 요청도 원분모에서 제외하지 않았다. At-write/current 합계를 final retention으로 부르지 않는다.', '',
        '## 3. 동일 문항 유지·손실·회복과 NLL tail', '',
        table([r for r in pairs if r['comparison'] in ('SL_ZFLOW_ATWRITE_TO_W10','SL_ZFLOW_W5_TO_W10_SAME_FIRST500','N4_W10_TO_SL_ZFLOW_W10')],
              [('comparison','비교'),('metric','지표'),('denominator','d'),('left_success','이전 성공'),('right_success','이후 성공'),('lost','성공→실패'),('gained','실패→성공'),('delta_pp','Δpp')]), '',
        'Lost/gained는 exact item identity/order join이다. W10 first500은 W10 full1000의 동일 raw rows를 CPU로 잘랐고 중복 forward가 없다. 반복 관측 수를 독립 표본 수로 세지 않는다. Input-only superseded 후보/동일 batch 충돌/미확정 그룹은 paired.csv에 따로 남겼으며 canonical 분모를 삭제하지 않는다.', '',
        '실제 관측 inventory: `' + str(verification['observation_inventory']) + '`.', '',
        '입력 population: `' + str(verification['request_population_summary']['counts']) + '`.', '',
        table([r for r in metrics if r['scope'] in ('historical-n4','seen-full','first500')], [('batch','상태'),('scope','범위'),('metric','지표'),('new_nll_median','new NLL median'),('new_nll_p95','new NLL p95'),('new_nll_p99','new NLL p99'),('true_nll_median','true NLL median'),('success_margin_p05','성공방향 margin p05')]), '',
        '전체 strict/token·new/true NLL quantile은 metrics.csv, exact-paired NLL 변화 quantile은 paired.csv에 있다. NS를 전체 pretrained capability 보존으로 확대하지 않는다.', '',
        '## 4. 실제 Llama·state·resume 검증', '',
        f"기술 gate `{member(technical/'TECHNICAL_VALID.json')['sha256']}`: full-write↔all-token suffix logits/NLL/X-gradient와 global microbatch weight, actual FP32 Δ cost, entry rollback/nonselected guard, inner append0/terminal append1을 확인했다. 별도 Python process에서 W/M/context/RNG를 복원한 다음 batch 첫 request logits max-abs={gate['next_entry_logits_max_abs']}; exact-same={gate['exact_same_logits']}. 동일 commit 재시도는 no-op였다.", '',
        '범위: native source compute_ks replay는 실제 첫 2개 요청, gradient calibration/독립 고정 perturbation 검사는 첫 2개 요청의 7개 packed caches(모든 token), microbatch partition 검사는 같은 2개 요청의 microbatch1/2였다. Writer 좌표는 B100의 100개를 유지했다. 이후 terminal physical parity/cost와 flow는 전체 기술 B100에서 수행했다. 모든 1,000개 요청마다 full-write X-gradient를 별도 재검증했다고 주장하지 않는다.', '',
        '10개 MAIN checkpoint는 W/M/X/B/S/K/config/context/RNG/ledger/parent/next-index/cache-binding을 담고 있으며 CPU에서 full file/tensor/RNG SHA와 chain continuity를 재검산했다. 저장된 W와 이전 W의 FP64 차이로 실제 cost를 다시 계산하고, 저장 M이 이전 M+CPU FP32 K@K.T 한 번의 결과와 정확히 일치하는지도 확인했다. 이는 saved end-state 검증이며 단독으로 모든 중간 연산을 계수했다는 주장은 아니다. Actual model resume 실험은 기술 checkpoint에서 수행했으며 모든 MAIN checkpoint를 별도 GPU replay했다고 주장하지 않는다. 완성 manifest 없는 partial bundle은 resume 대상이 아니다.', '',
        '수치 threshold는 품질 평가 전에 fixed calibration으로 봉인했으며 chain 중 변경하지 않았다. Tokenizer는 source-native add_bos attribute=False 대입을 보존했으나 Fast tokenizer의 실제 backend BOS 제거와 같지 않다. 실제 token IDs와 source key 경로를 확인했으며 속성값만으로 BOS 부재를 주장하지 않는다.', '',
        '## 5. 비용·오류와 자원', '',
        table(jobs['jobs'], [('job_id','Job'),('scope','범위'),('state','상태'),('exit_code','Exit'),('allocated_gpu_seconds','할당 GPU-sec')]), '',
        f"마지막 MAIN process wall={verification['latest_process_seconds']:.6f}s. 할당 GPU 시간은 위 scheduler ledger이며 pure compute와 다르다. 초기 기술 metadata 실패도 비용에 포함한다. Plan의 4.74h/15h wall은 추정/요청값이지 actual elapsed가 아니다.", '',
        table(batches, [('batch','Batch'),('preparation_seconds','준비s'),('flow_seconds','flow s'),('commit_seconds','commit s'),('evaluation_seconds','평가s'),('total_seconds','batch s'),('peak_gpu_allocated','peak allocated B')]), '',
        'compute.csv는 prefix/teacher, native keys/LU, initial/accepted/rejected suffix F+B, terminal parity, 실제 cost, durable prepare, 평가를 분리한다. 중첩 timer를 더하지 않는다. 새 25 whole-batch sweep를 native max25 forward/24 backward 및 요청별 early-stop과 같은 단위로 취급하지 않는다. 기존 N4 비용은 새 allocation으로 청구하지 않았다.', '',
        '## 6. source·재현·artifact', '',
        f"실행 HEAD `{source['source_head']}`, tree `{source['source_tree']}`, input lock `{source['input_lock_sha256']}`. 분석 source는 verification.json의 analysis_source에 별도 기록했다. Source/hash 검증은 실제 model parity의 대체물이 아니다.", '',
        '- 집계: `aggregates/{node,batch,metrics,paired,compute}.csv`, verification.json 및 manifest.json.',
        '- 기술: `' + str(technical) + '`.',
        '- N4 재사용 근거: `' + str(Path(n4_reuse).absolute()) + '`.',
        '- MAIN raw/checkpoint 경로는 allocation-receipt.json의 main_output에 있다. Raw/model/teacher/prompt/checkpoint/full log는 Git 밖 local-only 보존, NO_BROADCAST_NOT_REQUIRED.', '',
        '재현 명령은 package README와 figures/plot-receipt.json에 있다. 분석은 sealed input lock+MAIN root+N4 exact raw SHA를 요구하며 모델/evaluator를 호출하지 않는다.', '',
        '## 7. 코드 생성 그림과 미실행 경계', '',
        '![Current 및 final 비교](figures/quality.png)', '', '![계산 및 실제 비용](figures/work.png)', '',
        '그림은 집계 CSV와 Python matplotlib 코드에서 생성했다. 입력·코드·PNG SHA 및 재현 명령을 결속하고 같은 입력의 PNG byte 재현을 검증한다. 수동 수정이나 이미지 생성 도구를 사용하지 않았다.', '',
        'Adam(same actual-write objective) 및 fixed/exponential barrier ablation은 NOT_RUN이다. 후속에는 objective/entry/sample/계산예산을 맞추어 optimizer 차이와 constraint 차이를 분리해야 하지만 이 보고로 추가 실행하지 않는다. 단일 MAIN으로 integrator 고유 기여를 인증하지 않으며 C 감소를 output locality 보장으로 부르지 않는다.', '']
    output.mkdir(parents=True, exist_ok=True)
    with report_path.open('x') as handle: handle.write('\n'.join(sections))
    _write_json(output / 'allocation-receipt.json', jobs)
    evidence = [member(report_path), member(output/'allocation-receipt.json'), member(aggregates/'manifest.json'), member(figures/'plot-receipt.json')]
    receipt = dict(status='FACTUAL_REPORT_COMPLETE', scientific_promotion=False,
        source_analysis_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).parent, text=True).strip(),
        members=evidence, member_root=digest(evidence), raw_broadcast='NO_BROADCAST_NOT_REQUIRED')
    _write_json(output/'rooted-receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aggregates','figures','output','technical','n4-reuse','allocation'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    publish(args.aggregates,args.figures,args.output,args.technical,args.n4_reuse,args.allocation)


if __name__ == '__main__': main()
