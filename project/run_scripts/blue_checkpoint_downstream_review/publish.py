"""Create-once raw-free Korean report and reproducible figures from audited rows."""
import argparse,csv,json,subprocess,shutil
from pathlib import Path
from metrics import TASKS
from review import load,save,member,digest,sha
from plots import render

def mdtable(headers,rows):
    return '\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def publish(reduction,dest,scratch):
    dest.mkdir(parents=True,exist_ok=False);scratch.mkdir(parents=True,exist_ok=True)
    for p in reduction.iterdir():
        if p.suffix in ('.csv','.json'):shutil.copyfile(p,dest/p.name)
    rows=list(csv.DictReader((dest/'metrics.csv').open()));s=load(dest/'verification.json')
    pair=list(csv.DictReader((dest/'paired-transitions.csv').open()))
    lookup={(r['family'],r['variant'],int(r['edits']),r['task'],r['branch']):r for r in rows}
    render(dest/'metrics.csv',dest/'figures');render(dest/'metrics.csv',scratch/'figure-reproduction')
    figure_checks=[]
    for p in sorted((dest/'figures').glob('*.png')):
        same=sha(p)==sha(scratch/'figure-reproduction'/p.name)
        if not same:raise ValueError('FIGURE_REPRODUCIBILITY')
        figure_checks.append(dict(name=p.name,sha256=sha(p),byte_exact_repeat=True))
    save(dest/'figure-reproduction.json',dict(status='PASS',figures=figure_checks))
    text=['# BLUE checkpoint downstream 평가 — 최종 한국어 검산 보고서\n',
    '작성: Server2 / 2026-09-12. `scientific_promotion=false`. 신규 GPU 평가 없이 job42706의 완료 결과만 CPU로 검산했다. 기존 전용 branch에는 준비 감사만 있었고 최종 보고서는 없었다. 원 실행 source와 raw는 수정하지 않았다.\n',
    '## 1. 완료 범위와 읽는 법\n',
    f'단발 scheduler 확인은 `COMPLETED / 0:0`, GPU allocation {s["scheduler"]["allocated_gpu_seconds"]:,}초였다. INITIAL_VALID는 W0+첫 checkpoint의 초기 gate였으며, 이번에는 전체 terminal과 각 상태의 완료/복원 기록을 별도로 확인했다. **73 상태(W0 1 + 72 checkpoint), 438 state-task cells, 43,800 task-example 관측**이 실제 존재하고 누락/실패 0이다. 두 예측 분기의 876개 집계 모두 독립 계산과 저장값의 최대 절대 차이 {s["metric_max_absolute_error"]}였다.\n',
    '평가 문항은 task당 고정 100개, 총 600개다. 이를 73개 상태에서 반복 측정했으므로 43,800개의 독립 표본이 아니다. W0는 한 번 측정한 공통값을 두 계열 표에서 참조한다. 이 문서의 edit 10,000은 편집 checkpoint 위치이며 downstream 10,000문항 평가를 뜻하지 않는다. 각 task는 full benchmark가 아닌 봉인된 100-row reference panel이다.\n',
    '주표의 F1은 **support-weighted F1×100**이다. generation은 원본 생성 parser의 예측, alternative는 고정 답안 문자열의 teacher-forced 확률 비교다. 둘은 서로 다른 지표이고 좋은 쪽만 선택하지 않는다. 아래 Δ는 공통 W0 대비 percentage points(pp)다. BLUE는 L4+L8; L4_ONLY/L8_ONLY는 각 단일 layer variant이며 Official native baseline이라는 뜻이 아니다.\n']
    for family in ('MEMIT','AlphaEdit'):
        text.append(f'## 2.{1 if family=="MEMIT" else 2}. {family} — final10k 독립 비교표\n')
        for branch in ('generation','alternative'):
            text.append(f'### {branch}: weighted F1 % (W0 대비 Δpp)\n')
            rr=[]
            for variant in ('W0','BLUE','L4_ONLY','L8_ONLY'):
                result=[]
                for task in TASKS:
                    r=lookup[('W0','W0',0,task,branch)] if variant=='W0' else lookup[(family,variant,10000,task,branch)]
                    result.append(f'{100*float(r["weighted_f1"]):.2f} ({float(r["delta_f1_pp"]):+.2f})')
                rr.append([variant]+result)
            text.append(mdtable(['상태']+list(TASKS),rr))
        text.append('### Final10k 정확도·invalid·paired 변화\n\n셀은 `정답/100; invalid; lost/gained`다. lost는 W0 정답→해당 상태 오답, gained는 W0 오답→정답이며 invalid도 오답 분모에 남긴다. F1 변화와 lost−gained를 동일시하지 않는다.\n')
        for branch in ('generation','alternative'):
            rr=[]
            for variant in ('BLUE','L4_ONLY','L8_ONLY'):
                result=[]
                for task in TASKS:
                    r=lookup[(family,variant,10000,task,branch)];q=next(p for p in pair if p['family']==family and p['variant']==variant and p['edits']=='10000' and p['task']==task and p['branch']==branch)
                    result.append(f'{r["correct"]}/100; {r["invalid"]}; {q["lost"]}/{q["gained"]}')
                rr.append([variant]+result)
            text.append(f'**{branch}**\n'+mdtable(['상태']+list(TASKS),rr))
    text.extend(['## 3. 평가 계약·label·parser 경계\n',
    '자료는 사용자 지정 AlphaEdit `b84624f44dfe8fc6cd9e41df916c44124a0c46dc` 계보의 dataset 복사본이다. Server2 `EasyEdit/glue_eval/dataset`를 읽었지만 upstream EasyEdit에 본 evaluator가 원래 포함되었다는 주장이 아니다. 평가 코드는 BLUE `311b076a92e4ed0f14f5c8b4909732da781bc5f7` reference를 별도 private source로 사용했다. 원본 파일·label·prompt·parser는 보존했고 RTE scoring-only adapter만 승인 적용했다.\n',
    mdtable(['task','파일 전체 rows','실행 rows','gold 의미/support'],[[t,s['data'][t]['rows'],'[10:110], 100',('A/B/C/D=0/1/2/3; 각25' if t=='mmlu' else 'entailment=1 / not_entailment=0; 각50' if t=='nli' else 'GLUE entailment=0 / not_entailment=1; 각50' if t=='rte' else 'source label 0/1; 각50')] for t in TASKS]),
    'SST2는 negative=0/positive=1, MRPC는 non-equivalent=0/equivalent=1, CoLA는 unacceptable=0/acceptable=1이다. NLI는 이 파일의 binary entailment/not_entailment이며 MNLI 3-class로 부르지 않는다. MMLU는 고정 100개 choice 문제의 reference 평가다. 전체 57-subject benchmark/공식 subject-macro accuracy를 측정했다는 주장은 하지 않는다. 원 파일의 공식 train/validation/test provenance와 유사문장 중복 제거는 미검증이다. 저장된 exact record/order 및 각 상태의 전체 input_prompt 순서 동일성은 이번에 확인했다. 예약 10행을 fewshot로 사용하지 않았고 fewshot=0, generation length=5, source 순서 그대로다.\n',
    'Weighted F1 = Σ_k support_k/N × 2TP_k/(2TP_k+FP_k+FN_k). Accuracy=정답/N. MCC는 전체 confusion matrix(예측 invalid=-1 포함)의 multiclass 상관계수이며 분모0이면0이다. 원 source는 generation MCC만 기록했고 이번 CPU 표는 alternative MCC도 같은 정의로 별도 계산했다. F1에는 단일 정답 numerator가 없으므로 `confusion.csv`에 class별 support/TP/FP/FN을 제공한다. `metrics.csv`에는 ACC numerator/denominator, F1, MCC, invalid, Δ 모두 있다.\n',
    'Alternative 점수는 각 고정 답안의 exp(−mean token NLL)를 비교한다. 이는 선택지 사이 합1로 정규화한 확률이 아니다. Binary는 strict >로 비교하며 동률은 False/No/negative, MMLU는 유일한 strict maximum이 없으면 invalid다. Generation은 생성문에 대한 원본 substring parser를 유지한다. 따라서 instruction-following/출력 형식과 task 정답 능력이 분리되지 않는다.\n',
    'RTE는 `RTE_LABEL_MAPPING_CORRECTED_V1`: source semantic True(1)→GLUE entailment(0), False(0)→not_entailment(1); invalid -1은 유지한다. 두 분기에 동일 적용했으며 raw prediction/semantic/canonical/raw gold는 원본 metrics.records에 별도 보존되어 있다. 원 source의 `correct` 및 교정 전 F1은 bug diagnostic으로만 검산하고 위 main 점수에 섞지 않았다. 이 교정으로 오른 점수를 method 개선으로 해석하지 않는다.\n',
    'MMLU generation은 원본 parser가 `A\\n` 등 newline 포함 문자에 반응하고 bare `A`는 invalid가 될 수 있다. 이번에 parser를 고치거나 생성문을 다시 해석하지 않았다. generation F1과 alternative f1_new를 각각 공개하고 `mmlu-parser.csv`에 invalid 및 bare-letter invalid 관측을 보존했다. 이 한계로 invalid가 많다는 사실을 곧 지식 전체 소실로 치환할 수 없다.\n',
    '## 4. 관측과 가능한 설명\n',
    '관측: final10k L8_ONLY에서는 두 계열 모두 generation F1이 매우 낮고 invalid가 많다. 그러나 alternative 점수는 일률적으로 0이 아니다. 예를 들어 AlphaEdit L8_ONLY SST2는 generation 0/100, invalid100이지만 alternative 58/100이다. MEMIT L8_ONLY MMLU는 generation 1/100, alternative26/100이다. 따라서 생성 parser/짧은 출력 형식 영향과 선택지 discrimination 감소를 함께 보고해야 한다.\n',
    '관측: MEMIT L4_ONLY의 final10k alternative weighted F1은 6task 중 5개에서 MEMIT BLUE보다 높고 MRPC에서는 낮다. AlphaEdit도 단순한 동일 순위가 아니다. L4_ONLY는 SST2/CoLA에서 BLUE보다 높지만 MRPC/RTE/MMLU/NLI에서는 낮다. MEMIT L4_ONLY RTE alternative는 W0보다 소폭 높은 반면 나머지 task는 W0보다 낮다. 모든 경우의 정확한 lost/gained와 분모는 위 표 및 CSV에 있다. 점수 상승 자체가 W0에서 맞았던 모든 문항 보존을 뜻하지 않는다.\n',
    '가능한 설명: 반복 편집 후 출력 형식, 답안 선호, task별 decision boundary가 서로 다르게 변했을 수 있다. L8-only의 저점은 이 조건에서 측정된 association이며 layer 자체의 보편적 열등성이나 편집 알고리즘 전체의 인과 효과를 증명하지 않는다. NLL 편집 efficacy/RS/PS/NS, history, key drift를 이번 downstream 점수로 대체하지 않는다. 원 BLUE lifelong 6chain 결과와 provenance를 연결할 수 있지만, 추가 L5/6/7/native/다른14chain downstream 점수는 NOT_MEASURED다.\n',
    '미분리 한계: task당100개 단일 고정 panel, 단일 source/parser 및 checkpoint trajectory다. 새로운 seed·prompt·parser·fullsplit·다른 하드웨어 반복 평가가 없으므로 통계적 독립 43,800 표본 또는 일반 capability/lifelong 우위로 확대하지 않는다. 곡선은 측정된12점만 연결한 도식이며 보간된 점수를 표에 추가하거나 best checkpoint를 사후 선택하지 않았다.\n',
    '## 5. 계산량·실제 복원과 비변조\n',
    mdtable(['항목','실측/범위'],[['scheduler allocated GPU seconds',s['scheduler']['allocated_gpu_seconds']],['전체 프로그램 wall seconds',s['wall_seconds']],['6task 평가 wall 합',s['task_wall_seconds']],['73state wall 합',s['state_wall_seconds']],['state 내 비평가 차이(복원·hash·CP load·I/O 혼합)',s['state_wall_seconds']-s['task_wall_seconds']],['entry setup(순수 model load 아님)',s['entry_setup_seconds']],['forward calls',s['forwards']],['input tokens(호출별, 재처리 포함)',s['input_tokens']],['peak GPU bytes',s['peak_gpu_bytes']],['runtime peak host KiB',s['peak_host_kib']],['sacct batch MaxRSS KiB',s['scheduler']['batch_MaxRSS_KiB']]]),
    '할당 시간은 약10.347 GPUh이며 순수 write/학습 비용이 아니다. 초기 512.91초와 잔여10.12h는 당시 추정이므로 최종 실측과 구분한다. `compute-tasks.csv`/`compute-states.csv`는 task/state별 실제 시간이다. restore/hash/CP읽기 각각의 독립 timer는 NOT_RECORDED이며 혼합 잔차를 순수 restore 시간으로 주장하지 않는다. runtime host peak와 sacct MaxRSS는 집계 경계가 달라 서로 일치하지 않으며 어느 하나로 대체하지 않았다. 하드웨어간 numerical parity 반복검증은 하지 않았다.\n',
    'Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`; FP32 parameters, eager attention, torch2.9.1+cu128 / transformers4.44.2, matmul TF32=false/cudnn TF32=true. Seed20260907, threads8, eval/비학습 inference, greedy do_sample=False. Deterministic-algorithms 또는 cross-hardware bitwise repeat PASS는 주장하지 않는다. Tokenizer padding_side=right/pad=eos128009이며 task source는 문항별 단일 입력으로 평가했다. 경로 이름을 canonical model 이름에 bind한 것은 source lookup 경계이고 weight 변경이 아니다.\n',
    '각 checkpoint는 W0 위에 지정 selected key(두 layer 또는 singleton)만 overwrite했다. Alpha cache/history는 보존 provenance이고 forward에 적용하지 않았다. 원 runner는 전 parameter pointer/version 및 streamed tensor byte hash를 endpoint 전후, W0 복원 시 검증했다. 이번 CPU 분석은 W0 전체 parameter hash dictionary에 manifest의 selected tensor hashes를 대입해 73개 expected parameter root를 독립 재구성하여 entry/metrics/terminal과 비교했다. 새 model load/GPU replay는 0이다.\n',
    '## 6. Provenance와 검산 범위\n',
    f'Execution HEAD `{s["execution_source"]}` / tree `{s["execution_tree"]}`; execution.lock SHA `{s["execution_lock_sha256"]}`. Analysis source는 `project/run_scripts/blue_checkpoint_downstream_review/`이며 아래 rooted receipt의 Git base/파일 SHA로 별도 결속한다. 과거 완료 runtime commit만 재사용 통합하며 현재 multilayer 코드는 포함하지 않는다.\n',
    f'CP manifest SHA `{s["checkpoint_manifest_sha256"]}`, 72개/62,011,141,768B. Server2 retained imports 경로는 `checkpoints.csv`에 전부 기록했다. Server4 삭제된 원본 경로는 provenance일 뿐 읽거나 재전송하지 않았다. CP와 4개 multiGB pretrained shards는 기존 full SHA·schema/실행 검증을 결속하고 현재 size/readability를 확인했으며 이번에 대형 bytes 재해시를 반복하지 않았다. 그 외 {s["fresh_small_member_hashes"]}개 source/data/dependency member 검산(중복 경로 포함)은 전체 SHA 재계산했다.\n',
    f'Output {s["raw_members"]}개 파일을 full SHA 재계산하고 분석 후 다시 비교하여 변경0을 확인했다. 원 terminal이 봉인한 438 metrics SHA는 전부 일치한다. **raw rows 자체는 원 terminal에서 개별 SHA를 남기지 않았으므로**, 현재 full hash·dataset/order/저장 metrics 일치가 원래 row 생성 시점의 독립 서명이라는 주장은 하지 않는다. 새 raw member root `{s["raw_member_root"]}`는 이번 CPU recall 시점의 결속이다.\n',
    'Dataset10member root `e9328a5d351816cb9ba89454d228f7ade841c526313dae9c0d1a0e72a8ab00fc`. 편집 provenance fixed10k dataset SHA `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1` / order root `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`는 downstream 데이터 SHA와 다르다. 원 raw는 `local/blue-checkpoint-downstream/20260909-v1/attempt-v1/output/`; source/data/CP exact 경로는 JSON/CSV manifest에 있다. Raw prompt/prediction/log/tensor는 Git에 포함하지 않는다. `NO_BROADCAST_NOT_REQUIRED`: 기존 Server2 보존으로 충분하여 신규 대형 broadcast 0.\n',
    '## 7. 재현·파일 안내\n',
    'CPU reducer는 원본 bytes를 읽고 새 출력 디렉터리를 요구한다. 모델을 import/호출하지 않는다. 아래 python은 기존 EasyEdit venv의 CPU plotting/metric libraries만 재사용하며 환경 수정은 없다.\n',
    '```bash\npython3 project/run_scripts/blue_checkpoint_downstream_review/review.py --output /absolute/new-private-reduction\n/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest discover -s project/run_scripts/blue_checkpoint_downstream_review -p "test_*.py" -v\n/mnt/raid5/janghj/EasyEdit/.venv/bin/python project/run_scripts/blue_checkpoint_downstream_review/publish.py --reduction /absolute/new-private-reduction --output /absolute/new-report --scratch /absolute/new-private-plot-check\n```\n',
    '`metrics.csv`: 876 branch summaries; `paired-transitions.csv`: 864 W0-paired rows; `confusion.csv`: class counts; `checkpoints.csv`: 72 retained identities; `compute-*.csv`: 실제 비용; `mmlu-parser.csv`: parser 한계; `verification.json`: 전체 gate; `raw-member-manifest.json`: private raw 경로/hash만. `analysis-manifest.json`/`rooted-receipt.json`은 report/code/test/input을 결속한다.\n',
    'PNG는 위 CSV를 입력으로 직접 작성한 `plots.py`를 실행했다. 아래4개는 같은 입력으로 별도 디렉터리에 다시 생성해 byte SHA 동일성을 확인했다. 추가 GPU 실험/외부 이미지 도구/수동 수치 수정 0.\n'])
    for family in ('MEMIT','AlphaEdit'):
        for branch in ('generation','alternative'):text.append(f'![{family} {branch}](figures/{family}-{branch}.png)\n')
    text.append('## 부록. 전체12 checkpoint 경로\n\n모든 값은 weighted F1 %이며 정확 ACC/MCC/invalid/paired n/d는 연결된 CSV 전체행으로 제공한다. W0 공통값은 앞 표를 참조한다.\n')
    for family in ('MEMIT','AlphaEdit'):
        for branch in ('generation','alternative'):
            text.append(f'### {family} / {branch}\n');rr=[]
            for variant in ('BLUE','L4_ONLY','L8_ONLY'):
                for edits in (100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000):
                    rr.append([variant,edits]+[f'{float(lookup[(family,variant,edits,t,branch)]["weighted_f1"])*100:.2f}' for t in TASKS])
            text.append(mdtable(['variant','edits']+list(TASKS),rr))
    (dest/'diagnostic-report-ko.md').write_text('\n'.join(text))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reduction',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();publish(a.reduction,a.output,a.scratch)
