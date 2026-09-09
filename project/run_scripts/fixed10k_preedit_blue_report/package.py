"""Create-once v3 integration, original-table regression, and rooted publication."""
import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from .reduce import TASK, DATA_SHA, ORDER_ROOT, read, save, sha, digest, member, require, verify_members, write_csv
from .plots import generate

V2_SHA='65645e52fb19f6aa4ac30ee00ce36334c5c3a9e22ed1ce92c142ed9c251239b8'
REPO=Path(__file__).resolve().parents[3]
V2_REL='experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v2'
V3_REL='experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v3'


def csvread(p):
    return list(csv.DictReader(Path(p).open()))


def text(path, value):
    with Path(path).open('x') as f:f.write(value)


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join('---' for _ in headers)+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])+'\n'


def frac(n,d):return f'{int(n)}/{int(d)} ({100*int(n)/int(d):.3f}%)'


def integrated_body(old):
    """Keep all v2 science; only supersede five exact W0-availability passages."""
    updates={
      '|Pre-edit W0 (full10000)|NOT_RECORDED|—|—|未測定|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|':
      '|Pre-edit W0 (full10000; v3 추가)|TERMINAL_VALID|42673|edit0|10000|791/10000 (7.910%)|1997/20000 (9.985%)|89212/100000 (89.212%)|34/10000|72/20000|',
      'Base/W0 공란은 실패0점이 아니라 동일10k 측정 부재다.':
      'Base 공란은 실패0점이 아니라 동일10k 편집 endpoint 미확인이다. W0는 v3에서 job42673 실측으로 추가했고 두 표는 동일 평가 한 번을 참조한다.',
      '동일 full10000 pre-edit도 기록되지 않았다.':
      'v2 작성 때 없던 동일 full10000 pre-edit는 v3에서 job42673 실측으로 추가했다.',
      '현재6chain의 W0 전체10k 성능은 NOT_RECORDED다. 이전1k W0 점수를 이번10k의 W0 reference로 대입하지 않는다.':
      'v3 공통 W0 전체10k는 RS791/10000, PS1997/20000, NS89212/100000이다. 기존1k reference를 확대하지 않고 job42673 전체 원문 pair를 독립 재집계했다.',
      '|W0 전체10k RS/PS/NS|NOT_RECORDED; 이전1k 점수로 대입0|':
      '|W0 전체10k RS/PS/NS|v3 job42673 실측 추가; RS791/10000, PS1997/20000, NS89212/100000; cross-hardware numeric parity 미검증|',
    }
    log=[]
    for before,after in updates.items():
        require(before in old,'V2_AVAILABILITY_PASSAGE_CHANGED')
        log.append(dict(before=before,after=after,occurrences=old.count(before)))
        old=old.replace(before,after)
    # All v2 tables/assets remain byte-exact in inherited-v2; local image links resolve there.
    old=old.replace('](figures/','](inherited-v2/figures/')
    return old,log


def build(local, v2, out):
    require(not out.exists(),'CREATE_ONCE '+str(out))
    require(sha(v2/'factual-report-ko.md')==V2_SHA,'V2_REPORT_SHA')
    v2m=read(v2/'analysis-manifest.json');v2r=read(v2/'rooted-receipt.json')
    require(sha(v2/'analysis-manifest.json')==v2r['manifest_sha256'],'V2_MANIFEST_ROOT')
    verify_members(v2,v2m['members'])
    require(digest(v2m['members'])==v2m['member_root']==v2r['member_root'],'V2_MEMBER_ROOT')
    reduction=local/'reduction';paired=local/'paired';comp=local/'compatibility'
    validation=read(reduction/'independent-validation.json')
    require(validation['status']=='PASS' and read(paired/'paired-validation.json')['status']=='PASS','INPUT_VALIDATION')
    ds=csvread(reduction/'w0-distributions.csv'); w0={r['metric']:r for r in ds if int(r['prefix_requests'])==10000}
    six=csvread(v2/'final-summary.csv');old_dist=csvread(v2/'final-distributions.csv')
    require(len(six)==6 and len(old_dist)==18,'SIX_CHAIN')
    allsummary=[dict(arm='PRE_EDIT_W0',arm_display_label='PRE_EDIT (W0)',method='SHARED_W0',variant='PRE_EDIT',job='42673',
        status='TERMINAL_VALID',batches=0,requests=10000,
        **{f'{k}_{field}':w0[k][field] for k in w0 for field in ('numerator','denominator','rate')})]+six
    for r in allsummary:
        for k in w0:
            require(int(r[k+'_denominator'])==int(w0[k]['denominator']),'MATCHED_DENOMINATOR')
            r[k+'_delta_W0_pp']=100*(int(r[k+'_numerator'])-int(w0[k]['numerator']))/int(w0[k]['denominator'])
        r['scope']='PRE_EDIT_W0_FULL10000' if r['arm']=='PRE_EDIT_W0' else 'FINAL_W100_FULL10000'
    paired_rows=csvread(paired/'w0-final-paired-transitions.csv')
    for r in paired_rows:
        source=next(x for x in six if x['arm']==r['arm']);metric=r['metric']
        require(int(r['after_num'])==int(source[metric+'_numerator']),'BLUE_FINAL_REDUCTION')
        require(int(r['before_num'])==int(w0[metric]['numerator']),'W0_PAIRED_REDUCTION')
    out.mkdir(parents=True)
    shutil.copytree(v2,out/'inherited-v2')
    for src,names in [(reduction,['w0-distributions.csv','w0-part-distributions.csv','independent-validation.json']),
                      (paired,['w0-final-paired-transitions.csv','w0-final-paired-cohorts.csv','paired-validation.json']),
                      (comp,['w0-blue-compatibility.csv','shared-source-asset-identity.csv','compatibility-validation.json','blue-runtime-extracts.json'])]:
        for name in names:shutil.copyfile(src/name,out/name)
    write_csv(out/'final-with-W0.csv',allsummary)
    for family in ('MEMIT','AlphaEdit'):
        write_csv(out/f'final-{family}-with-W0.csv',[r for r in allsummary if r['arm']=='PRE_EDIT_W0' or r['method']==family])
    combined=[dict(arm='PRE_EDIT_W0',arm_display_label='PRE_EDIT (W0)',batch=0,**r) for r in w0.values()]+old_dist
    write_csv(out/'final-distributions-with-W0.csv',combined)
    availability=csvread(v2/'baseline-availability.csv')
    for r in availability:
        if r['method']=='PRE_EDIT' and r['scope']=='SAME_FULL10000':
            r.update(status='MEASURED_TERMINAL_VALID',reason='job42673; independent 260000 raw NLL target rows ->130000 exact prompt pairs; one shared W0 evaluation')
    write_csv(out/'baseline-availability.csv',availability)
    runtime=read(reduction/'runtime-copy.json');compute=read(reduction/'compute-copy.json');terminal=read(reduction/'terminal-copy.json')
    compute_row=dict(job=42673,scheduler_seconds=4440,allocated_GPU_hours=4440/3600,script_seconds=terminal['seconds'],
       part_evaluator_wall_sum=sum(float(read(Path(validation['terminal']['path']).parent/f'part-{i:03d}.json')['seconds']) for i in range(1,101)),
       model_load_seconds=runtime['model_load_seconds'],forward_calls=compute['forward_calls'],evaluator_calls=compute['calls'],
       input_tokens_with_padding=compute['input_tokens_including_padding'],input_tokens_without_padding=compute['nonpadding_input_tokens'],
       target_tokens=compute['target_tokens'],peak_GPU_allocated_bytes=compute['peak_gpu_allocated_bytes'],peak_GPU_reserved_bytes=compute['peak_gpu_reserved_bytes'],
       peak_host_RSS_KiB=compute['peak_host_rss_kib'],edit=0,backward=0,writer=0,compute_z=0,history_append=0,
       wall_components_overlap='part evaluator includes existing state guard/file IO; not pure GPU kernel time',new_evaluation_during_analysis=0)
    write_csv(out/'preedit-compute.csv',[compute_row])
    save(out/'scheduler-terminal-validation.json',dict(job=42673,owner='janghj',job_name='odeedit_fixed10k_preedit_s2',
       source='single bounded sacct call in this recall',state='COMPLETED',exit_code='0:0',elapsed='01:14:00',
       alloc_tres='billing=8,cpu=8,gres/gpu=1,mem=59G,node=1',batch_state='COMPLETED0:0',extern_state='COMPLETED0:0',
       scontrol='Invalid job id specified (purged live record); used accounting + sealed runtime/lock instead',
       runtime_slurm_job=runtime['slurm_job'],terminal_status=terminal['status'],
       downstream42706_read=0,other_jobs_read=0,submit_cancel_requeue=0))
    save(out/'preedit-runtime-summary.json',dict(runtime=runtime,compute=compute,observer=read(reduction/'observer-copy.json'),
       terminal={k:v for k,v in terminal.items() if k!='manifest_members'}))
    lines=['# Llama3 BLUE lifelong B100×100 상세 리뷰 v3 — 실측 PRE_EDIT W0 통합',
      '', '작성: Server2 / SH2. 기존 Server4 v2 상세 결과를 보존한 사용자 승인 통합판. 기준: 2026-09-09 KST.',
      'PRE_EDIT는 **편집 전 W0 한 번의 전체10,000 평가**이며 아래 두 계열 표에서 공통 참조한다. 두 W0 실험으로 합산하지 않는다. 편집 arm은 각자의 final W100 전체10,000 결과다. 새 GPU/evaluator 실행0; scientific_promotion=false.','']
    for family in ('MEMIT','AlphaEdit'):
        lines+=['## '+family+' 계열 — full10k 독립 비교표','',table(['arm / state','RS n/d (%)','ΔRS vs W0 pp','PS n/d (%)','ΔPS pp','NS n/d (%)','ΔNS pp'],
          [[r['arm_display_label']]+[v for k in ('RS','PS','NS') for v in (frac(r[k+'_numerator'],r[k+'_denominator']),f"{r[k+'_delta_W0_pp']:+.3f}")] for r in allsummary if r['arm']=='PRE_EDIT_W0' or r['method']==family])]
    lines+=['W0 RS/PS는 편집 요청의 **새 target 선호**를 세므로 낮은 W0 RS 7.910%를 pretrained 모델 전체 능력 저하로 읽지 않는다. NS는 true target 선호다. 6개 BLUE arm의 원래 값은 모두 불변이며 stock/base MEMIT·Official AlphaEdit가 아니다. 편집된 동일10k BASE_MEMIT/BASE_ALPHAEDIT 값은 여전히 **NOT_AVAILABLE**이고 해당 job은 조회하지 않았다.',
       '', '![W0 and final BLUE family comparison](figures/w0-final-comparison.png)',
       '', '## 1. W0 실측·분모·정의',
       '', 'job42673은 sacct COMPLETED/0:0, elapsed01:14:00; source `12ad4cbe417d9935b1e7e55600214d14ba0d6360` / tree `be4ee8bede7e4e962fd13064814bd2fbd500b56a`다. Execution lock SHA `7ea4991018fb18b1cb4dc520acf3e6adb386ad39112440fa2f57af9e33bdf167`.',
       '600개 원문 JSON의 new/true 260,000 target 행을 데이터 문장·target·case·prompt index와 대조했다. 독립 CPU reducer의 pair130,000행이 기존 part100개 및 full10000 저장행/비트해시/numden와 정확히 같다. missing/duplicate/nonfinite/imputation0; ties RS/PS/NS 모두0. 첫 B100을10k로 확장한 값이 아니다.',
       f"입력 source/environment/model asset **{validation['source_asset_environment_rehash']['files']}개 / {validation['source_asset_environment_rehash']['bytes']}B**, terminal manifest output **{validation['output_member_rehash']['files']}개 / {validation['output_member_rehash']['bytes']}B**를 이번 recall에서 full SHA 재검산했다. terminal 자체는 별도 SHA로 결속한다. raw/log 원본 수정0.",
       'NLL은 source의 teacher-forced target token −log p 평균, nats/token이다. 저장 NLL로 strict new<true(RS/PS), true<new(NS)를 다시 계산한 것이며, 저장하지 않은 per-token log-prob나 logits를 재생성했다는 주장이 아니다. 정확한 `<`만 성공; tie 실패. 새 evaluator/rebatch/forward0.',
       'TF strict는 target 모든 token의 teacher-forced top1 일치, token accuracy는 correct/실제 target token 수다. 자유 생성 정확도와 다르다. Request pair-strict는 한 request의 모든 category prompt 선호가 성공(RS1/PS2/NS10)한 수이며 TF strict와 다르다.','',
       table(['W0 category','primary n/d (%)','request pair-strict /10000','new TF strict n/d','true TF strict n/d','new token correct/den','true token correct/den'],
        [[k,frac(r['numerator'],r['denominator']),f"{r['pair_strict_num']}/10000",frac(r['new_strict_num'],r['new_strict_den']),frac(r['true_strict_num'],r['true_strict_den']),frac(r['new_token_correct'],r['new_token_den']),frac(r['true_token_correct'],r['true_token_den'])] for k,r in w0.items()]),
       '## 2. W0 / final W100 NLL·secondary 비교','']
    for family in ('MEMIT','AlphaEdit'):
        selected=[r for r in combined if r['arm']=='PRE_EDIT_W0' or r['arm'].startswith(family)]
        lines+=['### '+family,'',table(['arm','metric','new NLL mean / median / p90','true NLL mean / median / p90','margin mean / median / p90','new TF strict / prompt den','new correct tokens / token den'],
          [[r['arm_display_label'],r['metric']]+[' / '.join(f"{float(r[prefix+s]):.6f}" for s in ('mean','median','p90')) for prefix in ('new_nll_prompt_','true_nll_prompt_','margin_prompt_')]+[frac(r['new_strict_num'],r['new_strict_den']),frac(r['new_token_correct'],r['new_token_den'])] for r in selected])]
    lines+=['전체 new/true token·TF strict, q25/q75/max, request-cluster 평균 후 분포는 `final-distributions-with-W0.csv`에 21행으로 보존한다. Request-cluster mean은 prompt 수가 동일해 전체 prompt mean과 같을 수 있으나 median/quantile은 다르다. margin=true−new를 NS에서도 부호 변경하지 않는다.',
       '', '## 3. 정확한 같은 문항 W0→final NS 전환','',
       table(['arm','W0 success→post failure /89212','W0 failure→post success /10788','retained','both failed','post success /100000'],
        [[next(s['arm_display_label'] for s in six if s['arm']==r['arm']),frac(r['lost'],89212),frac(r['gained'],10788),r['retained'],r['both_failed'],frac(r['after_num'],100000)] for r in paired_rows if r['metric']=='NS']),
       '분모89,212는 공통 W0 NS 성공 문항, 10,788은 W0 실패 문항이다. 각 행 retained+lost+gained+both_failed=100,000, 89,212−lost+gained=post numerator를 독립 검산했다. 총점만으로 문항 보존을 추정하지 않았다. 이 전환은 NLL 선호 변화이며 전체 pretrained capability 보호를 증명하지 않는다.',
       'RS/PS/NS 전체18 paired 행과 new/true NLL paired delta·증가/감소 수는 `w0-final-paired-transitions.csv`, 고정 B100 cohort×100×6×3=1,800행은 `w0-final-paired-cohorts.csv`다. 각 pair는 case_id/prompt_index/문장+target identity와 순서까지 일치했다. SH4의 봉인 final6 파일 399,737,324B만 read-only CPU로 읽었고 대형 raw는 옮기지 않았다. W0 scalar identity/NLL/bit만 메모리로 전달; prompt/token/tensor 전송0, 원격 파일쓰기0.',
       '', '## 4. 같은 W0의 prefix1k/3k — 추가 forward0','',
       table(['fixed prefix requests','RS n/d (%)','PS n/d (%)','NS n/d (%)'],[[n]+[frac(next(r for r in ds if int(r['prefix_requests'])==n and r['metric']==k)['numerator'],n*m) for k,m in [('RS',1),('PS',2),('NS',10)]] for n in (1000,3000,10000)]),
       '이는 동일 full10k 원문 rows[:N]의 CPU 부분집계다. v2의 과거1k W0 71/1000,227/2000,8820/10000 reference와 출처를 혼합하지 않는다. 과거1k reference/환경 차이는 하단 기존 상세 본문에 그대로 유지한다. Prefix W0, lifelong W10, final W100의 first1000은 서로 다른 state/scope다.',
       '', '## 5. Source·W0·evaluator 호환성 및 검증 한계','',
       table(['항목','검증 수준'],[
        ['fixed dataset',f'counterfact SHA {DATA_SHA}; order root {ORDER_ROOT}; unique10000 및 exact prompt identity PASS'],
        ['source/model assets','1757 common members SHA/size exact: 4 safetensor shards/config/index/tokenizer와 evaluator source 포함; source-asset 전체1777 현지 재해시'],
        ['W0','6 BLUE runtime의 selected entry W0 SHA가 PRE_EDIT L4..8 selected SHA와 일치; 각1/2keys. 나머지 model parameter는 pointer/version guard이며 full bytes NOT_CLAIMED'],
        ['revision','Llama3-8B-Instruct 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'],
        ['precision/backend','FP32, eager, torch2.9.1+cu128 / transformers4.44.2; TF32 matmulFalse/cudnnTrue. eval/inference_mode; seed20260907'],
        ['tokenizer/padding','same assets; tokenizer property right/pad=eos128009, evaluator explicit LEFT padding+attention_mask; position_ids native default; add_bos runtime NOT_EXPOSED'],
        ['NLL kernel','4f5af6dbf8854c79aedc5e36134fddfcb2995b60f05d270604f8ea735672d93e; shared evaluation wrapper d82bc6f0090c3f572ecc23a2de03e8af544e910e1f3534b331ffd8486f96085e'],
        ['batch packing WARN','microbatch16 동일. PRE_EDIT는100×B100 호출; BLUE W100은 past9900 + current100 재사용 merge. 카테고리 호출경계/패딩 묶음 완전 동일은 아님; source를 바꾸거나 재평가하지 않았다'],
        ['hardware WARN','Server2 RTX A6000 vs Server4 RTX PRO6000 Blackwell; same-W0 cross-device forward/logit parity NOT_TESTED'],
        ['contexts','W0는 봉인된 dataset prompt만 평가; compute_z/context generation0. BLUE native target context와 동일한 편집경로라는 주장은 하지 않음'],
        ['nonmutation','100/100 parts selected5 byte/pointer/version 및 모든 parameter pointer/version/requires_grad exact runtime assertions; optimizer/backward/key/solve/history append/edit0'],
       ]),
       '동일 source/data/entry selected bytes에 근거한 기술자료 통합이며 cross-hardware/호출묶음 차이를 제거한 인과 추정이 아니다. 별도 parity 측정을 실행하지 않았고 작은 차이의 원인을 단정하지 않는다. `w0-blue-compatibility.csv`와 `shared-source-asset-identity.csv`에 경로/SHA를 보존한다.',
       '', '## 6. PRE_EDIT 계산량 / 이번 분석 action 경계','',
       table(['항목','값'],[[k,v] for k,v in compute_row.items()]),
       'GPUh는 allocated 시간이며 utilization 측정이 아니다. script/part/model-load 시간은 중첩 범위가 있어 합으로 재구성하지 않는다. 분석 자체는 CPU이며 위 GPU 비용은 완료42673의 기존 계측이다. downstream42706/source/output/모니터링 조회0, native baseline/L567/중지ORBODE 조회·변경0. 새 submit/cancel/requeue/model/evaluator0.',
       '', '## 7. 산출물·검산·재현','',
       '원본 v2 package 77개 파일은 `inherited-v2/`에 byte-exact 복사하고 모든 원래 CSV 값/순서와10개 PNG를 유지했다. 아래 상세 본문에서는 W0 가용성 문장5종(총7 occurrence)만 명시적으로 갱신했다. 원래6chain 수치와 누적/forgetting/NLL/cost/checkpoint/한계 내용은 삭제하지 않았다. inherited-v2의 NOT_RECORDED는 당시 기록으로서 v3의 현 상태가 아니다.',
       '현재 canonical availability는 루트 `baseline-availability.csv`; 기존 BASE 측정 부재는 불변이다. source/input/output identity, regression/tests, member root는 `analysis-manifest.json` 및 `rooted-receipt.json`으로 결속한다. 원본 raw/log/model/prompt/cache는 Git 제외. local contract/SHA는 audit 및 manifest에만 참조한다.',
       '재현 코드: `project/run_scripts/fixed10k_preedit_blue_report/`. 독립 reducer는 `python3 -m project.run_scripts.fixed10k_preedit_blue_report.reduce --task-root /mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1 --dataset /mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json --out <new-local-reduction>`; GPU/import/evaluator0.',
       f'PNG 입력 `final-MEMIT-with-W0.csv`, `final-AlphaEdit-with-W0.csv`; 재현 `/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.fixed10k_preedit_blue_report.plots --package {V3_REL} --destination <new-output.png>`. Python matplotlib로만 생성하고 두 번 실행 SHA 동일을 검사했다. 기존10개 PNG는 불필요 재생성0.',
       f'패키지 검산 `python3 -m project.run_scripts.fixed10k_preedit_blue_report.package verify --out {V3_REL}`. 테스트 `python3 -m unittest -q project.run_scripts.fixed10k_preedit_blue_report.test_focused`.',
       '', '---', '', '# 기존 BLUE lifelong 상세 본문 — v2 계승 / W0 가용성만 v3 반영','',
       '아래의 v2 자체 소개·historical reference·기존 test/action counts는 당시 분석 provenance이며, 이번 추가 CPU 검산은 상단 및 루트 receipt를 따른다. 하단 파일명은 별도 표시가 없으면 `inherited-v2/` 기준이다. 원문 자체는 `inherited-v2/factual-report-ko.md`에 byte-exact 보존한다.','']
    body,changes=integrated_body((v2/'factual-report-ko.md').read_text())
    text(out/'factual-report-ko.md','\n\n'.join(lines)+body)
    save(out/'v2-text-amendments.json',changes)
    save(out/'input-provenance.json',dict(instruction_id=TASK,authored_server='server2',authorized_server4_v3_write=True,
       authoritative_contract=member(local/'authoritative-instruction.txt'),protocol=member(REPO/'PROTOCOL.md'),
       base_origin_main='4d70b8b8c7ce2fcfdc1513618b0ce25386dc0aaf',
       v2_report=member(v2/'factual-report-ko.md'),v2_manifest=member(v2/'analysis-manifest.json'),v2_receipt=member(v2/'rooted-receipt.json'),
       local_reduction_receipt=member(reduction/'independent-validation.json'),local_pair_receipt=member(paired/'paired-validation.json'),
       raw_auto_broadcast='not needed: local source/model/260000 raw rows retained; remote BLUE raw retained; only raw-free publication broadcast',
       scientific_promotion=False))
    generate(out,out/'figures/w0-final-comparison.png')
    with tempfile.TemporaryDirectory(prefix='w0-plot-repro-') as d:
        other=Path(d)/'same.png';generate(out,other)
        require(sha(other)==sha(out/'figures/w0-final-comparison.png'),'PNG_REPRODUCIBILITY')
    save(out/'plot-receipt.json',dict(code='project/run_scripts/fixed10k_preedit_blue_report/plots.py',
       source_sha256=sha(Path(__file__).with_name('plots.py')),input=[member(out/f'final-{f}-with-W0.csv') for f in ('MEMIT','AlphaEdit')],
       output=member(out/'figures/w0-final-comparison.png'),byte_identical_reproduction=True,imagegen_visualize=0,
       inherited_figures_byte_preserved=10,matplotlib_version=__import__('matplotlib').__version__))
    checks=validate(out,v2)
    save(out/'focused-checks.json',checks)
    print(json.dumps(checks,ensure_ascii=False))


def validate(out,v2=None):
    v2=v2 or out/'inherited-v2'
    require(sha(out/'inherited-v2/factual-report-ko.md')==V2_SHA,'INHERITED_REPORT')
    inherited=[p for p in v2.rglob('*') if p.is_file()]
    for p in inherited:require(sha(p)==sha(out/'inherited-v2'/p.relative_to(v2)),'INHERITED_MEMBER')
    before=csvread(v2/'final-summary.csv');after=csvread(out/'final-with-W0.csv')
    require(len(after)==7,'ONE_SHARED_W0_PLUS_SIX')
    shared=csvread(out/'shared-source-asset-identity.csv')
    require(sum(m['server4_path'].endswith('.safetensors') for m in shared)==4,'FOUR_WEIGHT_SHARD_IDENTITIES')
    for r in before:
        got=next(x for x in after if x['arm']==r['arm'])
        require(all(got[k]==v for k,v in r.items()),'SIX_SCIENTIFIC_VALUES_CHANGED')
    for family in ('MEMIT','AlphaEdit'):
        r=csvread(out/f'final-{family}-with-W0.csv');require(len(r)==4 and r[0]['arm']=='PRE_EDIT_W0','FAMILY_TABLE')
        for row in r:
            for k,m in [('RS',1),('PS',2),('NS',10)]:
                n,d=int(row[k+'_numerator']),int(row[k+'_denominator'])
                require(0<=n<=d==10000*m and abs(float(row[k+'_rate'])-n/d)<1e-12,'NUMDEN_RATE')
                require(abs(float(row[k+'_delta_W0_pp'])-100*(n-int(r[0][k+'_numerator']))/d)<1e-12,'DELTA_PP')
    for name,count in [('w0-final-paired-transitions.csv',18),('w0-final-paired-cohorts.csv',1800)]:
        rows=csvread(out/name);require(len(rows)==count,'PAIRED_ROWS')
        for r in rows:
            require(sum(int(r[k]) for k in ('retained','lost','gained','both_failed'))==int(r['denominator']),'PARTITION')
            require(int(r['before_num'])-int(r['lost'])+int(r['gained'])==int(r['after_num']),'TRANSITION_ARITHMETIC')
    suite=unittest.defaultTestLoader.loadTestsFromName('project.run_scripts.fixed10k_preedit_blue_report.test_focused')
    result=unittest.TextTestRunner(stream=io.StringIO()).run(suite);require(result.wasSuccessful(),'FOCUSED_TESTS')
    for p in Path(__file__).parent.glob('*.py'):compile(p.read_text(),str(p),'exec')
    forbidden={'.pt','.bin','.safetensors','.npz','.tar','.gz','.log','.out','.err','.pyc'}
    require(not any(p.suffix in forbidden for p in out.rglob('*') if p.is_file()),'RAW_CLASS_IN_PACKAGE')
    for p in out.glob('*.csv'):
        fields=csvread(p)[0].keys();require(not {'prompt','target','token_predictions','target_token_ids','raw_text'} & set(fields),'RAW_COLUMN')
    # The full inherited prose is preserved except the enumerated W0-availability edits.
    expected,_=integrated_body((out/'inherited-v2/factual-report-ko.md').read_text())
    require((out/'factual-report-ko.md').read_text().endswith(expected),'DETAIL_BODY_REGRESSION')
    return dict(status='PASS_WITH_DOCUMENTED_WARN',focused_tests=result.testsRun,failures=0,errors=0,
       original_v2_files_byte_identical=len(inherited),original_v2_CSV_byte_identical=len(list(v2.glob('*.csv'))),
       original_PNG_byte_identical=10,original_six_all_columns_values_order_exact=True,shared_W0_evaluations=1,
       paired_full_rows=18,paired_cohort_rows=1800,denominator_pp_partition_checks='PASS',raw_free='PASS',
       cross_hardware_and_packing_parity='NOT_TESTED_WARN',nonselected_full_parameter_bytes='NOT_CLAIMED',
       scientific_promotion=False,new_GPU_model_evaluator_Slurm=0)


def seal(out):
    checks=validate(out)
    sources=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(Path(__file__).parent.glob('*.py'))]
    members=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p),**({'rows':len(csvread(p))} if p.suffix=='.csv' else {}))
       for p in sorted(out.rglob('*')) if p.is_file() and p not in (out/'analysis-manifest.json',out/'rooted-receipt.json')]
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    man=dict(instruction_id=TASK,authored_server='server2',analysis_source_head=head,
       analysis_source_tree=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD^{tree}'],text=True).strip(),
       source=sources,inputs=read(out/'input-provenance.json'),members=members,member_root=digest(members),
       experimental_source_head='12ad4cbe417d9935b1e7e55600214d14ba0d6360',
       unique_W0_requests=10000,shared_W0_evaluations=1,original_BLUE_chains=6,dataset_sha256=DATA_SHA,order_root=ORDER_ROOT,
       scientific_promotion=False)
    save(out/'analysis-manifest.json',man)
    rec=dict(status='REVIEW_READY',instruction_id=TASK,authored_server='server2',member_root=man['member_root'],
       manifest_sha256=sha(out/'analysis-manifest.json'),report_sha256=sha(out/'factual-report-ko.md'),member_count=len(members),
       checks=checks,raw_rehash=read(out/'independent-validation.json'),paired_read=read(out/'paired-validation.json'),
       scientific_promotion=False,new_GPU_model_evaluator_Slurm=0)
    rec['root_sha256']=digest(rec);save(out/'rooted-receipt.json',rec)
    verify(out);print(json.dumps({k:rec[k] for k in ('status','member_count','report_sha256','manifest_sha256','root_sha256')},ensure_ascii=False))


def verify(out):
    m=read(out/'analysis-manifest.json');r=read(out/'rooted-receipt.json')
    require(sha(out/'analysis-manifest.json')==r['manifest_sha256'],'MANIFEST_SHA')
    require(sha(out/'factual-report-ko.md')==r['report_sha256'],'REPORT_SHA')
    require(digest(m['members'])==m['member_root']==r['member_root'],'ROOT')
    verify_members(out,m['members']);verify_members(REPO,m['source'])
    require({str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}=={x['path'] for x in m['members']}|{'analysis-manifest.json','rooted-receipt.json'},'PACKAGE_CLOSURE')
    root=r.pop('root_sha256');require(digest(r)==root,'RECEIPT_ROOT')
    validate(out)
    print('PACKAGE_REHASH_ACCESS_PASS',len(m['members']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','seal','verify']);p.add_argument('--local',type=Path);p.add_argument('--v2',type=Path,default=REPO/V2_REL);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.stage=='build':build(a.local,a.v2,a.out)
    elif a.stage=='seal':seal(a.out)
    else:verify(a.out)
