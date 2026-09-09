"""v2 label/family/base/W0 presentation revision; sealed aggregates only.

No runtime/evaluator/model imports, no raw result mutation or new scientific reduction.
"""
import argparse,json,re,shutil,subprocess
from .common import *
from .revision_labels import LABELS,ORDER,CAPTION,display,labeled_rows
from .revision_references import V1,inspect
from .plots import generate

INSTRUCTION_V2='ODEEDIT-S06-BLUE-L4-L8-LIFELONG-REPORT-BASELINE-LABEL-REVISION-SH4-V1'
OUT=Path('experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v2')
COMMAND='python -m project.run_scripts.blue_lifelong_analysis.revision plots --out PACKAGE --dest NEW_DIRECTORY'

def rates(r):
    return {t:f"{r[t+'_num']}/{r[t+'_den']} ({float(r[t+'_rate'])*100:.3f}%)" for t in MULT}

def family_tables(out):
    final=csvread(out/'final-summary.csv');text=[]
    for method in ['MEMIT','AlphaEdit']:
        rows=[];csvrows=[]
        for arm in [a for a in ORDER if a.startswith(method+'_')]:
            r=next(r for r in final if r['arm']==arm)
            z=dict(display_label=LABELS[arm],status=r['status'],job=r['job'],batches=r['batches'],requests=r['requests'])
            for t in MULT:z[t]=f"{r[t+'_numerator']}/{r[t+'_denominator']} ({100*float(r[t+'_rate']):.3f}%)"
            z['rewrite_TF_exact']=f"{r['RS_new_strict_num']}/{r['RS_new_strict_den']}";z['rephrase_TF_exact']=f"{r['PS_new_strict_num']}/{r['PS_new_strict_den']}"
            rows.append(z);csvrows.append(dict(display_label=LABELS[arm],**r))
        for label in ['Base MEMIT' if method=='MEMIT' else 'Official AlphaEdit','Pre-edit W0 (full10000)']:
            rows.append(dict(display_label=label,status='NOT_AVAILABLE' if label.startswith(('Base','Official')) else 'NOT_RECORDED',job='—',batches='—',requests='未測定',**{k:'NOT_MEASURED' for k in ['RS','PS','NS','rewrite_TF_exact','rephrase_TF_exact']}))
        text+=['### '+method+' 계열: 최종 W100 full10000\n',table(rows,list(rows[0])),CAPTION+' RS/PS/NS 단위는 prompt, 분모는 실측10,000/20,000/100,000. Base/W0 공란은 실패0점이 아니라 동일10k 측정 부재다.\n']
        csvwrite(out/('final-'+method+'-family.csv'),csvrows)
        # All original final NLL/strict/token fields remain in the family CSV.
        nll=[]
        for arm in [a for a in ORDER if a.startswith(method+'_')]:
            for r in csvread(out/'cumulative-metrics.csv'):
                if r['arm']!=arm or r['batch']!='100':continue
                z=dict(display_label=LABELS[arm],category=r['metric'],prompt_n=r['denominator'])
                for side in ['new','true']:
                    z[side+'_NLL_mean/median/p90/max']='/'.join(f'{float(r[side+"_nll_prompt_"+s]):.6g}' for s in ['mean','median','p90','max'])
                z['margin_mean/median/p90/max']='/'.join(f'{float(r["margin_prompt_"+s]):.6g}' for s in ['mean','median','p90','max'])
                nll.append(z)
        text+=['#### '+method+' 최종 NLL/secondary reading table\n',table(nll,list(nll[0])),
               'NLL은 target token 평균(nats/token), 위 분포는 prompt 단위다. Margin=true−new, NS 성공은 음수. Request-cluster 분포와 모든 strict/token numerator/denominator는 해당 family CSV 및 §7에 별도 보존한다.\n']
    return '\n'.join(text)

def references_text(out):
    pre=csvread(out/'prefix1000-references.csv');hist=csvread(out/'historical-base-references.csv');scope=read(out/'baseline-availability-audit.json')
    def show(rows):return table([dict(reference=r['reference'],scope=r['scope'],**rates(r)) for r in rows],['reference','scope','RS','PS','NS'])
    return '\n'.join(['### 기존 base baseline 및 pre-edit 가용성\n',
      '동일 BLUE10k sample/order에서 측정된 **base MEMIT/Official AlphaEdit final W100**은 감사 범위의 자료에서 찾지 못했다. 동일 full10000 pre-edit도 기록되지 않았다. 이는 전체 서버의 측정 부재를 증명하는 전역 검색 주장이 아니다. 새 evaluator/backfill/replay0.\n',
      '#### 동일 prefix1k reference — not final10k\n',show(pre),
      'Pre-edit W0는 편집방법이 아니다. 기존 BLUE 세1k run과 JVP publication의 W0 네행 모두 RS71/1000, PS227/2000, NS8820/10000을 기록했다. 여기서는 중복분모로 합산하지 않고 한 reference행으로 표시한다. 이는 **full1000** 평가이며 full10000 W0로 확대하지 않는다. Official AlphaEdit O_NATIVE는 source77358b1546d1baf83b3e251afcce663b08d7bfd7의 stock5layer reference이며 JVP controller arm이 아니다.\n',
      '첫1k IDs/order/record·target hashes는 exact다. 그러나 Official vs BLUE의 layer/target 정책, contexts hash, seed(20260906 vs20260907), GPU(A6000 vsBlackwell), tokenizer/backend 정책이 달라 method-only causal comparison이 아니다. 숫자 차이의 원인 또는 모델 byte parity를 주장하지 않는다. Prefix t1000 BLUE 값은 이번lifelong W10이지 final W100의첫cohort평가가 아니다.\n',
      '#### Historical base10k — 다른 stream / non-paired\n',show(hist),
      f"Historical v6는 native MEMIT/AlphaEdit의 다른10000 stream `{scope['historical_sample_root']}` / order `{scope['historical_order']}`의 **실제 final W100 all10000 canonical NLL pair**다. BLUE10000과 case 교집합{scope['case_intersection']}개지만 order/stream은 다르며 duplicate/collision selection policy도 다르다. 교집합만 골라 성능을 추정하거나 paired Δpp를 만들지 않았다.\n",
      '과거종단 current-B100/teacher-forced Loc를 재명명한 값이 아니라 v6 보충평가 후 확정된 RS/PS/NS다. Historical LM/LA edit source85a05d0a831db667fc68c27e634302eb30cc7d79, jobs32424/32380; 후속 frozen-state evaluation jobs33306/33539. 이번에는 봉인publication과 작은source/config/sample metadata만 재해시했으며 과거 raw 전체나 GPU state를 다시 검증했다고 주장하지 않는다.\n',
      '#### Reference compatibility / 검증 수준\n',table(csvread(out/'reference-compatibility.csv'),['reference','source_head','model_revision','sample_root','config_sha','layers','context_hash','seed','dtype','gpu','comparison','validation']),
      '상세 source/input path·SHA는 reference-input-inventory.csv, search 범위와 미측정 상태는 baseline-availability-audit.json 및 baseline-availability.csv. 원본v1의 raw/state audit는 그대로 계승하되 이번 revision의 검증 수준과 구분한다.\n'])

def regroup_sections(body):
    # Reorder per-arm subsections within their original semantic section only.
    chunks=re.split(r'(?=^## )',body,flags=re.M);out=[]
    for chunk in chunks:
        headings=list(re.finditer(r'^### (?:\d+\.\d+ )?('+ '|'.join(map(re.escape,ORDER))+r')\n',chunk,re.M))
        if len(headings)==6:
            start=headings[0].start();end=headings[-1].end()
            tailmatch=re.search(r'^### (?!'+ '|'.join(map(re.escape,ORDER))+r')',chunk[end:],re.M)
            end=len(chunk) if tailmatch is None else end+tailmatch.start()
            pieces={}
            for i,h in enumerate(headings):pieces[h[1]]=re.sub(r'^#### ', '##### ',chunk[h.start():headings[i+1].start() if i<5 else end],flags=re.M)
            chunk=chunk[:start]+''.join('\n### '+m+' 계열\n'+''.join(re.sub(r'^### .*\n','#### '+LABELS[a]+'\n',pieces[a],count=1) for a in ORDER if a.startswith(m+'_')) for m in ['MEMIT','AlphaEdit'])+chunk[end:]
        out.append(chunk)
    return ''.join(out)

def split_method_tables(body):
    # Family-specific tables for integrity, current/forgetting, updates and cost.
    def split(m):
        lines=m[0].strip().splitlines();header=lines[:2];rows=lines[2:]
        if not rows or not all(any('|'+a+'|' in row for a in ORDER) for row in rows):return m[0]
        if not any('|MEMIT_' in row for row in rows) or not any('|AlphaEdit_' in row for row in rows):return m[0]
        return '\n\n'.join('**'+family+' 계열**\n\n'+'\n'.join(header+[row for row in rows if '|'+family+'_' in row])+'\n\n'+CAPTION for family in ['MEMIT','AlphaEdit'])+'\n'
    return re.sub(r'(?:^\|.*\|\n)+',split,body,flags=re.M)

def render(repo,out):
    original=(repo/V1/'factual-report-ko.md').read_text()
    title='# Llama3 BLUE / L4-only / L8-only lifelong 상세 리뷰 v2 — B100×100\n'
    body=original[original.index('## 2.'):]
    # Inventory is regenerated, not stale SHA references to relabeled tables/PNGs.
    body=body[:body.index('## 11.')]+body[body.index('## 12.'):]
    body=display(split_method_tables(regroup_sections(body)))
    body=re.sub(r'(?<![A-Za-z])original(?![A-Za-z])','BLUE (L4+L8)',body,flags=re.I)
    body=body.replace('원본1k report','기존 BLUE1k report').replace('원본1k','기존 BLUE1k')
    # New appendix supplements rather than silently removing the prior missing10k note.
    final=family_tables(out)
    prefix=references_text(out)
    mapping=table([dict(raw_arm_id=a,display_label=LABELS[a],blue=True) for a in ORDER],['raw_arm_id','display_label','blue'])
    text=title+'\n본 v2는 v1의 명칭·계열별 표·base/pre-edit reference 보강본이다. v1 봉인 report SHA `9c2a520d26e3bab27ebac3a6eab01653f5825e2294fd5194a74fe8df3b6bc38d` 및 원본bytes는 보존한다. 모든6chain 측정값·분모·미측정 상태는 불변이며 통계/evaluator 재평가0. scientific_promotion=false.\n\n## 1. 최종 actual W100 full10000 — 계열별 주표\n'+final+prefix+'\n![Family final performance](figures/final-full10000.png)\n'+body
    text+='\n## 13. Display mapping / raw provenance\n'+CAPTION+' raw ORIGINAL은 식별자만 보존하며 base 방법을 뜻하지 않는다.\n'+mapping
    text+='\n## 14. Revision inventory / reproducibility\n'+table([dict(file=p.name,rows=len(csvread(p)),sha256=sha(p)) for p in sorted(out.glob('*.csv'))],['file','rows','sha256'])
    for f in read(out/'figures/plot-manifest.json'):text+=f"\n- `figures/{f['path']}` SHA `{f['sha256']}`; {f['caption']}\n"
    text+='\n```bash\npython -m project.run_scripts.blue_lifelong_analysis.revision build --repo REPO --out NEW_PACKAGE\n'+COMMAND+'\n```\nPNG는 Python/Agg 코드만으로 생성한다. 기존 집계값 불변, 명칭/계열/기준선 범위 검사, PNG byte 재현과 package full rehash를 이번 receipt에 기록한다. inherited-evidence는 과거 검산 receipt이며 이번 revision 검증과 구분한다.\n'
    (out/'factual-report-ko.md').write_text(text)

def build(repo,out):
    out.mkdir(parents=True,exist_ok=False)
    inspect(repo,out)
    for p in sorted((repo/V1).glob('*.csv')):
        rows=csvread(p);csvwrite(out/p.name,labeled_rows(rows))
    evidence=out/'inherited-evidence';evidence.mkdir()
    for p in (repo/V1).glob('*.json'):shutil.copyfile(p,evidence/p.name)
    csvwrite(out/'display-labels.csv',[dict(raw_arm_id=a,display_label=LABELS[a],blue=True) for a in ORDER])
    # Explicit per-family complete scalar tables preserve all columns and exact values.
    for name in ['cumulative-metrics','current-metrics','prompt-transitions','compute-summary']:
        rr=csvread(out/(name+'.csv'))
        for method in ['MEMIT','AlphaEdit']:csvwrite(out/(method+'-'+name+'.csv'),[r for r in rr if r['arm'].startswith(method+'_')])
    generate(out,out/'figures',LABELS,COMMAND)
    render(repo,out)
    print('BUILD_COMPLETE',str(out),sha(out/'factual-report-ko.md'),flush=True)

def verify(repo,out,repro):
    for p in (repo/V1).glob('*.csv'):
        old=csvread(p);new=csvread(out/p.name);assert len(old)==len(new)
        for a,b in zip(old,new):assert all(b[k]==v for k,v in a.items()),p.name
    a=generate(out,repro,LABELS,COMMAND);b=read(out/'figures/plot-manifest.json')
    assert {x['path']:x['sha256'] for x in a}=={x['path']:x['sha256'] for x in b}
    assert len(b)==10
    report=(out/'factual-report-ko.md').read_text()
    assert all('### '+m+' 계열: 최종' in report for m in ['MEMIT','AlphaEdit'])
    before_mapping=report.split('## 13.')[0]
    assert not re.search(r'\bORIGINAL\b|\boriginal\b|MEMIT_ORIGINAL|AlphaEdit_ORIGINAL',before_mapping)
    assert 'not final10k' in report and 'NOT_MEASURED' in report and 'Pre-edit W0 (1k)' in report
    for f in b:
        for a in f['axes']:assert not re.search('ORIGINAL|original',a['title'])
    save(out/'revision-checks.json',dict(status='PASS',old_csv_count=30,all_original_columns_values_order_exact=True,figures_byte_identical=10,base_same10k='NOT_AVAILABLE',preedit_full10k='NOT_RECORDED',new_GPU_model_evaluator_Slurm=0))

def seal(repo,out):
    members=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    forbidden={'.pt','.npz','.bin','.safetensors','.log','.err','.out','.tar'}
    assert not any(Path(m['path']).suffix in forbidden for m in members)
    code=[dict(path=str(p.relative_to(repo)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((repo/'project/run_scripts/blue_lifelong_analysis').glob('*.py'))]
    m=dict(instruction_id=INSTRUCTION_V2,source_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),source=code,inputs=read(out/'baseline-availability-audit.json'),members=members,member_root=digest(members),v1_report_sha256=sha(repo/V1/'factual-report-ko.md'),scientific_promotion=False)
    save(out/'analysis-manifest.json',m)
    for r in members:assert sha(out/r['path'])==r['sha256']
    rec=dict(status='REVIEW_READY',instruction_id=INSTRUCTION_V2,report_sha256=sha(out/'factual-report-ko.md'),manifest_sha256=sha(out/'analysis-manifest.json'),member_root=m['member_root'],members=len(members),CSV_count=len(list(out.glob('*.csv'))),PNG_count=10,original_measurements_unchanged=True,checks=read(out/'revision-checks.json'),tests=read(out/'revision-tests.json'),new_GPU_model_evaluator_Slurm=0,next='MAIN_PUSH_GH_REPORT_STOP')
    save(out/'rooted-receipt.json',rec);print(json.dumps(rec,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','plots','verify','seal']);p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--out',type=Path,required=True);p.add_argument('--dest',type=Path);a=p.parse_args()
    if a.stage=='build':build(a.repo,a.out)
    elif a.stage=='plots':generate(a.out,a.dest,LABELS,COMMAND)
    elif a.stage=='verify':verify(a.repo,a.out,a.dest)
    else:seal(a.repo,a.out)
