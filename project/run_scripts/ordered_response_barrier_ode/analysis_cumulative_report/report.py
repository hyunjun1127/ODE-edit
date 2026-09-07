"""Single raw-free Korean report and rooted package from completed CPU stages."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
from .common import *

TITLE='ORBODE B100×10 cumulative rerun — 20-arm exhaustive factual report'
REPORT='orbode-cumulative-exhaustive-factual-ko.md'
GROUPS={'performance':'performance-v1','mechanism':'mechanism-v1','comparisons':'comparisons-v1','integrity':'integrity-v1','tail':'tail-v1'}
PRIVATE={'request-state-records.csv.gz','prompt-pair-records.csv.gz'}

def table(df,cols=None):
    f=df if cols is None else df[cols]
    def v(x):
        if x is None or (isinstance(x,(float,np.floating)) and np.isnan(x)):return 'NOT_RECORDED'
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['|'+'|'.join(f.columns)+'|','|'+'|'.join(['---']*len(f.columns))+'|']+['|'+'|'.join(v(x) for x in row)+'|' for row in f.itertuples(index=False,name=None)])

def rates(df,extra=()):
    rows=[]
    for r in df.to_dict('records'):
        x={k:r[k] for k in ['cell','arm',*extra]}
        for m in ('RS','PS','PS_strict','NS','rewrite_acc','rephrase_acc'):
            x[m]=f"{int(r[m+'_num'])}/{int(r[m+'_den'])} ({100*r[m+'_rate']:.2f}%)"
        rows.append(x)
    return table(pd.DataFrame(rows))

def build(work,out):
    work,out=Path(work),Path(out);require(not out.exists(),'create-once package');out.mkdir(parents=True)
    integ=read(work/'integrity-v1/integrity-receipt.json');require(integ['status']=='FULL_REHASH_CPU_RECOVERY_PASS','integrity release')
    require(integ['checkpoints']==200 and integ['cumulative_request_states']==110000,'complete denominator')
    code=Path(__file__).parent;repo=code.parents[3]
    source=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip(),modules=[member(p,relative_to=repo) for p in sorted(code.glob('*.py'))])
    inputs=[];private=[];frames={};inventory=[]
    for group,directory in GROUPS.items():
        dest=out/'tables'/group;dest.mkdir(parents=True)
        for p in sorted((work/directory).glob('*')):
            if not p.is_file():continue
            meta=member(p);inputs.append(meta)
            if p.name in PRIVATE:
                private.append(dict(**meta,scope='LOCAL_ONLY_HASHED_SCALAR_ROWS_NOT_BROADCAST',reason='Full request/prompt-state volume; complete raw-free aggregate tables published'));continue
            if p.suffix not in ('.csv','.json') and not p.name.endswith('.csv.gz'):continue
            shutil.copyfile(p,dest/p.name)
            if p.name.endswith(('.csv','.csv.gz')):
                f=pd.read_csv(p);frames[(group,p.name)]=f
                inventory.append(dict(path=str((dest/p.name).relative_to(out)),rows=len(f),bytes=p.stat().st_size,sha256=sha256_file(p)))
    def f(group,name):return frames[(group,name)]
    plotdir=out/'figures';plotdir.mkdir();pm=read(work/'plots-v2/plot-manifest.json')
    for item in pm['figures']:
        p=work/'plots-v2'/item['path'];require(sha256_file(p)==item['sha256'],'plot SHA');shutil.copyfile(p,plotdir/p.name)
    shutil.copyfile(work/'plots-v2/plot-manifest.json',plotdir/'plot-manifest.json')
    write_json_once(out/'analysis-source.json',source);write_json_once(out/'local-only-inputs.json',private)
    write_json_once(out/'table-inventory.json',inventory)
    docs=Path('/data/janghj/ODE-edit/local/state/orbode-sequential-b100x10-v1/authoritative-docs')
    expected=['03a6fc61258e7643fc281fab8ab0d3ca700bb80f09aaa3eb34591ce5827087be','8548d016fda343f8b8917bcbe43647b58ad99ab6f796ece2fe4ef661faa4c431','e134ac708c556482b912d7103e12a12347a78958943e7fa3d3da0a473908aa40']
    docmeta=[member(p) for p in sorted(docs.glob('*.md'))];require(sorted(x['sha256'] for x in docmeta)==sorted(expected),'authoritative docs SHA')
    runtime=Path('/data/janghj/ODE-edit/local/worktrees/orbode-cumulative-rerun-v1');runtime_dir=runtime/'project/run_scripts/ordered_response_barrier_ode'
    claim=member(runtime_dir/'cumulative_claim_measurement_map.md');src=[member(p,relative_to=runtime) for p in sorted(runtime_dir.rglob('*.py')) if '__pycache__' not in str(p)]
    # Executed source identity remains separate from analysis and integration lineage.
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()==SOURCE[0],'runtime HEAD')
    exclusions=read(Path(RAW).parent/'preparation-exclusion.json')
    preflight=read(RAW/'preflight.json')
    for m in preflight['source']['members']:
        require(sha256_file(runtime/m['path'])==m['sha256'],'executed source member '+m['path'])
    for k in ['seal_path']:
        require(sha256_file(Path(preflight['stream'][k]))==preflight['stream']['seal_sha256'],'sealed stream file')
    write_json_once(out/'execution-preflight.json',preflight)
    provenance=dict(instruction_id=INSTRUCTION,executed_source_head=SOURCE[0],executed_source_tree=SOURCE[1],analysis_source=source,authoritative_documents=docmeta,claim_map=claim,runtime_python_members=src,stream_root=STREAM_ROOT,order_root=ORDER_ROOT,technical_exclusion=exclusions,scheduler=read(work/'scheduler-once.json'),publication_policy='NEW_ANALYSIS_ONLY_MAIN_INTEGRATION_AUTHORIZED_20260907; GH_OWNED_HANDOFFS_NOT_REMERGED',prior_sequential_v1='HISTORICAL_IMMUTABLE_NOT_A_SOURCE_OF_ACTUAL_CUMULATIVE_RETENTION',counterfactual_scope='Different arm batch-entry W/z; end-to-end paired sample, NOT same-state causal comparison')
    write_json_once(out/'provenance.json',provenance)
    final=f('performance','final-20-arm.csv');cum=f('performance','cumulative-core.csv');dist=f('performance','category-distributions.csv');ages=f('performance','age-rates.csv');cmd=f('mechanism','current-command-realization.csv');hist=f('mechanism','historical-checkpoint-residual.csv');act=f('mechanism','layer-actual-action.csv');nodes=f('mechanism','node-mechanism.csv');batch=f('mechanism','batch-mechanism.csv');cost=f('mechanism','cost-by-batch.csv');cellcost=f('mechanism','cell-compute.csv')
    text=[f'# {TITLE}','',f'Instruction: `{INSTRUCTION}`. 새 누적 rerun의 실제 W_B 평가를 분석한 독립 canonical package다. 이전 sequential-v1의 online aggregate를 누적 retention으로 재명명하지 않았다. 이번 분석은 CPU-only; 새 model/evaluator/GPU/Slurm action=0. Scientific promotion=false.','',
          '## 1. 최종 성능 — 각 final W_B10에서 전체 1,000개 재평가','',rates(final),'',
          '공통 unique sample은 1,000개이며 20개 독립 arm-chain에서 반복 사용했다. 위 표는 20,000 arm-request endpoints이며 20,000개의 서로 다른 샘플이 아니다. Final W10 평가를 200개 누적 checkpoint 중 마지막 member로 한 번만 센다. RS/PS/NS는 canonical NLL pair, accuracy/strict는 secondary다.','',
          '![final](figures/final-full1000.png)','',
          '### 1.1 Official O 대비 final 차이 (percentage points)','',table(f('performance','paired-arm-deltas.csv').query("batch==10 and reference=='O'")),'',
          '### 1.2 셀별 산술 관찰','']
    for cell in CELLS:
        g=final[final.cell==cell].set_index('arm');text.append(f'**{cell}**: '+ '; '.join(f"{a} RS/PS/NS={100*g.loc[a,'RS_rate']:.2f}/{100*g.loc[a,'PS_rate']:.2f}/{100*g.loc[a,'NS_rate']:.2f}%" for a in ARMS)+'.')
        text.append('')
        for arm in ARMS[1:]:
            d=[100*(g.loc[arm,m+'_rate']-g.loc['O',m+'_rate']) for m in ('RS','PS','NS')]
            text.append(f'- {arm}−O: RS {d[0]:+.2f}, PS {d[1]:+.2f}, NS {d[2]:+.2f} pp. 동일 요청의 서로 다른 end-to-end W/z 결과이며 동일 상태에서 controller만 바꾼 효과가 아니다.')
        text.append('')
    text += ['## 2. 지표 읽는 법과 단위','',
    '|지표|정의·단위·방향|분모/주의|','|---|---|---|',
    '|RS|rewrite에서 length-normalized NLL(new)<NLL(true); 높을수록 많은 성공|B_k까지100k rewrite prompts|',
    '|PS|각 rephrase prompt에서 new<true; 높을수록 많은 성공|200k prompts; request 평균 NLL 비교가 아님|',
    '|NS|각 neighborhood prompt에서 true<new; 높을수록 더 많은 원사실 선호|1000k prompts; PP token 보존과 다름|',
    '|tie|strict 부등식이므로 동일 NLL은 실패|tolerance나 tie 개선 없이 원 reducer 그대로|',
    '|PS_strict|request의 두 rephrase가 모두 canonical pair 성공|100k requests; PS와 분리|',
    '|rewrite_acc/rephrase_acc|teacher-forced target-new의 모든 target token top-1 정답인 prompt 수|100k/200k prompts; token-mean accuracy가 아님|',
    '|token accuracy|correct_token_num/target_token_den|category-distributions.csv; prompt strict와 다른 단위|',
    '|PP|batch-entry 대비 neighborhood predicted-token preservation|CURRENT_B100에서만 기록; cumulative에는 NOT_RECORDED|',
    '|NLL(new/true)|target span token별 NLL 평균, nat/token; 낮을수록 높은 해당 target 확률|prompt와 request_cluster를 별도 집계; latter는 request 안 prompts 평균 후 분포|',
    '|margin|NLL(true)−NLL(new)|positive는 rewrite/rephrase new 선호; locality는 negative 선호. logit margin 아님|',
    '|mean/median/IQR/p90/max|산술평균/50%/25–75%/90%/최댓값, linear quantile|request-cluster와 prompt 분포 혼합 금지. raw n=0은 NOT_RECORDED|',
    '|R, q|해당 cohort의 원래 frozen z*−현재 terminal activation; q=||R||/||R_origin|||cohort마다 원래 entry scale; zero denominator typed missing|',
    '|A, Y, E|current command A, 실제 activation 변화 Y, E=A−Y|Official A=현재 residual/remaining layers; dynamic A는 h·u native field command. 모든 norm은 별도|',
    '|M|actual minus predicted response의 norm 및 원래 residual 정규화|dynamic JVP 예측 모델 오차; Official M 미기록|',
    '|rho, tau|rho=<Y,A>/||A||²; tau=||Y−rho A||/||A|||0<rho<1 under, rho>1 over, rho<0 opposite; current command에만 적용|',
    '|g,r,u|g=mean(<R,D>/||R0||²), r=mean(||D||²/||R0||²), D는 native-direction terminal JVP; u는 accepted multiplier|r 자체가 squared-response 통계; 다시 r²로 제곱하지 않는다. a*=max(g,0)/r, 실제 workload α=h·u|',
    '|potential V|평균 1/2 normalized residual squared|current batch frozen origin; finite defect=max(V_after−V_before,0) 별도|',
    '|과거 cohort damage|기존 z*/origin 대비 q와 자신의 immediate-post activation으로부터 거리|새 command 없음; current rho를 historical realization ratio로 적용 금지|',
    '|Layer-wise Update Magnitude|실제 stored ΔW_l의 Frobenius norm|batch-net, W0-net, nominal low-rank path를 분리. layer block은 batch-level, request100개 독립 업데이트 아님|',
    '|share / path / workload|magnitude share 및 squared share 구별; path Σ||αB||, work Σh·u|dynamic path는 nominal, 실제 FP32 rounding 차이 아님. Native C_reg action=UNBOUND|','',
    '## 3. 실행 provenance·분모·무결성','',table(f('integrity','chain-integrity.csv')),'',
    f"전체 raw {integ['raw_member_count']:,} members / {integ['raw_bytes']:,} bytes를 SHA/size 재해시했다. Raw member root `{integ['raw_member_root']}`. 200 checkpoint files의 5개 FP32 edited tensor를 CPU reload하여 W hash exact 확인; 200 frozen target/origin 파일도 재해시. W/M chain 각180 links, compute_z20,000 request calls, inner recompute0, terminal restore4/4. 중복/비유한/결과 imputation0. 관측된 finite scientific defect나 낮은 성능은 분모 제외하지 않았다.",'',
    f'Executed source `{SOURCE[0]}` / tree `{SOURCE[1]}`; analysis source `{source["head"]}` / tree `{source["tree"]}`. Run root `{RAW}`. Job37649_[0-3] 모두 COMPLETED0; 세부 child/timing/resource는 provenance.json. 1GPU/8CPU/59G, source/command/attention/backend/parameter inventory/EasyEdit/model bindings는 tables/integrity/source-runtime-provenance.json에 원 schema대로 포함한다.','',
    'LM/LA= Llama MEMIT/AlphaEdit, QM/QA=Qwen MEMIT/AlphaEdit. Llama parameter inventory291 tensors /8,030,261,248 elements, Qwen339 tensors /7,615,616,512 elements 모두 torch.float32. Autocast/TF32/quantization/BF16·FP16 cast0. Pinned stock EasyEdit HEAD14cea8245f06715684592ab55184939b99d70784/tree9c52aadbc0883da422badf0a730fff21aaa3a8a7, tracked-clean receipt. Eager attention은 terminal JVP/evaluator 공통이며 observation별 backend switch0. 원 runtime의 tokenizer 설정과 evaluator manual left-padding/attention/target-span code identity를 그대로 재사용한 결과이며 이번 CPU 분석에서 padding을 바꾸거나 forward parity를 새로 실행하지 않았다.','',
    'AlphaEdit: 각 arm cold 시작, 매 성공 B100 append100, 0→1000; arm 사이 이월0. MEMIT: static covariance 고정, history append0. Checkpoint는 five edited weights + pinned original model로 **평가 복구 가능**, cache는 hash만 있어 edit-resume snapshot이라고 주장하지 않는다. Source semantics를 재실행으로 검증한 것이 아니라 봉인 실행 receipt+파일/tensor 복구로 검증했다.','',
    '## 4. 모든 W_B에서 실제 all-seen 누적 성능 — 200 checkpoints','',
    'B_k의 동일 물리적 W에서 앞100k개 전체를 평가했다. 분모 증가와 W 변화가 함께 있으므로 인접 checkpoint rate 차이를 동일 cohort forgetting으로 단정하지 않는다. 동일 cohort의 변화는 §6 retention matrix로 분리한다.','']
    for cell in CELLS:
        text += [f'### 4.{CELLS.index(cell)+1} {cell}: 5 arms × 10 checkpoints','',rates(cum[cum.cell==cell],('batch',)),'']
        for arm in ARMS:
            g=cum[(cum.cell==cell)&(cum.arm==arm)].sort_values('batch');a,z=g.iloc[0],g.iloc[-1]
            text.append(f"- {arm}: B1→B10 RS/PS/NS 변화 "+'/'.join(f"{100*(z[m+'_rate']-a[m+'_rate']):+.2f}" for m in ('RS','PS','NS'))+f" pp; 누적 PS 최솟값 {100*g.PS_rate.min():.2f}% (B{int(g.loc[g.PS_rate.idxmin(),'batch'])}), NS 최솟값 {100*g.NS_rate.min():.2f}% (B{int(g.loc[g.NS_rate.idxmin(),'batch'])}).")
        text.append('')
    for m in ('RS','PS','NS'):text += [f'![cumulative {m}](figures/cumulative-{m}.png)','']
    text += ['### 4.5 Final W10 NLL / margin 분포','',
    '아래는 request_cluster 단위: request별 prompt 평균 후 1,000 request 분포. 모든 checkpoint·current·entry의 prompt-level/cluster 별 full mean/median/IQR/p90/max 및 token numerator는 category-distributions.csv 7,200행에 보존한다. Margin은 true−new이며 new/true target 행에서 같은 pair margin을 중복 metric으로 세지 않는다.','']
    for cell in CELLS:
        d=dist[(dist.cell==cell)&(dist.batch==10)&(dist.stage=='CHECKPOINT_W_ON_ALL_SEEN_REQUESTS')&(dist.unit=='request_cluster')]
        text += [f'#### {cell} (5 arms × 6 categories)','',table(d,['arm','category','target','prompt_den','strict_num','nll_n','nll_mean','nll_median','nll_p90','nll_max','margin_true_minus_new_mean','margin_true_minus_new_median','margin_true_minus_new_p90','margin_true_minus_new_max']),'']
    text += ['## 5. Official 및 claim별 paired 비교','',
    'paired-prompt-transitions.csv는 동일 request/category/prompt index와 new/true input hash를 exact match한다. Loss=O 성공→arm 실패, recovery=O 실패→arm 성공이다. 각 arm은 이미 누적된 W/z가 달라서 controlled same-state causal 효과가 아니다.','',table(f('performance','paired-prompt-transitions.csv').query('batch==10'),['cell','arm','category','prompt_n','O_success','arm_success','loss','recovery','delta_rate','nll_new_delta_mean','nll_new_delta_median','nll_new_delta_p90','nll_new_delta_max']),'',
    'Quota/response/JAC 대비도 같은 sample의 end-to-end 산술차이다. NQFIX−QCL, ORBFH−NQFIX, ORBFH−JAC 모든 checkpoint 차이는 paired-arm-deltas.csv에 완전 수록. NQFIX가 큰 업데이트/낮은 성능을 보여도 제외하거나 다른 arm으로 대체하지 않는다.','',
    '## 6. Retention·forgetting·recovery·충돌','',
    'rewrite-retention matrix 1,100행은 각100-request cohort를 W_B에서 추적한다. 최초 at-write 실패와 이후 성공의 소실을 분리하고, 실패 후 recovery도 보존한다.','',table(f('performance','atwrite-final-partitions.csv').drop(columns=['overwrite'])),'',
    '![retention](figures/rewrite-retention-20arms.png)','',
    '### 6.1 정확한 neighborhood prompt loss/recovery','',table(f('comparisons','own-write-final-prompt-transitions.csv').query("category=='locality'"),['cell','arm','prompts','atwrite_success','final_success','loss','recovery','conditional_loss_den','nll_new_change_mean','nll_true_change_mean','margin_true_minus_new_change_mean']),'',
    '전체 NS 점수가 비슷해도 서로 다른 prompt의 loss와 recovery가 상쇄될 수 있다. 위 분해는 원래 prompt input/order hash가 일치하는 범위만 계산했다. Rewrite/rephrase 동형 40행도 동일 CSV에 존재한다.','',
    '### 6.2 Exact metadata 충돌','',table(f('comparisons','overwrite-conflicts.csv')),'',
    '한 쌍의 exact subject+relation / 다른 target metadata 충돌을 발견했다. Raw prompt/subject/target 문자열은 publish하지 않고 case/request/metadata hash만 남긴다. 인과적 overwrite 원인으로 확정하지 않는다. 60-row atwrite-final-conflict-strata.csv에 전체/metadata 노출/비노출을 분리하며 primary exclusion0. 초기 scalar extraction의 metadata-unavailable 표시는 이 후속 exact dataset audit로 보완했다.','',
    '### 6.3 Early/middle/recent','',
    '각 checkpoint seen prefix의 first20%/middle60%/last20%로 사전 고정. B1은20/60/20, B10은200/600/200 requests다. 상대 위치이므로 checkpoint 사이 age label이 고정 cohort를 뜻하지 않는다. 600-row 모든 checkpoint age rates와 7,200-row age category 분포를 제공한다.','',rates(ages[ages.batch==10],('age','request_n')),'',
    '![age](figures/age-strata-NS.png)','',
    '### 6.4 Online-at-write·current-B100와 혼동하지 않기','',
    'online-own-write.csv는 각 request 자신의 편집 직후 값을 pooling한200행이며 하나의 W에서 평가한 성능이 아니다. current-B100.csv는 매 batch 새100개만이다. cumulative-vs-online-gaps.csv는 산술차이를 보이되 동형 prompt identity가 확인된 것이지 상태가 같은 것은 아니다.','',table(f('comparisons','cumulative-vs-online-gaps.csv').query('batch==10')),'',
    '## 7. Proposal claim → 측정 → 분모 → 한계','',
    '|claim/비교|실제 기록 및 table|분모|허용 범위·미측정|','|---|---|---|---|',
    '|Quota vs native strength: QCL/NQFIX|g/r/u, A/Y/E, cumulative rate, layer action|2arms×4cells×10B×20visits|W/z가 다른 end-to-end 비교; quota 이외 누적 상태 효과도 포함|',
    '|Realization-aware response: NQFIX/ORBFH|predicted/actual reduction, defect, JVP M, u/workload|20visits/B, request100/visit|finite-step monotonicity는 보장 아님; defect 그대로 공개|',
    '|Frozen-sweep vs fresh-layer: ORBFH/JAC|build_state, key/writer revisit, JVPcount, cost|4sweeps×5layers/B|같은 W의 isolated Jacobian ablation 아님|',
    '|과거 편집 보존|old frozen z/origin residual, own immediate activation distance|110000 checkpoint request states; layer probe1870000 rows|과거 target은 새command 없음. rho_command로 잘못 해석 금지|',
    '|allocation/realization gap|current-command-realization A/Y/E/M, rho/tau|3400 visit summaries×100requests|Official A와 dynamic A 의미 차이 명시|',
    '|층별 workload/감속·집중|Σh·u, nominal path, actual batch-net/W0-net|1000 layer×batch rows|u 감소와 다른 layer 실제 write 증가를 별개로 측정|',
    '|first hit / horizon semantics|batch status, ORBHit prefix inventory|160dynamicB; ORBHit40 후보|prefix는 materialized diagnostic, 별도 chain/primary denominator0|',
    '|Native quadratic action|native_Creg_action UNBOUND|0 bound native metric|Frobenius를 native metric으로 부르지 않음|',
    '|정확한 causal locality/speedup·ODE necessity|동일상태 matched-no-revisit/static 미실행|NOT_RECORDED|이 campaign에서 확인 불가; 추가실험 권고/자동실행0|','',
    '## 8. 현재 command realization과 과거 fixed-target damage','',
    '### 8.1 Current A/Y/E/M·rho/tau (visit 내100 request → layer summary)','',
    '다음 평균은 request별 원 residual scale로 정규화한 visit mean들을 동등 평균한다. Median/p90/max가 필요한 경우 current-command-realization.csv의3,400행에서 각visit n/분포를 그대로 읽는다. 여기의 visit-summary 평균을 pooled request quantile이라고 부르지 않는다.','']
    layer_cmd=cmd.groupby(['cell','arm','layer'],sort=False).agg(A=('A_over_origin_mean','mean'),Y=('Y_over_origin_mean','mean'),E=('E_over_origin_mean','mean'),M=('M_over_origin_mean','mean'),rho=('rho_command_mean','mean'),tau=('tau_command_mean','mean'),under=('under_realization','sum'),over=('overshoot','sum'),opposite=('opposite_direction','sum'),zero=('zero_command','sum'),request_visits=('request_rows','sum')).reset_index()
    text += [table(layer_cmd),'',
    '### 8.2 B10의 원래 cohort residual: current와 historical 분리','',table(hist[hist.batch==10],['cell','arm','cohort','relationship','request_rows','q_after_mean','q_after_median','q_after_p90','q_after_max','q_gt_one','q_worsened','distance_from_own_immediate_activation_mean']),'',
    'historical q_worsened는 이번 batch entry 대비 post의 증가 count다. own-immediate 거리는 처음 편집 완료 activation에서 얼마나 달라졌는지이며 성공의 원인 또는 failure 그 자체가 아니다. 모든 B/cohort 및 모든 node/layer는 각각1,100/18,700 group CSV에 있다.','',
    '![historical](figures/historical-fixed-z-residual.png)','',
    '## 9. Layer action·workload·response·drift','',
    '### 9.1 최종 B10 actual batch-net / 누적 W0-net 및 path','',table(batch[batch.batch==10],[c for c in ['cell','arm','batch_net_actual_frobenius','W0_net_actual_frobenius','physical_euler_path_frobenius_length','terminal_net_frobenius','resolution_stable_path_frobenius','u_zero','u_interior','u_one','final_V','defect_count','defect_sum','status'] if c in batch]),'',
    'Actual batch/W0 net은 CPU-reloaded stored FP32 weights에 결속됐다. Dynamic visit path는 nominal αB low-rank norm으로 FP32 materialization rounding delta와 다르다. Path와 net이 다르다고 native regularized-action 차이로 치환하지 않는다.','',
    '### 9.2 모든 layer의 B1/B10 실제 분포 및 workload','',table(act[act.batch.isin([1,10])],['cell','arm','batch','layer','batch_net_update_magnitude','batch_magnitude_share','batch_squared_share','net_from_original_W0_magnitude','path_magnitude_sum_batch_increment','dynamic_workload_sum_h_u_batch_increment','dynamic_signed_potential_progress_sum_batch_increment']),'',
    '![weights](figures/layer-wise-update-magnitude.png)','',
    '막대는 arm별10 sequential batches의 실제 layer update norm 평균이다. Equal-share/ideal line은 없다. L8 집중 여부는 squared share와 magnitude share를 구분하고 Llama/Qwen raw norm을 pooling하지 않는다.','',
    '### 9.3 Node별 response와 유한-step defect','',table(nodes.groupby(['cell','arm'],sort=False).agg(visits=('g','size'),g_mean=('g','mean'),g_min=('g','min'),r_mean=('r','mean'),u_mean=('coefficient_u','mean'),predicted=('predicted_reduction','sum'),actual=('actual_reduction','sum'),defect_sum=('discretization_defect','sum'),defect_max=('discretization_defect','max'),entry_violation_max=('entry_sublevel_violation','max')).reset_index()),'',
    '3,200 dynamic visit rows의 g/r/u, state/build identity, actual/predicted reduction, command/response/model-error, per-request q statistics를 node-mechanism.csv에 완전 수록한다. Finite defect>0이나 HORIZON_SEMANTIC_MISS를 기술 실패로 제외하지 않았다.','',
    '![defect](figures/finite-step-defect.png)','',
    '### 9.4 Key/writer revisit drift','',table(f('mechanism','key-writer-revisit.csv').groupby(['cell','arm'],sort=False)[['key_revisit_relative_difference','writer_revisit_relative_difference']].agg(['count','mean','median','max']).reset_index().pipe(lambda d:d.set_axis(['_'.join(x).strip('_') for x in d.columns],axis=1))),'',
    '같은 layer의 이전 visit 대비 drift이며 이전 batch-entry를 고정한 counterfactual 비교가 아니다. Build state/version과 factor rank, solve residual, key hash를 같이 제공한다.','',
    '## 10. ORBHit 및 horizon 관찰','',table(f('comparisons','derived-prefix-inventory.csv'),['cell','batch','arm','primary_arm_denominator','status','state_version','RS_num','RS_den','PS_num','PS_den','NS_num','NS_den']),'',
    'ORBHit는 ORBFH 실제 경로에서 발견·materialize한 prefix diagnostic이다. 해당 prefix를 계속 sequential로 실행한 여섯 번째 arm이 아니다. Prefix 결측/미발견은 NOT_RECORDED로 보존한다.','',
    '## 11. 성능–메커니즘 association과 outlier','',
    'Spearman300행은20arms×5기록량×RS/PS/NS, arm당10개 반복 checkpoint다. Shared cumulative history와 증가하는 seen denominator가 있어 독립표본 상관이나 인과효과가 아니다. 아래는 original-target q와 canonical 성능의 계수 전부다.','',table(f('comparisons','mechanism-cumulative-associations.csv').query("mechanism=='q_after_mean'")),'',
    '![association](figures/mechanism-cumulative-association.png)','',
    '메커니즘 outlier ledger4,400행은 각checkpoint/cohort에서 q_after 및 own-immediate distance 상위2개를 request SHA로 식별한다. 전체 source identity/nonfinite0 검증에 포함되었고 값이 크다는 이유로 제외하지 않았다. 대상 raw prompt는 publish하지 않는다. 아래 각arm 최댓값의 원래 frozen-origin residual norm도 CPU에서 재계산했다. 작은 양의 분모로 큰 normalized tail이 생기는 경우 raw post residual과 구분하며 임의 epsilon/clip/exclusion을 적용하지 않는다.','',table(f('tail','normalized-tail-origin-context.csv'),['cell','arm','batch','cohort','request_sha256','value','original_residual_norm','implied_post_residual_norm','original_target_norm']),'',
    '## 12. Compute ledger — 겹치는 bracket을 합산하지 않기','',table(cellcost,['cell','model_load_seconds','wall_seconds','model_forward_invocation_count','memory_peak_allocated_bytes','fixed_z_compute_count','fixed_z_recompute_count','inter_batch_weight_link_count','inter_batch_method_state_link_count']),'',
    '각 source cell당 model-load1, fixed-z5,000=5arms×10B×100request. JVP 내부 model forward는 model_forward_invocation_count와 겹치므로 더하지 않는다. GPU elapsed는 scheduler allocation 시간, 아래 endpoint call/observer/evaluation bracket과 다른 단위다. Hardware/software/backend provenance는 runtime receipt에 기록된 것만 사용한다.','',
    table(cost.groupby(['cell','arm'],sort=False)[['endpoint_wall_seconds','cumulative_evaluation_checkpoint_wall_seconds','observer_probe_wall_seconds_increment','endpoint_model_forward_calls','jvp_jvp_call_count','adapter_key_capture_count','adapter_solve_count','adapter_inner_history_append_count']].sum(min_count=1).reset_index()),'',
    'Endpoint wall에는 runtime observer와 endpoint evaluator가 포함된다. Cumulative checkpoint wall에는 all-seen evaluation/geometry/storage가 포함된다. Observer-probe incremental wall은 겹치는 하위 bracket이므로 합산하지 않는다. Compute_z optimization, key/solve/controller/materialization/storage 개별 wall bracket은 NOT_RECORDED_SEPARATE_BRACKET. 기록된 count와 전후 bracket을 명시하는 것이며 controlled speedup 주장0.','',
    '## 13. 기술 exclusions·미기록·분석 범위','',
    'Preparation source867bc755…의 precreated result/log namespace 충돌은 PRE_GPU PURE_TECHNICAL_PREPARATION_EXCLUDED, model/GPU/Slurm/scientific denominator0으로 보존됐다. execution-r1은 봉인 preparation만 있고 실행 job 없음; 실제 completed campaign은 execution-r2/job37649. 원문 exclusion은 provenance.json에 bytes 내용 그대로 기록했다. 이 campaign의20 primary chain runtime technical failure0.','',
    '미기록/구분: full-cache edit-resume state; native C_reg action; original W0로 전체1,000 neighborhood PP pre/post paired panel; 별도 compute_z/storage wall; matched static/no-revisit causal control. Null/zero command는 rho undefined를0으로 채우지 않는다. CP geometry의 residual/action은 action argument0인 경우 unavailable이며 layer-visit action-normalized 결과만 해석한다. 최종 성능은 heldout selection 또는 tolerance 변경에 사용되지 않았다.','',
    '## 14. FACT / INFERENCE 경계 / DECISION','',
    'FACT: 네 cell의20 chains가200 B100 commits와200 actual all-seen checkpoints를 완료했다. 모든 primary arm과 낮은 성능·finite defect를 포함했다. 표의RS/PS/NS는NLL pair 기준이며 전token 정확도와 다르다.','',
    'INFERENCE 경계: 같은 요청의 end-to-end 비교이고 batch-entry W/z가 달라진다. 과거 fixed target residual, current command realization, actual weight net, nominal path workload를 각각 보고하며 어느 하나를 다른 것으로 대체하지 않는다. Architecture/method별 상반된 숫자는 그대로 유지한다. Universal bottleneck/causal locality benefit/ODE necessity/controlled speedup은 이 자료에서 확정하지 않는다.','',
    'DECISION: 분석·공유 완료 범위. Scientific promotion=false; 신규 실험/재평가/자동 follow-up0. GH의 이미 통합한 runtime handoff는 중복merge하지 않고 이번 새 analysis/report만 최신 main에 통합한다.','',
    '## 15. 완전 table / figure inventory와 재현','',
    'CSV에 NOT_RECORDED는 빈 numeric cell 또는 typed string으로 보존한다. Python/pandas missing을0으로 impute하지 않는다. 아래는 공개 aggregate 전부의 정확한 row count와SHA다. Raw request/prompt-state 대용량 scalar records는 local-only-inputs.json의 경로/SHA로 남기고 Git에 올리지 않았다.','',table(pd.DataFrame(inventory)),'',
    '### 재현 명령','',
    '```bash\n/data/janghj/EasyEdit/.venv/bin/python -m unittest -q project.run_scripts.ordered_response_barrier_ode.analysis_cumulative_report.test_focused\n/data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.ordered_response_barrier_ode.analysis_cumulative_report.plots --performance <PACKAGE>/tables/performance --mechanism <PACKAGE>/tables/mechanism --comparisons <PACKAGE>/tables/comparisons --output <NEW_FIGURE_DIR>\n```','',
    'Plot code/source/input/output SHA, seed20260907, Agg backend, pinned style/order/DPI/axis ranges는 figures/plot-manifest.json에 기록했다. 각 figure를 동일 command로2회 그려 PNG bytes exact를 검사했다. Codex visualization/imagegen/manual-image-edit0. Tests/hash/access/source seal은 tests-and-audit.json에 둔다.','']
    for item in pm['figures']:
        text += [f"- [{item['path']}](figures/{item['path']}): {item['caption']} SHA `{item['sha256']}`; {item['bytes']} bytes."]
    with (out/REPORT).open('x') as h:h.write('\n'.join(text)+'\n')
    # Test/access receipt is generated by the explicit package verifier before sealing.
    write_json_once(out/'build-inputs.json',dict(inputs=inputs,private_inputs=private,analysis_source=source,figures=len(pm['figures']),tables=len(inventory),status='BUILT_AWAITING_FINAL_VERIFY'))
    print(json.dumps(dict(report=str(out/REPORT),sha256=sha256_file(out/REPORT),tables=len(inventory),figures=len(pm['figures']))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.work,a.output)
