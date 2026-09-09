"""Raw-free final synthesis of three independently sealed stage packages.

No model evaluation, interpolation, endpoint selection by heldout outcome, or
rewriting of parent packages. The caller supplies a factual interpretation memo
only after reading the completed B/C observations.
"""
import argparse
import csv
import io
import json
import platform
import stat
import subprocess
from pathlib import Path
from .analysis import write_csv
from .discussion import num
from .import_assets import sha
from .package_stage import verify,call
from .records import save,digest
from .report import table,rate

def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))

def endpoint_rows(parent,stage,completion):
    rows=read(parent/'paired-summary.csv')
    chosen={'N-full',*[f"{support}-alpha-{completion['selections'][support]['alpha']}/eval-032"
                        for support in ['B','C']]} if stage=='A' else None
    return [dict(stage=stage,**row) for row in rows if row['resolution']=='full'
            and row.get('reference','') in ['', 'N'] and (chosen is None or row['endpoint'] in chosen)]

def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,'figure.dpi':120,'savefig.dpi':120}):
        fig,axes=plt.subplots(3,3,figsize=(14,11))
        for i,stage in enumerate(['A','B','C']):
            for j,panel in enumerate(['Current100','Fixed100','Past100']):
                ax=axes[i,j]
                for entry,marker in [('Early','o'),('Middle','s'),('Late','^')]:
                    points=[r for r in rows if r['stage']==stage and r['entry']==entry and r['panel']==panel and r['metric']=='NS']
                    lookup={(r['entry'],r['endpoint'],r['metric']):r for r in rows if r['stage']==stage and r['panel']=='Current100'}
                    for index,r in enumerate(points):
                        x=lookup[(entry,r['endpoint'],'RS')]
                        ax.scatter(float(x['new_nll_mean']),100*float(r['rate']),marker=marker,s=26,
                                   color={'Early':'#2878b5','Middle':'#c95c25','Late':'#3c9166'}[entry],
                                   label=entry if index==0 else None,alpha=.8)
                ax.set_title(f'{stage}: {panel}')
                ax.set_xlabel('Current rewrite target-new NLL')
                ax.set_ylabel('Full NS (%)')
                ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('A / B / C measured endpoints — no matched-strength interpolation')
        fig.tight_layout();buffer=io.BytesIO()
        fig.savefig(buffer,format='png',metadata={'Software':'ODE-edit cumulative-risk final synthesis'})
        plt.close(fig);return buffer.getvalue()

def amplitude_figure(rows):
    """All three predeclared amplitudes at the same curve denominator."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['resolution']=='curve' and r['reference']=='N']
    lookup={(r['entry'],r['endpoint'],r['panel'],r['metric']):r for r in selected}
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':120,'savefig.dpi':120}):
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for col,entry in enumerate(['Early','Middle','Late']):
            for direction in ['GFminus','GFplus','LFminus','Random1','Random2','OPminus','COVminus']:
                amplitudes=[.03,.1,.3]
                for row,(panel,metric,field,scale) in enumerate([
                    ('Current100','RS','new_nll_mean',1.),('Past100','NS','rate',100.)]):
                    base=lookup[(entry,'N_REUSED',panel,metric)]
                    values=[scale*(float(lookup[(entry,f'{direction}-amplitude-{a}/eval',panel,metric)][field])-float(base[field])) for a in amplitudes]
                    axes[row,col].plot(amplitudes,values,marker='o',label=direction,linewidth=1)
                    axes[row,col].axhline(0,color='black',linewidth=.4)
                    axes[row,col].set_xticks(amplitudes);axes[row,col].set_xlabel('Extra action / native Frobenius norm')
                    axes[row,col].set_ylabel('Current new NLL delta vs N' if row==0 else 'Past curve NS delta vs N (pp; n=200)')
                    axes[row,col].set_title(entry);axes[row,col].grid(alpha=.2)
            axes[0,col].legend(fontsize=6,ncol=2)
        fig.suptitle('B direction × amplitude — markers measured; lines only visual guides')
        fig.tight_layout();buffer=io.BytesIO()
        fig.savefig(buffer,format='png',metadata={'Software':'ODE-edit cumulative-risk final synthesis'})
        plt.close(fig);return buffer.getvalue()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--interpretation',type=Path,required=True)
    args=parser.parse_args();parents={};rows=[];cost=[];contrasts=[]
    for stage in ['A','B','C']:
        directory=args.parent/stage;identity=verify(directory)
        completion=json.loads((directory/'completion.json').read_text())
        if completion['status']!='COMPLETE' or completion['remaining_mandatory']!=0:
            raise ValueError('ALL_THREE_STAGES_MUST_BE_COMPLETE')
        parents[stage]=dict(path=str(directory),**identity,completion_sha=sha(directory/'completion.json'))
        rows+=endpoint_rows(directory,stage,completion)
        cost += [dict(stage=stage,**r) for r in read(directory/'allocation/job-gpu-hour-ledger.csv')]
        if stage!='A':contrasts+=read(directory/'contrasts/paired-method-contrasts.csv')
    args.output.mkdir(parents=True,exist_ok=False)
    # A was already published before this additional paired analysis; never edit it.
    call('contrast_analysis','--stage','A','--request-table',args.parent/'A/request-metrics.csv.gz',
         '--completion',args.parent/'A/completion.json','--output',args.output/'A-paired-contrasts')
    contrasts+=read(args.output/'A-paired-contrasts/paired-method-contrasts.csv')
    write_csv(args.output/'full-endpoint-main-table.csv',rows)
    write_csv(args.output/'paired-method-contrasts.csv',contrasts)
    if len({r['JobID'] for r in cost})!=len(cost):raise ValueError('CROSS_STAGE_ALLOCATION_DOUBLE_COUNT')
    write_csv(args.output/'allocation-cost.csv',cost)
    payload=figure(rows)
    if payload!=figure(rows):raise ValueError('PNG_BYTE_REPRODUCTION_FAILED')
    plot=args.output/'abc-measured-endpoints.png'
    with plot.open('xb') as stream:stream.write(payload)
    amplitude_rows=read(args.parent/'B/paired-summary.csv')
    amplitude_payload=amplitude_figure(amplitude_rows)
    if amplitude_payload!=amplitude_figure(amplitude_rows):raise ValueError('AMPLITUDE_PNG_REPRODUCTION_FAILED')
    amplitude_plot=args.output/'B-direction-amplitude.png'
    with amplitude_plot.open('xb') as stream:stream.write(amplitude_payload)
    import matplotlib
    import numpy
    save(args.output/'plot-reproduction.json',dict(byte_stable=True,actual_renders_per_png=2,
         input_sha=sha(args.output/'full-endpoint-main-table.csv'),output_sha=sha(plot),
         amplitude_input_sha=sha(args.parent/'B/paired-summary.csv'),amplitude_output_sha=sha(amplitude_plot),
         python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=numpy.__version__,
         command='python -m project.run_scripts.single_layer_cumulative_risk.final_synthesis '
                 f'--parent {args.parent} --output <new-create-once-path> --interpretation {args.interpretation}',
         plot_code_sha=sha(Path(__file__)),model_action=0))
    save(args.output/'parent-packages.json',parents)
    interpretation=args.interpretation.read_text()
    save(args.output/'interpretation-input.json',dict(path=str(args.interpretation),sha256=sha(args.interpretation)))
    lines=['# 단일 layer 누적위험 A→B→C 최종 진단 보고서','',
      'A/B/C 모두 완료. 각 stage의 immutable 상세 보고서·원본 identity·요구사항 evidence를 아래 parent package로 결속한다. '
      'Scientific promotion=false. 낮은 성능, risk 증가, strength 불일치를 제외하거나 성공 gate로 바꾸지 않았다.','',
      '## 설계와 분모','',
      'A: native-assisted Direct-B/Direct-C support 비교(13 writers,320 SGD steps,12 추가 native scales). '
      'B: 공통 WN에서7방향×3amplitudes×3entry=63 trials. C: Middle WN에서5방식×8steps=40 steps. '
      'A의 Direct-B/Direct-C 이름과 상위 B/C 단계는 다르다.','',
      'Full 각 panel은 RS100/PS200/NS1000, Current·Fixed·Past 전체3900쌍이다. '
      'RS/PS는 new NLL<true NLL, NS는 true NLL<new NLL이며 tie는 실패다. '
      'Curve1100쌍 및 full에서 재사용한 동일 관측행은 별도 독립 분모로 합산하지 않는다. '
      'Conditional loss/recovery, all-prompt success, TF exact 및 literal generation을 서로 구분한다.','',
      'W0=원본, We=historical entry, WN=native endpoint. NS inherited margin은 We−W0, '
      'additional은post−We이며 N 대비 변화는 별도 reference delta다. '
      '모든 margin은 true NLL−new NLL이다. 따라서 rewrite/rephrase에서는 큰 margin이 새 target 선호이고, '
      'neighbor에서는 작은 margin이 원래 정답 선호다. NLL은 해당 정답 token 평균으로 낮을수록 그 문자열에 높은 확률을 준다. '
      'Joint sequence probability나 literal generation accuracy와 동일한 지표가 아니다.','',
      '## Current / Fixed / Past — 사전 지정 full endpoints','']
    for stage in ['A','B','C']:
        subset=[r for r in rows if r['stage']==stage]
        lookup={(r['entry'],r['endpoint'],r['panel'],r['metric']):r for r in subset};data=[]
        for entry,endpoint,panel in sorted({(r['entry'],r['endpoint'],r['panel']) for r in subset}):
            metrics=[lookup[(entry,endpoint,panel,m)] for m in ['RS','PS','NS']]
            data.append([entry,endpoint,panel,*map(rate,metrics),num(metrics[0]['new_nll_mean']),
                         num(metrics[1]['new_nll_mean']),num(metrics[1]['new_nll_p90'])])
        lines += [f'### {stage}','',table(['entry','endpoint','panel','RS n/d (%)','PS n/d (%)','NS n/d (%)',
                  'rewrite new NLL','rephrase new NLL','rephrase new p90'],data),'']
    lines += ['## 관측에 근거한 해석과 미분리 요인','',interpretation,'',
      '## 계산량과 출처','',table(['stage','job','entry','allocated GPU seconds','GPU hours'],
            [[r['stage'],r['JobID'],r['entry'],r['allocated_gpu_seconds'],r['allocated_gpu_hours']] for r in cost]),'',
      f"합계 {sum(float(r['allocated_gpu_seconds']) for r in cost):.0f} GPU-seconds / "
      f"{sum(float(r['allocated_gpu_hours']) for r in cost):.6f} GPU-hours. 모델 준비·평가·생성·process residency를 포함한다. "
      '각 stage auxiliary/compute-summary.csv의 process total과 child ledger를 중복 합산하지 않는다. FLOPs는 NOT_RECORDED이며 시간에서 추정하지 않았다.', '',
      '![ABC measured endpoints](abc-measured-endpoints.png)','',
      '![B direction amplitude](B-direction-amplitude.png)','',
      '두 번째 그림은 curve에서 같은 Past NS200쌍을 사용하며, full NS1000쌍과 섞지 않는다. '
      '연결선은 눈금을 읽기 위한 안내일 뿐 중간 amplitude의 측정·보간값이 아니다.','',
      '그림은 관측점만 나타내며 색/marker는 entry이다. 선형보간, NS 기반 선택, 동일 strength 또는 age-only 인과효과를 주장하지 않는다. '
      '전체 방향·amplitude 표와 내부 step 궤적은 각 stage CSV/PNG를 함께 본다.','',
      '## 봉인된 상세 보고서','']
    for stage,item in parents.items():
        path=Path(item['path'])/f'{stage}-factual-report-ko.md'
        lines += [f"- [{stage} 상세 보고서]({path}): SHA256 `{item['report_sha']}`; members root `{item['members_root']}`."]
    report=args.output/'final-diagnostic-report-ko.md'
    with report.open('x') as stream:stream.write('\n'.join(lines)+'\n')
    members=[dict(path=str(p.relative_to(args.output)),sha256=sha(p),bytes=p.stat().st_size,
                  mode=oct(stat.S_IMODE(p.stat().st_mode))) for p in sorted(args.output.rglob('*')) if p.is_file()]
    manifest=dict(stage='final',members=members,members_root=digest(members),parents=parents,scientific_promotion=False)
    save(args.output/'package-manifest.json',manifest)
    receipt=dict(stage='final',report_sha=sha(report),manifest_sha=sha(args.output/'package-manifest.json'),
                 members_root=manifest['members_root'],scientific_promotion=False,
                 source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip())
    save(args.output/'rooted-receipt.json',dict(**receipt,identity=digest(receipt)))
    print(json.dumps(verify(args.output)))

if __name__=='__main__':main()
