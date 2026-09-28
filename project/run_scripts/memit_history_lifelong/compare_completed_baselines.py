"""CPU-only four-method comparison of published W100 summaries; no GPU/raw loads."""
import argparse,csv,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args()
 def csvrows(name):return list(csv.DictReader((a.baseline/name).open()))
 s=json.loads((a.report/'summary.json').read_text());methods=[('MEMIT','BASE_MEMIT','42658','MEMIT'),('MEMIT history','MEMIT_HISTORY_NATIVE','54007',None),('AlphaEdit','BASE_ALPHAEDIT','42657','AlphaEdit'),('AlphaEdit-BLUE','AlphaEdit_ORIGINAL','39283_1','AlphaEdit')]
 compat=csvrows('source-config-compatibility.csv')+csvrows('new-source-config-compatibility.csv');rows=[];bindings=[];curves=[]
 for label,arm,job,family in methods:
  if family:
   c=next(c for c in compat if c['arm']==arm);hp=json.loads(c['hparams']);assert c['job']==job
   assert c['sample_root']=='5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729'
   assert c['model_revision']=='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2' and c['seed']=='20260907'
   source=csvrows(family+'-cumulative-metrics.csv');source=[r for r in source if r['arm']==arm]
   curves.extend(dict(method=label,batch=int(r['batch']),metric=r['metric'],rate=float(r['rate'])) for r in source)
   bind=dict(method=label,source_arm=arm,job=job,layers=hp['layers'],blue=hp['blue'],L2=hp.get('L2'),nullspace_threshold=hp.get('nullspace_threshold'),gpu=c['gpu'],model_revision=c['model_revision'],sample_root=c['sample_root'],seed=c['seed'])
  else:
   bind=dict(method=label,source_arm=arm,job=job,layers=[4,5,6,7,8],blue=False,L2=None,nullspace_threshold=None,gpu=s['gpu'],model_revision='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2',sample_root='5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729',seed='20260907')
   curves.extend(dict(method=label,batch=int(r['batch']),metric=r['metric'],rate=float(r['rate'])) for r in csv.DictReader((a.report/'all-seen-metrics.csv').open()))
  bindings.append(bind)
  for t,n in [('RS',10000),('PS',20000),('NS',100000)]:
   d='true' if t=='NS' else 'new';h=s['final'][t]
   if family:
    r=next(r for r in source if r['batch']=='100' and r['metric']==t);assert int(r['denominator'])==n
    v=dict(success=int(r['numerator']),denominator=n,rate=float(r['rate']),tf_correct=int(r[d+'_token_correct']),tf_count=int(r[d+'_token_den']),tf_strict_num=int(r[d+'_strict_num']),tf_strict_den=int(r[d+'_strict_den']),new_nll=float(r['new_nll_prompt_mean']),true_nll=float(r['true_nll_prompt_mean']),tf_prompt_macro='NOT_RECORDED_IN_REUSED_TABLE')
   else:
    v=dict(success=h['numerator'],denominator=n,rate=h['rate'],tf_correct=h['tf_token_correct'],tf_count=h['tf_token_count'],tf_strict_num=h['tf_strict_numerator'],tf_strict_den=n,new_nll=h['new_nll'],true_nll=h['true_nll'],tf_prompt_macro=h['tf_prompt_macro'])
   assert v['tf_count']==h['tf_token_count'] and v['tf_strict_den']==n
   rows.append(dict(method=label,job=job,metric=t,desired=d,**v,tf_micro=v['tf_correct']/v['tf_count'],tf_strict=v['tf_strict_num']/n,desired_nll=v[d+'_nll'],history_minus_method_pp=100*(h['rate']-v['rate'])))
 with (a.report/'four-method-comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 inputs=['MEMIT-cumulative-metrics.csv','AlphaEdit-cumulative-metrics.csv','source-config-compatibility.csv','new-source-config-compatibility.csv']
 manifest=dict(scope='USER_RECALL_ADD_ALPHAEDIT_AND_ALPHAEDIT_BLUE',endpoint='W100_SAME_FIXED10K',methods=bindings,inputs=[dict(path=str(a.baseline/n),sha256=hashlib.sha256((a.baseline/n).read_bytes()).hexdigest(),bytes=(a.baseline/n).stat().st_size) for n in inputs],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),paired_cross_run='NOT_AVAILABLE_NO_LOCAL_RAW',new_GPU_jobs=0)
 (a.report/'four-method-comparison-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 def table(headers,body):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in body])
 text=['# MEMIT history / MEMIT / AlphaEdit / AlphaEdit-BLUE 완료 비교','','사용자 추가지시: “alphaedit, alphaedit-blue도 비교에 넣어봐”. 기존 완료자료의 W100/동일fixed10k만 재사용. 새 실험·GPU평가0.','',table(['방법','job','층','blue','L2'],[[b['method'],b['job'],str(b['layers']),str(b['blue']),str(b['L2']) if b['L2'] is not None else '해당없음'] for b in bindings]),'',
 '과거 arm ID `AlphaEdit_ORIGINAL`은 실제 blue=true L4+L8이며 표시명 **AlphaEdit-BLUE**로 결속한다. `BASE_ALPHAEDIT`는 blue=false L4–L8/L2=10이다. 두 AlphaEdit 모두 history를 사용하지만 MEMIT history의 15000C0+H writer와 식이 다르다. AlphaEdit projector/threshold0.02 및 L2, BLUE의 층/target 정책을 동일 조건으로 취급하지 않는다.','',
 table(['방법','RS % (n=10000)','PS % (n=20000)','NS % (n=100000)'],[[m[0],*[f"{next(r['rate'] for r in rows if r['method']==m[0] and r['metric']==t)*100:.3f}" for t in ['RS','PS','NS']]] for m in methods]),'',
 'MEMIT history에서 각 비교값을 뺀 산술차(%p):', '',table(['비교 대상','RS','PS','NS'],[[m[0],*[f"{next(r['history_minus_method_pp'] for r in rows if r['method']==m[0] and r['metric']==t):+.3f}" for t in ['RS','PS','NS']]] for m in methods if m[0]!='MEMIT history']),'',
 '## Teacher-forced accuracy와 NLL','',
 'Rewrite/rephrase desired=new, neighborhood desired=true. Token-micro와 full-target strict를 분리했다. NS에 과거 new-target accuracy를 대입하지 않았다. TF는 자유생성 정확도가 아니다. 원 게시표에 prompt-macro가 없는 과거방법은 NOT_RECORDED로 남겼다.','',
 table(['방법','family','TF token correct/count (%)','TF strict correct/count (%)','true NLL','new NLL','desired NLL'],[[r['method'],r['metric'],f"{r['tf_correct']}/{r['tf_count']} ({r['tf_micro']*100:.3f})",f"{r['tf_strict_num']}/{r['tf_strict_den']} ({r['tf_strict']*100:.3f})",f"{r['true_nll']:.6f}",f"{r['new_nll']:.6f}",f"{r['desired_nll']:.6f}"] for r in rows]),'',
 '## 누적곡선','', '![동일 fixed10k all-seen 누적지표](four-method-trajectory.png)','',
 'PS/NS는 실제 full-observer batch 지점 사이를 선으로 연결했다. 모든방법의 curve 차이는 다른 denominator의 시간점 차이가 아니라 각 대응 batch의 동일prefix 관측이다. 원 CSV의 source/hash/분모는 manifest에 결속했다.','',
 '## 비교 경계','',
 '- 같은 model revision/seed20260907/fixed10k ordered root/최종분모를 확인했다. 실행시점·host·GPU는 동일하지 않다: history는S3H200NVL, 과거방법은S4PRO6000.',
 '- L4–8 vs L4+L8, projector/L2/BLUE target 정책 등 복수 차이가 있어 history 단독효과나 BLUE 단독효과의 인과 분해를 하지 않는다.',
 '- 각 run 사이 paired lost/gained는 로컬에 호환raw가 없어 NOT_AVAILABLE. 평균/총점 차이를 동일ID의 성공집합이라고 주장하지 않는다. 기존 history run 내부 paired유지 분석은 원보고서에 별도로 유지한다.',
 '- 과거표에는 TF/NLL이 존재해 재사용했으며 저장하지 않은 값·새GPU 재평가를 채우지 않았다. 교차host numerical certification은 NOT_ESTABLISHED.',
 '', '원상세보고: [job54007 완료리뷰](report-ko.md). 재현: `python project/run_scripts/memit_history_lifelong/compare_completed_baselines.py --report <completion-review-r1> --baseline experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1`.','']
 (a.report/'four-method-comparison-ko.md').write_text('\n'.join(text))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(13,3.8))
 for ax,t in zip(axes,['RS','PS','NS']):
  for label,_,_,_ in methods:
   rs=sorted([r for r in curves if r['method']==label and r['metric']==t],key=lambda r:r['batch']);ax.plot([r['batch'] for r in rs],[r['rate']*100 for r in rs],label=label)
  ax.set(title=t,xlabel='Batch',ylabel='All-seen NLL preference (%)',ylim=(40,102));ax.grid(alpha=.2)
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(a.report/'four-method-trajectory.png',dpi=160);plt.close(fig)
 print(table(['method','RS','PS','NS'],[[m[0],*[round(next(r['rate'] for r in rows if r['method']==m[0] and r['metric']==t)*100,3) for t in ['RS','PS','NS']]] for m in methods]))
if __name__=='__main__':main()
