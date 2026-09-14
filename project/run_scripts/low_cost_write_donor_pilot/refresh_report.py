"""Raw-free Korean factual report and deterministic code-generated PNGs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .refresh_review import ORDER

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def table(rows,fields):
 return '\n'.join(['| '+' | '.join(fields)+' |','|'+'|'.join(['---']*len(fields))+'|']+['| '+' | '.join(str(r.get(k,'NOT_RECORDED')) for k in fields)+' |' for r in rows])

def read(root,name):
 with (root/(name+'.csv')).open() as f:return list(csv.DictReader(f))

def plots(root):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':110,'savefig.dpi':150})
 final=read(root,'first-final-table');rates=read(root,'batchmetrics');sub=read(root,'subwrites');cost=read(root,'compute-ledger')
 def save(fig,name):
  fig.tight_layout();fig.savefig(root/name,metadata={'Software':'ODE-edit refresh_report.py'});plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for ax,m in zip(axes,['RS','PS','NS']):
  ax.bar([r['policy'] for r in final],[float(r[m+'_percent']) for r in final]);ax.set_title(f'W60 FullSeen6000 {m}');ax.set_ylabel('Canonical preference (%)');ax.tick_params(axis='x',rotation=35)
 save(fig,'final-six-policy.png')
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for ax,m in zip(axes,['RS','PS','NS']):
  for p in ORDER:
   rr=[r for r in rates if r['policy']==p and r['metric']==m and r['population']=='current' and r['group']=='ALL']
   ax.plot([int(r['batch']) for r in rr],[100*float(r['rate']) for r in rr],label=p,marker='.')
  ax.set_title('Current B100 '+m);ax.set_xlabel('Batch');ax.set_ylabel('%')
 axes[-1].legend(fontsize=7);save(fig,'current-batch-curves.png')
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for ax,m in zip(axes,['RS','PS','NS']):
  for p in ORDER:
   rr=[r for r in rates if r['policy']==p and r['metric']==m and r['population']=='first_suffix500' and r['group']=='ALL']
   ax.plot([int(r['batch']) for r in rr],[100*float(r['rate']) for r in rr],label=p,marker='o')
  ax.set_title('Fixed first suffix500 '+m);ax.set_xlabel('Actual checkpoint');ax.set_ylabel('%')
 axes[-1].legend(fontsize=7);save(fig,'fixed-first500-retention.png')
 tails=read(root,'nll-distributions')
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for ax,m in zip(axes,['RS','PS','NS']):
  field='true_nll' if m=='NS' else 'new_nll'
  selected={r['policy']:r for r in tails if r['batch']=='60' and r['population']=='suffix' and r['metric']==m and r['group']=='ALL' and r['field']==field and r['unit']=='PROMPT'}
  for q in ['median','p95','p99']:
   ax.plot(ORDER,[float(selected[p][q]) for p in ORDER],marker='.',label=q)
  ax.set_title('Suffix1000 desired-target NLL: '+m);ax.tick_params(axis='x',rotation=35);ax.set_ylabel('Mean-token NLL')
 axes[-1].legend();save(fig,'desired-target-nll-tails.png')
 fig,ax=plt.subplots(figsize=(9,4))
 for p in ORDER[2:]:
  rr=[r for r in sub if r['policy']==p]
  ax.plot(range(len(rr)),[float(r['actual_delta_norm']) for r in rr],label=p)
 ax.set_title('Layer-wise Update Magnitude');ax.set_xlabel('Chronological subwrite (L4)');ax.set_ylabel('Actual FP32 delta Frobenius norm');ax.legend();save(fig,'layer-update-magnitude.png')
 fig,ax=plt.subplots(figsize=(9,4))
 online=[sum(float(r['online_seconds']) for r in cost if r['policy']==p)/60 for p in ORDER]
 ev=[sum(float(r['evaluation_seconds']) for r in cost if r['policy']==p)/60 for p in ORDER]
 ax.bar(ORDER,online,label='Instrumented online');ax.bar(ORDER,ev,bottom=online,label='Evaluation');ax.set_ylabel('Minutes');ax.set_title('Recorded online and evaluation time (not allocated total)');ax.legend();save(fig,'recorded-compute-cost.png')
 return {p.name:sha(p) for p in sorted(root.glob('*.png'))}

def report(attempt,root):
 a=Path(attempt);lock=json.loads((a/'execution.lock.json').read_text());final=read(root,'first-final-table');rates=read(root,'batchmetrics');cost=read(root,'compute-ledger');trans=read(root,'paired-transitions');general=read(root,'terminal-general')
 def population(pop):
  return [r for r in rates if r['batch']=='60' and r['population']==pop and r['group']=='ALL']
 cost_summary=[]
 for policy in ORDER:
  rows=[r for r in cost if r['policy']==policy]
  result=dict(policy=policy,new_spending=policy not in lock['reference_reuse'])
  for field in ['online_seconds','evaluation_seconds','target_including_nested_IO_seconds','native_writer_seconds','history_seconds','materialization_seconds','snapshot_seconds']:
   values=[r.get(field) for r in rows]
   result[field]=round(sum(float(v) for v in values),4) if len(values)==10 and all(v not in (None,'','NOT_RECORDED') for v in values) else 'NOT_RECORDED'
  cost_summary.append(result)
 policyrows=[]
 for p in lock['policies']:
  policyrows.append(dict(policy=p['id'],caps=str(p['target_update_caps']),gammas=str(p['write_gammas']),target_mode=p['target_mode'],reuse=p['id'] in lock['reference_reuse']))
 paragraphs=['# REFIT4 write-refresh Middle SEQ1000 사실 보고',
  'Instruction: ODEEDIT-S06-REFIT4-WRITE-REFRESH-SEQ1000-SH4-V1. Scientific promotion=false; 최종 claim/후보 선택은 GH 소유다.',
  '## 완료 범위와 비교 기준',
  '공통 L4 W50/M50에서 B51–B60, 고정 ordinal [5000,6000)의 신규 unique1000을 각 정책이 자기 trajectory로 처리한다. 전체표는6정책60logicalbatch이며 N4/REFIT4의 기존20batch를 재사용한다. 신규 main은4정책40batch다. fullseen6000은 과거5000+신규1000이며 신규6000/full10k 실험이 아니다. 기술 native/I1 B100은 별도 분모·비용이다.',
  table(policyrows,['policy','caps','gammas','target_mode','reuse']),
  '여기서 N4는 BLUE-style AlphaEdit singleton L4/L2=1의 fresh native fit이다. blue=False 다섯 layer/L2=10의 원본 AlphaEdit baseline과 다르며 MEMIT 비교도 아니다. 모든 신규 정책은 L4 down_proj만 쓰고 P physical4→asset0→singleton0을 사용한다. L8 write 또는 M8 재구축은 없다.',
  '같은 sample/host/config/entry에서 시작했으나 B52 이후 각자의 W/M/target이 달라진다. 교차 정책 차이는 end-to-end trajectory 비교이며 동일 state에서 한 요소의 인과 기여율이 아니다. 원 reference execution5e96dcb와 새 execution '+lock['worktree_head']+'는 구분한다.',
  '## 최종 actual W60 fullseen6000',
  table(final,['policy','RS_n','RS_d','RS_percent','PS_n','PS_d','PS_percent','NS_n','NS_d','NS_percent','RS_delta_n_N4','PS_delta_n_N4','NS_delta_n_N4']),
  'RS/PS는 target-new 평균 token NLL<target-true, NS는 true<new이며 tie는실패다. TFstrict/tokenaccuracy는 보조이며 자유생성 의미정확도가 아니다. NS를 true-token preservation으로 부르지 않는다.',
  '## 신규1000, 과거5000, 현재B60 분리']
 for pop in ['suffix','entry_old','current','first_suffix500']:
  paragraphs += ['### '+pop,table(population(pop),['policy','metric','numerator','prompt_denominator','rate','new_strict_numerator','new_token_accuracy','two_P_request_strict_n','two_P_request_strict_d'])]
 paragraphs += ['Fullseen의 Current/suffix/old/Historical/first500 rows는 같은 endpoint identity에서 재사용했다. 별도 평가나 추가 독립 분모로 더하지 않는다. Online-at-write1000은 서로 다른10개state의 합이며 finalretention이 아니다.',
  '## 누적 retention·전이',
  table([r for r in trans if r['contrast']=='AT_WRITE_TO_W60' and r['population']=='suffix1000' and r['group']=='ALL'],['before','metric','before_numerator','after_numerator','lost','gained','retained','failed_both','conditional_loss_rate','prompt_denominator']),
  'first-suffix500의 W55→W60은 같은 문항의 시간 전이다. 공통성공/공통실패 strata는 W55의 사후 상태에 조건을 건 것이므로 인과 식별이 아니다. B60 cohort는 후속 노출0으로 별도 표기했다. oldentry 평가가 없는 모집단의 교차정책 W60 차이를 W50→W60 forgetting으로 부르지 않는다.',
  'ACTIVE_TARGET는 observed prefix의 exact(subject,relation) 최신target문자열과 같은 이벤트이며 같은target 재발행을 포함한다. SUPERSEDED/UNKNOWN을 버리지 않는다. 의미상 충돌의 완전 판별은 아니다. 저장되지 않은 중간 첫failure 시점은 추정하지 않는다.',
  '## NLL·strict·반대 방향 근거',
  'nll-distributions.csv는 new/true/desiredmargin의 mean/median/p90/p95/p99/min/max를 prompt와 request-cluster mean으로 분리한다. paired-transitions.csv의 new/true NLL harm tail은 동일 identity 교차정책 차이다. 성공률 증가를 모든 문항의 손실0 또는 strict 개선으로 바꾸지 않는다. P2/N10을 독립 request로 늘린 유의성 주장은 없다. 이번 표는 단일 고정 순서이며 별도 random-seed 반복 또는 full10k 안정성을 입증하지 않는다.',
  'request-cluster-uncertainty.csv/JSON은 동일 case의 P2/N10을 함께 resample한 paired request-cluster percentile 95% CI다. NumPy PCG64 seed20260914, 1000회이며 저장된 표본 내 불확실성만 나타낸다. 독립 편집순서 반복이나 checkpoint 독립성을 가정한 결과가 아니고 다중 탐색에 대한 확증적 유의성 선언도 아니다. CI가0을포함하는지여부는성능gate가아니다. 양의 preference Δpp는 gain, 양의 desired-target NLL Δ는 harm이다.',
  '## 작은 general 패널',table(general,['policy','wiki_nll','wiki_tokens','wiki_sequences','mmlu_correct','mmlu_d','mmlu_invalid']),
  'Wiki128 mask/입력과 MMLUdev32 identity를 재사용했다. MMLU는 alternative integercorrect/invalid이고 generation parser/F1/full57subject 점수와 다르다. Audit128/MMLU68/FutureN은 DEFERRED_NOT_EVALUATED, 신규평가·정책feedback0.',
  '## 실행·상태·기술 검증',
  'state-review-receipt.json 및 state-links/subwrites/checkpoints/target-counters CSV에 실제검산을 기록한다. innerhistoryappend0, native endpoint finalizer append1/batch; request100 target barrier 뒤 batchwrite1이다. u/Adam m/v/t는 request내chunk사이에만 유지하며 anchor/teacher/clamp는batchentry고정이다. Frozen후속chunk도 현재Y로residual을새로읽는다. oldreference optimizer내부미저장항목은 NOT_RECORDED이다.',
  'CPU weights_only/tensorSHA와 실제 GPU continuation/replay는 다르다. 신규 CP51/55/60은 selectedW4/M4/context/RNG를 보존하지만 전체모델checkpoint가 아니다. Base model/P/config/source/order closure가필요하다. Actualdelta journal의 exacttrajectory replay는 NOT_TESTED다. 원native및NativeSingletonFitter.fit 수정0.',
  '## 비용과 저장',
  table(cost_summary,['policy','new_spending','online_seconds','evaluation_seconds','target_including_nested_IO_seconds','native_writer_seconds','history_seconds','materialization_seconds','snapshot_seconds']),
  'compute-ledger.csv는 reused reference와 신규spending을 분리한다. 신규targettimer의 requeststate I/O는 nested이므로 총합에 다시더하지 않는다. 기존reference는 매batchgeneral, 신규는terminalgeneral이므로 평가포함wall을 같은정책online비용으로 치환하지 않는다. Instrumentedonline은purewriter가 아니다. Prepared/M8과기존N4/REFIT4의과거비용을새연구비로중복계상하지않는다. Scheduler allocation/queue/concurrency는별도receipt근거이며 측정되지않은component는NOT_RECORDED.',
  '## 재현·산출물·제한',
  '원시tensor/request/teacher/prompt/fullstdout은 local-only이고 Git에 포함하지 않는다. NO_BROADCAST_NOT_REQUIRED. 원본checkpoint/공유asset/다른run삭제0. Main에는 본namespace CPU분석코드와raw-free표/PNG/manifest만통합한다.',
  '재현 명령(저장된 terminal 자료만 CPU):\n\n```bash\npython -m project.run_scripts.low_cost_write_donor_pilot.refresh_review --attempt '+str(a)+' --out '+str(root)+'\npython -m project.run_scripts.low_cost_write_donor_pilot.refresh_state_review --attempt '+str(a)+' --out '+str(root)+'\npython -m project.run_scripts.low_cost_write_donor_pilot.refresh_report --attempt '+str(a)+' --out '+str(root)+'\n```',
  'PNG는 이 코드가 CSV를 읽어 생성한다: final-six-policy.png, current-batch-curves.png, fixed-first500-retention.png, desired-target-nll-tails.png, layer-update-magnitude.png, recorded-compute-cost.png. AI 이미지생성/수동그림수정0. 각PNG SHA는manifest와plot-reproduction.json에결속한다.',
  'Middle 결과 후 최대1후보/Late policy는GH가결정한다. 본보고서가후보선정/새Late제출/full10k/추가alpha·layer실험을승인하지않는다. finitepoor결과를제외하거나성능ANDgate로완료를판정하지않는다.']
 (root/'diagnostic-report-ko.md').write_text('\n\n'.join(paragraphs)+'\n')

def seal(attempt,root):
 a=Path(attempt);files=[]
 for p in sorted(root.iterdir()):
  if not p.is_file() or p.name in ['analysis-manifest.json','rooted-receipt.json']:continue
  assert p.suffix in ['.csv','.json','.md','.png']
  files.append(dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)))
 code=Path(__file__).resolve().parent;sources=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(code.glob('refresh*py'))]
 lock=json.loads((a/'execution.lock.json').read_text());m=dict(members=files,sources=sources,execution_commit=lock['worktree_head'],execution_lock_sha256=sha(a/'execution.lock.json'),reference_execution='5e96dcb3745977b1f273e3f5afbee61167248d49',raw_payload_in_git=False)
 (root/'analysis-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 (root/'rooted-receipt.json').write_text(json.dumps(dict(status='RAWFREE_PACKAGE_REHASH_PASS',manifest_sha256=sha(root/'analysis-manifest.json'),member_root=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),scientific_promotion=False,claim_decision='PENDING_GH_REVIEW'),indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 report(args.attempt,args.out);first=plots(args.out);second=plots(args.out);assert first==second
 (args.out/'plot-reproduction.json').write_text(json.dumps(dict(status='BYTE_IDENTICAL',plots=first,source_sha256=sha(__file__)),indent=2)+'\n')
 seal(args.attempt,args.out)
