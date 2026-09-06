"""First-B100 factual/cost preview from completed receipts only, no replay."""
import argparse,hashlib,json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from .reporting import read,sha,write,jwrite,csvwrite,metrics


def first_b100(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,mode=0o700)
    source=read(root/'source.lock.json');sample=read(root/'sample.lock.json')
    smoke=Path(read(root/'smoke-gates.lock.json')['root']);inputs=[];rows=[];records={}
    names=[(0,'llama3-8b-inst','O_NATIVE'),(1,'qwen2.5-7b-inst','O_NATIVE'),(2,'llama3-8b-inst','JV_NATIVE')]
    for i,alias,arm in names:
        b=root/f'chain-{i}-{alias}-{arm}'/'batch-01'
        paths=[b/'complete.json',b/'writer.json',b/'entry.json',b/'target-reference.json']
        x,w,e,z=map(read,paths);inputs+=paths
        if x['status']!='BATCH_COMMITTED' or x['requested']!=100 or x['commit']['history_append_count']!=1:
            raise RuntimeError('FIRST_B100_TERMINAL_GATE')
        row=metrics(w['endpoint']['evaluation'],dict(cell=i,alias=alias,arm=arm,batch=1,
            completed_requests=100,source_head=source['head'],W_entry=e['W_entry'],M_entry=e['M_entry'],
            fixed_z_sha256=z['fixed_z_sha256'],target_seconds=w['compute_z']['wall'],
            write_with_endpoint_seconds=w['write_including_endpoint']['wall'],
            endpoint_observation_seconds=w['endpoint_evaluation_seconds'],
            first_batch_total_seconds=x['full_batch_compute']['wall'],checkpoint_bytes=x['checkpoint']['bytes'],
            checkpoint_seconds=x['checkpoint_compute']['wall'],peak_gpu_bytes=x['peak_gpu_allocated'],
            peak_host_rss_kib=x['peak_host_rss_kib'],main_jvp_count=w.get('main_jvp_count',0)))
        rows.append(row);records[i]=(x,w,e,z)
    if any(records[0][2][k]!=records[2][2][k] for k in ('W_entry','M_entry','request_order_sha256')):
        raise RuntimeError('LLAMA_FIRST_B100_COMMON_ENTRY')
    if records[0][3]['fixed_z_sha256']!=records[2][3]['fixed_z_sha256']:
        raise RuntimeError('LLAMA_FIRST_B100_COMMON_Z')
    csvwrite(out/'first-b100-metrics.csv',rows)
    # Explicit engineering proxies, never scientific measurements or a resource cap.
    smoke_write={};eval_time={}
    for alias in ('llama3-8b-inst','qwen2.5-7b-inst'):
        p=smoke/f'smoke-{alias}'/'batch-01'/'writer.json';inputs.append(p)
        smoke_write[alias]=read(p)['write_including_endpoint']['wall']
        p=smoke/f'smoke-{alias}'/'W0-full.json';inputs.append(p);eval_time[alias]=read(p)['compute']['wall']
    ratio=smoke_write['qwen2.5-7b-inst']/smoke_write['llama3-8b-inst']
    estimates=[];cost={}
    for row in rows:
        cost[row['cell']]=row['first_batch_total_seconds']
    q=rows[1];l=rows[2]
    # Qwen JV B100 has not run yet; its write-only proxy is labeled as such.
    cost[3]=q['target_seconds']+l['write_with_endpoint_seconds']*ratio+max(0,
        q['first_batch_total_seconds']-q['target_seconds']-q['write_with_endpoint_seconds'])
    jvp=records[2][1]['jvp_ledger']['wall_seconds']
    cost[4]=cost[2]-.8*jvp;cost[5]=cost[3]-.8*jvp*ratio
    aliases={0:'llama3-8b-inst',1:'qwen2.5-7b-inst',2:'llama3-8b-inst',3:'qwen2.5-7b-inst',4:'llama3-8b-inst',5:'qwen2.5-7b-inst'}
    arms={0:'O_NATIVE',1:'O_NATIVE',2:'JV_NATIVE',3:'JV_NATIVE',4:'L8_ONLY_NATIVE',5:'L8_ONLY_NATIVE'}
    duration={}
    for i in range(6):
        extra_seen=(1300*26+3200*2)/(1000*26)*eval_time[aliases[i]]
        duration[i]=10*cost[i]+extra_seen
        estimates.append(dict(cell=i,alias=aliases[i],arm=arms[i],first_B100_seconds=cost[i],
            first_B100_status='MEASURED' if i in (0,1,2) else 'UNMEASURED_ENGINEERING_PROXY',
            ten_batch_plus_seen_proxy_seconds=duration[i],added_seen_eval_proxy_seconds=extra_seen,
            additional_seen_prompt_rows=40200,first_cost_checkpoint_overcount_not_corrected=True,
            history_evolution_cost_unmeasured=True,scientific_outcome_imputation=0))
    csvwrite(out/'planning-cost-proxies.csv',estimates)
    main_makespan=max(duration[0],duration[1],duration[2],min(duration[0],duration[1],duration[2])+duration[3])
    wall=main_makespan+max(duration[4],duration[5])
    checkpoint_bytes=9*(rows[0]['checkpoint_bytes']+rows[1]['checkpoint_bytes'])
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    text=['# Alpha-JV sequential — 첫 B100 및 비용 예고',f'\nKST: {stamp}',
        '\n세 cell의 첫 B100이 실제 완료됐다. Qwen JV는 cap3에 따라 queued이며 아래 성능 표에는 추정하지 않는다. 이는 at-write current-B100이며 sequential final-W10 retention 결과가 아니다.',
        '\n| Model | Arm | RS | PS | strict PS | NS | B100 total min |', '|---|---|---:|---:|---:|---:|---:|']
    for x in rows:text.append(f"| {x['alias']} | {x['arm']} | {x['RS_num']}/{x['RS_den']} | {x['PS_num']}/{x['PS_den']} | {x['PS_strict_num']}/{x['PS_strict_den']} | {x['NS_num']}/{x['NS_den']} | {x['first_batch_total_seconds']/60:.2f} |")
    text+=['\nLlama O/JV는 첫 W0/M0, request/order와 accepted-z가 exact 일치한다. 이후에는 각 arm의 자기 W/M에서 z를 다시 계산하므로 shared-z 인과 비교로 부르지 않는다.',
        '\nRS/PS/NS는 NLL preference이며 tie는 실패다. Token correctness는 CSV 별도 열이며 자유 생성 정확도와 혼합하지 않는다.',
        '\n## 첫 비용 기반 계획 (측정값과 proxy 분리)',
        f'\n6 chains의 단순 proxy 합: {sum(duration.values())/3600:.2f} GPU-hours. cap3와 main→L8 우선순위의 단순 wall proxy: {wall/3600:.2f} hours. 실제 보장/새 hard cap이 아니다.',
        '\nQwen JV는 아직 B100 실측이 없어 Llama JV write 비용에 두 모델의 이미 완료된 2-batch smoke writer 비율을 적용한 engineering proxy다. L8-only는 20→4 JVP 감소만 반영한 proxy이며 실제 overhead를 숨기지 않는다. 과학 metric imputation0.',
        '\n각 chain에 10×첫 B100 cost와 additional seen-prefix 약 40,200 prompt-row 비용을 W0 평가 row당 시간으로 더했다. 최초 cold setup 및 checkpoint 비용을 10번 세어 보수적으로 중복 포함했고, history 성장·동시 workload·prompt-length 변동은 아직 실측되지 않았다. B5/terminal 실측이 이를 대체한다.',
        f'\n실제 checkpoint 크기에 기반한 6-chain W/M checkpoint1/5/10 예상 합: {checkpoint_bytes/1024**3:.2f} GiB. Smoke/raw receipts/logs는 별도다. Full pretrained model은 중복 저장하지 않는다.',
        '\n## 불변조건',
        '\n2-model 2-batch smoke terminal-valid; actual W/M continuity, append1, commit F/B/solve0, checkpoint tensor reload, W0 restore PASS. Main source의 observer-only timing child는 수식/seed/sample/precision을 바꾸지 않았다. Main chains와 모든 finite endpoint는 계속하며 L8-only는 네 main chain 완주·main table 뒤 두 모델 모두 실행한다.',
        f"\nSource `{source['head']}` / tree `{source['tree']}`. Sample `{sample['ordered_root']}`.",f'\n실행 root `{root}`. Main merge/push0; dedicated branches only.']
    write(out/'first-b100-report-ko.md','\n'.join(text)+'\n')
    manifest=dict(source_head=source['head'],source_tree=source['tree'],sample_root=sample['ordered_root'],
        inputs=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in inputs],
        members=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.iterdir())],
        valid_first_B100_cells=[0,1,2],queued_cell=3,scientific_imputation_count=0,model_replay_count=0)
    jwrite(out/'first-b100-manifest.json',manifest)
    jwrite(out/'rooted-first-b100-receipt.json',dict(
        manifest_sha256=sha(out/'first-b100-manifest.json'),report_sha256=sha(out/'first-b100-report-ko.md'),
        root_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        status='FIRST_B100_PARTIAL_3_OF_4_MEASURED',scientific_imputation_count=0,model_replay_count=0))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(first_b100(a.root,a.output))
