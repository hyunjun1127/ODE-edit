from .common import *
from .metrics import reduce_eval

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    s=sample();ids=[r['case_id'] for r in s['records']]
    rows=[];summary=[];inventory=[]
    sched={r['job']:r for r in csvread(LOCAL/'scheduler-audit.csv')}
    for i,arm in enumerate(ARMS):
        r=root(i);t=read(r/'terminal.json');rt=read(r/'runtime.json')
        assert (t['status'],t['batches'],t['requests'],t['cell'])==('TERMINAL_VALID',100,10000,i)
        manifest={m['path']:m for m in t['manifest_members']}
        for rel in ['runtime.json','B100/commit.json','B100/seen-full.json']:
            inventory.append(dict(arm=arm,**checked_member(r,rel,manifest)))
        inventory.append(dict(arm=arm,path=str(r/'terminal.json'),sha256=sha(r/'terminal.json'),bytes=(r/'terminal.json').stat().st_size,status='SELF_ROOT_REHASH_NO_EXTERNAL_SIGNATURE'))
        c=read(r/'B100/commit.json');ev=read(r/'B100/seen-full.json')
        assert ev['evaluation_type']=='CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS' and ev['state']==c['endpoint']
        assert rt['lock_sha256']==sha(attempt(i)/'execution.lock.json')
        reduced=reduce_eval(ev,ids,arm,100,'FINAL_W100_FULL10000');rows+=reduced
        q=dict(arm=arm,method=rt['spec']['method'],variant=rt['spec']['variant'],job=JOBS[i],status=t['status'],batches=100,requests=10000,
               scheduler_seconds=int(sched[JOBS[i]]['elapsed_seconds']),gpu_hours=int(sched[JOBS[i]]['elapsed_seconds'])/3600,wall_seconds=t['seconds'],
               raw_bytes=sum(m['bytes'] for m in t['manifest_members'])+(r/'terminal.json').stat().st_size,
               verification='FINAL_MEMBER_SHA_NLL_PAIR_ORDER_STATE_PASS; whole-chain audit pending')
        for z in reduced:
            for k in ['numerator','denominator','rate','new_strict_num','new_strict_den','new_token_correct','new_token_den']:
                q[z['metric']+'_'+k]=z[k]
        summary.append(q)
        print(arm,[(z['metric'],z['numerator'],z['denominator']) for z in reduced],flush=True)
    csvwrite(out/'final-distributions.csv',rows);csvwrite(out/'final-summary.csv',summary)
    save(out/'first-table-inputs.json',inventory)
    show=[]
    for r in summary:
        q={k:r[k] for k in ['arm','job','status','batches','requests','gpu_hours']}
        for tag in MULT:q[tag]=f"{r[tag+'_numerator']}/{r[tag+'_denominator']} ({r[tag+'_rate']*100:.2f}%)"
        q['rewrite_TF_exact']=f"{r['RS_new_strict_num']}/{r['RS_new_strict_den']}"
        q['rephrase_TF_exact']=f"{r['PS_new_strict_num']}/{r['PS_new_strict_den']}"
        show.append(q)
    text='# 6-chain 첫 표: 실제 최종 W100에서 전체 10,000 요청 재평가\n\n'
    text+=table(show,list(show[0]))
    text+='\nRS/PS: new 평균-token NLL < true NLL. NS: true NLL < new NLL. Tie 실패. TF exact는 teacher-forced 모든 target token top-1 일치로 primary preference와 다르다. Current-B100/online 합계가 아니다. 6개 final 평가 member SHA, sample/order, prompt pair 및 endpoint identity 독립 대조 완료; 전체 chain/checkpoint CPU 검산은 후속 단계다. 원본1k/JVP와의 설정·target-policy 차이를 통제한 인과비교가 아니다. scientific_promotion=false.\n'
    (out/'first-sixarm-table-ko.md').write_text(text)
    save(out/'first-table-receipt.json',dict(instruction_id=INSTRUCTION,status='FIRST_TABLE_READY',sample_root=SAMPLE_ROOT,final_members=6,
         final_request_observations=60000,unique_requests=10000,report_sha256=sha(out/'first-sixarm-table-ko.md'),inputs_root=digest(inventory)))

if __name__=='__main__':main(cli().out)
