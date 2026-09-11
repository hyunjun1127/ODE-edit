"""Final family tables before expensive full-checkpoint audit."""
from .common import *

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'first-input-members.json').exists():
        for m in read(OUT/'first-input-members.json'):assert sha(m['path'])==m['sha256']
        finish(csvread(OUT/'first-final-summary.csv'));return
    assert sha(OLD/'factual-report-ko.md')=='839a903839743ad6de7c8cb08d255e7ad2e2c43099f6b81b27a3c91483fb991f'
    om=read(OLD/'analysis-manifest.json'); inherited=[]
    for m in om['members']:
        p=OLD/m['path'];assert sha(p)==m['sha256'] and p.stat().st_size==m['bytes'],str(p)
        inherited.append(dict(path=str(p),sha256=m['sha256'],bytes=m['bytes'],level='SEALED_PUBLICATION_MEMBER_REHASH'))
    if (OUT/'reused-publication-members.json').exists():assert read(OUT/'reused-publication-members.json')==inherited
    else:save(OUT/'reused-publication-members.json',inherited)
    ids=[r['case_id'] for r in sample()['records']]
    sched={line.split('|')[1]:line.split('|') for line in read(LOCAL/'initial.json')['commands'][0]['stdout'].splitlines()}
    oldsummary={r['arm']:r for r in csvread(INHERITED/'final-summary.csv')}
    summary=[];dist=[];inventory=[]
    for arm,job in zip(ARMS,JOBS):
        status=sched[job];assert status[2]=='janghj'
        if status[4]!='COMPLETED' or status[5]!='0:0':
            summary.append(dict(arm=arm,job=job,status='NOT_TERMINAL_'+status[4]));continue
        if arm in OLDARMS:
            r=oldsummary[arm].copy();r.update(verification='REUSED_V3_FULL_CHAIN_AUDIT',job=job)
            summary.append(r);continue
        rp=root(arm);t=read(rp/'terminal.json');rt=read(rp/'runtime.json')
        assert (t['status'],t['batches'],t['requests'])==('TERMINAL_VALID',100,10000)
        tm={m['path']:m for m in t['manifest_members']}
        for rel in ['runtime.json','B100/commit.json','B100/seen-full.json']:
            inventory.append(dict(arm=arm,**checked(rp,rel,tm)))
        inventory.append(dict(arm=arm,path=str(rp/'terminal.json'),bytes=(rp/'terminal.json').stat().st_size,sha256=sha(rp/'terminal.json'),level='TERMINAL_SELF_ROOT'))
        ev=read(rp/'B100/seen-full.json');co=read(rp/'B100/commit.json')
        assert ev['state']==co['endpoint'] and ev['evaluation_type']=='CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS'
        assert rt['lock_sha256']==sha(attempt(arm)/'execution.lock.json')
        rr=reductions(ev,ids,arm,100,'FINAL_W100_FULL10000');dist+=rr
        r=dict(arm=arm,method=family(arm),job=job,status='TERMINAL_VALID_FINAL_MEMBERS_VERIFIED',batches=100,requests=10000,scheduler_seconds=int(status[6]),gpu_hours=int(status[6])/3600,wall_seconds=t['seconds'],raw_bytes=sum(x['bytes'] for x in tm.values()),verification='NEW_FINAL_SHA_PAIR_ORDER_ENDPOINT_PASS; full chain pending')
        for z in rr:
            for k in ['numerator','denominator','rate','new_strict_num','new_strict_den','new_token_correct','new_token_den']:r[z['metric']+'_'+k]=z[k]
        summary.append(r);print(arm,[(z['metric'],z['numerator']) for z in rr],flush=True)
    csvwrite(OUT/'first-final-summary.csv',summary);csvwrite(OUT/'new-final-distributions.csv',dist)
    save(OUT/'first-input-members.json',inventory)
    finish(summary)

def finish(summary):
    text='# 첫 family별 최종 W100 전체10k 표\n\n현재 표는 final NLLpair·분모·state 검증이며 신규8개 전체 chain/CP 검증 완료와 구분한다. 기존6개는 봉인 v3 감사 재사용. W0는 공통1회이며 편집 arm수가 아니다.\n\n'
    for method in ['MEMIT','AlphaEdit']:text+='## '+method+'\n\n'+family_table(summary,method)+'\n'
    text+='\nRS/PS: new NLL<true NLL; NS: true NLL<new NLL, tie 실패. blue=False native와 BLUE는 target/layer/L2가 다르므로 sample-matched end-to-end 비교이며 인과통제 실험 아님. 새 model/eval/GPU0.\n'
    with (OUT/'first-family-final-ko.md').open('x') as f:f.write(text)
    save(OUT/'first-receipt.json',dict(status='FIRST_FAMILY_FINAL_TABLE_READY',rows=len(summary),new_final_files=8,reused=6,w0_publications=1,first_table_sha256=sha(OUT/'first-family-final-ko.md')))
    print('FIRST_TABLE',sha(OUT/'first-family-final-ko.md'),flush=True)

if __name__=='__main__':main()
