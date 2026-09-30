"""Seal raw-free CPU review and user-directed cancellation evidence; no scheduler calls."""
import csv
import io
from pathlib import Path
from .common import read, record, save
from .review_b010 import PUB

def run():
    local=Path('/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1/review-b010-20260930-v1')
    cancel=read(local/'cancellation-and-terminal-receipt.json')
    jobs=list(csv.DictReader(io.StringIO(cancel['sacct']),delimiter='|'))
    assert all(next(r for r in jobs if r['JobID']==j)['State'].startswith('CANCELLED') for j in cancel['canceled'])
    assert all(next(r for r in jobs if r['JobID']==f'55116_{i}')['State']=='COMPLETED' for i in range(3))
    audit=Path('audits/servers/server2/joint-multilayer-bs1-20260929-v1/b010-review-20260930-v1')
    save(audit/'cancellation-receipt.json',dict(utc=cancel['utc'],user_instruction=record(local/'user-recall.txt'),
        source_receipt=record(local/'cancellation-and-terminal-receipt.json'),jobs=jobs,
        commands=cancel['commands'],commands_exit=cancel['commands_exit'],target_queue_empty=True,
        unaffected=['55116_3','55116_4','55117'],canceled_partial_commits=10,canceled_partial_weight_snapshots=0,
        raw_deleted=False,rerun=False,process_disappearance='독립 OS process 검사 미수행'))
    v=read(PUB/'verification.json')
    save(audit/'postrun-owner-review.json',dict(status='PASS_WITH_RECORDED_LIMITS',owner='SH2',independent_red_agent=False,
        checked=['동일 B010 부모 state','100요청 순서와 세 arm token catalog','300 commit/history/인접 state 연결',
                 '독립 NLL reducer 및 기존 reducer 일치','snapshot12 full SHA와 원 reload receipt',
                 '최종과 at-write 분리','NS 실제10문항 분모','지정4job 취소','원본 raw 보존','코드 PNG byte 재현'],
        limits=['최종 전체100요청 NS 미측정','greedy 전량32token censored','GPU 재평가 없음','M/RNG full editor resume 없음',
                '누적 delta 인과적 attribution 없음','exclusive timer 일부 미기록','B050/B090 비교 제외'],
        reducer_tests=6,verification=record(PUB/'verification.json'),supplement=record(PUB/'supplement-verification.json'),
        runtime_mutation=False,new_gpu=0,new_submission=0,scientific_promotion=False))
    report=record(PUB/'report-ko.md')
    save(audit/'artifact-manifest.json',dict(report=report,execution_source=v['execution_source'],
        analysis_source=[record(Path(__file__).parent/f) for f in ('review_b010.py','review_b010_extra.py','review_b010_seal.py','test_review_b010.py')],
        outputs=[record(p) for p in sorted(PUB.iterdir()) if p.is_file()],
        NO_BROADCAST_NOT_REQUIRED=True,raw_local=str(local),publication_source='이 manifest를 포함하는 Git commit; 실행 source와 별개'))
    msg=Path('messages/server-heads/server2/2026-09-30-joint-bs1-b010-review.md');msg.parent.mkdir(parents=True,exist_ok=True)
    with msg.open('x') as f:f.write('# B010 세 arm 리뷰와 사용자 지정 취소\n\n'
        '사용자 recall에 따라 B010 세 arm의 완료100step 및 지표를 CPU 검산했다. B050 JOINT_CUM 55116_5와 B090 55116_6/7/8은 사용자 명시 취소로 CANCELLED 확인, raw 보존. B050 나머지2와 collector55117 변경0.\n\n'
        f'보고 `{PUB}/report-ko.md` SHA `{report["sha256"]}`. 근거 audit `{audit}`. 원실행 source `{v["execution_source"]}`. 신규 GPU/submit0, 전체9경로 완료 주장0, GH의 과학적 해석과 별개 사실 리뷰. 이후 자동 monitoring/재개0.\n')
    print(report)

if __name__=='__main__':run()
