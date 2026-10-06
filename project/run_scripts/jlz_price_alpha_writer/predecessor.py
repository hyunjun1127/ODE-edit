"""Exact predecessor submission/source/resource graph; no science/result polling."""
import getpass,json,re,subprocess
from pathlib import Path
from .common import ROOT,require,sha,verify,member

PREVIOUS=Path('/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k/attempt')
SOURCE='87a5a736c455d5082f5f666ae562f9edc4e5d3b2'
LANES=((59931,59932,59933),(59934,59935,59936))
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def bind():
    path=ROOT/'runs/jlz-price-cap-base-repair-2k/submission.json'
    require(sha(path)=='5854663fbdba6c15bd5539d386a8c96101d5c5a0b76f8305e62fe7db3f6acb3e','MEMIT_SUBMISSION_SHA')
    submission=json.loads(path.read_text());lockpath=PREVIOUS/'execution.lock.json'
    require(sha(lockpath)=='ea45afde90a88f315edb61c35807a9b3419c2c4dd29710870fe269afd4db42b9','MEMIT_LOCK_SHA')
    lock=json.loads(lockpath.read_text());require(lock['source_commit']==SOURCE and lock['owner']==getpass.getuser(),'MEMIT_OWNER_SOURCE')
    require(sha(PREVIOUS/'config.json')==lock['config_sha256'],'MEMIT_CONFIG_SHA')
    for row in lock['launchers']:verify(row)
    ids=[str(j) for lane in LANES for j in lane];records={};absent=[]
    for role,j in submission['jobs'].items():
        j=str(j)
        if j not in ids:continue
        r=subprocess.run(['scontrol','show','job',j,'--oneliner'],text=True,capture_output=True)
        if r.returncode:
            absent.append((role,j));continue
        detail=r.stdout.strip();fields=dict(re.findall(r'(?:^| )([A-Za-z/]+)=([^ ]*)',detail))
        require(fields.get('UserId','').startswith(getpass.getuser()+'(') and fields.get('JobName')=='jlz-price-cap-base-repair-2k-'+role
            and fields.get('ReqNodeList')=='server4' and fields.get('Command')==str(PREVIOUS/(role+'.sh')),'MEMIT_EXACT_JOB_IDENTITY')
        expected=submission['mapping'][role]['dependency'];actual=fields.get('Dependency','(null)')
        expected_ids=set(expected.split(':')[1:]) if expected else set()
        actual_ids=set(re.findall(r'(?:^|:)(\d+)(?=\(|:|$)',actual)) if actual!='(null)' else set()
        require(actual_ids<=expected_ids and (not actual_ids or actual.startswith('afterany:')),'MEMIT_GRAPH_CHANGED')
        script=subprocess.run(['scontrol','write','batch_script',j,'-'],text=True,capture_output=True)
        require(script.returncode==0 and script.stdout.strip()==(PREVIOUS/(role+'.sh')).read_text().strip(),'MEMIT_SOURCE_ARGV')
        records[j]=dict(role=role,state=fields['JobState'],dependency=actual,
            registered_dependency=expected,resource_detail=detail,scheduler_present=True)
    if absent:
        r=subprocess.run(['sacct','-n','-P','-j',','.join(j for _,j in absent),
            '--format=JobIDRaw,User,JobName%100,State,ExitCode'],text=True,capture_output=True)
        require(r.returncode==0,'MEMIT_ACCOUNTING_QUERY')
        rows={x.split('|')[0]:x.split('|') for x in r.stdout.splitlines() if x.split('|')[0] in ids}
        for role,j in absent:
            v=rows.get(j);require(v and v[1]==getpass.getuser() and v[2]=='jlz-price-cap-base-repair-2k-'+role
                and v[3].split()[0].rstrip('+') in TERMINAL,'MEMIT_AGED_OUT_NOT_PROVEN_TERMINAL')
            records[j]=dict(role=role,state=v[3],exit=v[4],scheduler_present=False,
                registered_dependency=submission['mapping'][role]['dependency'],accounting_terminal=True)
    require(set(records)==set(ids),'MEMIT_SIX_EXACT_RECORDS')
    graph_ok=True
    for lane in LANES:
        for i,j in enumerate(lane):
            expected=None if i==0 else 'afterany:'+str(lane[i-1])
            graph_ok=graph_ok and records[str(j)]['registered_dependency']==expected
    frontier=[str(lane[-1]) for lane in LANES] if graph_ok else ids
    # A removed terminal ID cannot be used as a fresh dependency; its exact
    # accounting is the completed barrier, not an inferred successful result.
    frontier=[j for j in frontier if records[j]['scheduler_present']]
    return dict(submission=member(path),lock=member(lockpath),execution_source=SOURCE,
        jobs=records,frontier=frontier,all_six_covered_by_tails=graph_ok,
        no_science_gate=True,no_collector_dependency=True,no_job_mutations=True,
        no_scientific_results_or_logs_read=True)
