"""One afterany CPU collection: failures/blocked/aliases are retained as facts."""
import argparse,json,subprocess,traceback
from pathlib import Path
from project.run_scripts.memit_history_lifelong.io import save,file_sha
from .plan import cells
from .reducer import reduce

def collect(lock_path,mapping_path):
 lock=json.loads(Path(lock_path).read_text());mapping=json.loads(Path(mapping_path).read_text())
 assert mapping['source_commit']==lock['source_commit'] and mapping['lock_sha256']==file_sha(lock_path)
 out=Path(lock['output']);dest=out/'collection';dest.mkdir(parents=True,exist_ok=False)
 analysis_path=Path(__file__).resolve().parents[3]/'analysis-source.json'
 analysis=json.loads(analysis_path.read_text()) if analysis_path.exists() else dict(source_commit=lock['source_commit'],kind='same_source')
 if analysis_path.exists():
  assert analysis['runtime_source_commit']==lock['source_commit'] and analysis['execution_lock_sha256']==file_sha(lock_path)
  for member in analysis['members']:assert file_sha(member['path'])==member['sha256']
 save(dest/'analysis-source.json',analysis)
 jobs=[str(x['job_id']) for x in mapping['jobs'] if x['group']!='CPU']
 p=subprocess.run(['sacct','-j',','.join(jobs),'--noheader','--parsable2','--format=JobIDRaw,JobName,User,State,ExitCode,ElapsedRaw,AllocTRES,MaxRSS'],text=True,capture_output=True)
 save(dest/'accounting.json',dict(job_ids=jobs,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,allocation_parent_only=True))
 groups={}
 for g in 'PABCD':
  root=out/'groups'/g
  if (root/'failure.json').exists():groups[g]=dict(status='TECHNICAL_FAILURE',receipt=json.loads((root/'failure.json').read_text()))
  elif (root/'terminal.json').exists():groups[g]=json.loads((root/'terminal.json').read_text())
  else:groups[g]=dict(status='NO_RUNTIME_TERMINAL',interpretation='scheduler_dependency_failure_or_process_termination; see exact accounting')
 for group,record in groups.items():
  if record['status']=='COMPLETED_GROUP':assert record['bindings']==lock['bindings']
 coverage={}
 for c in cells(lock['cells']):
  path=out/'cells'/c['cell_id']/'terminal.json'
  coverage[c['cell_id']]=json.loads(path.read_text()) if path.exists() else dict(status='NOT_COMPLETED',cell=c['cell_id'])
 save(dest/'coverage.json',coverage);save(dest/'groups.json',groups)
 expected=all(v['status'] in ['COMPLETED','NOT_FIRED','BLOCKED_Z_CALIBRATION'] for v in coverage.values())
 error=None;result=None
 if expected and all(v['status']=='COMPLETED_GROUP' for v in groups.values()):
  try:
   result=reduce(out,lock['cells'],str(Path(lock['dataset_root'])/'counterfact.json'))
   from .figures import make
   make(out)
   from .secondary import make as secondary
   result['secondary']=secondary(out,cells(lock['cells']),json.loads((Path(lock['dataset_root'])/'counterfact.json').read_text()))
  except BaseException as ex:error=dict(error=repr(ex),traceback=traceback.format_exc());save(dest/'reducer-failure.json',error)
 status='COMPLETED_REGISTERED_PLAN' if result is not None and error is None else 'TECHNICAL_INCOMPLETE'
 lines=['# MEMIT HJ v2 수집 상태','',f'상태: {status}', '', '|cell|상태|','|---|---|']
 lines += [f"|{k}|{v['status']}|" for k,v in coverage.items()]
 lines += ['', 'BLOCKED_Z_CALIBRATION은 새 Z 실험 완료가 아니며, NOT_FIRED는 부모 관측의 alias이다.',
  '배정 비용은 accounting의 parent allocation 행만 집계한다. batch/extern을 더하지 않는다.']
 (dest/'report-ko.md').write_text('\n'.join(lines)+'\n')
 members=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix not in ['.pt','.partial']]
 save(dest/'manifest.json',dict(bindings=lock['bindings'],members=members))
 # Completion is published only AFTER report, independent reducer and manifest.
 save(out/'terminal.json',dict(status=status,manifest_sha256=file_sha(dest/'manifest.json'),report_sha256=file_sha(dest/'report-ko.md'),
  result=result,analysis_source=analysis,unexecuted_or_blocked=[k for k,v in coverage.items() if v['status']!='COMPLETED'],reducer_error=error,
  permanent_checkpoints=False,remaining_temporary_CP=[str(p) for p in (out/'temporary-checkpoints').glob('*.pt')]))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--mapping',required=True);a=p.parse_args();collect(a.lock,a.mapping)
