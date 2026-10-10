"""Exact small context payload validation and create-once three-server delivery."""
import pathlib,json,hashlib,subprocess,sys,base64,datetime
R=pathlib.Path('/data/janghj/ODE-edit/local/native-context-server3-20261010/execution-r1');family=sys.argv[1];job={'qwen25':'62101','gptj':'62102','llama3':'62103'}[family]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
c=json.loads((R/f'{family}.json').read_text());o=pathlib.Path(c['out']);d=json.loads((o/'READY.json').read_text())
account=subprocess.check_output(['sacct','-X','-j',job,'-n','-P','-o','JobID,State,Elapsed,ExitCode'],text=True).strip();assert '|COMPLETED|' in account and account.endswith('|0:0')
assert d['job_id']==job and d['source']==c['source'] and d['config_sha256']==c['config_sha256'] and d['revision']==c['revision']
assert d['model_manifest_sha256']==c['model_manifest_sha256'] and d['model_weight_verification']=='RUNTIME_FULL_SHA_PASS'
assert d['seed']==0 and d['profile']==c['profile']
assert d['context_sha256']==sha(o/'contexts.json') and d['context_tokens_sha256']==sha(o/'context-token-ids.json')
ctx=json.loads((o/'contexts.json').read_text());tok=json.loads((o/'context-token-ids.json').read_text());assert list(map(len,ctx))==list(map(len,tok))==[1,5]
assert ctx[0]==['{}'] and all(x.count('{}')==1 for g in ctx for x in g)
assert all(isinstance(i,int) for g in tok for row in g for i in row)
expected='35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4'
if family!='llama3':assert d['generator_sha256']==expected
members=[]
for name,p in [('contexts.json',o/'contexts.json'),('context-token-ids.json',o/'context-token-ids.json'),('READY.json',o/'READY.json'),('config-provenance.json',R/f'{family}.json')]:
 b=p.read_bytes();members.append(dict(name=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),payload=base64.b64encode(b).decode()))
receipts=[]
for server,host,root in [('server1','rke-server1','/mnt/raid5/janghj/ODE-edit'),('server2','rke-server2','/mnt/raid5/janghj/ODE-edit'),('server4','rke-server4','/data/janghj/ODE-edit')]:
 dest=root+'/local/native-context-import/server3-20261010/job-'+job+'/'+family
 request=dict(destination=dest,members=members)
 code="import json,pathlib,hashlib,base64\nr=json.loads("+repr(json.dumps(request))+ ")\np=pathlib.Path(r['destination']);p.mkdir(parents=True,exist_ok=True)\nassert not p.is_symlink()\nfor m in r['members']:\n f=p/m['name'];b=base64.b64decode(m['payload']);assert len(b)==m['bytes'] and hashlib.sha256(b).hexdigest()==m['sha256']\n if f.exists():assert not f.is_symlink() and f.is_file() and f.read_bytes()==b\n else:\n  with f.open('xb') as z:z.write(b)\n assert hashlib.sha256(f.read_bytes()).hexdigest()==m['sha256']\nprint(json.dumps({'destination':str(p),'verified':True,'members':[{k:v for k,v in m.items() if k!='payload'} for m in r['members']]}))\n"
 result=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3','-'],input=code,text=True,timeout=45);receipts.append(dict(server=server,**json.loads(result)))
report=dict(family=family,job_id=job,status='READY_VERIFIED_DELIVERED',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),accounting=account,ready=d,source_paths=[str(o/'contexts.json'),str(o/'context-token-ids.json'),str(o/'READY.json')],deliveries=receipts,source_KEEP=True,frozen_run_injection=False)
out=pathlib.Path(__file__).parent/(family+'-completion.json')
with out.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'family':family,'job':job,'context_sha256':d['context_sha256'],'delivered':[r['server'] for r in receipts]}))
