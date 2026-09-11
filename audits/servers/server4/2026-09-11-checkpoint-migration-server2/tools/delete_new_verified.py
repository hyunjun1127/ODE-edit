"""Only verified new archive bundles; individual receipt-bound source unlink."""
from inventory import *
from delete_verified import identity,used
import argparse
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',choices=['new177','jvp1k'],required=True);ap.add_argument('--receipt',required=True);ap.add_argument('--receipt-sha',required=True);ap.add_argument('--catalog',required=True);ap.add_argument('--catalog-sha',required=True);a=ap.parse_args()
 sourcefile=C/('source-manifest.json' if a.bundle=='new177' else 'jvp-source-manifest.json')
 expected='f74e15e88ebb72b2dd62d21cec85caf7a119bc1b714b6945c561b64a24309a96' if a.bundle=='new177' else '5580c8c661c43061be54d99aad1a67dafe37a4a664bc102961b0363604200db6'
 assert sha(sourcefile)==expected
 es=[e for e in read(sourcefile)['members'] if not e['reuse']]
 rp=C/a.receipt;cp=C/a.catalog;assert rp.parent==cp.parent==C
 assert sha(rp)==a.receipt_sha and sha(cp)==a.catalog_sha
 receipt=read(rp);assert receipt['status']=='VERIFIED_DESTINATION' and receipt['full_sha256']
 if 'destination_manifest_sha256' in receipt:assert receipt['destination_manifest_sha256']==a.catalog_sha
 dest={m['source_path']:m['destination'] for m in read(cp)['members']}
 assert set(dest)=={e['path'] for e in es} and receipt['count']==len(es)
 finalroot='/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/'+a.bundle+'-v1/payload/'
 paths={e['path'] for e in es};first=used(paths);assert not first['hits'] and not first['denied_owned_pids'],first
 for e in es:
  d=dest[e['path']];assert d['path']==e['retained_path'] and d['path'].startswith(finalroot)
  assert (d['bytes'],d['sha256'])==(e['bytes'],e['sha256'])
  p=identity(e);assert sha(p)==e['sha256'];identity(e)
 last=used(paths);assert not last['hits'] and not last['denied_owned_pids'],last
 for e in es:identity(e)
 save(a.bundle+'-deletion-plan.json',dict(source_manifest=member(sourcefile),receiver_receipt=member(rp),retention_catalog=member(cp),members=es,in_use_before=first,in_use_immediately_before=last,source_permanent_unlink=True,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
 fs=os.statvfs(R);before=fs.f_bavail*fs.f_frsize;removed=[]
 with (C/(a.bundle+'-unlink-journal.jsonl')).open('x') as f:
  for e in es:
   p=identity(e);os.unlink(p);row=dict(source=e['path'],destination=e['retained_path'],bytes=e['bytes'],sha256=e['sha256'],removed=not p.exists(),reason='USER_APPROVED_VERIFIED_SERVER2_RETAINED_COPY');removed.append(row)
   f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
 fs=os.statvfs(R);after=fs.f_bavail*fs.f_frsize
 save(a.bundle+'-deletion-receipt.json',dict(status='SOURCE_EXACT_FILES_REMOVED_DESTINATION_PRESERVED',removed=removed,count=len(removed),logical_bytes=sum(e['bytes'] for e in es),filesystem_available_before=before,filesystem_available_after=after,observed_available_delta=after-before,delta_caveat='shared filesystem concurrent activity; not exclusively attributed',receiver_receipt_sha256=sha(rp),source_plan_sha256=sha(C/(a.bundle+'-deletion-plan.json')),recursive_delete=0,other_assets_changed=0,GPU=0,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
 print('DELETED',a.bundle,len(removed),sum(e['bytes'] for e in es),'AVAILABLE_DELTA',after-before,flush=True)
if __name__=='__main__':main()
