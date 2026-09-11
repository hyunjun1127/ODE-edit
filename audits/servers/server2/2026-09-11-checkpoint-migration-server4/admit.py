"""수신 manifest의 경로·중복·용량만 검산하고 private staging을 예약한다."""
import json
import os
from pathlib import Path
from verify_initial import CONTROL,ARCHIVE,INITIAL,save,sha

EXPECTED={
 'source-manifest.json':'f74e15e88ebb72b2dd62d21cec85caf7a119bc1b714b6945c561b64a24309a96',
 'companion-manifest.json':'b801ca86db22898bf6bb4221fcce3107020ee3f5c6caf9a306c6a1f61c3d6997',
 'jvp-source-manifest.json':'5580c8c661c43061be54d99aad1a67dafe37a4a664bc102961b0363604200db6',
 'jvp-companion-manifest.json':'d7cac7dc5acbca858f1fef91a32663bb624b09094395adb58405e496a7c5c291',
 'supplement-manifest.json':'728ffc5ffd167a6b6a7d4e5318b14533f3b310d85429a01f8c2611ded2eb70a7'}

def main():
    manifests={}
    for name,expected in EXPECTED.items():
        p=CONTROL/'incoming'/name;assert sha(p)==expected;manifests[name]=json.loads(p.read_text())
    old=json.loads((CONTROL/'initial72-retention-catalog.json').read_text())
    oldmap={r['source_path']:r['destination'] for r in old['members']}
    s=manifests['source-manifest.json'];j=manifests['jvp-source-manifest.json']
    paths=set();reuse=0;newbytes=0;newcount=0
    for m in s['members']+j['members']:
        assert m['path']==m['realpath'] and m['type']=='regular' and m['uid']==os.getuid() and m['nlink']==1
        assert m['path'] not in paths;paths.add(m['path'])
        if not m['schema']:assert m.get('bundle')=='blue-1k'  # CPU weights_only verification after receipt; not yet deletion-valid.
        if m['reuse']:
            o=oldmap[m['path']];assert (o['path'],o['sha256'],o['bytes'])==(m['retained_path'],m['sha256'],m['bytes']);reuse+=1
        else:
            dest=Path(m['retained_path']);assert dest.is_relative_to(ARCHIVE) and '..' not in dest.parts
            assert not dest.exists();newbytes+=m['bytes'];newcount+=1
    assert reuse==72 and newcount==111
    companionbytes=0;companioncount=0
    for name in ['companion-manifest.json','jvp-companion-manifest.json','supplement-manifest.json']:
        seen=set()
        for m in manifests[name]['members']:
            p=Path(m['relative']);assert not p.is_absolute() and '..' not in p.parts
            assert str(p) not in seen;seen.add(str(p))
            assert m['disposition']=='COPY_ONLY_SOURCE_PRESERVED'
            companionbytes+=m['bytes'];companioncount+=1
    v=os.statvfs(CONTROL);available=v.f_bavail*v.f_frsize;reserve=200*1024**3
    assert newbytes+companionbytes+reserve<available,'PARTIAL_CAPACITY_HOLD'
    assert v.f_favail>newcount+companioncount+10000
    for name in ['new177-v1','jvp1k-v1']:
        root=ARCHIVE/(name+'.partial');root.mkdir(mode=0o700,exist_ok=False)
        (root/'payload').mkdir(mode=0o700)
        (root/'closure').mkdir(mode=0o700)
        if name=='new177-v1':(root/'supplement').mkdir(mode=0o700)
    save(CONTROL/'admission-v1.json',dict(status='DESTINATION_READY_FOR_SH4_ONLY',manifest_hashes=EXPECTED,
      new_checkpoint_count=newcount,new_checkpoint_bytes=newbytes,reuse_count=reuse,reuse_bytes=62011141768,
      companion_count=companioncount,companion_bytes=companionbytes,available_bytes=available,
      available_inodes=v.f_favail,safety_reserve_bytes=reserve,exclusive_filesystem_reservation=False,
      reserved_task_bytes=newbytes+companionbytes,other_task_future_growth='NOT_KNOWN; no other monitoring resumed',
      staging=[str(ARCHIVE/(n+'.partial')) for n in ['new177-v1','jvp1k-v1']],sole_payload_writer='SH4',
      atomic_seal_owner='SH2_AFTER_FULL_HASH_AND_CLOSURE',overwrite=False))
    print('CAPACITY_READY',newcount,newbytes,companioncount,companionbytes,sha(CONTROL/'admission-v1.json'))

if __name__=='__main__':main()
