from inventory import *
def run():
 known={e['path'] for n in ['source-manifest.json','jvp-source-manifest.json'] for e in read(C/n)['members']}
 roots=[R/'local/blue-lifelong-b100x100/attempt-cell0-checkpoint-r3',R/'local/blue-lifelong-b100x100/attempt-checkpoint-r2',R/'local/blue-lifelong-b100x100-l567/attempt-v1',R/'local/fixed10k-native-baselines/attempt-v1',R/'local/blue-alphaedit-sequential-comparison/attempt-v1',R/'local/blue-alphaedit-l4-oneshot-sequential/attempt-v1',R/'local/blue-alphaedit-l8-oneshot-sequential/attempt-v1',R/'local/state/alpha-jv-migration-server4-20260907/tech-r1']
 found=[]
 for root in roots:
  for name in ['W-M.pt','W-method-state.pt']:
   for p in root.rglob(name):
    if str(p) in known:continue
    s=p.lstat();found.append(dict(path=str(p),bytes=s.st_size,uid=s.st_uid,nlink=s.st_nlink,symlink=p.is_symlink(),status='ADDITIONAL_INVENTORY_ONLY_NOT_IN_CANONICAL_MIGRATION',reason='smoke/other local copy; no verified destination bundle, preserved'))
 save('additional-inventory.json',dict(members=found,count=len(found),bytes=sum(m['bytes'] for m in found),search='exact known checkpoint filenames only within authorized roots; no generic PT classification',disposition='PRESERVED_NOT_DELETED'))
 print('ADDITIONAL',len(found),sum(m['bytes'] for m in found))
if __name__=='__main__':run()
