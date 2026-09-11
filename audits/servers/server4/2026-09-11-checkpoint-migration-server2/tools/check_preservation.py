from inventory import *
def run():
 manifest=read(C/'source-manifest.json');terminals={e['terminal']['path']:e['terminal'] for e in manifest['members']};count=0;size=0
 cp={e['path'] for e in manifest['members']}
 for path,m in terminals.items():
  p=Path(path);assert sha(p)==m['sha256'];t=read(p)
  for row in t['manifest_members']:
   f=p.parent/row['path']
   if str(f) in cp:continue
   assert f.is_file() and f.stat().st_size==row['bytes'],str(f)
   count+=1;size+=row['bytes']
 publications=[]
 for name in ['blue-native-lifelong-comprehensive-review-2026-09-11-v1','blue-l4-l8-lifelong-b100x100-review-2026-09-09-v3','blue-alphaedit-fivearm-sequential1000-review-2026-09-07-v1']:
  root=W/'experiment-reports/servers/server4'/name
  for p in root.iterdir():
   if p.name in ['diagnostic-report-ko.md','factual-report-ko.md','analysis-manifest.json','rooted-receipt.json']:publications.append(member(p))
 save('preservation-receipt.json',dict(status='NON_CHECKPOINT_FILES_PRESENT_SIZE_MATCH_TERMINALS_UNCHANGED',terminal_fullsha=len(terminals),non_checkpoint_members=count,non_checkpoint_bytes=size,publications=publications,limitation='non-checkpoint contents not unnecessarily rehashed; exact unlink scope excludes them',recursive_delete=0))
 print('PRESERVED',len(terminals),count,size)
if __name__=='__main__':run()
