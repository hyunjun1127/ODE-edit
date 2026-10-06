"""Compact only this task's generated metadata; never edits experiment artifacts."""
import hashlib
import json
from pathlib import Path
import inventory_readonly as inv

def main():
    p=inv.AUDIT/'coverage.json'; c=json.loads(p.read_text())
    assets=set()
    for task in inv.CURRENT:
        config=json.loads((inv.ROOT/'local'/task/'attempt/config.json').read_text())
        assets.update(a['path'] for a in config['assets'])
    all_bindings=c['asset_binding']
    c['asset_binding_record_count_before_compaction']=len(all_bindings)
    c['asset_binding_mismatches']=[x for x in all_bindings if 'error' in x or any(x.get(k) is False for k in ['size_match','inode_match','mtime_match'])]
    c['asset_binding']=[x for x in all_bindings if x['path'] in assets]
    c['full_binding_local']=str(inv.OUT/'initial-coverage.json')
    c['full_binding_local_sha256']=hashlib.sha256((inv.OUT/'initial-coverage.json').read_bytes()).hexdigest()
    p.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    p=inv.AUDIT/'summary.json';s=json.loads(p.read_text())
    s['hardlink_samples_only']=True;s['observed_hardlinks']=s['observed_hardlinks'][:10]
    s['owner_checks']={'AST_JSON_links_counts':'PASS','separate_reviewer':False,'scientific_status_review':False,'read_only_metadata_scope':True}
    p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
    # Generated CSV formatting only. Frozen/raw files are untouched.
    p=inv.AUDIT/'inventory.csv';p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    print(json.dumps({'compact_binding_records':len(c['asset_binding']),'source_binding_records':len(all_bindings),'mismatches':len(c['asset_binding_mismatches'])}))

if __name__=='__main__':main()
