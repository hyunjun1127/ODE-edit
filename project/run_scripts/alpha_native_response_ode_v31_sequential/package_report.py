"""Publish a small byte-closed report subset, retaining large raw-free tables locally."""
import argparse,hashlib,json,os,stat
from pathlib import Path
from .reporting import read,sha,jwrite,write


def package(source,destination):
    source=Path(source).resolve();destination=Path(destination).absolute()
    manifest=read(source/'analysis-manifest.json');receipt=read(source/'rooted-analysis-receipt.json')
    if sha(source/'analysis-manifest.json')!=receipt['manifest_sha256']:
        raise RuntimeError('SOURCE_REPORT_MANIFEST')
    if sha(source/'factual-report-ko.md')!=receipt['report_sha256']:
        raise RuntimeError('SOURCE_REPORT_IDENTITY')
    root=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if root!=receipt['root_sha256']:raise RuntimeError('SOURCE_REPORT_ROOT')
    members=[];external=[]
    for r in manifest['members']:
        p=source/r['path']
        if p.parent!=source or not stat.S_ISREG(p.lstat().st_mode) or p.is_symlink():raise RuntimeError('REGULAR_REPORT_MEMBER')
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise RuntimeError('SOURCE_REPORT_MEMBER_HASH')
        if p.suffix not in ('.md','.csv','.png','.json'):raise RuntimeError('RAW_FREE_REPORT_ALLOWLIST')
        # A packaging byte limit, never a scientific/outcome threshold.
        if r['bytes']>1024*1024 and p.suffix=='.csv':external.append(dict(r,path=str(p),classification='RAW_FREE_LARGE_TABLE_LOCAL_ONLY'))
        else:members.append((p,p.name))
    members.extend([(source/'analysis-manifest.json','source-analysis-manifest.json'),
        (source/'rooted-analysis-receipt.json','source-rooted-analysis-receipt.json')])
    destination.mkdir(parents=True,mode=0o700)
    for p,name in members:
        path=destination/name
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as out,p.open('rb') as src:
            for chunk in iter(lambda:src.read(1024*1024),b''):out.write(chunk)
        if sha(path)!=sha(p):raise RuntimeError('PUBLISHED_MEMBER_BYTES')
    jwrite(destination/'external-table-identities.json',external)
    write(destination/'PACKAGE.md',
        '# Raw-free report package\n\nThis is a small, byte-closed publication subset. '
        'Large per-request raw-free CSVs remain at the exact server2 paths in external-table-identities.json. '
        'They are not silently omitted from the full analysis: source-analysis-manifest.json and '
        'source-rooted-analysis-receipt.json bind the complete original package at '+str(source)+'. '
        'Those source manifests describe that original directory, not this subset. '
        'No raw prompts, target strings, generations, weights, cache, checkpoints or runtime logs are copied.\n')
    package_manifest=dict(source_report_path=str(source/'factual-report-ko.md'),
        source_report_sha256=receipt['report_sha256'],source_analysis_root=root,
        source_head=manifest['source']['head'],analysis_implementation=manifest.get('analysis_implementation'),
        completed_chains=manifest['complete_chains'],external_raw_free_tables=external,
        members=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(destination.iterdir())],
        raw_prompt_target_generation_model_cache_log_bytes=0,model_replay_count=0)
    jwrite(destination/'package-manifest.json',package_manifest)
    jwrite(destination/'rooted-package-receipt.json',dict(manifest_sha256=sha(destination/'package-manifest.json'),
        root_sha256=hashlib.sha256(json.dumps(package_manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        report_sha256=sha(destination/'factual-report-ko.md'),status='RAW_FREE_PUBLICATION_SUBSET_REHASH_PASS'))
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();print(package(a.source,a.destination))
