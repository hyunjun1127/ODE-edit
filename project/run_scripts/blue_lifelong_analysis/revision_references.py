"""Bounded publication/metadata-only baseline and pre-edit availability audit."""
import hashlib,subprocess
from .common import *
from .revision_labels import LABELS

S4=Path('experiment-reports/servers/server4')
V1=S4/'blue-l4-l8-lifelong-b100x100-review-2026-09-09-v1'
FIVE=S4/'blue-alphaedit-fivearm-sequential1000-review-2026-09-07-v1'
HIST=S4/'official-layer-realization-debt-lifelong-b100x100-2026-09-03-v6'
HIST3=S4/'official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3'
REF=Path('experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1')

def verify_package(repo,rel,full=True,selected=()):
    root=repo/rel;m=read(root/'analysis-manifest.json');inv=[]
    for r in m['members']:
        if not full and r['path'] not in selected:continue
        p=root/r['path'];assert p.is_file() and not p.is_symlink()
        assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],str(p)
        inv.append(dict(path=str(rel/r['path']),bytes=r['bytes'],sha256=r['sha256'],validation='PUBLICATION_MEMBER_REHASH; not new remote raw/state verification'))
    inv.append(dict(path=str(rel/'analysis-manifest.json'),bytes=(root/'analysis-manifest.json').stat().st_size,sha256=sha(root/'analysis-manifest.json'),validation='MANIFEST_FILE_IDENTITY'))
    return inv

def inspect(repo,out):
    inv=[]
    for p in [V1,FIVE,HIST]:inv+=verify_package(repo,p)
    inv+=verify_package(repo,HIST3,False,['execution-provenance.csv','arm-summary.csv'])
    def local(p):
        assert p.is_file() and not p.is_symlink();inv.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),validation='SMALL_SEALED_METADATA_ONLY'));return read(p)
    blue=local(BASE/'local/blue-lifelong-b100x100/attempt-v1/sample.lock.json')
    prior=local(repo/REF/'sample.lock.json')
    assert digest(blue['records'])==SAMPLE_ROOT and digest(prior['records'])==PREFIX_ROOT
    assert blue['records'][:1000]==prior['records']
    oldpath=BASE/'local/state/official-layer-realization-debt-lifelong-b100-v1/stream-seal-v1.json'
    old=local(oldpath);assert sha(oldpath)==read(repo/HIST3/'analysis-manifest.json')['stream']['sha256']
    b_ids=[str(r['case_id']) for r in blue['records']];o_ids=[str(r['case_id']) for r in old['training']]
    assert len(b_ids)==len(set(b_ids))==len(o_ids)==len(set(o_ids))==10000
    assert b_ids!=o_ids
    pfpath=BASE/'local/state/official-layer-realization-debt-lifelong-b100-v1/campaign-20260901-tech-r1/preflight.json'
    pf=local(pfpath)
    for hp in pf['models']['llama3-8b-inst']['hparams']:
        p=Path(hp['path']);assert sha(p)==hp['sha256'];inv.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),validation='HISTORICAL_NATIVE_CONFIG_BYTES'))
    # Only publications and small locked metadata are read: never scheduler/live outputs.
    root_hits=[]
    for p in (repo/'experiment-reports').rglob('*'):
        if p.suffix not in ('.json','.md') or p.stat().st_size>8_000_000:continue
        if SAMPLE_ROOT in p.read_text():root_hits.append(str(p.relative_to(repo)))
    scope=dict(search='Tracked experiment-reports JSON/MD <=8MB exact sample root, published v1/fivearm/v6 and small sample/preflight metadata',
               root_matches=root_hits,global_nonexistence_claim=False,matched_final10k_base='NOT_AVAILABLE_IN_AUDITED_ARTIFACTS',
               exact10k_W0='NOT_RECORDED_IN_BLUE_LIFELONG_ARTIFACTS',prefix_records_exact=1000,
               historical_unique=10000,blue_unique=10000,case_intersection=len(set(b_ids)&set(o_ids)),same_order=False,
               blue_sample_root=SAMPLE_ROOT,historical_sample_root=old['root_digest'],historical_order=old['training_order_sha256'],
               historical_collision_policy=old['collision_policy'],BLUE_collision_policy=blue['overwrite_policy'],
               new_GPU_evaluator_Slurm=0,original_raw_rehash='NOT_REPEATED; publication member rehash only',stopped_ORBODE_and_L567='NOT_ACCESSED')
    save(out/'baseline-availability-audit.json',scope)
    availability=[]
    for method in ['MEMIT','AlphaEdit']:
        availability.append(dict(method=method,reference='Base MEMIT' if method=='MEMIT' else 'Official AlphaEdit',scope='SAME_10K_FINAL_W100',status='NOT_AVAILABLE',reason='No measured base result with exact BLUE10000/order in audited publications; historical results not substituted'))
    availability.append(dict(method='PRE_EDIT',reference='Pre-edit W0',scope='SAME_FULL10000',status='NOT_RECORDED',reason='Existing W0 evaluation is full1000 only; no extrapolation/evaluation'))
    # Add exact prefix measurements, not a 10k baseline.
    fr=csvread(repo/FIVE/'final_metrics.csv');pr=csvread(repo/FIVE/'preedit.csv')
    official=next(r for r in fr if r['arm']=='O_NATIVE')
    pubofficial=next(r for r in csvread(repo/REF/'final_metrics.csv') if r['arm']=='O_NATIVE' and r['alias']=='llama3-8b-inst')
    for tag in MULT:
        for k in ['num','den','rate']:assert official[tag+'_'+k]==pubofficial[tag+'_'+k]
    # The three BLUE W0 measurements agree, but are not silently promoted to full10k.
    wr=[r for r in pr if r['arm'] in ['BLUE','BLUE_L4_ONLY','BLUE_L8_ONLY','PRE_EDIT_ORIGINAL_W0']]
    assert len(wr)==4
    prefix=[]
    for r in [wr[0],official]:
        q=dict(reference='Pre-edit W0 (1k)' if r in wr else 'Official AlphaEdit (1k)',scope='1k matched-prefix reference; not final10k',status='SEALED_PUBLICATION_REHASHED',raw_arm_id=r['arm'])
        for tag in MULT:
            for k in ['num','den','rate']:q[tag+'_'+k]=r[tag+'_'+k]
        prefix.append(q)
    cum=csvread(repo/V1/'cumulative-metrics.csv')
    for arm,label in LABELS.items():
        q=dict(reference=label,raw_arm_id=arm,scope='Lifelong W10 on first1000',status='V1_SEALED_AGGREGATE_REHASHED')
        for tag in MULT:
            r=next(r for r in cum if r['arm']==arm and r['batch']=='10' and r['metric']==tag)
            for a,b in [('num','numerator'),('den','denominator'),('rate','rate')]:q[tag+'_'+a]=r[b]
        prefix.append(q)
    for r in wr:
        for tag in MULT:assert r[tag+'_num']==prefix[0][tag+'_num'] and r[tag+'_den']==prefix[0][tag+'_den']
    hist=[]
    for r in csvread(repo/HIST/'counterfact-finalw-full10k.csv'):
        if r['arm'] not in ['LM','LA']:continue
        q=dict(reference='Base MEMIT (historical)' if r['arm']=='LM' else 'Official AlphaEdit (historical)',raw_arm_id=r['arm'],scope='HISTORICAL_DIFFERENT_STREAM_NOT_PAIRED',sample_root=old['root_digest'],order_hash=r['request_order_sha256'],status='SEALED_PUBLICATION_REHASHED')
        for tag in MULT:
            for a,b in [('num','numerator'),('den','denominator'),('rate',None)]:q[tag+'_'+a]=r[tag.lower()+'_'+b] if b else r[tag.lower()]
        for k in ['rewrite_acc_numerator','rewrite_acc_denominator','rephrase_acc_numerator','rephrase_acc_denominator']:q[k]=r[k]
        hist.append(q)
    compat=[]
    sources=csvread(repo/FIVE/'source-config-compatibility.csv')
    for r in sources:
        if r['arm'] not in ['O_NATIVE','BLUE']:continue
        compat.append(dict(reference='Official AlphaEdit prefix1k' if r['arm']=='O_NATIVE' else 'BLUE pre-edit prefix1k',**{k:r.get(k,'NOT_RECORDED') for k in ['source_head','model_revision','sample_root','config_sha','layers','context_hash','seed','dtype','gpu','writer_tokenizer','evaluator_tokenizer','projector_file_sha']},
                           evaluation='canonical NLL pair; final W10 own1000 / W0 own1000',comparison='SAME_PREFIX; different context/seed/source/GPU; no method-only causal delta',validation='Published source lock hashes; remote raw not rehashed'))
    prov=csvread(repo/HIST3/'execution-provenance.csv')
    for r in prov:
        if r['model']!='llama3-8b-inst':continue
        hp=next(h for h in pf['models'][r['model']]['hparams'] if Path(h['path']).name.startswith(r['method']+'-'))
        compat.append(dict(reference='Historical '+r['method'],source_head=r['source_head'],model_revision=pf['models'][r['model']]['revision'],sample_root=old['root_digest'],config_sha=hp['sha256'],layers='[4,5,6,7,8]',
             context_hash='NOT_REVERIFIED_FOR_HISTORICAL_REFERENCE',seed='order_seed_index=1; RNG equality NOT_ESTABLISHED',dtype='FULL_FP32 model; FP64 scalar reduction',gpu='historical runtime backend NOT_REVERIFIED',
             evaluation='v6 canonical NLL pairs on final frozen W100; evaluation jobs33306/33539 (not edit replay)',comparison='DIFFERENT_STREAM/ORDER/COLLISION_POLICY/CONFIG; NON_PAIRED; no causal delta',validation='v3/v6 publication + preflight/hparams rehash; no new raw/GPU parity'))
    source_findings=[]
    for head,path,terms in [
        ('77358b1546d1baf83b3e251afcce663b08d7bfd7','project/run_scripts/alpha_native_response_ode_v31_sequential/runtime.py',["if arm=='O_NATIVE':result=dict",'official=old._bootstrap_easyedit']),
        ('85a05d0a831db667fc68c27e634302eb30cc7d79','project/run_scripts/official_layer_realization_debt/runtime.py',['import run_official_memit_apply','import run_official_native_apply','return Method.MEMIT','def _run_apply']),
    ]:
        data=subprocess.check_output(['git','-C',str(repo),'show',head+':'+path]);sh=hashlib.sha256(data).hexdigest()
        for n,line in enumerate(data.decode().splitlines(),1):
            if any(t in line for t in terms):source_findings.append(dict(head=head,path=path,sha256=sh,line=n,expression=line.strip(),conclusion='BASE_ENTRYPOINT_DISPATCH; not model-level parity proof'))
    assert len(source_findings)>=5
    for name,rows in [('baseline-availability',availability),('prefix1000-references',prefix),('historical-base-references',hist),('reference-compatibility',compat),('reference-input-inventory',inv),('reference-source-dispatch',source_findings)]:csvwrite(out/(name+'.csv'),rows)
    return scope
