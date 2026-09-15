"""Only five sealed local W10 references plus published W0 prefix aggregate."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review_nogate import ROOT,read,ref,save,table,summary,panel_rows,transition,sha,key

def run(w,out):
    w,out=Path(w),Path(out);out.mkdir(parents=True,exist_ok=False)
    report=w/'experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1'
    sources=list(csv.DictReader((report/'source-config-compatibility.csv').open()))
    published=list(csv.DictReader((report/'cumulative-metrics.csv').open()))
    lock=read(ROOT/'execution.lock.json');ep=read(ROOT/'scientific-v1/B010/selected-evaluation.json')['seen_full']
    config=read(lock['config4']);context=read(lock['contexts'])
    labels={'AlphaEdit_L4_ONLY':'AlphaEdit_BLUE_L4_ONLY','AlphaEdit_ORIGINAL':'AlphaEdit_BLUE (L4+L8)',
        'MEMIT_ORIGINAL':'MEMIT_BLUE (L4+L8)','BASE_ALPHAEDIT':'BASE_ALPHAEDIT (blue=False L4-L8)',
        'BASE_MEMIT':'BASE_MEMIT (blue=False L4-L8)'}
    metrics=[];paired=[];compat=[];members=[]
    for arm,label in labels.items():
        x=next(z for z in sources if z['arm']==arm);base=Path(x['raw_root']);path=base/'B010/seen-full.json'
        if not path.exists():
            compat.append(dict(label=label,status='NOT_AVAILABLE',expected_path=str(path)));continue
        old=read(path);members.append(ref(path));hp=json.loads(x['hparams'])
        context_path=base/'B001/contexts.json';equal_context=read(context_path)==context if context_path.exists() else 'NOT_AVAILABLE'
        if context_path.exists():members.append(ref(context_path))
        identitymatched=True
        for m in ('RS','PS','NS'):
            rr=panel_rows(old,m);ee=panel_rows(ep,m)
            s=summary(rr,m,policy=label,state='W10_FULL_PREFIX1000',verification='NEW_LOCAL_JSON_REDUCTION_PLUS_SEALED_PUBLICATION')
            prior=next(y for y in published if y['arm']==arm and y['batch']=='10' and y['metric']==m)
            assert s['numerator']==int(prior['numerator']) and s['denominator']==int(prior['denominator'])
            metrics.append(s)
            if {key(z) for z in rr}=={key(z) for z in ee}:
                paired.append(transition(rr,ee,m,comparison=label+'_TO_EP_TW1',population='SAME_PROMPT_TARGET_FIRST1000_DIFFERENT_TRAJECTORY',
                    baseline=label,seed_matched=str(x['seed'])==str(lock['seed'])))
            else:identitymatched=False
        diffs={k:dict(reference=hp.get(k),EP=config.get(k)) for k in set(hp)|set(config) if hp.get(k)!=config.get(k)}
        compat.append(dict(label=label,status='AVAILABLE_W10_FIRST1000_NOT_FINAL10K',job=x['job'],
            model_revision=x['model_revision'],model_revision_equal=x['model_revision']==lock['model_revision'],
            sample_root=x['sample_root'],sample_root_equal=x['sample_root']=='5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729',
            case_prompt_target_hash_pairing=identitymatched,contexts_parsed_equal=equal_context,seed=x['seed'],EP_seed=lock['seed'],
            dtype=x['dtype'],attention=x['attention'],torch=x['torch'],transformers=x['transformers'],GPU=x['gpu'],
            tf32_matmul=x['tf32_matmul'],tf32_cudnn=x['tf32_cudnn'],layers=hp['layers'],blue=hp['blue'],L2=hp.get('L2','NA'),
            native_scalar_policy=x.get('native_scalar_policy','MEMIT FP64 solve/Alpha native FP32'),
            tokenizer=x['tokenizer'],evaluator_tokenizer=x['evaluator_tokenizer'],
            source_head=x['source_head'],source_archive_sha256=x['source_archive_sha256'],runtime_sha256=x['runtime_sha256'],
            config_sha256=x['config_sha256'],hparam_differences=json.dumps(diffs,sort_keys=True),
            same_state_causal_comparison=False,C4_D64='NOT_AVAILABLE_NO_NEW_EVALUATION',baseline_raw_sha256=sha(path)))
    w0=w/'experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v3/w0-distributions.csv'
    members.extend([ref(w0),ref(report/'cumulative-metrics.csv'),ref(report/'source-config-compatibility.csv')])
    for x in csv.DictReader(w0.open()):
        if x['prefix_requests']!='1000':continue
        metrics.append(dict(policy='PRE_EDIT W0 (SH2 common reference)',state='W0_FIRST1000',metric=x['metric'],
            numerator=int(x['numerator']),denominator=int(x['denominator']),percent=float(x['rate'])*100,
            new_nll_mean=x['new_nll_prompt_mean'],true_nll_mean=x['true_nll_prompt_mean'],
            verification='SEALED_AGGREGATE_REUSED_NO_PROMPT_RECONSTRUCTION'))
    compat.append(dict(label='PRE_EDIT W0',status='SEALED_FIRST1000_PUBLICATION_AVAILABLE',job='42673 SH2',
        case_prompt_target_hash_pairing='AGGREGATE_ONLY_NO_NEW_W0_PAIRING',same_state_causal_comparison=False,
        C4_D64='FIXED_TEACHER_NOT_BASELINE_COUNTERFACT_EVAL',model_revision=lock['model_revision']))
    table(out/'baseline-first1000.csv',metrics);table(out/'baseline-paired.csv',paired);table(out/'compatibility.csv',compat)
    save(out/'baseline-input-manifest.json',dict(members=members,
        forbidden_population_excluded=['W50_to_W60_full6000','W100_final10k','unrelated older methods'],
        scope='FIVE_SAVED_W10_LOCAL_JSONS_PLUS_EXISTING_W0_PREFIX_AGGREGATE',remote_transfers=0,new_evaluations=0))
    print(json.dumps({'rows':len(metrics),'paired_rows':len(paired),'identities':[x.get('case_prompt_target_hash_pairing') for x in compat]}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.worktree,a.output)
