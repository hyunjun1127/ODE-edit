"""Outcome-independent exact relation-template neighbor subject locator."""
from collections import defaultdict
import torch
from project.run_scripts.jlz_pilot.prompts import subject_last
from .common import require,digest


def panel(records,all_records,bench):
    templates=defaultdict(set)
    for record in all_records:
        rw=record['requested_rewrite'];template=rw['prompt']
        if template.count('{}')==1 and template!='{}':templates[rw['relation_id']].add(template)
    rows=[]
    for record in records:
        rw=record['requested_rewrite']
        for index,prompt in enumerate(record['neighborhood_prompts']):
            identity=digest([record['case_id'],'N',index,prompt,rw['target_new']['str'],rw['target_true']['str']])
            matches={}
            for template in sorted(templates[rw['relation_id']]):
                prefix,suffix=template.split('{}')
                if not prompt.startswith(prefix) or (suffix and not prompt.endswith(suffix)):continue
                end=len(prompt)-len(suffix) if suffix else len(prompt)
                subject=prompt[len(prefix):end]
                if not subject or template.format(subject)!=prompt:continue
                pos=subject_last(bench.tokenizer,template,subject)
                p=bench.evaluation_ids(prompt,rw['target_new']['str'])[0]
                prefix_ids=list(bench.tokenizer.encode(prefix+subject))
                # Native subject_last token location must be in PROMPT, not target.
                if not (0<=pos<len(p) and p[:pos+1]==prefix_ids):continue
                matches.setdefault((subject,pos),[]).append(template)
            row=dict(case_id=record['case_id'],kind='N',prompt_index=index,identity=identity,
                prompt_sha=digest(prompt),relation_id=rw['relation_id'],match_count=len(matches))
            if len(matches)==1:
                (subject,pos),used=next(iter(matches.items()))
                row.update(identifiable=True,subject=subject,lookup=pos,templates=used,
                    reason='EXACT_RELATION_TEMPLATE_NATIVE_SUBJECT_LAST')
            else:row.update(identifiable=False,reason='AMBIGUOUS_EXACT_TEMPLATES' if matches else 'NO_EXACT_TEMPLATE_OR_TOKEN_PREFIX')
            for label in ('new','true'):
                p,t=bench.evaluation_ids(prompt,rw['target_'+label]['str'])
                row[label+'_token_identity']=digest([p,t]);row[label+'_input_length']=len(p)+len(t)-1
                if row['identifiable']:require(row['lookup']<len(p),'LOOKUP_NOT_IN_PROMPT')
            rows.append(row)
    require(len(rows)==sum(len(r['neighborhood_prompts']) for r in records),'LOOKUP_DENOMINATOR')
    return dict(rows=rows,denominator=len(rows),identifiable=sum(r['identifiable'] for r in rows),
        method='all fixed10k same-relation exact templates; unique subject+token locator; no fuzzy fallback',
        template_source=digest({k:sorted(v) for k,v in templates.items()}),model_results_used=False,
        common_four_mask_subset=[r['identity'] for r in rows if r['identifiable']])


def masks(encoded,lookups,width,device):
    require(len(encoded)==len(lookups),'LOOKUP_ROW_COUNT')
    valid=torch.zeros((len(encoded),width),dtype=torch.bool,device=device);subject=torch.zeros_like(valid)
    for i,((prompt,target),pos) in enumerate(zip(encoded,lookups)):
        require(type(pos) is int and 0<=pos<len(prompt),'LOOKUP_NOT_PROMPT')
        length=len(prompt)+len(target)-1;offset=width-length
        require(offset>=0,'MASK_WIDTH');valid[i,offset:]=True;subject[i,offset+pos]=True
    other=valid & ~subject
    require(not bool((subject&other).any()) and torch.equal(subject|other,valid),'MASK_PARTITION')
    return dict(NONE=torch.zeros_like(valid),SUBJECT_ONLY=subject,NONSUBJECT_ONLY=other,ALL=valid)
