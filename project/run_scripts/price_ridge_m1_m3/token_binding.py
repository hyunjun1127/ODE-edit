"""Saved input/tokenizer CPU check for the four named GPT-J slots."""
import json
from pathlib import Path


def check():
    from transformers import AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
    from .phase0 import member
    p=Path('/data/janghj/ODE-edit/local/jlz-price-gptj-2k/checkpoint-repair-20261007')
    c=json.loads((p/'config.json').read_text());r=json.loads((p/'inputs/ready.json').read_text())
    tok=AutoTokenizer.from_pretrained(c['models']['MEMIT']['model'],local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='right'
    bench=CounterFactAdapter(tok,json.loads(Path(r['contexts']).read_text()))
    records=load_prefix(Path(c['stream']).parent,2000);result=[]
    for b,slot in ((3,47),(5,36),(7,26),(10,25)):
        pack=bench.prepare(records[(b-1)*100:b*100])
        path=p/'MEMIT_CAP075'/f'batch-{b:02d}'/'entry.json';entry=json.loads(path.read_text())
        if pack['identity']!=entry['native_pack']:raise ValueError('PHASE0_PACK_IDENTITY')
        row=pack['canonical_rows'][slot];rec=records[(b-1)*100+slot]['requested_rewrite']
        if pack['lookup'][row]!=0 or pack['row_request'][row]!=slot:raise ValueError('M1_POSITION_PREMISE')
        result.append(dict(batch=b,slot_zero_based=slot,canonical_row=row,lookup=0,
            pack_identity=pack['identity'],entry_member=member(path),
            subject_is_Spain=rec['subject']=='Spain',relation_is_P463=rec['relation_id']=='P463',
            subject_token_count=len(tok.encode(rec['subject'],add_special_tokens=False))))
    return dict(status='PASS_SAVED_PACK_BINDING',rows=result,new_model_load=0,new_forward=0,
        prefix_hidden_norm_mean='NOT_RECORDED',raw_prompt_or_tokens_published=False)


if __name__=='__main__':print(json.dumps(check(),ensure_ascii=False,indent=2))
