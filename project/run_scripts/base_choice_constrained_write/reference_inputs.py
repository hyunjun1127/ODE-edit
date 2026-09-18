"""CPU-only R512 input selection; no model, score, answer or future-edit access."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from project.run_scripts.bg_tw_reference import builder as b
from .provenance import ROOT, create_json, sha

OLD = Path('/data/janghj/ODE-edit/local/bg1-c4-ours-first/20260915-v1/attempt-v1/reference-v1')
EXPECTED = 'f5791dd3c986261a252796bd5d7293ece46339d261609449dcfe674970bbfde0'

def build():
    started=time.monotonic()
    out=ROOT/'reference-inputs-v2'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    sources=json.loads((OLD/'source-manifest.json').read_text())
    splits=json.loads((OLD/'splits.json').read_text())
    assert sources['reference_identity_sha256']==splits['reference_identity_sha256']==EXPECTED
    oldseal=json.loads((OLD/'build-status.json').read_text())
    for name in ('reference-documents.jsonl','reference-tokens.npz','source-manifest.json','splits.json'):
        match=next(m for m in oldseal['members'] if Path(m['path']).name==name)
        if sha(OLD/name)!=match['sha256']:raise ValueError('REFERENCE_INPUT_SHA:'+name)
    contract_path=Path(sources['contract']['path'])
    if sha(contract_path)!=sources['contract']['sha256']:raise ValueError('C4_CONTRACT_SHA')
    contract=json.loads(contract_path.read_text());b.validate_contract(contract)
    # Input-only metadata is used for exclusion, never Report answer/model evaluation.
    metadata=[json.loads(line) for line in (OLD/'reference-documents.jsonl').read_text().splitlines()]
    with np.load(OLD/'reference-tokens.npz',allow_pickle=False) as z:
        oldids=z['input_ids'].copy();rowids=z['source_row_ids'].tolist();roles=z['split_roles'].tolist()
    assert len(metadata)==len(oldids)==768 and oldids.shape==(768,257)
    previous=[]
    for i,m in enumerate(metadata):
        assert m['source_row_id']==rowids[i] and m['split_role']==roles[i]
        assert b.digest(oldids[i].astype('<i8').tobytes())==m['window_input_sha256']
        previous.append(b.Document(m,oldids[i].tolist(),b.Fingerprint.make(
            m['raw_text'],m['decoded_window_for_diagnostics_only'],m['canonical_url'])))
    # Seal immutable sampling policy before running selection or any W0 decoding.
    policy=dict(existing_train_roles=['S64','Reserve320'],existing_train_count=384,
        extra_count=128,seed=b.SEED,corpus_revision=b.REVISION,
        extra_order='same whole pinned train shard lowest document-priority SHA, stable source-ID tie',
        retain=16384,initial_limit=8192,window_rule='original C4-WebRef-v2 window hash',
        duplicate_rules='same normalized URL/full/window + exact13gram Jaccard>=.8; exclude all old768 and previous extra',
        prompt='original BOS plus first128 natural tokens of exact locked256-token window',
        native_current_or_future_access=False,model_outcome_access=False,
        new_overlap_allowlist=['Wiki128'],
        legacy_overlap_excluded='CounterFact-first1000-input-only contains future and official P/N; not used by v2 selection',
        original_S64_Reserve320='unchanged prior identity; historical sampler provenance not erased',
        Report256_access='INPUT_ONLY_EXCLUSION_FINGERPRINTS; no capsule/model/evaluation',
        fixed8='lowest SHA BPCW-v2|20260918|spot|source_row_id',
        nested256='lowest SHA BPCW-v2|20260918|nested|source_row_id; metadata only')
    create_json(out/'selection-policy.json',policy)
    from transformers import AutoTokenizer
    tokpath=sources['tokenizer']['path']
    tokenizer=AutoTokenizer.from_pretrained(tokpath,local_files_only=True)
    tokidentity=b.tokenizer_identity(tokpath,tokenizer)
    if tokidentity['member_root']!=sources['tokenizer']['member_root']:raise ValueError('TOKENIZER_IDENTITY')
    spec=next(s for s in contract['corpus']['source_files'] if s['split']=='train')
    original=next(s for s in sources['source_files'] if s['split']=='train')
    pool,scan=b.scan_shard(original['path'],dict(spec,full_file_sha256=original['sha256']),retain=16384,corpus=contract['corpus'])
    overlap_path=Path(sources['overlap_input']['path'])
    if sha(overlap_path)!=sources['overlap_input']['sha256']:raise ValueError('OVERLAP_INPUT_SHA')
    payload=json.loads(overlap_path.read_text())
    filtered=dict(schema_version=1,inspected_scopes=['Wiki128'],unavailable_scopes=['CounterFact omitted by BPCW no-future/no-PN firewall'],
        evaluator_inputs=[r for r in payload['evaluator_inputs'] if r['scope']=='Wiki128'],diagnostic_inputs=[])
    extra,counts=b.select_documents(pool,tokenizer,contract,needed=128,initial_limit=8192,
        previous=previous,overlap=b.OverlapChecker(filtered))
    if len(extra)!=128:raise ValueError('INSUFFICIENT_EXTRA128')
    train=[d for d in previous if d.metadata['split_role'] in ('S64','Reserve320')]+extra
    dev=[d for d in previous if d.metadata['split_role']=='Dev128']
    assert len(train)==512 and len(dev)==128
    ids=set();tokens=set();rows=[]
    for role,docs in (('R512',train),('Dev128',dev)):
        for i,doc in enumerate(docs):
            rid=doc.metadata['source_row_id'];prompt=doc.input_ids[:129]
            tokenhash=b.digest(np.array(prompt,dtype='<i8').tobytes())
            if rid in ids or tokenhash in tokens:raise ValueError('R_DEV_INPUT_DUPLICATE')
            ids.add(rid);tokens.add(tokenhash)
            rows.append(dict(role=role,ordinal=i,source_row_id=rid,input_ids=prompt,
                prompt_token_sha256=tokenhash,window_token_sha256=doc.metadata['window_input_sha256'],
                source_role=doc.metadata.get('split_role','AdditionalTrain128'),
                source_text_sha256=doc.metadata['raw_text_sha256'],window_start=doc.metadata['window_start']))
    create_json(out/'inputs.json',rows)
    # Additional raw provenance remains local, never enters compact publication.
    create_json(out/'additional128-provenance.json',[d.metadata for d in extra])
    trainids=[r['source_row_id'] for r in rows if r['role']=='R512']
    choose=lambda salt,n:sorted(trainids,key=lambda x:(b.sha_text(f'BPCW-v2|20260918|{salt}|{x}'),x))[:n]
    receipt=dict(status='INPUTS_SEALED_ANSWERS_NOT_GENERATED',train=512,dev=128,inputs_sha256=sha(out/'inputs.json'),
        old_reference_identity=EXPECTED,oldsource_sha256=sha(OLD/'source-manifest.json'),
        additional_scan=scan,selection_counts=counts,tokenizer=tokidentity,
        policy_sha256=sha(out/'selection-policy.json'),nested256=choose('nested',256),fixed8=choose('spot',8),
        no_Report_answer_or_model_access=True,model_forwards=0,cpu_seconds=time.monotonic()-started)
    create_json(out/'manifest.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','train','dev','inputs_sha256','cpu_seconds')},indent=2))

if __name__=='__main__':build()
