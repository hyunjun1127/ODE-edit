"""Pre-outcome fixed panels; raw text/tokens stay local-only."""
import collections
import gzip
import heapq
import json
from pathlib import Path
import re
import sys
import unicodedata
from .common import ROOT, DATA, DEPS, MODEL, PANELS, INSTRUCTION, digest, read, save, file_record, sha

SEED = 'NATIVE-DELAYED-WRITE-E3|20260924|v1'
FULL_CF = Path('/data/janghj/EasyEdit/data/counterfact/counterfact.json')
C4 = Path('/data/janghj/ODE-edit/local/bg1-c4-ours-first/20260915-v1/attempt-v1/acquisition/c4-validation.00000-of-00008.json.gz')


def norm(s):
    return re.sub(r'[^\w]+', ' ', unicodedata.normalize('NFKC', str(s)).casefold()).strip()


def fact(r):
    q = r['requested_rewrite']
    return norm(q['subject']), q['relation_id']


def prompt(r):
    q = r['requested_rewrite']
    return q['prompt'].format(q['subject'])


def encode_target(tok, text):
    ids = list(tok.encode(text if text.startswith(' ') else ' ' + text, add_special_tokens=False))
    while ids and ids[0] in {tok.bos_token_id, tok.unk_token_id}:
        ids.pop(0)
    assert ids
    return ids


def make_pair(tok, r, panel, kind, index, text, extra=None):
    q = r['requested_rewrite']
    pair = dict(panel=panel, kind=kind, case_id=r['case_id'], subject=q['subject'], relation=q['relation_id'], prompt_index=index,
                prompt=text, true_target=q['target_true']['str'], new_target=q['target_new']['str'], fact=list(fact(r)))
    pair.update(extra or {})
    pair['pair_id'] = digest([panel, kind, r['case_id'], index, text, pair['true_target'], pair['new_target']])
    pair['prompt_cluster'] = digest(norm(text))
    rows = []
    pids = list(tok(text, add_special_tokens=True)['input_ids'])
    for label in ('true', 'new'):
        target = pair[label+'_target']
        tids = encode_target(tok, target)
        row = dict(pair, label=label, prompt_ids=pids, target_ids=tids, input_ids=(pids+tids)[:-1], positions=list(range(len(pids)-1, len(pids)+len(tids)-1)))
        row['row_id'] = digest([pair['pair_id'], label, row['input_ids'], tids, row['positions']])
        rows.append(row)
    return rows


def build():
    sys.path.insert(0, str(DEPS))
    from transformers import AutoTokenizer
    import transformers
    assert transformers.__version__ == '4.44.2'
    assert sha(DATA) == '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
    assert sha(FULL_CF) == 'd017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f'
    assert sha(C4) == '1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f'
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    stream = read(DATA)
    full = read(FULL_CF)
    assert len(stream) == 10000
    rows = []; selected_n = []
    for quintile in range(5):
        pool = stream[quintile*2000:(quintile+1)*2000]
        chosen = sorted(pool, key=lambda r: digest([SEED, 'N', r['case_id']]))[:20]
        for r in chosen:
            assert len(r['neighborhood_prompts']) == 10
            selected_n.append(r['case_id'])
            for i, text in enumerate(r['neighborhood_prompts']):
                rows.extend(make_pair(tok, r, 'N_diag1000', 'N', i, text, dict(quintile=quintile)))
    for r in stream[:100]:
        rows.extend(make_pair(tok, r, 'H_diag_B1_R100_P200', 'R', 0, prompt(r)))
        assert len(r['paraphrase_prompts']) == 2
        for i, text in enumerate(r['paraphrase_prompts']):
            rows.extend(make_pair(tok, r, 'H_diag_B1_R100_P200', 'P', i, text))
    stream_subjects = {norm(r['requested_rewrite']['subject']) for r in stream}
    stream_facts = {fact(r) for r in stream}
    stream_prompts = {norm(s) for r in stream for s in [prompt(r), *r['paraphrase_prompts'], *r['neighborhood_prompts']]}
    relations = {r['requested_rewrite']['relation_id'] for r in stream}
    object_count = collections.Counter(r['requested_rewrite']['target_new'].get('id', r['requested_rewrite']['target_new']['str']) for r in stream)
    groups = collections.defaultdict(list); excluded = collections.Counter()
    seen_subjects = set()
    for r in sorted(full, key=lambda r: digest([SEED, 'Base', r['case_id']])):
        q = r['requested_rewrite']; subject = norm(q['subject'])
        # Same canonical subject with punctuation/case/unicode variations is one entity.
        # No external entity/alias resolver was provided: expose this limitation.
        if subject in stream_subjects or fact(r) in stream_facts:
            excluded['subject_or_fact'] += 1; continue
        if norm(prompt(r)) in stream_prompts:
            excluded['prompt'] += 1; continue
        if subject in seen_subjects:
            excluded['duplicate_base_subject'] += 1; continue
        seen_subjects.add(subject)
        obj = q['target_true'].get('id', q['target_true']['str'])
        n = object_count[obj]
        group = 'other_relation' if q['relation_id'] not in relations else ('same_relation_zero_object' if n == 0 else 'same_relation_low_object' if n <= 5 else 'same_relation_high_object')
        groups[group].append((r, n))
    # Availability-balanced deterministic round robin; no performance/teacher consulted.
    selected = []
    names = sorted(groups)
    offsets = {g: 0 for g in names}
    while len(selected) < 256:
        added = False
        for g in names:
            j = offsets[g]
            if j < len(groups[g]) and len(selected) < 256:
                r, n = groups[g][j]; selected.append((r, g, n)); offsets[g] += 1; added = True
        if not added:
            raise RuntimeError(('BLOCKED_ASSET_BASE_SUBJECT_DISJOINT', len(selected)))
    for r, g, n in selected:
        rows.extend(make_pair(tok, r, 'BaseEval256', 'BASE', 0, prompt(r), dict(exposure_stratum=g, edited_object_exposure=n)))
    # GeneralEval is a fresh diagnostic selection of existing C4 validation text,
    # not old generated W0 answer capsules or a train bank. Select by document SHA.
    heap = []; docs = set(); scanned = 0
    with gzip.open(C4, 'rt') as f:
        for line_no, line in enumerate(f):
            doc = json.loads(line); text = doc['text']; h = digest(text)
            if h in docs or len(text.split()) < 160:
                continue
            docs.add(h); scanned += 1
            rank = int(digest([SEED, 'General', h]), 16)
            item = (-rank, line_no, h, text)
            if len(heap) < 512: heapq.heappush(heap, item)
            elif item[0] > heap[0][0]: heapq.heapreplace(heap, item)
    general = []
    for neg, line_no, h, text in sorted(heap, reverse=True):
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < 128: continue
        # BOS plus 128 natural tokens; observe next natural token on positions64..127.
        seq = [tok.bos_token_id] + ids[:128]
        row = dict(panel='GeneralEval128', kind='GENERAL', label='natural', case_id='C4:'+h,
                   subject='C4:'+h, prompt_cluster=h, document_sha256=h, source_line=line_no,
                   input_ids=seq[:-1], positions=list(range(64,128)), target_ids=seq[65:129],
                   token_window=[0,128], prompt=text, fact=None)
        row['pair_id'] = digest(['GeneralEval128', h]); row['row_id'] = digest([row['pair_id'], seq, row['positions']])
        general.append(row)
        if len(general) == 128: break
    assert len(general) == 128
    rows += general
    active_variants = set()
    for batch in (1,5,10,20,30,40,50,60,70,80,90,100):
        latest = {fact(r):r for r in stream[:batch*100]}
        for r in stream[:100]:
            now = latest[fact(r)]
            if now['requested_rewrite']['target_new'] == r['requested_rewrite']['target_new']:
                continue
            identity = (r['case_id'], digest(now['requested_rewrite']['target_new']))
            if identity in active_variants: continue
            active_variants.add(identity)
            clone = dict(r, requested_rewrite=dict(r['requested_rewrite'], target_new=now['requested_rewrite']['target_new']))
            for kind, pi, text in [('R',0,prompt(r))]+[('P',i,x) for i,x in enumerate(r['paraphrase_prompts'])]:
                rows.extend(make_pair(tok,clone,'H_active_variants',kind,pi,text,dict(active_target_hash=identity[1],target_origin_case=now['case_id'])))
    # Stable homogeneous categories for bounded original head/microbatch ordering.
    rows.sort(key=lambda r: (r['panel'],r['kind'],r['label'],str(r['case_id']),r.get('prompt_index',0)))
    assert len({r['row_id'] for r in rows}) == len(rows) == 3240 + 6*len(active_variants)
    inputs = PANELS
    panel = save(inputs/'rows.json', rows)
    ledger = []
    for batch in (0,1,5,10,20,30,40,50,60,70,80,90,100):
        latest = {fact(r):r for r in stream[:batch*100]}
        for r in stream[:100]:
            active = latest.get(fact(r))
            ledger.append(dict(batch=batch,case_id=r['case_id'],fact=list(fact(r)),
                status='NOT_RECEIVED' if active is None else 'ACTIVE' if active['case_id']==r['case_id'] else 'SUPERSEDED',
                active_case_id=None if active is None else active['case_id'],
                active_target=None if active is None else active['requested_rewrite']['target_new'],
                original_target=r['requested_rewrite']['target_new']))
    ledger_rec = save(inputs/'active-overwrite-ledger.json', ledger)
    # A small fidelity panel must preserve the original first16 batch/microbatch composition.
    fidelity = []
    for label in ('true','new'):
        fidelity += [rr for r in stream[:16] for rr in make_pair(tok,r,'G10_original_first16','R',0,prompt(r)) if rr['label']==label]
    fidelity_rec = save(inputs/'fidelity-rows.json', fidelity)
    manifest = dict(instruction_id=INSTRUCTION, status='SEALED_BEFORE_OUTCOMES', seed=SEED, source=[file_record(DATA), file_record(FULL_CF), file_record(C4)],
        rows=panel, active_overwrite=ledger_rec, fidelity=fidelity_rec, row_count=len(rows), pair_counts=dict(N_diag1000=1000,H_diag_R=100,H_diag_P=200,BaseEval256=256,GeneralEval128=128),
        selected_N_case_ids=selected_n, BaseEval_strata=dict(collections.Counter(g for r,g,n in selected)), BaseEval_pool={g:len(v) for g,v in groups.items()}, excluded=dict(excluded),
        subject_alias_check='NFKC+casefold+punctuation canonical names; external entity aliases NOT_AVAILABLE; no stronger claim',
        fit_bank='NOT_USED_DIAGNOSTIC_ONLY', GeneralEval_documents_scanned=scanned, GeneralEval_tokens_per_document=64, GeneralEval_full_vocab_kl=True,
        H_active_supplementary_variants=len(active_variants), H_active_policy='select original or active variant by prefix ledger; fixed original denominator remains separate',
        previous_panel_manifest=file_record(ROOT/'inputs/panels/panel-manifest.json'),
        patch_positions='ALL_VALID_TF_INPUT_TOKENS_PAD_EXCLUDED', patch_rotation_seeds=[2026092401,2026092402,2026092403],
        evaluation_microbatch=16, tokenizer=dict(path=str(MODEL),special_tokens=True,pad=tok.pad_token_id,bos=tok.bos_token_id,eos=tok.eos_token_id),
        new_model_calls=0, new_native_fits=0, raw_not_for_git=True)
    print(save(inputs/'panel-manifest.json', manifest))


if __name__ == '__main__': build()
