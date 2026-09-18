"""Bounded CPU artifact audit; no model loading or new forward evaluation."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import torch
from review import ARMS, read, dump, sha, csvout, arm_ledger


def tensor_sha(t, header=False):
    h=hashlib.sha256()
    if header: h.update(f'{tuple(t.shape)}|{t.dtype}|'.encode())
    h.update(t.contiguous().numpy().tobytes()); return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); out=args.input; dest=args.output; torch.set_num_threads(8)
    task=out.parents[2]; imported=task/'imports/server4-b001-r1/sealed'
    old=imported/'single-layer-edit-preserving-correction/20260918-v1/M/attempt-metadata-r1/episodes/b001/attempt-v1'
    capsule=imported/'local-z-adaptive-allocation/20260916-v1/arms/N4/attempt-v1/output/B001/proposals/own-N4.pt'
    native=torch.load(capsule,weights_only=True,map_location='cpu',mmap=True)
    wn=native['weight']; native_sha=tensor_sha(wn)
    rows=[]; members=[]; events=[]; state=[]
    seen=set()
    def member(f, expected=None):
        f=Path(f); assert f.is_file() and not f.is_symlink()
        got=dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f),mode=oct(f.stat().st_mode & 0o777))
        if expected:
            assert got['bytes']==expected['bytes'] and got['sha256']==expected['sha256'], str(f)
        if str(f) not in seen: members.append(got); seen.add(str(f))
        return got
    member(capsule)
    ids=None; context=None; w0sha=None
    for a in ARMS:
        ledger,endpoint,reuse=arm_ledger(out,a); parent=endpoint.parent
        endpoint_member=member(endpoint,ledger['endpoint'])
        lmember=member(parent/'selection-ledger.json');seal=read(parent/'selection-seal.json');member(parent/'selection-seal.json')
        assert lmember['sha256']==seal['selection_ledger_sha256']
        saved=torch.load(endpoint,weights_only=True,map_location='cpu',mmap=True);w=saved['weight']
        assert list(w.shape)==[4096,14336] and w.dtype==torch.float32 and torch.isfinite(w).all()
        rawsha=tensor_sha(w);headersha=tensor_sha(w,True)
        assert rawsha==seal['endpoint_weight_sha256'] and headersha==ledger['selected_weight_sha256']
        assert saved['WN_sha']==native_sha and saved['history']==0 and saved['episode']==0
        assert saved['weight_name']=='model.layers.4.mlp.down_proj.weight'
        if ids is None: ids=saved['case_ids'];context=saved['context_identity'];w0sha=saved['W0_sha']
        assert saved['case_ids']==ids and saved['context_identity']==context and saved['W0_sha']==w0sha
        assert len(ids)==len(set(ids))==100
        obs=read(out/f'observers/{a}.json')
        assert obs['selection_seal']['endpoint_weight_sha256']==rawsha
        assert obs['selection_seal_verified_before_P_N_access']
        assert obs['RNG_unchanged_and_restored'] and obs['entry_selected_weight_restored_exact']
        delta=w.double()-wn.double();norm=float(delta.norm());expected=ledger['actual_delta_norm']
        assert abs(norm-expected)<1e-10*max(1,norm)
        rows.append(dict(arm=a,optimization=reuse,endpoint_path=str(endpoint),file_sha256=endpoint_member['sha256'],
            bytes=endpoint_member['bytes'],tensor_raw_sha256=rawsha,tensor_header_sha256=headersha,shape='4096x14336',dtype='float32',finite=True,
            correction_norm=norm,correction_changed_fraction=float(torch.count_nonzero(delta))/delta.numel(),
            correction_maxabs=float(delta.abs().max()),native_delta_norm=native['receipt']['actual_delta_norm'],
            cumulative_W_minus_W0_norm='NOT_RECORDED_LOCAL_W0_TENSOR_ABSENT',
            context_sha=context,WN_sha=native_sha,W0_sha=w0sha,history_appends=0,CPU_save_reload='PASS_NOT_GPU_CONTINUATION'))
        for ref in ledger.get('event_refs',[]):
            # Original S4 absolute paths may only be mapped into the sealed local import.
            f=parent/'events'/Path(ref['path']).name; member(f,ref)
            event=read(f);record=event['record']
            if record['event'] in ('direction','gradient_observed','quality_guard_checked','actual_invariant_checked','proposal_geometry_checked','controller_start'):
                scalars={k:v for k,v in record.items() if isinstance(v,(str,int,float,bool)) or v is None}
                details=record.get('details',{})
                scalars.update({f'detail_{k}':v for k,v in details.items() if isinstance(v,(str,int,float,bool)) or v is None})
                scalars['detail_reasons']=json.dumps(details.get('reasons',[]))
                events.append(dict(scalars,arm=a,source_event_sha=sha(f)))
        state.append(dict(arm=a,**ledger.get('counters',{}),reject_reasons=json.dumps(dict(Counter(t['reason'] for t in ledger.get('trials',[]) if not t['accepted'])),sort_keys=True)))
        del saved,w,delta
    assert rows[0]['tensor_raw_sha256']==rows[1]['tensor_raw_sha256']==rows[2]['tensor_raw_sha256']
    before=read(out/'nonselected-before.json');after=read(out/'nonselected-after.json');assert before==after
    ep=read(out/'episode-integrity.json');terminal=read(out/'terminal.json')
    assert not(out/'failure.json').exists() and terminal['status']=='COMPLETE_WITH_T_SKIPPED'
    assert terminal['history_appends']==terminal['native_fit_new']==0
    assert ep['independent_next_episode_reset']['W']==w0sha
    newmeta=read(out/'protected-provenance.json');oldmeta=read(old/'protected-provenance.json')
    prefix_alias_equal=newmeta['key_aliases']==oldmeta['key_aliases']
    assert prefix_alias_equal
    def strip_actual(d):return [{k:v for k,v in x.items() if k!='actual_key_column'} for x in d['actual_key_aliases']]
    assert strip_actual(newmeta)==strip_actual(oldmeta)
    binding=dict(native_capsule_sha=sha(capsule),native_WN_sha=native_sha,request_count=100,unique_primary_weights=len(set(r['tensor_raw_sha256'] for r in rows)),
        nonselected_before_after_manifest_equal=True,nonselected_evidence='RUNTIME_HASH_RECEIPTS_ONLY_NOT_CPU_FULL_MODEL_RELOAD',
        endpoint_count=8,history_appends=0,native_fit_new=0,full_numerical_validation='NOT_ESTABLISHED',
        protected_token_prefix_aliases_equal=prefix_alias_equal,valid_position_inventory=len(newmeta['key_aliases']),
        S4_distinct_FP32_columns=oldmeta['actual_distinct_key_columns'],S1_distinct_FP32_columns=newmeta['actual_distinct_key_columns'],
        K_payload_byte_equal=False,cross_hardware_numerical_equivalence='NOT_TESTED',
        RAND_final_weight_files='NOT_RETAINED_SEED_FACTORS_NORM_HASH_ONLY',W0_tensor_local='NOT_AVAILABLE_NO_MODEL_READ')
    geometry=[]
    for name in ['CA','EN-S','EN-F','CA-EXACT']:
        d=read(out/f'geometry/{name}.json');g=read(out/f'geometry/{name}-gradient.json')
        geometry.append(dict(arm=name,**{k:v for k,v in d.items() if isinstance(v,(str,int,float,bool)) or v is None},**g))
    csvout(dest/'endpoint-artifacts.csv',rows);csvout(dest/'controller-events.csv',events);csvout(dest/'controller-counts.csv',state)
    csvout(dest/'geometry.csv',geometry);dump(dest/'artifact-checks.json',binding)
    # File-level hashing of saved factors/gradients, no redundant teacher/model rehash.
    for root in [out/'geometry',out/'gradients',old/'geometry',old/'gradients']:
        for f in sorted(root.glob('*.pt')):member(f)
    for f in [out/'nonselected-before.json',out/'nonselected-after.json',out/'episode-integrity.json',out/'ALL_SELECTIONS_SEALED.json',out/'terminal.json',old/'protected-provenance.json']:
        member(f)
    dump(dest/'artifact-manifest.json',members)
    print(json.dumps(binding,indent=2))


if __name__=='__main__':main()
