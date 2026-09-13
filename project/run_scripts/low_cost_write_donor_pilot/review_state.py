"""CPU-only independent saved-state audit; no model/evaluator import."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def tsha(t, header=True):
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    if header:
        h.update(str((str(t.dtype), list(t.shape))).encode())
    b = t.reshape(-1).view(torch.uint8).numpy()
    for i in range(0, b.size, 8 << 20):
        h.update(memoryview(b[i:i+(8 << 20)]))
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def audit(root):
    torch.set_num_threads(4)
    root = Path(root)
    read = lambda p: json.loads((root / p).read_text())
    load = lambda p: torch.load(root / p, map_location='cpu', weights_only=True, mmap=True)
    cap, term, hist = read('comparison-capsule.json'), read('core-terminal.json'), read('history-provenance.json')
    pre = load('prepared.pt')
    # Whole-file hashes are owned by the independent package inventory pass.
    assert digest(pre['rng']) == cap['entry_state']['rng']
    assert digest(pre['contexts']) == cap['entry_state']['contexts']
    assert pre['metadata']['P4']['source_index'] == 0 and pre['metadata']['P4']['local_index'] == 0
    assert pre['metadata']['P8']['source_index'] == 4 and pre['metadata']['P8']['local_index'] == 0
    assert len(hist['steps']) == 50 and sum(x['events'] for x in hist['steps']) == 5000
    assert [x['batch'] for x in hist['steps']] == list(range(1, 51))
    assert hist['alpha_reconstructions'] == 1
    assert tsha(pre['M8']) == hist['M8_sha256'] == cap['entry_state']['M8']
    native = load('N4/endpoint.pt')['weights'][4]
    reports = []
    for arm in ('N4', 'S875', 'S75', 'FULL8', 'RES8', 'REFIT4'):
        c, e = read(arm+'/commit.json'), read(arm+'/evaluation.json')
        p = load(arm+'/endpoint.pt')
        assert e['endpoint_state'] == c['state'] == p['metadata']['state']
        assert digest(p['rng']) == c['state']['rng'] == cap['entry_state']['rng']
        assert digest(p['contexts']) == c['state']['contexts'] == cap['entry_state']['contexts']
        assert e['evaluation_nonmutation'] is True
        assert sha(root/arm/'evaluation.json') == term['endpoints'][arm]['evaluation']['sha256']
        row = {'arm': arm, 'snapshot_file_sha256': c['endpoint']['sha256'], 'tensors': []}
        for label, t in [('W4', p['weights'][4]), ('W8', p['weights'][8]), ('M4',p['M4']), ('M8',p['M8'])]:
            assert t.dtype == torch.float32 and torch.isfinite(t).all()
            expected = c['state']['weights'][label[1:]] if label.startswith('W') else c['state'][label]
            assert tsha(t) == expected
            row['tensors'].append({'name':label,'shape':list(t.shape),'dtype':str(t.dtype),'sha256':expected})
        alpha = c['alpha']
        expected = native if alpha == 1 else pre['weights'][4] + alpha * (native-pre['weights'][4])
        assert tsha(expected, False) == c['materialization']['weight_sha256']
        if arm != 'REFIT4':
            assert torch.equal(p['weights'][4], expected)
        if arm not in ('FULL8','RES8'):
            assert torch.equal(p['weights'][8], pre['weights'][8]) and torch.equal(p['M8'],pre['M8'])
        assert len(c['history']) == (2 if arm in ('FULL8','RES8') else 1)
        for h in c['history']:
            layer = h['layer']
            assert h['history_append'] == 1 and h['compute_ks'] == 1
            assert h['before_sha256'] == tsha(pre['M'+str(layer)], False)
            assert h['after_sha256'] == tsha(p['M'+str(layer)], False)
            assert h['weight_sha256'] == tsha(p['weights'][layer],False)
        row['append_count'] = len(c['history'])
        row['delta_norm'] = {str(l): float((p['weights'][l].double()-pre['weights'][l].double()).norm()) for l in (4,8)}
        if arm in ('N4','FULL8','RES8','REFIT4'):
            f = read(arm+'/fit.json')
            assert f['compute_z']==100 and f['compute_ks']==f['solve']==f['get_module_input_output_at_words']==1 and f['history_append']==0
            layer=f['layer']
            assert f['endpoint_weight_sha256'] == tsha(p['weights'][layer],False)
            if arm != 'N4':
                partial=f['actual_partial_state']
                for key in ('M4','M8','P4','P8','contexts','rng'):
                    assert partial[key]==cap['entry_state'][key]
                assert partial['weights']['4']==tsha(expected)
                assert partial['weights']['8']==tsha(pre['weights'][8])
                start=expected if layer==4 else pre['weights'][8]
            else:
                start=pre['weights'][4]
            assert f['entry_weight_sha256']==tsha(start,False)
            captures=load(arm+'/native-target-key-readout.pt')
            assert len(captures['compute_z'])==100 and len(captures['compute_ks'])==len(captures['get_module_input_output_at_words'])==1
            row['z_capture_order_root']=hashlib.sha256(''.join(tsha(x) for x in captures['compute_z']).encode()).hexdigest()
            row['actual_z_calls']=f['compute_z']
        reports.append(row)
        del p
    assert len(set(x['z_capture_order_root'] for x in reports if 'z_capture_order_root' in x))==4
    assert read('process-restore.json')['selected_W0_exact'] is True
    return {'status':'SAVED_STATE_CPU_AUDIT_PASS','snapshots':7,'endpoint_count':6,'request_z_calls':400,'fit_solves':4,'endpoint_history_appends':8,'M8_reencoding_batches':50,'M8_reencoding_events':5000,'rows':reports,'alpha0':'SOURCE_ENDPOINT_COPY_AND_CPU_TEST_ONLY_NOT_A_CORE_ENDPOINT','GPU_continuation_replay':'NOT_RUN','model_level_observer_off_parity':'NOT_TESTED','actual_z_optimizer_iteration_counts':'NOT_RECORDED_IN_STRUCTURED_FIT_RECEIPTS'}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.root)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
