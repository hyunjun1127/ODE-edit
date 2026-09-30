"""CPU-only terminal review of the three frozen T100 evaluation outputs."""
import argparse
import json
import math
from pathlib import Path
from .common import read, record, save, digest
from .review_b010 import ARMS, PUB, pairs, summarize, transitions, csvsave

ROOT = Path('/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1/final-full-eval-20260930-v1')
SOURCE = '62005204cf2b0a5abf5ef2830849bc9103199515'
LOCK = '95d898d31c1f8f4fa41a7b8af74dbfbaf0771b5b19dc8fde266d80a8a1b338ef'

def run(repo, scratch):
    pub = Path(repo)/PUB/'final-eval-update-r1'
    pub.mkdir(parents=True, exist_ok=False)
    scratch = Path(scratch); scratch.mkdir(parents=True, exist_ok=False)
    inventory = {}; metrics = []; costs = []; joined = {}; identities = None
    def load(path):
        before = record(path); obj = read(path)
        assert record(path) == before
        inventory[str(path)] = before
        return obj
    lock = load(ROOT/'execution.lock.json')
    assert inventory[str(ROOT/'execution.lock.json')]['sha256'] == LOCK
    assert lock['source'] == SOURCE
    cfg = load(lock['configuration']['path'])
    assert inventory[lock['configuration']['path']]['sha256'] == lock['configuration']['sha256']
    ids = cfg['execution_ids']; assert len(ids) == len(set(ids)) == 100
    for i, arm in enumerate(ARMS):
        spec = lock['arms'][i]; assert spec['arm'] == arm
        d = ROOT/'output'/arm; t = load(d/'terminal.json'); restore = load(d/'restore.json')
        assert t['status'] == 'COMPLETED' and t['source'] == SOURCE and t['arm'] == arm
        assert t['scope'] == 'FINAL_T100_FULL100_R100_P200_N1000'
        assert t['requests'] == 100 and t['raw_rows'] == 2600 and t['nonmutation']
        assert t['write'] == t['backward'] == t['history_append'] == 0 and not t['checkpoint_saved']
        assert t['weight_hashes_before'] == t['weight_hashes_after'] == restore['weight_hashes']
        assert restore['selected_exact'] and restore['nonselected_pointer_version_unchanged']
        sr = load(spec['snapshot_receipt']['path'])
        assert inventory[spec['snapshot_receipt']['path']]['sha256'] == spec['snapshot_receipt']['sha256']
        assert sr['tensor_hashes'] == t['weight_hashes_before'] and sr['exact_logp']
        actual = Path(spec['snapshot_file']['path']).stat()
        assert actual.st_size == spec['snapshot_file']['bytes'] and actual.st_mtime_ns == spec['snapshot_file']['mtime_ns']
        raw = load(d/'full-metrics.json'); rr = inventory[str(d/'full-metrics.json')]
        assert (rr['sha256'], rr['bytes']) == (t['raw']['sha256'], t['raw']['bytes'])
        assert len(raw) == len({r['row_id'] for r in raw}) == 2600
        concatenated = []
        for n, cid in enumerate(ids, 1):
            rows = load(d/'requests'/f'{n:03d}.json')
            assert len(rows) == 26 and {r['case_id'] for r in rows} == {cid}
            concatenated.extend(rows)
        assert concatenated == raw
        pp = pairs(raw); joined[arm] = pp
        identity = {k:v['identity'] for k,v in pp.items()}
        if identities is None: identities = identity
        else: assert identity == identities
        ss = summarize(pp); old = {r['panel']:r for r in load(d/'summary.json')}
        assert {r['panel']:r['denominator'] for r in ss} == {'continuation:R':100,'continuation:P':200,'neighborhood:N':1000}
        assert t['summary'] == list(old.values())
        for s in ss:
            o = old[s['panel']]
            assert (s['numerator'],s['denominator'],s['strict'],s['token_correct'],s['tokens']) == tuple(o[k] for k in ('success','prompts','strict','token_correct','tokens'))
            assert math.isclose(s['mean_true_nll'],o['mean_true'],abs_tol=1e-12)
            assert math.isclose(s['mean_new_nll'],o['mean_new'],abs_tol=1e-12)
            metrics.append(dict(arm=arm,**s))
        prior = load(spec['old_final']['path'])
        assert inventory[spec['old_final']['path']]['sha256'] == spec['old_final']['sha256']
        rp = {k:v for k,v in pp.items() if v['panel'].startswith('continuation:')}
        assert rp == pairs(prior)  # Same IDs, packing, NLL, strict and token results.
        assert t['prior_rp_max_abs_nll_difference'] == 0
        costs.append(dict(arm=arm,job=f'55331_{i}',state='COMPLETED',exit_code='0:0',
            allocated_gpu_seconds=(173,168,166)[i],program_seconds=t['total_seconds'],
            forward_seconds=t['timers']['final_full_evaluation'],calls=t['calls']['final_full_evaluation'],
            tokens=t['tokens']['final_full_evaluation'],model_load_seconds=restore['model_load_seconds'],
            restore_seconds=restore['restore_seconds'],cuda_peak_bytes=t['cuda_peak_bytes'],host_maxrss_kib=t['host_maxrss_kib']))
    trans = [dict(comparison='NATIVE_to_'+a,**r) for a in ARMS[1:] for r in transitions(joined['NATIVE'],joined[a])]
    trans += [dict(comparison='JOINT_STEP_to_JOINT_CUM',**r) for r in transitions(joined['JOINT_STEP'],joined['JOINT_CUM'])]
    for name, rows in [('metrics',metrics),('paired-transitions',trans),('compute',costs)]: csvsave(pub/(name+'.csv'),rows)
    save(scratch/'input-members.json',list(inventory.values()))
    save(scratch/'paired-rows.json',joined)
    save(pub/'verification.json',dict(status='THREE_TERMINAL_OUTPUTS_CPU_VERIFIED',source=SOURCE,
        lock_sha256=LOCK,ordered_ids_sha256=digest(ids),raw_rows=7800,pairs_per_arm=1300,
        unique_requests=100,unique_task_prompts=1300,arm_prompt_observations=3900,
        raw_inventory=record(scratch/'input-members.json'),raw_members=len(inventory),
        raw_root=digest(list(inventory.values())),prior_RP_exact=True,all_arm_pair_identity_exact=True,
        all_terminal_nonmutation=True,snapshot_verification='Execution full SHA/restore receipts reused plus current size/mtime; no new GPU reload or multi-GB hash.',
        new_forward=0,new_submission=0,independent_red_agent=False,scientific_promotion=False,
        analysis_source=[record(Path(__file__)),record(Path(__file__).with_name('review_b010.py'))]))
    print(json.dumps(dict(metrics=metrics,transitions=trans,compute=costs),ensure_ascii=False,indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--scratch',required=True)
    a=p.parse_args();run(a.repo,a.scratch)
