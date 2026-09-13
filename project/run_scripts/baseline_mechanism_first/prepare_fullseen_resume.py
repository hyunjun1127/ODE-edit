"""CPU admission of saved endpoints and exact original/reused performance."""
import json
from pathlib import Path
import subprocess
import torch
from scripts.fixed_counterfact import load_prefix
from .contracts import member,save,digest
from .fixtures import tensor_sha,SingletonSpec
from .performance_schema import normalize
from .terminal_performance import paired


def main():
    torch.set_num_threads(8)
    root=Path.cwd();rel=Path('local/baseline-mechanism-first-e01/20260912-v1')
    old=Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-terminal-performance-repair-r1')/rel
    new=root/rel/'fullseen-schema-repair-r3';head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    records=load_prefix('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1',10000)
    results=[]
    for n,job,cost in [(5000,45914,9727),(9000,45915,9748)]:
        end=n//100+10;attempt=old/f'attempts/warm-l4-n{n}-performance-r2';out=attempt/'output'
        parent=old/f'locks/warm-l4-n{n}-performance-r2/input.lock.json';lock=json.loads(parent.read_text())
        fail=json.loads((out/'failure.json').read_text());assert fail['completed_batches']==10 and fail['error']=="KeyError('request_order')"
        assert fail['restore']['pointer_bytes_exact'] and fail['restore']['rng_restored_exact']
        commit=json.loads((out/f'B{end:03d}/commit.json').read_text());endpoint=member(commit['endpoint']['path'],expected=commit['endpoint']['sha256'])
        cp=torch.load(endpoint['path'],map_location='cpu',weights_only=False,mmap=True);spec=SingletonSpec(4)
        obs=json.loads((out/f'B{end:03d}/native-observation.json').read_text())
        assert tensor_sha(cp['weights'][spec.weight_name])==obs['receipt']['endpoint_sha256']==cp['metadata']['state']['weights'][spec.weight_name]
        assert tensor_sha(cp['cache_c'])==obs['receipt']['history_sha256']==cp['metadata']['state']['cache']
        assert cp['metadata']['batch']==end and cp['metadata']['seen_ids']==[r['case_id'] for r in records[:end*100]]
        assert torch.isfinite(cp['weights'][spec.weight_name]).all() and torch.isfinite(cp['cache_c']).all()
        assert {'contexts','rng','covariance','state','base_model_revision'}<=cp['metadata'].keys()
        original={tag:normalize(json.loads(Path(m['path']).read_text()),records[end*100-100:end*100] if tag=='current' else records[:end*100]) for tag,m in lock['terminal_performance'].items()}
        reused=normalize(json.loads((out/'terminal-performance/current.json').read_text()),records[end*100-100:end*100])
        paired_current=paired(original['current'],reused,'current_REUSED')
        preserved=[member(p) for p in sorted(attempt.rglob('*')) if p.is_file() and p.suffix!='.pt']
        # Final W/M checkpoint full SHA independently read; other native tensor
        # files retain their existing per-batch sealed identities, no pointless rehash.
        preserved.append(endpoint)
        case=new/f'n{n}';preservation=save(case/'preserved-members.json',dict(members=preserved,root=digest(preserved),unmodified=True))
        members=[member(parent),member(out/'failure.json'),member(out/f'B{end:03d}/commit.json'),member(out/f'B{end:03d}/native-observation.json'),
                 endpoint,member(out/'terminal-performance/current.json'),*lock['terminal_performance'].values()]
        inp=dict(source_head=head,parent_input=member(parent),parent_attempt=str(attempt),endpoint=endpoint,
                 reuse_current=member(out/'terminal-performance/current.json'),terminal_batch=end,expected_W=obs['receipt']['endpoint_sha256'],expected_M=obs['receipt']['history_sha256'],
                 members=members,science_changes=0,new_native_batches=0,new_z=0,new_history_append=0,
                 evaluator='same historical microbatch16; past128-request shards preserve all candidate16 batch boundaries; final Current100 reused',
                 initial_gate='first actual past128 fullseen schema/paired/endpoint guarded shard; not model load',preserved=preservation)
        input_ref=save(case/'input.lock.json',inp)
        receipt=save(case/'repair-receipt.json',dict(RCA='terminal_performance assumed current.json request_order header in original seen-full.json; source intentionally merges rows without header',
                    original_error=fail,old_job=job,old_allocation_GPU_seconds=cost,parent_source='b51dcf5ab825608bee81dd13549318d8d267e835',new_source=head,
                    CPU_actual_schema_validation='PASS_ALL_ORDERED_CASE_PROMPT_TARGET_HASHES',CPU_saved_endpoint='PASS_SHA_FINITE_METADATA_W_M',
                    repair='validated schema adapter + endpoint-only observation runner',new_native_batches=0,reused_native_batches=10,
                    reused_current_requests=100,new_past_requests=end*100-100,paired_current=paired_current,input=input_ref,
                    fullseen_previous_complete_shards=0,source_equivalence='UNRESOLVED_NOT_RECLASSIFIED',preserved=preservation,
                    wall_request_hours=6,estimated_GPU_hours=[.8,2.] if n==5000 else [1.3,3.],estimate_not_measurement=True,GPU_hour_cap=None,
                    initial_policy='MONITORING_PAUSED_AWAITING_USER after first actual repaired path',scientific_promotion=False))
        results.append(dict(entry_n=n,input=input_ref,repair=receipt))
        del cp
    save(new/'prepared.json',dict(cells=results,source=head,old_allocated_GPU_seconds=19475,cpu_only=True))
    print(json.dumps(results))


if __name__=='__main__':main()
