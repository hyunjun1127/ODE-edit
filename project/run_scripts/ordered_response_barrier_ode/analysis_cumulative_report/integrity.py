"""Full member rehash and CPU tensor recovery of already-terminal 37649."""
import argparse,json,stat
from pathlib import Path
import torch
from ..fp32_overlay import tensor_sha256,tensor_set_sha256
from .common import *

class Ledger:
    def __init__(self,root):self.root=Path(root);self.members={};self.identities=0
    def bind(self,p,expected=None):
        p=Path(p);key=str(p.relative_to(self.root))
        if key not in self.members:self.members[key]=member(p,relative_to=self.root)
        row=self.members[key]
        if expected is not None:require(row['sha256']==expected,'member SHA: '+str(p))
        return row
    def load(self,p,expected=None):
        self.bind(p,expected);v=read(p);self.identities+=deep_identity(v);return v
    def ref(self,root,r):
        p=root/r['path'];v=self.load(p,r['sha256']);require(v['identity_sha256']==r['identity_sha256'],'reference identity');return v

def build(root,out):
    root,out=Path(root),Path(out);require(not out.exists(),'create-once output');out.mkdir(parents=True)
    torch.set_num_threads(2)
    ledger=Ledger(root);chains=[];checkpoints=[];sources=[];common_order=None;total_eval=0
    submission=ledger.load(root/'submission-receipt.json')
    require((submission['source_head'],submission['source_tree'])==SOURCE,'submission source')
    pre=ledger.load(root/'preflight.json',submission['preflight_sha256'])
    ledger.bind(root/'dry-plan.json',submission['dry_plan_sha256'])
    for cid,cell in enumerate(CELLS):
        cr=root/'results'/f'task-{cid}';term=ledger.load(cr/'terminal-receipt.json');r=ledger.load(cr/'result.json',term['result_sha256'])
        require(r['identity_sha256']==term['result_identity_sha256'],'cell result root')
        require(r['cell_id']==cid and r['status']=='SEQUENTIAL_CELL_TERMINAL_VALID','terminal status')
        require((r['source']['head'],r['source']['tree'])==SOURCE,'executed source')
        require(r['arms']==list(ARMS) and r['primary_endpoint_count']==5000,'arm inventory')
        require(r['stream_root']==STREAM_ROOT and r['order_root']==ORDER_ROOT,'stream/order')
        for key in ['nonfinite_count','imputation_count','retry_count','technical_failure_count','scientific_failure_count','inter_arm_state_carry_count','fixed_z_recompute_count']:
            require(r[key]==0,cell+' '+key)
        require(r['fixed_z_compute_count']==5000 and r['full_fp32'],'fixed z/FP32')
        require(r['terminal_w0_pointer_bytes_restore_pass'] and r['terminal_method_state_restore_pass'],'final restore')
        sources.append(dict(cell=cell,result={k:v for k,v in r.items() if k!='arm_receipts'},terminal=term))
        entry=None;entry_m=None
        for ar in r['arm_receipts']:
            a=ledger.load(cr/ar['relative_path'],ar['file_sha256']);arm=a['arm'];cum=cr/f'arm-{arm}'/'cumulative'
            require(a['identity_sha256']==ar['identity_sha256'] and a['status']=='SEQUENTIAL_ARM_TERMINAL_VALID','arm terminal')
            if entry is None:entry=a['entry_weight_sha256'];entry_m=a['entry_method_state_sha256']
            require((a['entry_weight_sha256'],a['entry_method_state_sha256'])==(entry,entry_m),'arm isolation')
            ct=ledger.ref(cum,a['cumulative_terminal_receipt']);require(ct['checkpoints']==10 and ct['evaluator_cohort_calls']==55,'cumulative terminal')
            require(ct['source_targets_recomputed']==ct['controller_feedback']==0,'observer feedback')
            frozen={}
            for m in ct['members']:
                if m['path'].endswith('.pt'):
                    p=cum/m['path'];ledger.bind(p,m['sha256']);t=torch.load(p,map_location='cpu',weights_only=True)
                    for key in ('target','origin'):
                        require(t[key].dtype==torch.float32 and bool(torch.isfinite(t[key]).all()),'target dtype/finite')
                        require(tensor_sha256(t[key])==m[key+'_sha256'],'frozen tensor hash')
                    frozen[len(frozen)+1]=(t['request_sha256'],m['target_sha256'],m['origin_sha256']);del t
                else:ledger.ref(cum,m)
            require(len(frozen)==10,'frozen cohorts')
            last_w,last_m=entry,entry_m;orders=[];nseen=0
            for idx,j in enumerate(a['journals'],1):
                b=ledger.load(cr/j['relative_path'],j['file_sha256']);require(b['identity_sha256']==j['identity_sha256'],'journal identity')
                require(b['batch_index']==idx-1 and b['arm']==arm and b['status']=='SEQUENTIAL_BATCH_TERMINAL_VALID','journal mapping')
                require(len(b['request_sha256'])==len(set(b['request_sha256']))==100,'batch unique100')
                require(canonical_hash(b['request_sha256'])==b['request_order_sha256'],'batch order')
                require((b['entry_weight_sha256'],b['entry_method_state_sha256'])==(last_w,last_m),'W/M interbatch chain')
                last_w,last_m=b['committed_weight_sha256'],b['committed_method_state_sha256']
                require(frozen[idx][:2]==(b['request_sha256'],b['fixed_z']['identity_sha256']),'frozen target/request binding')
                require(b['fixed_z']['compute_count']==100 and b['fixed_z']['recompute_count']==0,'fixed z count')
                for key in ['controller_evaluator_influence_count','dynamic_z_recompute_count','retry_count','backtracking_count','fallback_count','nonfinite_count','imputation_count']:
                    require(b[key]==0,'batch '+key)
                width=b['alpha_history_width']
                if r['writer_family']=='AlphaEdit':require((width['entry'],width['exit'],width['append'])==((idx-1)*100,idx*100,100),'history append/width')
                else:require(b['memit_covariance_static'] and width['append']==0 and last_m==entry_m,'MEMIT covariance')
                require(b['endpoint']['selected_weight_endpoint_sha256']==last_w,'immediate endpoint W')
                orders+=b['request_sha256']
                cp=ledger.ref(cum,b['cumulative_evaluation_receipt']);directory=cum/Path(b['cumulative_evaluation_receipt']['path']).parent
                require(cp['status']=='CUMULATIVE_CHECKPOINT_VALID' and cp['evaluation_type']=='CHECKPOINT_W_ON_ALL_SEEN_REQUESTS','checkpoint type')
                require((cp['seen_requests'],cp['rewrite_prompts'],cp['rephrase_prompts'],cp['locality_prompts'])==(100*idx,100*idx,200*idx,1000*idx),'checkpoint counts')
                require((cp['weight_sha256'],cp['method_state_sha256'])==(last_w,last_m),'checkpoint state binding')
                require(cp['request_order_sha256']==canonical_hash(orders),'seen prefix order')
                for key in ['observer_compute_z_writer_key_solve_backward_history_mutation','controller_feedback','nonfinite','duplicate','imputation']:require(cp[key]==0,'checkpoint '+key)
                for key in ['weight_pointer_version_bytes_unchanged','cache_unchanged','checkpoint_cpu_reload_exact','full_fp32']:require(cp[key] is True,'checkpoint '+key)
                require(len(cp['evaluation_members'])==idx,'evaluation cohort inventory')
                for ci,em in enumerate(cp['evaluation_members'],1):
                    e=ledger.ref(directory,em)
                    require((e['at_batch'],e['cohort_batch'],e['W_sha256'])==(idx,ci,last_w),'cohort W identity')
                    require(e['evaluation']['request_order_sha256']==canonical_hash(frozen[ci][0]),'cohort order')
                    require(e['evaluation']['request_count']==100 and e['evaluation']['row_count']==2600,'evaluation row count')
                    total_eval+=100;nseen+=100
                for key in ['geometry_member','key_writer_drift_member']:ledger.ref(directory,cp[key])
                for cm in cp['command_members_relative_to_cumulative_root']:ledger.ref(cum,cm)
                p=cum/cp['layer_response_file'];ledger.bind(p,cp['layer_response_sha256'])
                with p.open() as f:
                    count=0;seen=set()
                    for line in f:
                        v=json.loads(line);deep_identity(v);key=(v['cohort_batch'],v['request_sha256'],v['visit'])
                        require(key not in seen,'layer duplicate');seen.add(key);count+=1
                    require(count==cp['layer_visits_this_batch']*100*idx,'layer observation cardinality')
                p=directory/cp['checkpoint_file'];m=ledger.bind(p,cp['checkpoint_sha256']);require(m['bytes']==cp['checkpoint_bytes'],'checkpoint size')
                tensors=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
                require(set(tensors)==set(cp['checkpoint_tensor_sha256']) and len(tensors)==5,'checkpoint tensor keys')
                for name,t in tensors.items():
                    require(t.dtype==torch.float32 and bool(torch.isfinite(t).all()),'checkpoint tensor FP32 finite')
                    require(tensor_sha256(t)==cp['checkpoint_tensor_sha256'][name],'checkpoint tensor SHA')
                require(tensor_set_sha256(tensors)==last_w,'recovered W hash');del tensors
                checkpoints.append(dict(cell=cell,arm=arm,batch=idx,seen_requests=idx*100,W=last_w,M=last_m,checkpoint_sha256=m['sha256'],checkpoint_bytes=m['bytes'],CPU_tensor_rehash=True,evaluation_request_rows=idx*100,cache_recovery='HASH_ONLY_NOT_EDIT_RESUME'))
            require(last_w==a['terminal_weight_sha256'] and last_m==a['terminal_method_state_sha256'],'arm final W/M')
            require(len(set(orders))==1000 and canonical_hash(orders)==ORDER_ROOT,'global1000order')
            if common_order is None:common_order=orders
            require(common_order==orders,'crossarm/cell order')
            require(nseen==5500,'allseen5500')
            chains.append(dict(cell=cell,arm=arm,batches=10,requests=1000,W_links=9,M_links=9,fixed_z_compute=1000,recompute=0,history_append_batches=10 if r['writer_family']=='AlphaEdit' else 0,terminal_restore=True))
            print(f'INTEGRITY {cell}/{arm}: 10 checkpoint file/tensor + 5500 seen rows PASS',flush=True)
    # Inventory every remaining sealed file, including logs; do not publish contents.
    for p in sorted(root.rglob('*')):
        if p.is_file():ledger.bind(p)
    inventory=sorted(ledger.members.values(),key=lambda m:m['path'])
    require(len(checkpoints)==200 and total_eval==110000,'full denominator')
    write_json_once(out/'raw-member-inventory.json',inventory)
    frame_write(out/'raw-member-inventory.csv',inventory)
    frame_write(out/'checkpoint-integrity.csv',checkpoints);frame_write(out/'chain-integrity.csv',chains)
    write_json_once(out/'source-runtime-provenance.json',sources)
    receipt=dict(status='FULL_REHASH_CPU_RECOVERY_PASS',instruction_id=INSTRUCTION,cells=4,primary_arms=20,checkpoints=200,cumulative_request_states=total_eval,unique_requests=1000,raw_member_count=len(inventory),raw_bytes=sum(x['bytes'] for x in inventory),raw_member_root=canonical_hash(inventory),schema_identity_count=ledger.identities,W_links=180,M_links=180,fixed_z_compute=20000,recompute=0,nonfinite=0,duplicate=0,imputation=0,terminal_restore=4,checkpoint_scope='EVALUATION_ONLY_NOT_CACHE_RESUME',source=SOURCE,model_load=0,GPU=0,evaluator=0,scheduler=0)
    receipt['identity_sha256']=canonical_hash(receipt);write_json_once(out/'integrity-receipt.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RAW);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.root,a.output)),flush=True)
