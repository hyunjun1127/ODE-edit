"""보존 완료 후 local 검증 기록에서 raw-free catalog/map/보고서를 만든다."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
from verify_initial import CONTROL,REPO,save,sha

OUT=REPO/'transfers/verifications/2026-09-11-checkpoint-migration-server2-destination'
AUDIT=REPO/'audits/servers/server2/2026-09-11-checkpoint-migration-server4'

def main(deletion_receipts):
    OUT.mkdir(parents=True,exist_ok=False)
    mappings=[];receipts=[]
    initial=json.loads((CONTROL/'initial72-retention-catalog.json').read_text())
    for r in initial['members']:
        d=r['destination'];mappings.append(dict(bundle='initial72',arm=r['arm'],batch=r['batch'],source=r['source_path'],destination=d['path'],bytes=d['bytes'],sha256=d['sha256'],mode='IN_PLACE_REUSE'))
    names=['initial72-VERIFIED_DESTINATION-v2.json']
    for name in ['new177-v1','jvp1k-v1']:
        cat=json.loads((CONTROL/(name+'-retention-catalog.json')).read_text())
        for r in cat['checkpoint_members']:
            d=r['source_stat'];mappings.append(dict(bundle=name,arm=r['arm'],batch=r['batch'],source=r['source_path'],destination=r['destination_path'],bytes=d['bytes'],sha256=d['sha256'],mode='NEW_VERIFIED_ARCHIVE'))
        names.append(name+'-VERIFIED_DESTINATION.json')
    assert len(mappings)==183 and len({r['source'] for r in mappings})==183
    assert sum(r['bytes'] for r in mappings)==240176147811
    for n in names:
        p=CONTROL/n;d=json.loads(p.read_text());assert d['status']=='VERIFIED_DESTINATION'
        save(OUT/n,d);receipts.append(dict(local_path=str(p),publication_path=str(OUT/n),sha256=sha(p)))
    text=io.StringIO();w=csv.DictWriter(text,fieldnames=list(mappings[0]),lineterminator='\n');w.writeheader();w.writerows(mappings)
    with (OUT/'migration-map.csv').open('x') as f:f.write(text.getvalue())
    source_by={r['source']:r for r in mappings};removed=[];deletion_refs=[]
    for p in deletion_receipts:
        d=json.loads(p.read_text());assert d.get('recursive_delete',0)==0
        assert d['status']=='SOURCE_EXACT_FILES_REMOVED_DESTINATION_PRESERVED'
        assert d['receiver_receipt_sha256'] in {x['sha256'] for x in receipts}
        assert len(d['removed'])==d['count']
        assert sum(x['bytes'] for x in d['removed'])==d['logical_bytes']
        for row in d['removed']:
            m=source_by[row['source']]
            assert row['removed'] and (row['destination'],row['sha256'],row['bytes'])==(m['destination'],m['sha256'],m['bytes'])
            removed.append(row)
        deletion_refs.append(dict(path=str(p),sha256=sha(p),count=d['count'],logical_bytes=d['logical_bytes'],observed_available_delta=d.get('observed_available_delta'),source_owner='SH4',independent_remote_query_by_SH2=False))
    assert len({r['source'] for r in removed})==len(removed)
    assert len(removed)==183, 'FINAL_PUBLICATION_REQUIRES_ALL_SOURCE_RECEIPTS'
    summary=dict(instruction_id='ODEEDIT-S06-SERVER4-CHECKPOINT-MIGRATION-SERVER2-V1',
        status='DESTINATION_PRESERVATION_COMPLETE',retained_count=183,retained_bytes=240176147811,
        reused_count=72,reused_bytes=62011141768,new_count=111,new_bytes=178165006043,
        new_companion_members=6273,new_companion_bytes=135254070,source_deletion_receipts=deletion_refs,
        source_removed_count_reported=len(removed),source_removed_bytes_reported=sum(r['bytes'] for r in removed),
        additional_preserved_inventory_count=4,additional_preserved_inventory_bytes=6341797900,
        local_control=str(CONTROL),destination_receipts=receipts,
        model_gpu_evaluator_slurm_actions=0,rsync_actions=0,source_unlink_actions=0,
        other_task_monitoring_resumed=False,scientific_outcome_changed=False,
        limitations=['No GPU continuation replay','BLUE1k historical RNG not saved','SH4 deletion facts attributed to source receipts; shared filesystem delta not exclusive recovery attribution'])
    save(OUT/'preservation-summary.json',summary)
    print('RAW_FREE_PUBLICATION_READY',len(mappings),len(removed),sha(OUT/'preservation-summary.json'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--deletion-receipts',nargs='+',type=Path,required=True);main(ap.parse_args().deletion_receipts)
