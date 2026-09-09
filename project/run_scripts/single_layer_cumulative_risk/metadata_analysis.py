"""Outcome-blind strata and optional subject/relation clustered uncertainty."""
import argparse
import csv
import gzip
import io
import json
from collections import defaultdict
from pathlib import Path
from scripts.fixed_counterfact import load_prefix
from .analysis import aggregate,bootstrap,write_csv
from .binding import DATA
from .metadata import panel_metadata
from .records import save,digest
from .import_assets import sha

BOOLS={'success','entry_success','W0_success','new_strict','true_strict','entry_success_to_failure','entry_failure_to_success'}
NUMBERS={'new_nll','true_nll','margin','new_nll_delta','true_nll_delta','inherited_margin','additional_margin'}
def decode(row):
    for name in BOOLS:
        if name in row:row[name]=row[name]=='True'
    for name in NUMBERS:
        if name in row:row[name]=float(row[name])
    row['case_id']=int(row['case_id'])
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--request-table',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=load_prefix(DATA,10000)
    metadata=[r for e in ['Early','Middle','Late'] for r in panel_metadata(records,e)]
    lookup={(r['entry'],r['panel'],r['case_id']):r for r in metadata}
    raw=gzip.decompress(a.request_table.read_bytes())
    groups=defaultdict(list)
    for row in csv.DictReader(io.StringIO(raw.decode())):
        row=decode(row)
        group=tuple(row.get(k,'') for k in ['entry','endpoint','resolution','reference','panel','metric'])
        groups[group].append(row)
    cluster_rows=[];strata=[]
    for keys,rows in sorted(groups.items()):
        labels=dict(zip(['entry','endpoint','resolution','reference','panel','metric'],keys))
        mapping={r['case_id']:lookup[(r['entry'],r['panel'],r['case_id'])]['subject_relation_cluster'] for r in rows}
        cluster_rows.append(dict(**labels,**bootstrap(rows,clusters=mapping),
                           unique_requests=len({r['case_id'] for r in rows}),denominator=len(rows)))
        for flag in ['later_conflicting_seen_target','current_conflicting_target','exact_neighbor_rewrite_overlap_count']:
            for value in [False,True]:
                subset=[r for r in rows if bool(lookup[(r['entry'],r['panel'],r['case_id'])][flag])==value]
                if subset:strata.append(dict(**labels,stratum=flag,present=value,parent_denominator=len(rows),**aggregate(subset)))
    a.output.mkdir(parents=True,exist_ok=False)
    outputs=[write_csv(a.output/'panel-metadata.csv',metadata),write_csv(a.output/'subject-relation-bootstrap.csv',cluster_rows),
             write_csv(a.output/'overwrite-overlap-strata.csv',strata)]
    save(a.output/'metadata-manifest.json',dict(input_request_table=str(a.request_table),input_sha=sha(a.request_table),
        outputs=outputs,output_root=digest(outputs),metadata_root=digest(metadata),
        endpoint_exclusions=0,semantic_conflict_detection=False,causal_claim=False,
        limits='Exact subject/relation and string overlap metadata only; no semantic conflict inference. Strata are descriptive, not candidate selection.'))

if __name__=='__main__':main()
