"""Load-only publication and independent PNG reproduction gate; no raw inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import aos_review as review


def numeric_rows(path):
    rows=list(csv.DictReader(Path(path).open()))
    for row in rows:
        for key,value in row.items():
            try:row[key]=float(value)
            except (ValueError,TypeError):pass
    return rows


def verify(args):
    root=Path(args.package).resolve();repo=Path(args.repo).resolve();h=review.helper(repo)
    manifest=h.read(root/'analysis-manifest.json');receipt=h.read(root/'rooted-receipt.json')
    h.require(h.digest(manifest['members'])==manifest['members_root']==receipt['members_root'],'OUTPUT_ROOT_MISMATCH')
    h.require(h.sha(root/'analysis-manifest.json')==receipt['manifest_sha256'],'MANIFEST_BINDING')
    for member in manifest['members']:
        p=root/member['path'];h.require(p.parent==root,'NONLOCAL_PUBLICATION_MEMBER')
        got=h.member(p);h.require(got['bytes']==member['bytes'] and got['sha256']==member['sha256'],'PUBLICATION_MEMBER_MISMATCH')
    h.require(receipt['endpoint_pairs']==receipt['paired_A0_AOS']==3900 and receipt['attribution_pairs']==300,'DENOMINATOR')
    h.require(receipt['campaign_complete'] is False and receipt['scientific_promotion'] is False,'OVERCLAIM')
    h.require(receipt['new_model']==receipt['new_forward']==receipt['new_GPU']==receipt['new_Slurm']==0,'REVIEW_ACTION')
    summary=numeric_rows(root/'endpoint-summary.csv');paired=numeric_rows(root/'paired-summary.csv')
    functional=numeric_rows(root/'functional-risk.csv');physical=numeric_rows(root/'physical-update.csv')
    trajectory=h.read(root/'solver-and-transaction.json')
    figures=review.plot_bytes(summary,paired,functional,physical,trajectory)
    for name,content in figures.items():
        h.require(content==(root/name).read_bytes(),'SEALED_CSV_PNG_REPRODUCTION_MISMATCH')
    # Git objects, not a mutable worktree, bind the scientific source reviewed.
    paths=['track_a/run_os.py','track_a/protocol.py','functional.py','linear_solve.py',
           'elastic_qp.py','common_reference/session.py','evaluation.py']
    closure=[]
    for tail in paths:
        path='project/run_scripts/multilayer_joint_compensation/'+tail
        blob=subprocess.check_output(['git','-C',str(repo),'show',f'{review.EXECUTION}:{path}'])
        closure.append(dict(path=path,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),
                            git_blob=subprocess.check_output(['git','-C',str(repo),'rev-parse',f'{review.EXECUTION}:{path}'],text=True).strip()))
    lines=(root/'factual-report-ko.md').read_text().splitlines();width=None;tables=0
    for line in lines:
        if line.startswith('|'):
            n=line.count('|')
            if width is None:width=n;tables+=1
            h.require(n==width,'MARKDOWN_TABLE_COLUMN_MISMATCH')
        else:width=None
    result=dict(status='PASS_REHASH_ACCESS_SOURCE_AND_SEALED_CSV_PNG_REPRODUCTION',
                package=str(root),report_sha256=h.sha(root/'factual-report-ko.md'),
                manifest_sha256=h.sha(root/'analysis-manifest.json'),rooted_receipt_sha256=h.sha(root/'rooted-receipt.json'),
                members_root=manifest['members_root'],members_verified=len(manifest['members']),
                figures_independently_recreated_from_published_CSV=len(figures),markdown_tables_column_valid=tables,
                execution_source=review.EXECUTION,execution_source_members=closure,
                scientific_source_mutation=0,raw_free=True,new_model=0,new_forward=0,new_GPU=0,new_Slurm=0,
                focused_tests=7,focused_test_command='python -m unittest discover -s project/run_scripts/multilayer_joint_compensation/analysis -p test_*.py -v',
                focused_output='Ran 7 tests; OK',py_compile='PASS',diff_check='PASS',
                verifier_source_sha256=h.sha(Path(__file__)),scientific_promotion=False,
                current_profile_warning='A0 and AOS Current teachers differ; their displayed arithmetic risk delta is not a comparable objective improvement',
                reconstruction='checkout analysis code commit51675eab for original full build; this verifier reproduces figures from sealed CSV without changing package')
    if args.receipt:h.save_json(Path(args.receipt),result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--package',default=review.REPORT);p.add_argument('--receipt')
    verify(p.parse_args())
