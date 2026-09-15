"""Bounded CPU package validation; no scientific raw reload or scheduler access."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from .begin import LOCAL, TASK, WT, digest, save
from .publication import CAKE, CAP, OVER


def rows(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))


def markdown_check(path):
    """Check GFM table width, heading separation and local link existence."""
    lines = path.read_text().splitlines()
    errors, tables, links = [], 0, 0
    width = None
    in_code = False
    for i, line in enumerate(lines):
        if line.startswith('```'):
            in_code = not in_code
        if in_code:
            continue
        if re.match(r'^#{1,6} ', line) and i+1 < len(lines) and lines[i+1].strip():
            errors.append(f'{i+1}: missing blank after heading')
        if line.startswith('|'):
            n = len(re.split(r'(?<!\\)\|', line.strip().strip('|')))
            if width is None:
                width = n
                tables += 1
                if i and lines[i-1].strip():
                    errors.append(f'{i+1}: missing blank before table')
            elif n != width:
                errors.append(f'{i+1}: table width {n} != {width}')
        else:
            width = None
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', line):
            if target.startswith(('https://', 'http://', '#')):
                continue
            links += 1
            q = (path.parent / target.split('#')[0].strip('<>')).resolve()
            if not q.exists():
                errors.append(f'{i+1}: missing local link {target}')
    return dict(path=str(path), tables=tables, local_links=links, errors=errors)


def verify_ref(ref):
    actual = digest(ref['path'])
    assert actual['sha256'] == ref['sha256'], ref['path']
    assert actual['bytes'] == ref['bytes'], ref['path']


def verify_sweep_seal():
    receipt = json.loads((CAP/'rooted-receipt.json').read_text())
    verify_ref(receipt['manifest']); verify_ref(receipt['report'])
    manifest = json.loads((CAP/'analysis-manifest.json').read_text())
    for ref in manifest['package'] + manifest['analysis_source']:
        verify_ref(ref)
    assert manifest['numerical_validation'] == 'NOT_ESTABLISHED'
    return dict(members=len(manifest['package']), receipt=digest(CAP/'rooted-receipt.json'),
                parent_instruction=manifest['instruction_id'], current_review_instruction=TASK,
                historical_execution_status='SOURCE_INPUT_FROZEN_NOT_SUBMITTED is preserved freeze-time metadata, not current terminal status')


def reproduce_pngs(output):
    result = []
    for label, module, package in [('cake','project.run_scripts.server4_completed_review.plot_cake', CAKE),
                                 ('caps','project.run_scripts.bg_tw_reference.ep_tw.plot_sweep', CAP)]:
        dest = output/label
        command = [sys.executable, '-B', '-m', module, '--package', str(package), '--output', str(dest)]
        proc = subprocess.run(command, cwd=WT, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr
        for p in sorted(dest.glob('*.png')):
            q = package/'figures'/p.name
            assert p.read_bytes() == q.read_bytes(), p
            result.append(dict(**digest(p), publication=digest(q), byte_identical=True))
    assert len(result) == 9
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest',
        'project.run_scripts.server4_completed_review.test_review',
        'project.run_scripts.bg_tw_reference.ep_tw.test_sweep_review',
        'project.run_scripts.bg_tw_reference.ep_tw.test_review_nogate'],
        cwd=WT, text=True, capture_output=True, check=False)
    save(output/'focused-tests.json', dict(returncode=tests.returncode, stdout=tests.stdout, stderr=tests.stderr))
    assert tests.returncode == 0, tests.stderr
    compiled = []
    for p in sorted((WT/'project/run_scripts/server4_completed_review').glob('*.py')):
        compile(p.read_text(), str(p), 'exec'); ast.parse(p.read_text()); compiled.append(digest(p))
    md = [markdown_check(p) for root in (CAKE,CAP,OVER) for p in sorted(root.rglob('*.md'))]
    assert not any(x['errors'] for x in md), md
    packages = []
    for root in (CAKE,CAP,OVER):
        for p in sorted(root.rglob('*')):
            if not p.is_file(): continue
            assert not p.is_symlink()
            assert p.suffix in {'.md','.csv','.json','.png','.patch'}, p
            assert p.stat().st_size < 15_000_000, p
            if p.suffix == '.json': json.loads(p.read_text())
            if p.suffix == '.csv':
                rr = rows(p)
                assert all(None not in r for r in rr), p
            packages.append(digest(p))
    expected = {'CAKE_NATIVE': (9840,17755,62935), 'CAP1': (998,1942,8056),
                'CAP10': (999,1947,8075), 'CAP100': (998,1945,8052), 'NORM_ONLY': (999,1938,8042)}
    for package in (CAKE,CAP):
        rr=rows(package/'first-final-table.csv')
        for r in rr:
            m=('RS','PS','NS').index(r['metric']); factor=10 if package == CAKE else 1
            assert int(r['numerator']) == expected[r['arm']][m]
            assert int(r['denominator']) == (1000,2000,10000)[m]*factor
    cake=json.loads((CAKE/'summary.json').read_text())
    assert (cake['batches'],cake['requests'],cake['links'],cake['saved_tensor_CP']) == (100,10000,99,0)
    states=[]
    for arm in ('CAP10','CAP100','NORM_ONLY'):
        p=LOCAL/'analysis/caps'/arm/'state/state-summary.json'
        s=json.loads(p.read_text());states.append(digest(p))
        assert (s['checkpoints'],s['adjacent_links'],s['targets'],s['solves'],s['final_history_appends'],s['inner_history_appends']) == (10,9,1000,10,10,0)
    plot_refs=reproduce_pngs(output/'plots')
    seal=verify_sweep_seal()
    result=dict(instruction_id=TASK, status='CPU_PACKAGE_VALIDATED',
        analysis_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip(),
        tests=digest(output/'focused-tests.json'),compiled=compiled,markdown=md,files=packages,
        PNG_byte_reproduction=plot_refs,sweep_seal=seal,new_cap_state_receipts=states,
        raw_free_check='extension/size/CSV-schema plus source and content review; paths/hashes/aggregates only',
        review_level='SINGLE_SH_SELF_AUDIT_WITH_INDEPENDENT_NLL_REDUCTION_NOT_SECOND_AGENT_AUDIT',
        scheduler_queries_in_validator=0,new_model_forwards=0,new_GPU_seconds=0,
        numerical_validation='NOT_ESTABLISHED',automatic_resume=False)
    save(output/'validation.json', result)
    return dict(validation=digest(output/'validation.json'),tests=tests.stderr.strip(),PNG_byte_equal=len(plot_refs),markdown_files=len(md),package_files=len(packages))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(json.dumps(run(p.parse_args().output)))
