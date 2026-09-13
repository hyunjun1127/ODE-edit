"""Read-only Git inventory; generated metadata contains no source/raw payload."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = '7d5bae2e3be8dba87c92b86a727d5de9ed549af3'

def git(*args, check=True):
    p = subprocess.run(['git', *args], capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError((args, p.stderr))
    return p.stdout.strip()

def tree(ref):
    result = {}
    for row in git('ls-tree', '-r', ref).splitlines():
        meta, path = row.split('\t', 1)
        mode, kind, blob = meta.split()
        result[path] = {'mode': mode, 'kind': kind, 'blob': blob}
    return result

def scan():
    main = tree(BASE)
    worktrees = {}
    for block in git('worktree', 'list', '--porcelain').split('\n\n'):
        d = dict(line.split(' ', 1) for line in block.splitlines() if ' ' in line)
        if 'branch' in d:
            worktrees[d['branch']] = d['worktree']
    refs = git('for-each-ref', '--format=%(refname)', 'refs/heads/codex', 'refs/remotes/origin/codex').splitlines()
    rows = []
    for ref in refs:
        wt = worktrees.get(ref, '')
        name = ref.replace('refs/heads/', '').replace('refs/remotes/', '')
        if 'owned-branches-main-integration-20260914' in name:
            continue
        head = git('rev-parse', ref)
        authors = git('log', '-1', '--format=%an|%ae', ref)
        known = ('odeeditsh1' in wt or '/server1-' in name)
        ambiguous = (ref.startswith('refs/heads/') and not known and
                     not any(x in name for x in ['gh-', '-gh-', 'session-registry', 'app-server', 'fzcb-edit', 'slurm-memory', 'layer-update-share']))
        if not known and not ambiguous:
            continue
        ancestry = subprocess.run(['git', 'merge-base', '--is-ancestor', head, BASE]).returncode == 0
        base = git('merge-base', head, BASE)
        patches = git('cherry', BASE, head) if not ancestry else ''
        files = git('diff', '--name-only', base, head).splitlines()
        branch_tree = tree(head)
        members = []
        for path in files:
            b, m = branch_tree.get(path), main.get(path)
            members.append({'path': path, 'branch': b, 'main': m,
                            'comparison': 'BYTE_MODE_EQUAL' if b == m else 'ABSENT_FROM_MAIN' if m is None else 'DIFFERENT'})
        patch_equivalent = bool(patches) and all(line.startswith('- ') for line in patches.splitlines())
        all_equal = bool(members) and all(x['comparison'] == 'BYTE_MODE_EQUAL' for x in members)
        status = 'ALREADY_IN_MAIN' if ancestry or patch_equivalent or all_equal else 'PENDING_REVIEW'
        if ambiguous:
            status = 'OWNER_UNRESOLVED'
        reflog = git('reflog', 'show', '--format=%H %gs', ref, check=False).splitlines()
        creation = next((x for x in reversed(reflog) if 'branch: Created from' in x), '')
        rows.append({'branch': name, 'ref': ref, 'head': head, 'tree': git('rev-parse', head+'^{tree}'),
                     'owner': 'SH1' if known else 'OWNER_UNRESOLVED', 'owner_evidence': wt or 'server1 branch namespace',
                     'head_author': authors, 'worktree': wt, 'original_base_reflog': creation or 'NOT_RECORDED',
                     'comparison_merge_base': base, 'ancestry': ancestry, 'patch_equivalent': patch_equivalent,
                     'all_changed_file_bytes_equal': all_equal, 'patches': patches.splitlines(),
                     'status': status, 'members': members,
                     'worktree_tracked_dirty': git('-C',wt,'status','--porcelain','--untracked-files=no') if wt else 'NO_LOCAL_WORKTREE'})
    out = {'base_main':BASE, 'base_tree':git('rev-parse',BASE+'^{tree}'), 'branches':rows,
           'protocol_sha256':hashlib.sha256(subprocess.check_output(['git','show',BASE+':PROTOCOL.md'])).hexdigest(),
           'policy_sha256':hashlib.sha256(subprocess.check_output(['git','show',BASE+':messages/head/2026-09-14-all-sh-owned-branches-main-integration.md'])).hexdigest()}
    target=ROOT/'inventory-before.json'
    with target.open('x') as f: json.dump(out,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps({'refs':len(rows),'unique_heads':len(set(r['head'] for r in rows)),
                      'counts':{s:sum(r['status']==s for r in rows) for s in sorted(set(r['status'] for r in rows))}},ensure_ascii=False))

if __name__ == '__main__': scan()
