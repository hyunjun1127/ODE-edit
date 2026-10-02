"""User-directed removal of the exact attempt-r1 W0 dependency graph."""
import getpass
import json
import subprocess
from .common import LOCAL, write, require, member


def command(argv):
    p = subprocess.run(argv, capture_output=True, text=True)
    require(p.returncode == 0, repr(argv) + p.stderr)
    return p.stdout


def main():
    root = LOCAL / 'attempt-r1'
    receipt = LOCAL / 'user-remove-w0-r1' / 'cancellation.json'
    require(not receipt.exists(), 'ALREADY_CANCELLED')
    submission = json.loads((root / 'submission.json').read_text())
    expected = {'shared-SHARED': '57282', 'pilot-JLZ_A': '57283',
                'pilot-JLZ_B': '57284', 'main-JLZ_A': '57285',
                'main-JLZ_B': '57286', 'collector': '57287'}
    require(submission['jobs'] == expected, 'EXACT_MAPPING')
    before = {}
    for name, job in expected.items():
        detail = command(['scontrol', 'show', 'job', job, '--oneliner'])
        for text in [f'JobId={job} ', 'UserId=' + getpass.getuser() + '(',
                     'JobName=odeedit_jlz_v4_s4_' + name.replace('-', '_') + ' ',
                     'Command=' + str(root / (name + '.sh')) + ' ',
                     'WorkDir=' + str(root / 'source') + ' ']:
            require(text in detail, 'IDENTITY:' + text)
        require(('JobState=RUNNING ' in detail) if name == 'shared-SHARED'
                else ('JobState=PENDING ' in detail), 'UNEXPECTED_STATE')
        before[name] = detail
    write(receipt.parent / 'before.json', dict(jobs=before, submission=member(root / 'submission.json')))
    cancelled = []
    for name in ['collector', 'main-JLZ_A', 'main-JLZ_B', 'pilot-JLZ_A', 'pilot-JLZ_B', 'shared-SHARED']:
        argv = ['scancel', expected[name]]
        output = command(argv)
        row = dict(name=name, job=expected[name], argv=argv, output=output, actor='SH4')
        write(receipt.parent / (name + '.json'), row)
        cancelled.append(row)
    write(receipt, dict(user_instruction='W0 실험 중단/제외, 이전 W0 결과 재사용, A/B 제출',
                       cancelled=cancelled, raw_preserved=True, no_other_job_mutation=True,
                       reason='Replace immutable graph without shared W0 evaluation'))
    print(json.dumps(cancelled))


if __name__ == '__main__':
    main()
