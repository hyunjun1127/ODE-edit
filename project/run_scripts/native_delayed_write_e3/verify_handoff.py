"""CPU-only link/table/raw-free audit of the blocked pre-submission package."""
import importlib.util
from pathlib import Path
import re
import subprocess
from .common import save, sha, file_record


def main():
    repo=Path(__file__).resolve().parents[3]
    report=repo/'experiment-reports/servers/server4/native-delayed-write-e3-20260924-v1'
    files=[report/'report-ko.md',repo/'messages/acks/server4/2026-09-24-native-delayed-write-e3.md',
        repo/'messages/server-heads/server4/2026-09-24-native-delayed-write-e3.md']
    links=[];table_rows=0
    for p in files:
        text=p.read_text();n=None
        for line in text.splitlines():
            if line.startswith('|'):
                count=len(line.split('|'))-2
                if n is None:n=count
                assert count==n,('GFM_TABLE_COLUMN_MISMATCH',str(p),line)
                table_rows+=1
            else:n=None
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target:continue
            q=(p.parent/target.split('#')[0]).resolve();assert q.is_file(),(p,target)
            links.append(dict(document=str(p.relative_to(repo)),target=target,status='EXISTS'))
    renderer={x:importlib.util.find_spec(x) is not None for x in ('markdown','markdown_it','mistune')}
    changed=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=repo,text=True).splitlines()
    for rel in changed:
        p=repo/rel;assert p.stat().st_size<250000,('UNEXPECTED_LARGE_TRACKED_PAYLOAD',rel)
        assert p.suffix in ('.py','.md','.json','.csv'),rel
        if p.suffix=='.json':
            text=p.read_text()
            assert '"input_ids"' not in text and '"token_predictions"' not in text and '"test_stderr"' not in text,('RAW_GIT',rel)
    result=dict(status='PASS_STATIC_ONLY',relative_links=links,GFM_rows=table_rows,
        renderer_available=renderer,actual_HTML_render='NOT_RUN_RENDERER_NOT_INSTALLED',
        figures='NOT_APPLICABLE_NO_SCIENTIFIC_OUTPUT',raw_free=True,independent_agent=False,
        shared_access_helper='NOT_PASS: runs/ pattern absent; exact envelope §5 explicitly permits this one task path; shared helper unmodified',
        no_scheduler_or_model_calls=True)
    print(save(repo/'audits/servers/server4/native-delayed-write-e3-20260924-v1/package-checks.json',result))


if __name__=='__main__':main()
