"""GFM/link/compact-file checks, with honest optional HTML renderer status."""
import csv
import importlib.util
import io
from pathlib import Path
import re


def check(directory):
    directory=Path(directory).resolve();report=directory/'diagnostic-report-ko.md'
    text=report.read_text();fenced=False;table_width=None;tables=0;links=[]
    for line in text.splitlines():
        if line.startswith('```'):
            fenced=not fenced;table_width=None;continue
        if fenced:continue
        if line.startswith('|'):
            if not line.rstrip().endswith('|'):raise ValueError('GFM_UNCLOSED_ROW')
            width=line.count('|')-1
            if table_width is None:table_width=width;tables+=1
            elif table_width!=width:raise ValueError('GFM_INCONSISTENT_WIDTH')
        else:table_width=None
        for target in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)',line):
            if target.startswith(('https://','http://')):raise ValueError('UNPLANNED_EXTERNAL_REPORT_LINK')
            path=(directory/target.split('#',1)[0]).resolve()
            if not path.is_relative_to(directory) or not path.is_file():raise ValueError('REPORT_LINK_MISSING_OR_ESCAPE')
            links.append(target)
    if fenced:raise ValueError('UNCLOSED_CODE_FENCE')
    if not tables or not links:raise ValueError('REQUIRED_REPORT_TABLES_LINKS_MISSING')
    csv_rows={};files=[]
    for path in directory.iterdir():
        if not path.is_file() or path.suffix not in ('.md','.json','.csv','.png'):raise ValueError('PUBLICATION_UNEXPECTED_MEMBER')
        if path.stat().st_size>4*2**20:raise ValueError('COMPACT_MEMBER_SIZE_REVIEW_REQUIRED')
        if path.suffix=='.csv' and path.stat().st_size:
            rows=list(csv.reader(io.StringIO(path.read_text())))
            if any(len(row)!=len(rows[0]) for row in rows):raise ValueError('CSV_WIDTH')
            csv_rows[path.name]=max(0,len(rows)-1)
        files.append(path.name)
    rendered=None
    if importlib.util.find_spec('markdown_it'):
        from markdown_it import MarkdownIt
        rendered=MarkdownIt('commonmark').enable('table').render(text)
        if rendered.count('<table>')!=tables:raise ValueError('HTML_TABLE_RENDER_COUNT')
    return dict(status='GFM_LINK_CSV_COMPACT_CHECKS_PASS',GFM_tables=tables,local_links=links,CSV_data_rows=csv_rows,
        checked_members=sorted(files),HTML_renderer='markdown_it_COMMONMARK_TABLE' if rendered else 'NOT_RUN_NOT_INSTALLED',
        actual_HTML_render_PASS=rendered is not None,PNG='NONE_NO_REDUNDANT_PLOT_REQUESTED',
        raw_free='EXTENSION_SIZE_AND_COMPACT_SCHEMA_CHECKS; MANUAL_GIT_DIFF_ALSO_REQUIRED')
