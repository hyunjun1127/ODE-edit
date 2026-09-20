"""Read-only package structure checks. Never substitutes for HTML rendering."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import re
import shutil


def check(package):
    root=Path(package);report=root/'report-ko.md';text=report.read_text(encoding='utf-8')
    if '\ufffd' in text:raise ValueError('UTF8_REPLACEMENT_CHARACTER')
    table_width=None;tables=0
    for line in text.splitlines():
        if line.startswith('|'):
            cells=re.split(r'(?<!\\)\|',line)[1:-1]
            if table_width is None:table_width=len(cells);tables+=1
            if len(cells)!=table_width:raise ValueError('GFM_COLUMN_COUNT')
        else:table_width=None
    links=[]
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        if '://' in target or target.startswith('#'):continue
        path=(root/target.split('#',1)[0]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('LOCAL_LINK:'+target)
        links.append(target)
    csvs={}
    for path in sorted(root.glob('*.csv')):
        with path.open(newline='') as handle:rows=list(csv.reader(handle))
        if rows and any(len(row)!=len(rows[0]) for row in rows):raise ValueError('CSV_COLUMNS:'+path.name)
        csvs[path.name]=max(0,len(rows)-1)
    tools={name:shutil.which(name) for name in ('pandoc','cmark','commonmark','chromium','google-chrome')}
    modules={name:importlib.util.find_spec(name) is not None for name in ('markdown','markdown_it')}
    return dict(UTF8='PASS',GFM_column_counts='PASS',tables=tables,local_links=links,csv_row_counts=csvs,
        renderer_tools=tools,renderer_modules=modules,
        actual_HTML_render='NOT_RUN' if any(tools.values()) or any(modules.values()) else 'NOT_AVAILABLE',
        PNG_visual_inspection='OWNER_SEPARATE_REQUIRED',raw_free='OWNER_SCOPE_DIFF_INSPECTION_REQUIRED')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('package');print(json.dumps(check(p.parse_args().package),ensure_ascii=False,indent=2))
