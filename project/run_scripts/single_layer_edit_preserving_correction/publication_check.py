"""CPU publication integrity and actual Markdown-to-HTML render; no model."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Tables(HTMLParser):
    def __init__(self):super().__init__();self.rows=[];self.current=None
    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.current=0
        if tag in ('td','th') and self.current is not None:self.current+=1
    def handle_endtag(self,tag):
        if tag=='tr':self.rows.append(self.current);self.current=None


def main():
    p=argparse.ArgumentParser();p.add_argument('--report-root',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--html',type=Path,required=True);a=p.parse_args()
    root=a.report_root;report=root/'submission-and-paired-stop-ko.md'
    manifest=json.loads((root/'artifact-manifest.json').read_text());checked=[]
    for m in manifest['members']:
        f=root/m['relative']
        if f.stat().st_size!=m['bytes'] or sha(f)!=m['sha256']:raise ValueError('MANIFEST_MEMBER:'+str(f))
        checked.append(m['relative'])
    receipt=json.loads((root/'rooted-receipt.json').read_text())
    if sha(root/'artifact-manifest.json')!=receipt['manifest']['sha256']:raise ValueError('ROOTED_MANIFEST')
    value=report.read_text();links=re.findall(r'\[[^\]]+\]\(([^)]+)\)',value)
    for link in links:
        if not (root/link).is_file():raise ValueError('LINK:'+link)
    from markdown_it import MarkdownIt
    html=MarkdownIt('commonmark').enable('table').render(value)
    dom=Tables();dom.feed(html)
    if dom.rows!=[4,4,4,4]:raise ValueError('HTML_TABLE_COLUMNS:'+repr(dom.rows))
    a.html.parent.mkdir(parents=True,exist_ok=True)
    with a.html.open('x') as f:f.write('<!doctype html><meta charset="utf-8">'+html)
    result=dict(status='PASS',scope='CPU publication only; no numerical or scientific promotion',
        members_checked=checked,relative_links_checked=links,actual_HTML_renderer='markdown_it commonmark + table enabled',
        HTML_table_column_counts=dom.rows,Korean_UTF8_decoded=True,HTML=dict(path=str(a.html),sha256=sha(a.html)),
        report_sha256=sha(report),manifest_sha256=sha(root/'artifact-manifest.json'),rooted_receipt_sha256=sha(root/'rooted-receipt.json'),
        new_GPU=0,model_load=0,browser_screenshot='NOT_REQUESTED_NOT_PERFORMED')
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    with a.receipt.open('x') as f:json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
