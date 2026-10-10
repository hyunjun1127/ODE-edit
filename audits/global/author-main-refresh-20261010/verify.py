"""소형 영수증→README 매핑/배율/이력보존 검산. GPU/원격조회/파일쓰기 없음."""
import hashlib
import json
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
read = lambda p: json.loads((ROOT / p).read_text())
old = subprocess.check_output(['git','show','8d717f7b:README.md'],cwd=ROOT,text=True)
new = (ROOT/'README.md').read_text()
def fmt(x):
    if isinstance(x,Fraction): x=Decimal(x.numerator)/Decimal(x.denominator)
    return str(Decimal(str(x)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
def tables(text):
    output={}; model=None
    for line in text.splitlines():
        if line.startswith('### '): model={'### Llama3-8B-Instruct':'llama3','### Qwen2.5-7B-Instruct':'qwen25','### GPT-J-6B':'gptj'}.get(line)
        elif model and line.startswith('| '):
            cells=[x.strip() for x in line.split('|')[1:-1]]
            if len(cells)==10:output[(model,cells[0])]=cells[1:]
    return output
a,b=tables(old),tables(new)
label='MEMIT-FE (FE author hparams + history)'
s1=read('audits/servers/server1/author-main-refresh-20261010/table-rows.json')
s2=read('audits/servers/server2/author-main-refresh-20261010/table-rows.json')
assert hashlib.sha256((ROOT/'audits/servers/server1/author-main-refresh-20261010/table-rows.json').read_bytes()).hexdigest()=='87b8f2ee01405abad2f09b3f987168f68f8b5a1fa91fc9c99b43ad4d3d1882ab'
assert hashlib.sha256((ROOT/'audits/servers/server2/author-main-refresh-20261010/table-rows.json').read_bytes()).hexdigest()=='8bbbcbdaffc683bb568ff2d9fcefcd92d58c489d35bca7ad742ce9df8742f6c5'
rows={r['job_id']:r for r in s1['rows']+s2['rows']}
counts=read('audits/global/author-main-refresh-20261010/exact-counts.json')
for model,cf,zs in [('llama3','62529','62530'),('qwen25','62531','62532')]:
    r=rows[cf];c=counts[cf]
    assert r['commits']==20 and r['metrics'].get('requests',r.get('requests'))==2000
    exact=[Fraction(n,d)*100 for n,d in zip(c['success_counts'],c['prompt_counts'])]
    assert exact==list(map(Fraction,c['exact_pct']))
    score=3/sum(1/x for x in exact);assert score==Fraction(c['score_exact'])
    expected=[fmt(score)]+list(map(fmt,exact))+['DEFERRED']*2
    expected+=['ING: '+rows[zs]['job_name']+' ('+zs+')']*3
    assert b[(model,label)]==expected
    assert rows[zs].get('observed_state',rows[zs].get('scheduler_state'))=='RUNNING'
    name={'llama3':'Llama3','qwen25':'Qwen2.5'}[model]
    legacy='| '+name+' | MEMIT-FE (이전 native 설정) | '+' | '.join(a[(model,'MEMIT-FE')])+' |'
    assert legacy in new
assert '| MEMIT_FE_HISTORY (FE author hparams) |' not in new
assert b[('gptj','MEMIT-FE (legacy native; author 미실행)')]==a[('gptj','MEMIT-FE')]
generation_changes={('gptj','FT'),('gptj','MEMIT'),('gptj','AlphaEdit+SPHERE'),('qwen25','AlphaEdit')}
for key,values in a.items():
    if key[1]=='MEMIT-FE':continue
    if key in generation_changes:
        assert b[key][:4]==values[:4] and b[key][6:]==values[6:]
    else: assert b[key]==values,key
for method,jid in [('FT','62864'),('MEMIT','62865'),('AlphaEdit+SPHERE','62869')]:
    g=next(x for x in s2['generation'] if x['job_id']==jid)
    assert g['numeric_eligible'] and g['summary']['planned_count']==2000
    assert g['summary']['fluency_count']==g['summary']['consistency_count']==2000
    vals=[fmt(Decimal(str(g[k]['raw_value']))*100) for k in ['Flu','Con']]
    assert vals==[g[k]['paper_display_x100'] for k in ['Flu','Con']]==b[('gptj',method)][4:6]
g=rows['62061']['generation']
assert [g[k]['paper_display_x100'] for k in ['Flu','Con']]==['532.39','0.41']
history=next(x for x in new.splitlines() if x.startswith('| MEMIT_FE_HISTORY | Qwen2.5'))
assert history.endswith('| 532.39 | 0.41 |')
assert b[('qwen25','AlphaEdit')][4:6]==['ING: s2-flucon-qwen25-alphaedit-62075 (62871)']*2
assert sum(r['dataset']=='zsre' and bool(r.get('metrics')) for r in rows.values())==17
print('PASS: author CF8cells; author zsRE6status; generation8numeric+2status; legacy2rows preserved; PRICE/W0/other facts unchanged; 17 zsRE completed rows source-bound')
