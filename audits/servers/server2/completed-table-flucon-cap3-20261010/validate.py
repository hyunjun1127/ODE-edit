"""Additional metadata/source and CPU negative controls, with no scheduler query."""
from pathlib import Path
import math
from official.experiments.prepare import read,file_sha,digest,write_new
from official.runners.server1.common import member,verify
from official.runners.server1.submit import graph_width
from official.evaluation.generation.paper_display import paper_cell
OUT=Path(__file__).resolve().parent
repo=OUT.parents[3];data=read(OUT/'table-rows.json');cap=read(OUT/'cap-receipt.json')
code=repo/'audits/servers/server2/baseline-completed-zsre-audit-20261010/audit.py'
ns={'__file__':str(code)};exec(compile(code.read_text().split('\ndata=read(')[0],str(code),'exec'),ns)
independent=ns['independent'];sources=[];generation=[];live=[]
for row in data['rows']:
    if not row['numeric_eligible']:continue
    if row['dataset']=='zsre':
        for m in row['zsre_audit']['frozen_members']:verify(m);sources.append(m)
    if row.get('generation_raw'):
        raw=ns['verified'](row['generation_raw']);assert len(raw['rows'])==2000
        for metric,field,valid,unit in [('Flu','ngram_entropy','fluency_valid','bits'),('Con','reference_score','consistency_valid','cosine_0_to_1')]:
            values=[x['metrics'][field] for x in raw['rows'] if x['metrics'][valid]]
            mean=math.fsum(values)/len(values)
            assert all(math.isfinite(v) for v in values) and abs(mean-row[metric])<1e-12
            assert paper_cell(mean,metric=metric,raw_unit=unit)==row[metric+'_paper_x100']
            generation.append(dict(job=row['job_id'],metric=metric,raw_value=mean,raw_unit=unit,valid_count=len(values),paper_display_x100=row[metric+'_paper_x100']))
for r in cap['inventory']['project']:
    command=member(r['command']);work=Path(r['workdir']);root=work.parent if work.name=='source' else work
    lock=root/'source-lock.json'
    if not lock.exists():lock=root/'execution-lock.json'
    source=read(lock)
    live.append(dict(job=r['job'],state=r['state'],command=command,source_lock=member(lock),
                     source=source.get('code_commit',source.get('source_commit')),dependency=r['dependency']))
    assert '62534' not in r['dependency']
# Unequal token lengths force request macro != micro: (1 + 0)/2, not 1/4.
fixture=[]
for values in ([True],[False,False,False]):
    observations=[dict(predicted_token_id=int(bit),target_token_id=1,correct=bit) for bit in values]
    fixture.append({k+suffix:value for k in ('rewrite','paraphrase','neighborhood') for suffix,value in [('_observations',observations),('_prompts_correct',list(values))]})
macro,den,micro=independent(fixture)
assert set(macro.values())=={50.} and set(micro.values())=={25.}
fixture[0]['neighborhood_prompts_correct']=[False]
try:independent(fixture)
except ValueError:pass
else:raise AssertionError('tampered correctness accepted')
assert graph_width([dict(key=str(i),gpus=1,parents=[]) for i in range(3)])==3
assert graph_width([dict(key=str(i),gpus=1,parents=[]) for i in range(4)])==4
qrow=next(r for r in data['rows'] if r['job_id']=='62081')
cfg=read(Path(qrow['rerun_attempt'])/'configs/qwen25-zsre-memit.json')
assert cfg['config_sha256']==qrow['config_sha256']==digest({k:v for k,v in cfg.items() if k!='config_sha256'})
write_new(OUT/'validation.json',dict(CPU_controls=5,CPU_PASS=True,source_members_rechecked=sources,
    live_source_binding=live,generation_checks=generation,raw_metrics_unchanged=True,
    checkpoint_deserialized=False,model_forward=0,scheduler_queries=0,source=member(__file__),
    limitation='CP full payload rehash not repeated; recorded final hash/provenance and fresh presence/stat used. CPU query parity is not pretrained forward parity.'))
print({'CPU_controls':5,'source_members':len(sources),'generation_cells':len(generation),'live_sources':len(live)})
