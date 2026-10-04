"""Strict canonical JSONL schema; candidate relational audit independent of fit."""
import collections,json,math,sys
from datetime import datetime,timezone
import torch
from .common import ROOT,DESIGN,require
sys.path.append('/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/dependencies-r1')
from jsonschema import Draft7Validator

class Events:
    def __init__(self,path,run_id,batch=None):
        self.path=path;path.parent.mkdir(parents=True,exist_ok=True)
        require(not path.exists(),'EVENT_CREATE_ONCE')
        self.run_id,self.batch=run_id,batch
        self.schema=json.loads((ROOT/DESIGN/'telemetry-schema.json').read_text())
        self.validators={k:Draft7Validator(dict(v,definitions=self.schema['definitions'])) for k,v in self.schema['definitions'].items()}
    def emit(self,event,payload,request=None,candidate=None,layer=None):
        self.validators[event].validate(payload)
        row=dict(schema_version='1.0.0',run_id=self.run_id,event=event,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),batch_index=self.batch,
            request_id=None if request is None else str(request),candidate_index=candidate,
            layer_id=None if layer is None else str(layer),payload=payload)
        data=json.dumps(row,allow_nan=False,separators=(',',':'))
        with self.path.open('a') as f:f.write(data+'\n');f.flush()

def ratio(numerator,denominator,reason='ZERO_DENOMINATOR'):
    return dict(value=float(numerator/denominator),reason=None) if denominator else dict(value=None,reason=reason)

def cosine(x,y):
    return ratio(float((x.double()*y.double()).sum()),float(x.double().norm()*y.double().norm()))

def diagnostics(blocks,grads,price,radius=.75):
    fields=('task_l2','norm_l2','total_l2','task_radial','task_tangential_l2','active_stationarity_l2','inactive_excess')
    kfields=('mu_hat','estimator','stationarity_l2','primal_violation','complementarity_abs','diagnostic_active_count','diagnostic_boundary')
    if grads is None:
        return dict(availability='NO_BACKWARD_TERMINAL',**dict.fromkeys(kfields)),[
            dict(availability='NO_BACKWARD_TERMINAL',**dict.fromkeys(fields)) for u in blocks]
    norms=[float(u.double().norm()) for u in blocks];budget=sum(norms);active=[n>1e-10 for n in norms]
    radial=[float((g.double()*u.double()).sum())/n if n>0 else None for u,g,n in zip(blocks,grads,norms)]
    boundary=abs(budget-radius)<=1e-6*max(1,radius)
    mu=max(0.,sum(-r-price for r,flag in zip(radial,active) if flag)/sum(active)) if boundary and any(active) else 0.
    output=[];residuals=[]
    for u,g,n,flag,rad in zip(blocks,grads,norms,active,radial):
        u=u.double();g=g.double();ng=price*u/n if n>0 else torch.zeros_like(u)
        residual=float((g+(price+mu)*u/n).norm()) if flag else None
        excess=max(0.,float(g.norm())-price-mu) if not flag else None
        residuals.append(residual if flag else excess)
        output.append(dict(availability='AVAILABLE',task_l2=float(g.norm()),norm_l2=float(ng.norm()),
            total_l2=float((g+ng).norm()),task_radial=rad,
            task_tangential_l2=float((g-rad*u/n).norm()) if n>0 else None,
            active_stationarity_l2=residual,inactive_excess=excess))
    return dict(availability='AVAILABLE',mu_hat=mu,estimator='RADIAL_ACTIVE_MEAN_CLIPPED' if boundary and any(active) else 'INTERIOR_ZERO',
        stationarity_l2=math.sqrt(sum(r*r for r in residuals)),primal_violation=max(0.,budget-radius),
        complementarity_abs=abs(mu*(budget-radius)),diagnostic_active_count=sum(active),diagnostic_boundary=boundary),output

def validate_relations(path,layers,request_ids):
    schema=json.loads((ROOT/DESIGN/'telemetry-schema.json').read_text());validator=Draft7Validator(schema)
    candidates=collections.defaultdict(list);updates=collections.Counter();terminal={};layerrows=collections.Counter()
    for line in path.read_text().splitlines():
        r=json.loads(line);validator.validate(r);p=r['payload'];key=r['request_id'];c=r['candidate_index']
        if r['event']=='candidate_request':
            require(p['evaluation_ordinal']==c+1 and p['updates_completed']==c,'CANDIDATE_COUNTER')
            require(p['terminal_z_capture_candidate']==c,'TARGET_CANDIDATE')
            candidates[key].append((c,p['will_backward']))
        elif r['event']=='candidate_layer':layerrows[(key,c)]+=1
        elif r['event']=='optimizer_layer':updates[(key,c)]+=1
        elif r['event']=='terminal_request':
            require(key not in terminal,'DUPLICATE_TERMINAL');terminal[key]=p
    require(set(terminal)==set(map(str,request_ids)),'TERMINAL_COVERAGE')
    for key,p in terminal.items():
        n=p['accepted_candidate_index'];require(candidates[key]==[(c,c<n) for c in range(n+1)],'STOP_SEQUENCE')
        require(p['logical_evaluations']==n+1 and p['optimizer_updates']==p['backward_calls']==n,'TERMINAL_COUNTS')
        for c in range(n+1):
            require(layerrows[(key,c)]==len(layers),'LAYER_COVERAGE')
            require(updates[(key,c)]==(len(layers) if c<n else 0),'TERMINAL_BACKWARD')
    return dict(requests=len(terminal),request_evaluations=sum(p['logical_evaluations'] for p in terminal.values()),
                request_updates=sum(p['optimizer_updates'] for p in terminal.values()),schema_and_relations=True)
