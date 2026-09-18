"""Read-only CPU fixed-state sensitivity; never a trajectory or GPU sweep.

Inputs: sealed 80-node published full H/g/G and native q. Stdlib-only
independent active-face solve and symmetric Jacobi eigenvalues for n <= 5.
Outputs are derived observations; no model data, settings, or source modified.
"""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

INPUT = Path('/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-alpha-jv-review-20260907/experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1/layer_allocation_nodes.csv')
OUT = Path(__file__).resolve().parent

def dot(a, b):
    return sum(x*y for x,y in zip(a,b))

def mv(a, x):
    return [dot(r,x) for r in a]

def solve(a, b):
    n = len(b)
    w = [list(row)+[v] for row,v in zip(a,b)]
    for i in range(n):
        pivot = max(range(i,n), key=lambda j: abs(w[j][i]))
        if w[pivot][i] == 0: raise ArithmeticError('singular')
        w[i],w[pivot] = w[pivot],w[i]
        d = w[i][i]
        for k in range(i,n+1): w[i][k] /= d
        for j in range(n):
            if i == j: continue
            v = w[j][i]
            for k in range(i,n+1): w[j][k] -= v*w[i][k]
    return [row[-1] for row in w]

def eig(a):
    n=len(a)
    w=[[(a[i][j]+a[j][i])/2 for j in range(n)] for i in range(n)]
    scale=max(1.,max(abs(x) for row in w for x in row))
    for _ in range(500):
        p,q=max(((i,j) for i in range(n) for j in range(i+1,n)),key=lambda ij:abs(w[ij[0]][ij[1]]))
        if abs(w[p][q]) < 1e-14*scale: break
        phi=.5*math.atan2(2*w[p][q],w[q][q]-w[p][p])
        c,s=math.cos(phi),math.sin(phi)
        pp,qq,pq=w[p][p],w[q][q],w[p][q]
        w[p][p]=c*c*pp-2*s*c*pq+s*s*qq
        w[q][q]=s*s*pp+2*s*c*pq+c*c*qq
        w[p][q]=w[q][p]=0.
        for k in range(n):
            if k in (p,q): continue
            kp,kq=w[k][p],w[k][q]
            w[k][p]=w[p][k]=c*kp-s*kq
            w[k][q]=w[q][k]=s*kp+c*kq
    else: raise ArithmeticError('eigen nonconvergence')
    return sorted(w[i][i] for i in range(n))

def nnls(g,h,gram,lam):
    n=len(g)
    a=[[h[i][j]+lam*gram[i][j] for j in range(n)] for i in range(n)]
    tol=256*2**-52*n*max(1.,max(map(abs,g)),max(abs(x) for row in a for x in row))
    valid=[]
    for size in range(n+1):
        for face in itertools.combinations(range(n),size):
            c=[0.]*n
            if face:
                x=solve([[a[i][j] for j in face] for i in face],[g[i] for i in face])
                for i,v in zip(face,x):c[i]=v
            if min(c)<0:continue
            dual=[x-y for x,y in zip(mv(a,c),g)]
            if any(dual[i]<-tol for i in range(n) if i not in face):continue
            if any(abs(dual[i])>tol for i in face):continue
            valid.append((.5*dot(c,mv(a,c))-dot(g,c),face,c,dual))
    if not valid:raise ArithmeticError('KKT unresolved')
    return min(valid,key=lambda r:(r[0],r[1]))

def main():
    records=list(csv.DictReader(INPUT.open()))
    spectral=[];sensitivity=[];reproduction=[]
    for r in records:
        h,g,gram,c0,q=(json.loads(r[k]) for k in ('full_H','g','G','c','q_layers'))
        n=len(g)
        assert n==5 and gram==[[float(i==j) for j in range(n)] for i in range(n)]
        e=eig(h)
        _,support,c,dual=nnls(g,h,gram,.1)
        rel=math.sqrt(dot([a-b for a,b in zip(c,c0)],[a-b for a,b in zip(c,c0)]))/max(1e-30,math.sqrt(dot(c0,c0)))
        assert rel<1e-9,(r['alias'],r['batch'],r['node'],rel)
        reproduction.append(rel)
        v=float(r['V_before'])
        unc=solve(h,g)
        spectral.append(dict(alias=r['alias'],batch=int(r['batch']),node=int(r['node']),
            eigenvalues=json.dumps(e),condition=e[-1]/e[0],lambda_over_min_eigenvalue=.1/e[0],
            regularized_support=json.dumps([i+4 for i in support]),g=json.dumps(g),c=json.dumps(c0),
            response_cosine=json.dumps(json.loads(r['response_residual_cosine'])),
            nonnegative_predicted_response_sq=dot(c0,mv(h,c0)),
            unconstrained_explainable_fraction=dot(g,unc)/(2*v),
            normalized_model_error=float(r['model_error_normalized']),
            model_error_over_linear_step=float(r['model_error_normalized'])/(.5*math.sqrt(dot(c0,mv(h,c0)))) if dot(c0,mv(h,c0))>0 else None,
            coefficient_reproduction_relative_error=rel))
        for k in range(13):
            lam=10**(-3+k/4)
            obj,face,c,dual=nnls(g,h,gram,lam)
            change=[a-b for a,b in zip(c,c0)]
            cn,cn0=dot(c,c),dot(c0,c0)
            row=dict(alias=r['alias'],batch=int(r['batch']),node=int(r['node']),lambda_value=lam,
                support=json.dumps([i+4 for i in face]),c=json.dumps(c),
                raw_coefficients=json.dumps([v/math.sqrt(w) for v,w in zip(c,q)]),
                physical_native_relative_change=math.sqrt(dot(change,change)/cn0) if cn0>0 else None,
                native_cosine=dot(c,c0)/math.sqrt(cn*cn0) if cn>0 and cn0>0 else None,
                native_velocity_action=cn,gain=dot(g,c),response_sq=dot(c,mv(h,c)),
                objective_improvement=-obj,l8_native_velocity_share=c[-1]**2/cn if cn else None,
                lambda_over_min_eigenvalue=lam/e[0],
                analysis_type='FROZEN_RECORDED_STATE_DICTIONARY_RESPONSE_ONLY_NO_TRAJECTORY')
            sensitivity.append(row)
    for name,rows in [('llama-spectrum-all80.csv',spectral),('llama-lambda-shadow-all1040.csv',sensitivity)]:
        with (OUT/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(input=str(INPUT),input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),nodes=len(records),
        shadows=len(sensitivity),max_source_coefficient_reproduction_relative_error=max(reproduction),
        numerical_policy='Independent FP64 Python active-face elimination, original recorded matrices; Jacobi eigenvalues on symmetric part; no model/GPU or new data',
        llama_b10_spectrum=[r for r in spectral if r['alias']=='llama3-8b-inst' and r['batch']==10],
        llama_b10_max_relative_coefficient_change=max(r['physical_native_relative_change'] for r in sensitivity if r['alias']=='llama3-8b-inst' and r['batch']==10),
        outputs={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['llama-spectrum-all80.csv','llama-lambda-shadow-all1040.csv']})
    with (OUT/'llama-sensitivity-summary.json').open('x') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
