"""FP64 native-whitened exact elastic controller (design sections 8--11)."""
import torch

RANK_TOL = 1e-10

def dot(a,b): return (a.double()*b.double()).sum()

def spd_roots(b):
    values,vectors=torch.linalg.eigh(b.double())
    if not torch.isfinite(values).all() or values.min()<=0: raise ValueError('NATIVE_METRIC_NOT_SPD')
    root=(vectors*values.sqrt())@vectors.T
    invroot=(vectors*values.rsqrt())@vectors.T
    return root,invroot,values

def observer_space(j, invroot):
    white=(j.double()@invroot).flatten(1)
    norms=white.norm(dim=1)
    kept=norms>0
    if not kept.any(): return white.new_empty((0,white.shape[1])),dict(rank=0,zero_rows=len(j),singular_values=[])
    _,s,vh=torch.linalg.svd(white[kept]/norms[kept,None],full_matrices=False)
    rank=int((s>s[0]*RANK_TOL).sum())
    return vh[:rank],dict(rank=rank,zero_rows=int((~kept).sum()),singular_values=s.tolist(),rank_tolerance=RANK_TOL)

def project(a,j,invroot):
    aw=(a.double()@invroot).flatten()
    rows,receipt=observer_space(j,invroot)
    free=aw-rows.T@(rows@aw)
    w=free.reshape_as(a)@invroot
    return w,dot(free,free),dot(aw,aw),rows,receipt

def correction(v0,a,j,root,invroot,epsilon,hbar,arm):
    v0=v0.double();a=a.double();j=j.double()
    w,q,qall,rows,receipt=project(a,j,invroot)
    e=dot(a,v0)-2*hbar
    if arm=='H': factor=e.new_zeros(());velocity=v0;slack=e.clamp_min(0)
    elif arm=='R':
        w=(a@invroot)@invroot;q=qall
        factor=e.clamp_min(0)/(q+epsilon) if q+epsilon>0 else e.new_zeros(())
        velocity=v0-factor*w;slack=e.clamp_min(0)-factor*q
    elif arm=='EP-Free':
        vw=(v0@root).flatten();vf=(vw-rows.T@(rows@vw)).reshape_as(v0)@invroot
        numerator=dot(a,vf).clamp_min(0)
        factor=numerator/q if q>0 else q.new_zeros(())
        velocity=v0-factor*w;slack=(dot(a,velocity)-2*hbar).clamp_min(0)
    elif arm in ('EP','EP-J4','EP-N16'):
        factor=e.clamp_min(0)/(q+epsilon) if q+epsilon>0 else e.new_zeros(())
        velocity=v0-factor*w;slack=e.clamp_min(0)-factor*q
    else: raise ValueError(arm)
    if not torch.isfinite(velocity).all():raise FloatingPointError('NONFINITE_CONTROLLER')
    delta=velocity-v0
    receipt.update(q=float(q),q_all=float(qall),q_fraction=float(q/qall) if qall>0 else None,
                   e=float(e),epsilon=float(epsilon),slack=float(slack),factor=float(factor),
                   progress_leakage=(j*delta).flatten(1).sum(1).tolist(),
                   jv0=(j*v0).flatten(1).sum(1).tolist(),
                   correction_native_norm=float((delta@root).norm()),nominal_native_norm=float((v0@root).norm()),
                   correction_risk_derivative=float(dot(a,delta)),observer_empty=receipt['rank']==0,
                   equality_required=arm in ('EP','EP-J4','EP-N16','EP-Free'))
    return velocity,receipt

def calibrate(g,hn,u,native_norm,a0,hc):
    # A symmetric SPD solve here is coefficient geometry, NOT the nonsymmetric
    # Official Alpha normal equation. No dense coefficient-space matrix exists.
    unscaled=-torch.linalg.solve(hn.double(),g.double().T).T
    physical_norm=(unscaled@u.double().T).norm()
    stationary=bool(native_norm==0 or physical_norm==0)
    nu=float(.08*native_norm/physical_norm) if not stationary else 1.
    root,invroot,spectrum=spd_roots(hn/nu)
    v0=nu*unscaled if not stationary else torch.zeros_like(unscaled)
    qall=dot(a0@invroot,a0@invroot)
    curvature=torch.linalg.matrix_norm(invroot@hc@invroot,ord=2)
    reference=dot((v0/8)@root,(v0/8)@root)
    qref=qall+curvature.square()*reference
    return nu,root,invroot,.1*qref,dict(nu=nu,q_ref=float(qref),q_all_entry=float(qall),
          curvature_norm=float(curvature),reference_native_action=float(reference),d_ref_h=.125,
          stationary_nominal=stationary,flat_risk=bool(qall==0 and curvature==0),
          metric_eigen_min=float(spectrum.min()),metric_eigen_max=float(spectrum.max()))

def risk(x,cc,hc,sf):
    value=(dot(cc,x)+.5*dot(x@hc,x))/sf
    return value,(cc+x@hc)/sf

def groups(case_ids):
    import hashlib
    order=sorted(range(len(case_ids)),key=lambda i:(hashlib.sha256(('L4-EP-GROUP-V1|'+str(case_ids[i])).encode()).hexdigest(),str(case_ids[i])))
    assert len(order)==100
    return [order[k:k+25] for k in range(0,100,25)]
