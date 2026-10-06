"""Same MEMIT/Alpha operator choice, no hybrid or new solver."""
from project.run_scripts.jlz_realized_subject import geometry as memit
from . import alpha_geometry

def prior(a,path,history,layer):
    if a.profile['writer']=='alphaedit':
        return alpha_geometry.prior(path,history,a.device,a.alpha_projector,layer)
    return memit.prior(path,history,a.device,a.profile['lambda_C'])

def ridge(a,K,factor):
    return (alpha_geometry if a.profile['writer']=='alphaedit' else memit).ridge(K,factor)
