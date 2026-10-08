"""Explicit observation schedule; edit/RPN schedules are unchanged."""
LEGACY='CURRENT_AND_MILESTONES'
FINAL='W20_ONLY'

def generation_due(config,batch):
    schedule=config['generation'].get('schedule',LEGACY)
    if schedule not in (LEGACY,FINAL):raise ValueError('UNKNOWN_GENERATION_SCHEDULE')
    if not 1<=batch<=20:raise ValueError('GENERATION_BATCH_OUT_OF_RANGE')
    return schedule==LEGACY or batch==20

def first_generation_batch(config):
    return next(b for b in range(1,21) if generation_due(config,b))
