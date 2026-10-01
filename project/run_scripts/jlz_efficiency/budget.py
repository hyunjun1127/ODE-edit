"""One shared actual-call ledger. No free nested reference evaluations."""
class BudgetAccountant:
    def __init__(self,cap):
        if cap<2:raise ValueError('FINAL_RESERVE_REQUIRED')
        self.cap=cap;self.used=0;self.events=[];self.final=False
    @property
    def remaining_trials(self):return max(0,self.cap-self.used-1)
    def charge(self,role):
        if self.final:raise RuntimeError('AFTER_FINAL')
        if role=='final':
            if self.used>=self.cap:raise RuntimeError('FINAL_RESERVE_LOST')
            self.final=True
        elif self.remaining_trials<=0:raise RuntimeError('BUDGET_EXHAUSTED_FINAL_RESERVED')
        self.used+=1;self.events.append(role)
    def report(self):return dict(cap=self.cap,actual_calls=self.used,events=list(self.events),final_reserved=True,final_started=self.final)

class PanelBudget:
    def __init__(self):self.counts={'fixed':0,'reserve':0,'B100':0};self.short={}
    def charge(self,phase,route=''):
        if phase=='short':
            n=self.short.get(route,0)+1
            if n>12 or (route not in self.short and len(self.short)>=8):raise RuntimeError('SHORT_CAP')
            self.short[route]=n
        else:
            caps={'fixed':32,'reserve':32,'B100':8}
            if self.counts[phase]>=caps[phase]:raise RuntimeError('PANEL_CAP_'+phase)
            self.counts[phase]+=1
        if self.counts['fixed']+self.counts['reserve']+sum(self.short.values())>160:raise RuntimeError('SMALL160_CAP')
    def report(self):return dict(separate=self.counts,short=self.short,small_total=self.counts['fixed']+self.counts['reserve']+sum(self.short.values()))
