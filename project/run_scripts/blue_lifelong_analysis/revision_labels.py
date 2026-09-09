"""Human labels never alter raw arm identifiers or scientific measurements."""
LABELS={
    'MEMIT_ORIGINAL':'MEMIT_BLUE (L4+L8)',
    'MEMIT_L4_ONLY':'MEMIT_BLUE_L4_ONLY',
    'MEMIT_L8_ONLY':'MEMIT_BLUE_L8_ONLY',
    'AlphaEdit_ORIGINAL':'AlphaEdit_BLUE (L4+L8)',
    'AlphaEdit_L4_ONLY':'AlphaEdit_BLUE_L4_ONLY',
    'AlphaEdit_L8_ONLY':'AlphaEdit_BLUE_L8_ONLY',
}
ORDER=list(LABELS)
CAPTION='측정된 여섯 arm은 모두 blue=True이며 stock/base MEMIT·AlphaEdit가 아니다. Raw arm ID는 provenance로만 유지한다.'

def display(text):
    for raw,label in LABELS.items():text=text.replace(raw,label)
    return text

def labeled_rows(rows):
    result=[]
    for r in rows:
        d=dict(r)
        for key in ('arm','reference','arm_a','arm_b'):
            if r.get(key) in LABELS:d[key+'_display_label']=LABELS[r[key]]
        result.append(d)
    return result
