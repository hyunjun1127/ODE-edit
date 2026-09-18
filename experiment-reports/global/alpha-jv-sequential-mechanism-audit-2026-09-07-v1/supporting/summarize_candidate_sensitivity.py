"""Summarize sealed CPU shadows only; does not authorize a GPU campaign."""
import csv
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
data=list(csv.DictReader((ROOT/'llama-lambda-shadow-all1040.csv').open()))
baseline={(x['alias'],x['batch'],x['node']):x for x in data if float(x['lambda_value'])==.1}
groups={
    'Llama_all40_including_B10': lambda r:r['alias']=='llama3-8b-inst',
    'Llama_B1_B9_36_diagnostic_stratum':lambda r:r['alias']=='llama3-8b-inst' and r['batch']!='10',
    'Llama_B10_4_diagnostic_stratum':lambda r:r['alias']=='llama3-8b-inst' and r['batch']=='10',
    'Qwen_all40':lambda r:r['alias']=='qwen2.5-7b-inst',
}
summary=[]
for group,keep in groups.items():
    for lam in sorted({float(x['lambda_value']) for x in data}):
        rows=[x for x in data if keep(x) and float(x['lambda_value'])==lam]
        values=[]
        for r in rows:
            b=baseline[(r['alias'],r['batch'],r['node'])]
            cosine=float(r['native_cosine'])
            values.append(dict(
                gain_ratio=float(r['gain'])/float(b['gain']),
                qN_ratio=float(r['native_velocity_action'])/float(b['native_velocity_action']),
                response_norm_ratio=math.sqrt(float(r['response_sq'])/float(b['response_sq'])),
                l8_native_velocity_share=float(r['l8_native_velocity_share']),
                l8_native_velocity_share_delta_pp=100*(float(r['l8_native_velocity_share'])-float(b['l8_native_velocity_share'])),
                native_angle_degrees=math.degrees(math.acos(max(-1.,min(1.,cosine)))),
                native_turn_fraction=max(0.,1-cosine*cosine),
                physical_native_relative_change=float(r['physical_native_relative_change']),
                support_change=int(r['support']!=b['support']),
            ))
        row=dict(group=group,lambda_value=lam,node_denominator=len(rows),baseline_lambda=.1,
            analysis_type='FROZEN_RECORDED_STATE_NO_ENDPOINT_PREDICTION')
        for k in values[0]:
            v=[x[k] for x in values]
            row.update({k+'_min':min(v),k+'_median':statistics.median(v),k+'_max':max(v)})
        row['support_changed_count']=sum(x['support_change'] for x in values)
        summary.append(row)
with (ROOT/'candidate-sensitivity-summary.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print('groups=4; rows='+str(len(summary))+'; source_shadow_rows='+str(len(data)))
