"""GH 통합 영수증과 zsRE 표 상태 대조. 모델·scheduler 조회 없음."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
LABEL={'FT':'FT','MEMIT':'MEMIT','ALPHAEDIT':'AlphaEdit','ALPHAEDIT_BLUE':'AlphaEdit-BLUE',
       'MEMIT_FE':'MEMIT-FE','SPHERE':'AlphaEdit+SPHERE'}
TITLE={'llama3':'Llama3-8B-Instruct','gptj':'GPT-J-6B'}


def main():
    receipt=json.loads((Path(__file__).with_name('integration.json')).read_text())
    readme=(ROOT/'README.md').read_text();ids=[]
    for row in receipt['rows']:
        section=readme.split('### '+TITLE[row['model']]+'\n',1)[1].split('\n### ',1)[0]
        values=next([v.strip() for v in line.split('|')[1:-1]] for line in section.splitlines()
                    if line.startswith('| '+LABEL[row['method']]+' |'))
        assert len(values)==10
        if row.get('evaluation_job_id') is None:continue
        assert row['evaluation_status'] in ('PENDING','RUNNING')
        prefix='PENDING' if row['evaluation_status']=='PENDING' else 'ING'
        expected=prefix+': '+row['evaluation_job_name']+' ('+row['evaluation_job_id']+')'
        assert values[7:10]==[expected]*3,(row['model'],row['method'],values[7:10],expected)
        assert row['checkpoint']['sha256'] and row['evaluation_source']
        ids.append(row['evaluation_job_id'])
    assert len(ids)==len(set(ids))==12
    assert receipt['new_metric_rows']==0
    print(json.dumps(dict(status='PASS',registered_rows=12,zsre_status_cells=36,
                         new_metric_rows=0,job_ids=ids,model_forward_calls=0)))


if __name__=='__main__':main()
