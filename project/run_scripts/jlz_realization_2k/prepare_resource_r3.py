"""Explicit stricter Slurm memory binding; no input or science recomputation."""
import copy
import json
from .common import *

def main():
    original=LOCAL/'preparation-r2/configuration.json'
    c=json.loads(original.read_text());old=copy.deepcopy(c)
    require(c['resources']['host_mib']==60416,'ORIGINAL_MEMORY')
    c['resources']['host_mib']=59392
    require(sum(c['resources']['host_plan_GiB'].values())<=58,'HOST_PLAN_EXCEEDS_CURRENT_SITE_CAP')
    c['resource_repair']=dict(source=member(original),reason='Slurm submit plugin: 1GPU on server4 <=58G',
        first_attempt='attempt-r1',jobs_created=0,old_memory_mib=60416,new_memory_mib=59392,
        science_unchanged=True,original_error='CPU/RAM policy: use --mem up to 58G on server4 for 1 GPU(s)')
    restored=copy.deepcopy(c);restored.pop('resource_repair');restored['resources']['host_mib']=60416
    require(restored==old,'ONLY_RESOURCE_CHANGE')
    write(LOCAL/'preparation-r3/configuration.json',c)
    print(json.dumps(dict(status='RESOURCE_BINDING_ONLY',configuration=str(LOCAL/'preparation-r3/configuration.json'))))

if __name__=='__main__':main()
