"""Create-once policy/source preparation for the explicit user recall."""
from .common import *
from .control import freeze


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    authority=RUN/'authority';authority.mkdir(exist_ok=False)
    names=['messages/head/2026-09-20-sh2-checkpoint-mechanism-remove-gates-resume.md',
           'tasks/pending/server2-checkpoint-mechanism-no-gates-resume-20260920-v1.json']
    members=[]
    for name in names:
        src=REPO/name;dst=authority/src.name
        with dst.open('xb') as f:f.write(src.read_bytes())
        members.append(dict(source=name,path=str(dst),sha256=sha256(dst),bytes=dst.stat().st_size))
    pause=ATTEMPT/'receipts/user-implementation-only-20260920.json'
    write_json(authority/'recall.json',dict(instruction_id='ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1',
        explicit_user_recall=True,pause_receipt_sha256=sha256(pause),
        user_message='아니 저런 gate 자체를 아예 없애. 저런 것 때문에 실험이 너무 지연된다',
        members=members,**EXECUTION_POLICY))
    write_json(authority/'plan.json',dict(**EXECUTION_POLICY,prior_allocated_gpu_seconds=269,
        prior_inputs=str(ATTEMPT),prior_inputs_read_only=True,checkpoint_saved=False,
        common_preparation='keys: reuse B001/B002 W0 capture; remaining native and full geometry512 capture',
        lanes={'operator':[0,1,10,100,5,20,30,40,50,60,70,80,90],
               'activation':['1:10','50:100']},
        dependencies='Both lanes start after common key-bank/F00 artifact availability; no numerical prerequisite',
        each_resource=dict(gpu=1,cpu=8,mem_mib=60416,wall_hours=8,export='NONE',requeue=False),
        simultaneous_request=dict(gpu=2,cpu=16,mem_mib=120832),
        estimates=dict(runtime='UNMEASURED_ANALYSIS; 8h/job conservative reservation',
            additional_disk_reserve_bytes=30*(1<<30),shared_immutable_inputs=True),
        tests=dict(command='python -B -m unittest project.run_scripts.checkpoint_mechanism_audit.test_no_gates project.run_scripts.checkpoint_mechanism_audit.test_operator_lane -q',
                   executed_cases=22,status='PASS',scope='CPU routing/small tensor only; repeated imported fixture tests included')))
    freeze(RUN/'execution-keys-r1',mode='keys',analysis_output=RUN/'results/keys-r1',
        analysis_args=['--gate',str(ATTEMPT/'results/model-gate-tech-r1/terminal.json')])


if __name__=='__main__':prepare()
