"""CPU preparation/rehash CLI. This entrypoint cannot submit or load a model.

The injected D/S drivers are separate from resource admission. GPU execution
must bind a future user budget, exact committed source and asset/fidelity seals;
this preparation package is not a pre-GPU PASS or a budget authorization.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from .contracts import INSTRUCTION_ID,RUNTIME_HEAD,PUBLICATION_HEAD,CONTRACT_SHA256
from .publication import member,write_once,seal_package,verify_package,digest,safe_path
from .resources import estimate_from_publication
from .sampling import prepare_sample_manifest
from .sweep import dry_plan

PACKAGE='project/run_scripts/alpha_jv_llama_diagnosis'
REPORT='experiment-reports/servers/server1/alpha-jv-llama-diagnosis-sweep-2026-09-07-v1'
DESIGN='experiment-reports/global/alpha-jv-llama-diagnosis-and-sweep-2026-09-07-v1'
PLAN='experiment-reports/global/alpha-jv-parallel-research-plan-2026-09-07-v1'
MECHANISM='experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2'
DISPATCH=Path('/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-alpha-jv-diag-dispatch-20260907-v1/messages/head')
SESSION='01a04939-f93a-7b50-bca0-65438eab2062'


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE)


def full_read_inputs(repo):
    files=[repo/'PROTOCOL.md',repo/'servers/connection-inventory.md']
    files+=sorted((repo/DESIGN).iterdir())+sorted((repo/PLAN).iterdir())
    files += [repo/MECHANISM/p for p in ('gh-mechanism-review-ko.md','sweep-candidates.csv',
        'sweep-candidate-policy.json','supporting/llama-audit.md')]
    files += [repo/'experiment-reports/servers/server1/alpha-jv-sequential1000-layer-review-2026-09-07-v1/factual-report-ko.md',
        DISPATCH/'2026-09-07-alpha-jv-llama-diagnosis-sweep-sh1-v1.md',
        DISPATCH/'2026-09-07-alpha-jv-llama-diagnosis-sweep-operational-addendum.md']
    expected={str(repo/DESIGN/'gh-execution-prompt-ko.md'):CONTRACT_SHA256,
        str(files[-2]):'9b9ab0b4c7345d5e5a6793457852ece08022cdd100301e5e13213b04e55d75a1',
        str(files[-1]):'cef298e6c0b4e4613727483ca20b42d2bf787c09d712cc1099e12387a87b5603'}
    rows=[]
    for path in files:
        fact=member(path);data=path.read_bytes();data.decode('utf-8')
        if str(path) in expected and fact['sha256']!=expected[str(path)]:raise ValueError('AUTHORITATIVE_INPUT_SHA')
        rows.append(dict(fact,wc_lines=data.count(b'\n'),logical_records=len(data.splitlines()),
            terminal_newline=data.endswith(b'\n'),read_status='FULL_READ_PASS',
            reading_scope='main agent full read plus reproducible byte rehash'))
    return rows


def source_closure(repo):
    prefixes=['project/run_scripts/'+p for p in ('native_response_ode_v31',
        'alpha_native_response_ode_v31_sequential','ordered_response_barrier_ode')]
    names=git(repo,'ls-files',*prefixes).decode().splitlines()
    rows=[]
    for name in names:
        path=repo/name
        original=git(repo,'show',RUNTIME_HEAD+':'+name)
        fact=member(path,relative_to=repo)
        if hashlib.sha256(original).hexdigest()!=fact['sha256']:raise ValueError('REUSED_RUNTIME_CHANGED: '+name)
        rows.append(fact)
    return dict(runtime_head=RUNTIME_HEAD,members=rows,members_root=digest(rows),
                byte_unchanged_from_runtime=True,validation_scope='tracked package bytes, not remote raw rehash')


def cpu_checks(repo):
    tests=sorted('project.run_scripts.alpha_jv_llama_diagnosis.'+p.stem for p in (repo/PACKAGE).glob('test_*.py'))
    command=[sys.executable,'-m','unittest','-q',*tests]
    proc=subprocess.run(command,cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    text=proc.stdout+proc.stderr
    if proc.returncode:raise RuntimeError('FOCUSED_CPU_FAILURE\n'+text)
    count=re.search(r'Ran (\d+) tests?',text)
    if not count:raise ValueError('CPU_TEST_COUNT_MISSING')
    compile_command=[sys.executable,'-m','py_compile',*[str(p) for p in sorted((repo/PACKAGE).glob('*.py'))]]
    subprocess.run(compile_command,cwd=repo,check=True)
    session=subprocess.check_output(['bash','scripts/check-session-boundary.sh',SESSION],cwd=repo,text=True)
    subprocess.run(['git','diff','--check'],cwd=repo,check=True)
    return dict(status='PASS_CPU_ONLY',count=int(count[1]),command=command,output=text,
        py_compile_command=compile_command,session=session.strip(),python=platform.python_version(),
        CPU_fixture_actions_separate_from_experiments=True,GPU_fidelity='NOT_RUN_BUDGET_UNASSIGNED')


def prepare(repo,output):
    from .diagnosis_driver import read_publication_availability
    import torch
    if torch.cuda.is_initialized():raise ValueError('CPU_PREPARATION_CUDA_ALREADY_INITIALIZED')
    repo=Path(os.path.abspath(repo));output=safe_path(output,root=repo/REPORT,make_parents=True)
    if output.exists():raise FileExistsError('CREATE_ONCE_PREPARATION_EXISTS')
    inputs=full_read_inputs(repo);closure=source_closure(repo)
    sample=prepare_sample_manifest(repo,Path('/mnt/raid5/janghj/EasyEdit/data/counterfact/counterfact.json'))
    availability=read_publication_availability(repo);cost=estimate_from_publication(repo);plan=dry_plan()
    checks=cpu_checks(repo)
    new_sources=[member(p,relative_to=repo) for p in sorted((repo/PACKAGE).glob('*')) if p.is_file()]
    source=dict(parent_HEAD=git(repo,'rev-parse','HEAD').decode().strip(),
        parent_tree=git(repo,'rev-parse','HEAD^{tree}').decode().strip(),
        origin_main_observed=git(repo,'rev-parse','origin/main').decode().strip(),
        branch=git(repo,'branch','--show-current').decode().strip(),worktree=str(repo),
        instruction_id=INSTRUCTION_ID,contract_sha256=CONTRACT_SHA256,
        completed_publication_head=PUBLICATION_HEAD,implementation_members=new_sources,
        implementation_members_root=digest(new_sources),reused_runtime_closure=closure,
        execution_commit_status='NOT_SUBMITTED_DEDICATED_COMMIT_BOUND_ON_FUTURE_RELEASE',
        main_merge_authority='GH_REVIEW_ONLY')
    output.mkdir(mode=0o700)
    values={'full-read-inputs.json':inputs,'source.lock.json':source,'cpu_checks.json':checks,
        'sample.lock.json':sample,'artifact_availability.json':availability,'cost-estimate.json':cost,
        'dry-plan.json':plan,'resource.lock.json':dict(project_GPU_cap=2,mem_MiB_per_GPU=182272,
            gpu_hour_cap=None,budget_authority=None,status='GPU_HOUR_BUDGET_UNASSIGNED',
            local_registry=member(repo/'servers/local/gpu-caps.tsv'),
            canonical_registry=member(Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')),
            tracked_ceiling=member(repo/'control/gpu-concurrency-policy.tsv'),
            current_allocations_reserved=False,scheduler_recheck_required_before_submit=True,
            existing_job_mutation_count=0,old_budget_inheritance_count=0),
        'science.lock.json':dict(runtime_parent=RUNTIME_HEAD,stock_compute_z_unchanged=True,
            fixed_z_once_per_model_cohort=True,native_RHS_divisor=1,normalization_primary='N0_SOURCE',
            diagnosis_only_alternative='NRMS_ENTRY',actual_S_plan=plan,
            controller_dtype='FP64',model_forward_dtype='FP32',scientific_promotion=False),
        'runtime.lock.json':dict(status='CPU_MODULES_PREPARED_GPU_BINDING_NOT_RELEASED',
            model_load='NOT_RUN',GPU_fidelity='NOT_RUN',Slurm_submission='NOT_RUN',
            production_launcher='NOT_SEALED_PENDING_BUDGET_AND_ASSET_RELEASE',
            CPU_python=sys.executable,CPU_torch_version=torch.__version__,
            future_first_fidelity='normal-signal entry and nonzero node, source tolerances unchanged')}
    for name,value in values.items():write_once(output/name,value,root=output)
    rows=[]
    for model in plan['models']:
        for endpoint in plan['endpoints']:
            rows.append(dict(model=model,status='NOT_RUN_GPU_HOUR_BUDGET_UNASSIGNED',**endpoint))
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_once(output/'run_registry.csv',out.getvalue().encode(),root=output)
    status=availability['availability']['status']
    report=f'''# Alpha JV D/S CPU 준비 및 비용 보고 — GPU 결과 아님

상태: CPU_PREPARATION_COMPLETE / GPU_HOUR_BUDGET_UNASSIGNED. scientific_promotion=false.
이 패키지는 실행 결과나 PRE-GPU PASS가 아니다. 신규 model/GPU/Slurm/과학 endpoint 모두 0.

## 범위와 담당

SH1은 공통 config/fixture/trajectory 통합, D owner는 artifact/normalization/requestwise,
S owner는 sample/grid/driver, 독립 resource reviewer는 cap/budget 및 공통 경로 검사를 담당했다.
신규 소스는 `{PACKAGE}/`로 제한하며 기존 runtime {RUNTIME_HEAD}의
세 재사용 패키지 {len(closure['members'])}개 tracked member를 byte 비교했다. 수정 0.
main은 GH 검토 후에만 통합한다. 전용 branch push는 CPU 준비물만 포함한다.

## FULL_READ / CPU 검증

계약 SHA `{CONTRACT_SHA256}` 및 operational addendum을 포함한 전체 읽기/재해시는
`full-read-inputs.json`에 기록했다. focused CPU {checks['count']}/{checks['count']} PASS,
py_compile/session/diff PASS. Actual GPU fidelity는 아직 NOT_RUN이다.
기존 source N0/.1/T2/N4와 새 공통 루프는 CPU fixture에서 node 계수/q/V/E/endpoint가
일치했다. Prefix observer 및 raw/terminal sink의 RNG 개입은 fail-close하고 W0/RNG를 복원한다.
이 CPU 일치는 모델 fidelity 또는 과학적 효능의 증명이 아니다.

## D availability

`{status}`. Publication은 W1/W5/W10만 기재하고 W9/fixed-z tensor/journal exact replay는
검증되지 않았다. 명시된 W5/context 로컬 경로는 부재이며 Server2 연결/복구는 하지 않았다.
W10을 W9로 대체하거나 W0부터 재실행하지 않는다. D의 exact-state blocker는 S dependency가 아니다.
Raw가 없는 원인 attribution, NRMS intervention, old900 변화는 결과를 만들지 않고 NOT_RUN으로 둔다.

## S outcome-independent seal

양 모델 동일 DEV100, 별도 disjoint AUDIT300을 Git-only exclusion inventory와 hash-rank로
봉인했다. 모델 결과/target 길이로 replacement하지 않았다. DEV RS/PS/NS 분모는 실제 입력의
100/200/1000이며 임의 locality 중복/제거 0이다. ID·order·hash만 공개하고 raw prompt는 Git 제외.
다섯 λ `.01,.03162277660168379,.1,.31622776601683794,1`은 모두 actual 예정이며
양 모델 각 7 paths/34 nodes/170 main JVP, 9 JV endpoints + Official + entry다.
아직 관측 endpoint는 0/18이며 표의 NOT_RUN은 과학 실패 또는 endpoint로 집계하지 않는다.
T4/h.5의 T1/T2 prefix는 중간 평가만 수행하며 append/capture 0, 이후 trajectory 불변을 검사한다.
감사 표본 300개는 예약만 했으며 실행 권한/후속 선택 규칙을 자동으로 발명하지 않는다.

## 비용과 첫 wave

| 범위 | 과거 기록 기반 GPUh proxy | 제외/한계 |
|---|---:|---|
| S Llama | {cost['models']['llama3-8b-inst']['S_recorded_component_proxy_gpu_hours']['mean_component_proxy']:.6f} | 추가 FD·복원·reserve 미계측 |
| S Qwen | {cost['models']['qwen2.5-7b-inst']['S_recorded_component_proxy_gpu_hours']['mean_component_proxy']:.6f} | 위와 동일 |
| S 합계 | {cost['S_both_models_recorded_component_proxy_gpu_hours']['mean_component_proxy']:.6f} | 모델 로드/z/기존 writer/eval 성분 포함, 상한 아님 |
| D 기본 current100 | {cost['models']['llama3-8b-inst']['D_main_current100_recorded_component_proxy_gpu_hours']:.6f} | old900·capture·controls 제외 |
| 선택적 W5→W9 replay | {cost['models']['llama3-8b-inst']['D_optional_W5_to_W9_exact_replay_historical_gpu_hours']:.6f} | 파일 부재, 예약/실행 0 |

비용 산식/입력 SHA/미계측 항목은 `cost-estimate.json`에 전부 있다. 과거 host/state의 성분별
min/max는 신뢰구간·실행 상한이 아니다. 사용자 GPU-hour cap은 null이며 과거 8h/48h를 상속하지 않는다.
첫 GPU wave는 새 budget·source/asset·정상 신호 fidelity 결속 후 양 모델 S_DEV 비교를 cap2 내에서
시작한다. 먼저 양 모델 BASE/time/coarse λ, 이후 양 모델 intermediate λ actual을 수행할 계획이다.
기존 job은 그대로 두며 RUNNING/CONFIGURING/미해제 COMPLETING 포함 재계수, scheduler 실패는 HOLD다.
task 메모리 182272M/1GPU, local 상한183296M. 자원 조회 성공은 예약이 아니다.

## 남은 경계

GPU-hour authority 및 exact launch/source/asset/fidelity 결속은 아직 없다. production launcher는
이번 CPU 패키지에서 봉인하지 않았다. D 실제 historical 재생/old900 endpoint, S actual 표와
PNG/효능/원인 판단은 NOT_RUN이며 추정하지 않는다. source/CPU/GPU 검증을 구분해 인계한다.
Server4 live source/process/output, 기존 원본/원격 raw 변경·접근 0. Git은 code/tests/이 raw-free 준비물만 포함한다.
'''
    write_once(output/'factual-report-ko.md',report.encode(),root=output)
    audit=repo/'audits/servers/server1/2026-09-07-alpha-jv-llama-diagnosis-sweep/resource-review.md'
    write_once(output/'independent-CPU-review.json',dict(review=member(audit),
        scope='bounded CPU callback and source-exact trajectory plus resource accounting',
        status='PASS_CPU_ONLY',actual_GPU_fidelity='NOT_RUN'),root=output)
    after=full_read_inputs(repo)
    if inputs!=after or closure!=source_closure(repo):raise ValueError('INPUT_SOURCE_MUTATED_DURING_CPU_PREPARATION')
    write_once(output/'input-immutability.json',dict(input_member_count=len(inputs),
        reused_runtime_member_count=len(closure['members']),before_after_equal=True,
        external_server2_raw_rehash='NOT_PERFORMED_HOST_UNAVAILABLE',
        imputation_count=0,interpolation_count=0,scientific_endpoint_count=0),root=output)
    return seal_package(output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('prepare','verify'))
    parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args.repo,args.output) if args.operation=='prepare' else verify_package(args.output)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
