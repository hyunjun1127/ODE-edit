"""서버4 1회 CPU setup smoke 전용. 공통 실험 logger 구현이 아님."""
import contextlib
import io
import json
import netrc
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path('/data/janghj/ODE-edit/local/wandb-setup')
OUT = ROOT / 'smoke-receipt.json'
RUN_ID = 'server4-setup-20261006'


def main():
    # 재실행으로 중복 online run/point를 만들지 않는다.
    if OUT.exists():
        raise SystemExit('EXISTING_RECEIPT_REUSE_REQUIRED')
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['WANDB_DISABLED'] = 'false'
    os.environ['WANDB_MODE'] = 'online'
    os.environ['WANDB_CONSOLE'] = 'off'
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_QUIET'] = 'true'
    os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'
    ROOT.joinpath('spool').mkdir(exist_ok=True)
    receipt = dict(instruction_id='USER-GH-ALL-SH-WANDB-REALTIME-20261006-SERVER4',
                   server='server4', status='SETUP_STARTED', gpu=0, model_load=0,
                   slurm_writes=0, max_points=3, python=sys.executable,
                   entity='wkdguswns2256', project='layer allocation',
                   api_endpoint='https://api.wandb.ai', independent_reviewer=False)
    run = None
    started = time.monotonic()
    phase = 'sdk'
    try:
        import wandb
        receipt.update(sdk_version=wandb.__version__, sdk_path=wandb.__file__)
        # 존재만 기록. key/login/account 값 및 hash는 저장하거나 출력하지 않는다.
        auth = netrc.netrc().authenticators('api.wandb.ai')
        receipt['credential_present'] = bool(os.getenv('WANDB_API_KEY') or (auth and auth[2]))
        del auth
        if not receipt['credential_present']:
            receipt['status'] = 'SETUP_READY_NEEDS_USER_LOGIN'
            return
        settings = wandb.Settings(
            mode='online', base_url='https://api.wandb.ai', console='off',
            save_code=False, disable_code=True, disable_git=True,
            disable_job_creation=True, x_disable_meta=True, x_disable_stats=True,
            x_disable_machine_info=True, x_save_requirements=False,
            capture_loggers={}, init_timeout=30, login_timeout=15,
            finish_timeout=30, finish_timeout_raises=True,
            x_graphql_timeout_seconds=15, x_file_transfer_timeout_seconds=15,
            silent=True, quiet=True)
        receipt['privacy'] = {k: getattr(settings, k) for k in (
            'mode', 'console', 'save_code', 'disable_code', 'disable_git',
            'x_disable_meta', 'x_disable_stats', 'x_disable_machine_info',
            'x_save_requirements', 'capture_loggers', 'finish_timeout')}
        receipt['watch_calls'] = receipt['artifact_calls'] = 0
        # SDK terminal text is discarded, not published or copied into reports.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            phase = 'online_init_auth_project'
            run = wandb.init(entity='wkdguswns2256', project='layer allocation',
                             id=RUN_ID, name='server4-setup', group='wandb-realtime-setup',
                             job_type='cpu-setup-smoke', resume='never',
                             config={'server': 'server4'}, dir=str(ROOT / 'spool'),
                             settings=settings)
            assert run.settings.mode == 'online'
            receipt.update(run_id=run.id, run_url=run.url, auth_project_verified=True)
            phase = 'three_scalar_points'
            for step in range(3):
                payload = {'setup_ok': 1.0, 'step': step}
                assert set(payload) == {'setup_ok', 'step'}
                assert all(type(v) in (int, float) for v in payload.values())
                run.log(payload, step=step, commit=True)
            receipt['logged_points'] = 3
            phase = 'bounded_finish'
            run.finish()
            run = None
            receipt['finish_verified'] = True
            phase = 'remote_readback'
            api = wandb.Api(overrides={'base_url': 'https://api.wandb.ai'}, timeout=15)
            remote = api.run('wkdguswns2256/layer allocation/' + RUN_ID)
            rows = list(remote.scan_history(keys=['setup_ok', 'step', '_step'], page_size=10))
            assert len(rows) == 3 and [r['step'] for r in rows] == [0, 1, 2]
            assert [r['_step'] for r in rows] == [0, 1, 2]
            assert all(r['setup_ok'] == 1.0 for r in rows)
            files = [f.name for f in remote.files(per_page=20)]
            receipt.update(remote_points=rows, remote_files=files, remote_state=remote.state)
            assert remote.state == 'finished'
            assert not any(n.endswith(('.py', '.patch')) or n.startswith('code/')
                           or 'output.log' in n or 'requirements' in n or 'metadata' in n for n in files)
            receipt['status'] = 'READY_ONLINE_VERIFIED'
    except Exception as exc:
        receipt.update(status='LOGGING_BLOCKED', failed_phase=phase,
                       error_type=type(exc).__name__)
        # 예외 문자열은 인증/HTTP 내용을 포함할 수 있어 기록하지 않는다.
    finally:
        if run is not None:
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    run.finish(exit_code=1)
            except Exception as exc:
                receipt['finish_error_type'] = type(exc).__name__
        receipt['wall_seconds'] = time.monotonic() - started
        receipt['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        with OUT.open('x') as handle:
            json.dump(receipt, handle, indent=2, ensure_ascii=False)
        print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
