"""Execute exact frozen bash launcher; bounded job-lifetime telemetry companion."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main(attempt,ops):
    sys.path.insert(0,str(attempt/'source'))
    from project.run_scripts.experiment_tracking.schema import load_env
    c=json.loads((attempt/'config.json').read_text())['cells']['QWEN_M1_CAP075']
    settings=load_env(c['tracking']['env_file'])
    keys=('PATH','HOME','USER','LANG','LC_ALL','TMPDIR','SSL_CERT_FILE','REQUESTS_CA_BUNDLE',
          'NETRC','WANDB_API_KEY','WANDB_IDENTITY_TOKEN_FILE','WANDB_CREDENTIALS_FILE','WANDB_CONFIG_DIR')
    env={k:os.environ[k] for k in keys if k in os.environ}
    env.update(PYTHONPATH=str(attempt/'source'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
        WANDB_CONSOLE='off',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true')
    with (ops/'companion-private.log').open('x') as log:
        companion=subprocess.Popen([settings['ODEEDIT_WANDB_PYTHON'],str(Path(__file__).with_name('monitor.py')),
            '--attempt',str(attempt),'--ops',str(ops)],env=env,stdout=log,stderr=log)
        process=subprocess.run(['/bin/bash',str(attempt/'run.sh')],cwd=attempt)
        (ops/'process-finished.json').write_text(json.dumps({'exit_code':process.returncode})+'\n')
        try:companion.wait(timeout=180)
        except subprocess.TimeoutExpired:
            companion.terminate()
            try:companion.wait(timeout=5)
            except subprocess.TimeoutExpired:companion.kill();companion.wait()
    # Always CPU-reduce partial/terminal data even if startup/telemetry failed.
    sys.path.insert(0,str(Path(__file__).parent))
    from monitor import collect
    try:collect(attempt,ops,'terminal')
    except Exception as error:
        (ops/'collector-error.json').write_text(json.dumps({'type':type(error).__name__,'status':'NOT_VERIFIED'})+'\n')
    return process.returncode

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--ops',type=Path,required=True)
    a=p.parse_args();sys.exit(main(a.attempt.resolve(),a.ops.resolve()))
