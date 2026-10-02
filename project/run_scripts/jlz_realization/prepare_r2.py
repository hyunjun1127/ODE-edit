"""Create-once diagnostic-only config and explicit Q1 evidence bridge inputs."""
import json
from .common import LOCAL,write,sha

def main():
    old=LOCAL/'preparation-v1/configuration.json';config=json.loads(old.read_text())
    config['settings'].update(allocation_metric='ridge_optimal_value_root',
        telemetry_revision='r2: context direction, exact operator, radial and terminal KL; FP32epsilon descriptive rho flag')
    path=LOCAL/'repair-r2/configuration.json';write(path,config)
    print(json.dumps(dict(path=str(path),sha256=sha(path),new_GPU_fits=0)))

if __name__=='__main__':main()
