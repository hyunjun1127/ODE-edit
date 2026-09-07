"""Reuse sealed analysis reducers, separately from the executed runtime."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ..sequential_analysis import (ARMS,CELLS,require,deep_identity,canonical_v2_reducer,evaluation,summarize_requests,write_csv,stats)
from ..round0_analysis_contracts import (canonical_hash,sha256_file,member,write_json_once,STREAM_ROOT,ORDER_ROOT)

SOURCE=('8610faf0e114059a5e08116f1164baf31c573e80','bec9b180de905e40ccc9e810b0a5f6a0d7c7797a')
INSTRUCTION='ODEEDIT-S06-ORBODE-CUMULATIVE-RERUN-EXHAUSTIVE-REPORT-SH4-V1'
RAW=Path('/data/janghj/ODE-edit/local/state/orbode-cumulative-rerun-v1/execution-r2')

def read(p):return json.loads(Path(p).read_text())
def frame_write(p,rows):write_csv(Path(p),rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows))
def strict_stats(values):
    v=[float(x) for x in values if x is not None and not pd.isna(x)]
    require(all(np.isfinite(v)), 'nonfinite aggregate input')
    return stats(v)
def scalar_summary(rows,fields):
    return {f+'_'+k:v for f in fields for k,v in strict_stats(r.get(f) for r in rows).items()}
