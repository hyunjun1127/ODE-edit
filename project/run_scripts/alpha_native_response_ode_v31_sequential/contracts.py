"""Task-local constants; the v3.1 numerical primitives remain authoritative."""
from pathlib import Path

INSTRUCTION_ID = 'ODEEDIT-S06-ALPHA-JV-SEQUENTIAL-ROUTING-SH2-V1'
NAMESPACE = 'ALPHA-JV-SEQUENTIAL-ROUTING-20260906'
LAYERS = (4, 5, 6, 7, 8)
ARMS = ('O_NATIVE', 'JV_NATIVE', 'L8_ONLY_NATIVE')
ALIASES = ('llama3-8b-inst', 'qwen2.5-7b-inst')
CHAINS = tuple((m,a) for a in ARMS for m in ALIASES)
T, N, H, LAMBDA = 2., 4, .5, .1
CHECKPOINTS = (1, 5, 10)
ROOT = Path('/mnt/raid5/janghj/ODE-edit/local/alpha-native-response-v31-sequential-routing')
PILOT = Path('experiment-reports/servers/server1/native-response-v31-b10-warm-pilot-2026-09-06-v1/primary/sample.lock.json')
SCIENCE = dict(instruction_id=INSTRUCTION_ID, normalization='SOURCE_EXACT_N0',
    T=T,N=N,h=H,lambda_value=LAMBDA,layers=list(LAYERS),arms=list(ARMS),
    main_batch_size=100,batches=10,main_requested_edits=1000,
    first_hit=False,backtracking=False,gain=False,normalization_sweep=False,
    cold_initialization_count=1,history_append_per_batch=1,
    dynamic_z_per_batch=True,inner_z_reoptimization_count=0,inner_persistent_mutation_count=0,
    checkpoint_batches=list(CHECKPOINTS),main_priority=[0,1,2,3],secondary_priority=[4,5],
    model_dtype='float32',controller_dtype='float64',scientific_promotion=False)
