# Local Runtime Artifacts

This directory is intentionally Git-ignored except for this README.

Use it for server-local runtime state:

- `local/results/raw/`
- `local/datasets/`
- `local/checkpoints/`
- `local/logs/slurm/`
- `local/logs/run/`
- `local/transfers/`
- `local/secrets/`

Do not commit datasets, checkpoints, raw outputs, full logs, credentials,
private SSH material, or private connection inventory. Share ordinary project
artifacts between servers with `scripts/rsync-artifact-broadcast.sh`.
