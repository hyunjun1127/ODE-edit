# Artifact Paths

- run_id:
- task_id:
- server:
- agent_id:
- updated_at:

## Raw Logs

- stdout:
- stderr:

## Artifacts

| Path | Type | Size | Checksum | Durability | Notes |
| --- | --- | --- | --- | --- | --- |

## Notes

- Keep checkpoints, datasets, raw outputs, model weights, and full logs outside Git.
- Ordinary project artifacts under `local/` can be shared by automatic rsync
  broadcast. Use `transfers/` only for manual exceptions such as repo-external
  paths, destructive mirror behavior, sensitive material, or unusual overwrite
  risk.
