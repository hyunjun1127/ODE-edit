# SH4 checkpoint migration completed

Instruction: ODEEDIT-S06-SERVER4-CHECKPOINT-MIGRATION-SERVER2-V1.

183 checkpoints / 240176147811 bytes retained and independently verified on Server2; exact Server4 source copies removed. Existing 72 copies reused; 111 newly transferred. Four additional smoke checkpoints remain on Server4, outside the completed bundle. No directory deletion, model/GPU/evaluator/Slurm mutation or scientific changes.

Report: `experiment-reports/servers/server4/checkpoint-migration-server2-2026-09-11-v1/factual-migration-ko.md`
SHA256: `de091629c53deab0d50b475e511948a5d547e70da2ae60135c668547239bdd05`.

Full restoration mapping: `transfers/verifications/2026-09-11-checkpoint-migration-server4-source/migration-map.csv`.
Source deletion is permanent locally; verified Server2 retained copies provide recovery. Reports/code/NLL/logs/shared assets and companions remain unchanged. Historical BLUE1k RNG was not saved; GPU continuation replay was not performed. Native lifelong target telemetry stays copy-only on Server4 and was not required to restore committed W/M/context/RNG.

STOP after own-scope integration and direct handoff. No experiment monitoring resumed.
