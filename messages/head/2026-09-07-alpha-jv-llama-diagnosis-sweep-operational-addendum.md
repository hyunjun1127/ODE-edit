# Operational addendum — correct durable paths / prospective cap

Instruction ODEEDIT-S06-ALPHA-JV-LLAMA-DIAGNOSIS-SWEEP-SH1-V1, 2026-09-07.
The initial envelope stays immutable (SHA256
9b9ab0b4c7345d5e5a6793457852ece08022cdd100301e5e13213b04e55d75a1).
This corrects optional operational publication paths to the repository access
checker patterns; no scientific equation/sample/arm/gate changes.

- ACKs: `messages/acks/server1/2026-09-07-alpha-jv-llama-diagnosis-sweep*.md`.
- Task status: `tasks/status/alpha-jv-llama-diagnosis-sweep-20260907/server1.json`.
- Server-head notes, source, local output, reports and audits remain as initially
  allowed. Use audits/experiment-reports for job receipts; do not create a new
  runs/ path that the server-head access checker does not allow.
- Tracked new-run ceiling: `control/gpu-concurrency-policy.tsv`.
  Local task caps may be lower, never higher. Disabled/unregistered hosts remain
  disabled; server3 local cap0 is preserved beneath the universal ceiling2.
- Existing admitted runs are unchanged. New admission counts RUNNING,
  COMPLETING and CONFIGURING allocations. Failed scheduler query is not zero
  usage; remain pending.
- Server1's bounded GH check found project active0 and 2-GPU/memory admission
  possible, NOT a reservation. Recheck immediately before actual submission.
- SH4 handler ACK confirms prospective local cap2, memory60416M, existing
  37649/38433 changes0. Server2 app-server handshake timeout; cap propagation
  unconfirmed, not silently reported delivered. No resource recovery is authorized.
- GPU-hour cap is still null pending user. CPU preparation continues now;
  only model/GPU submission awaits budget. D artifact unavailability is not S hold.
