# Conflict Report Template

## Summary

State which repository/clone hit the conflict, what operation failed, and
whether automated sync has been paused.

## Context

- Time:
- Reporting agent:
- Server:
- Repository:
- Branch:
- Remote:
- Related plan/task/run:
- Local conflict report path:

## Evidence

Include the key `git status --short --branch` output, failed command, conflicted
paths, and relevant commit SHAs. Do not paste private local scratch logs.

## Impact

State which servers or tasks should stop pull/rebase/push activity and whether
the global pause marker `control/sync-paused` has been published.

## Requested Decision

Ask the global head/user for the specific resolution needed, such as choosing
one branch of changes, manually merging a file, recloning a server, or restoring
from a clean remote state.

## Next Owner

Name the agent or role expected to resolve or continue.
