# Server Head Handoff Template

## Header

- Time:
- From:
- To:
- Status: info/request/blocked/handoff/done
- Priority:
- Related plan:
- Related task:
- Related run:

## Summary

Describe the cross-server issue, request, or handoff in enough detail that the
target server head can act without reading private local logs.

## Checked Context

- Server inspected:
- Log path:
- Log coverage:
- Artifact path:
- Command or check performed:

## Evidence

Include key metrics, short error excerpts, file size, checksum, or the specific
observation that justifies the request. Do not paste full logs.

## Requested Action

State the exact action requested from the target agent or global head.

For file transfers, include:

- Source server:
- Source path:
- Destination server:
- Destination path requested or proposed:
- Overwrite policy:
- Validation after transfer:

## Blockers Or Risks

List missing paths, permissions, unknown environment details, version mismatch,
or resource constraints.

## Next Owner

Name the agent or role expected to respond or continue.
