# B1 gate technical RCA — projector hash shape header

Job51071/source3f65d170: FAILED1:0,230 allocated GPU seconds; failure stage B1_DENSE.
Original failure/raw/cost are immutable. C00 and B1 evaluation completed before failure.

Original `integrity.tensor_sha` prefixes dtype and shape. The pinned runtime binding hashes
selected P as shape `[1,14336,14336]`, whereas the initial gate compared `[14336,14336]`.
Independent current file hash is `6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec`.
Selected stack hash is `8c50a474f01a28afbe52f3212e79da3dc4a22f5c753a53dedc3d31ed10c2ec68`,
exact contract match; matrix hash is `ce668ee4587705173f8cecd4fc8245efdb4d6e5505f81192d88a5665502e698a`.
This is a header/orientation metadata bug, not changed P or a different numerical method.

Repair: validate original singleton stack then pass unchanged slot0 matrix to solve.
The same pre-execution check in operator_lane receives this narrow correction.
CPU fixture verifies same bytes/different shape hashes and rejects actual P changes.

Continuation uses a create-once SHA/size manifest of successful captures and existing
evaluation evidence. Runtime/model/W0 identities are rechecked; prior NS numeric FAIL is
retained unchanged. No endpoint evaluation is rerun to seek a better value. Only missing
physical/dense evidence is completed in a distinct attempt/source. Original run archive unchanged.

Separate unresolved numerical gate: B1 NS row237 maximum NLL difference0.00016117095947265625,
margin difference0.00010061264038085938 > fixed0.0001. Success counts100/190/867 and bits
match, repeats are exact, but that does not waive row parity. Dependent C02/O/N/E/F/G
claims remain blocked unless the existing scientific contract is actually satisfied.
