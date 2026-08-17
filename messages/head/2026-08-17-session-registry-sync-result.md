# 2026-08-17 새 session registry sync 결과

## 결론

- GH→SH1 direct ACK: PASS
- GH→SH2 direct ACK: PASS
- canonical remote `main` registry update: PASS
- SH1 `origin/main` update 확인: PASS
- SH2 local `main` fast-forward: PASS
- scientific/model/Slurm/GPU/result action: 0

## Canonical main

- commit: `1caed88867db3087fa5db26995c7e3719c064216`
- tree: `6b8796ccb8d97c0822ecc0247b0f419674aa7ca0`
- subject: `Update GH and SH session registry`
- remote: `origin/main`

## Model-profile user override

| 역할 | session | observed runtime | 정책 |
| --- | --- | --- | --- | --- |
| GH | `01a00e5f-63ef-7cc2-89ec-f2f7b23df40f` | `gpt-5.6-sol/xhigh` | user-managed, non-blocking |
| SH1 | `01a00e5d-29e8-7a01-822b-7acf43226035` | `gpt-5.6-sol/xhigh` | user-managed, non-blocking |
| SH2 | `01a00e5c-f7ae-72a2-98b2-b8b0907168b4` | `gpt-5.6-sol/max` | user-managed, non-blocking |

SH1은 ignored local boundary의 새 GH/SH ID를 갱신했고 detached HEAD와
user-owned untracked paths를 보존했다. `origin/main`은 canonical commit으로
확인했지만 model confirmation은 기록하지 않았고 checker를 실행하지 않았다.

사용자는 model family/reasoning effort를 직접 관리하며 repository hard boundary로
사용하지 않도록 명시했다. Checker와 protocol은 session ID, CWD, repository
identity만 검증하도록 갱신한다. SH2는 boundary ID/CWD/repository check를
PASS하고 clean main을 `d4206536d0c9629e1061b6575bc967e0cf1f742b`까지
ff-only로 동기화했다.

## 최종 ACK

- SH1: `origin/main=d4206536d0c9629e1061b6575bc967e0cf1f742b`,
  equivalent hard-boundary PASS, detached HEAD/untracked paths 보존, active job 0.
- SH2: before `6145406ae4b11e05b683c46aa604c972eb727f5a` → after
  `d4206536d0c9629e1061b6575bc967e0cf1f742b`, ff-only, clean, ahead/behind
  0/0, hard-boundary와 active registry verification PASS.

## 보존 범위

GH root dirty state, SH1 detached HEAD/untracked paths, SH2 clean old local main,
과거 reports/audits/receipts/completed launchers의 old session provenance는 모두
그대로 보존했다.
