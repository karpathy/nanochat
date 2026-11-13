# Progress

Delivery ledger for `docs/plan/` slices (non-nWave TRAIL adaptation).

## Decision Log

| date | slice_id | decision | why |
|------|----------|----------|-----|
| 2026-11-13 | fix-forward-mac-phase1 | Ship Phase 1 (echoes/skip/rms_norm) before Phase 2 (defaults/memory auto-fit); freeze echo contract | Phase 2 only adds auto-fit lines; dual-shipping would mix SSOT work with the current Mac train blocker |
| 2026-11-13 | fix-forward-mac-phase1 | Shell owns stage map; Python owns boundary/fail detail; markers written by eval scripts | Keeps orchestration readable in bash and avoids hot-loop echo spam |
| 2026-11-13 | fix-forward-mac-phase1 | Pack `next:` + `wrote:` + `mem:` on one DONE enrichment line | Honors “one enrichment line” discipline while still meeting mem/disk checklist items |

## Slice states

| slice_id | state | evidence |
|----------|-------|----------|
| fix-forward-mac-phase1 | PASSED | PR #4; pytest 51p/18s; Gate 4 APPROVED (software-crafter, acceptance, solution-architect); ddd N/A |
| fix-forward-mac-phase2 | PROPOSED | Blocked on Phase 1 merge; see plan Phase 2 section |
