# Bug-Review Log

Scheduled bug-review routine tracks reviewed commits here so subsequent runs only inspect new commits. Newest entry first.

## 2026-07-29 — reviewed by analyst

Range: `origin/main..HEAD` on `claude/modest-rubin-8j1hnn`, 18 commits.

Commits reviewed:
- bde4f9b logs: refresh baseline, eval, and self-play artifacts
- 06e6129 docs: restructure wiki into fresh topic pages
- df794b3 selfplay: fix doc-list indentation in eval task run() doc
- 924f6be selfplay: apply cargo fmt to new elo-promotion code
- 9450e38 baseline: extract candidate_elo from ladder_match and add to score
- 0c35f8f selfplay: wire elo ladder env vars + startup notices
- a93b077 selfplay: refactor eval task to per-opponent elo ladder
- 2a38e77 selfplay: plumb opponent inference server for elo ladder
- 9ddee3a selfplay: add archive pool enumeration helper
- 7b53e5d selfplay: add elo math module
- 8511b99 docs: revise elo-promotion plan per review
- 00107d7 docs: plan for elo-promotion
- 0ab40d4 docs: research for elo-promotion plan
- aff97fb iter-5: re-apply cosine LR
- 7b5dd87 iter-2: enable policy entropy regularization
- 68b29ef bench: dump HYZERO_* env vars at startup
- 2a3e6ee wip: framework refactor + wiki sync + replay subsystem + run artifacts
- 5f30ea8 mcts: add Gumbel-Top-K + sequential halving root selection

Confirmed bugs: 0

Latent concern (not a review-criteria bug, tracked for cleanup):
- `src/mcts/tree.rs::simulate_with_root_action` — in the leaf-expansion branch, a raw `*const MCTSNode` overlaps with `self.navigate_to_parent_mut(&path)`'s `&mut self`. The raw pointer is not dereferenced after the mut-borrow so behavior is correct today, but this violates stacked borrows and could bite under future refactors. Worth switching to an index/path-based re-lookup.
