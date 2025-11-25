# FRI-Binius Codebase Improvement Plan

## Objectives
- Improve presentation and organisation so modules are easy to follow.
- Standardise error handling with `anyhow::Result` while keeping code succinct.
- Keep public APIs small and predictable for prover/verifier usage.

## Quick Wins (implemented now)
- Introduced a crate-wide `Result<T> = anyhow::Result<T>` alias and added the `anyhow` dependency.
- Switched the Fiat–Shamir `Channel`, `prove`, `verify`, and integration test flows to return `Result` with `?` propagation instead of `expect`/`unwrap` chains.
- Trimmed unused imports and dead code paths to reduce noise and make module intent clearer.
- Normalised the test to return `Result<()>`, aligning the harness with the new error surface.

## Next Steps (recommended)
- **Public API clarity**: Re-export the primary types (`FriCommitment`, `EvalProof`, `Channel`) from `lib.rs` with minimal surface docs so downstream callers see a concise interface.
- **Error context**: Add targeted `context()` messages at outer boundaries (e.g., encoding/NTT preparation, Merkle queries) to pinpoint failures without flooding inner loops.
- **Module layout**: Group prover/verifier helpers under dedicated submodules (e.g., `fri::prove`, `fri::verify`, `fri::merkle`) to signal ownership and reduce cross-file hopping.
- **Data validation**: Replace `assert!`-heavy verification paths with checked errors where invariants depend on external input; keep asserts only for internal invariants.
- **Benchmark harness**: Add Criterion benches for commit/prove/verify to track performance regressions after refactors.
- **Docs pass**: Add short module-level docs describing data flow (poly → codeword → Merkle → FRI queries) and how `TAU`, `LOG_RATE`, and packing interact.

## Test & CI Suggestions
- Run `cargo fmt && cargo clippy --all-targets --all-features` to keep style consistent and catch edge cases early.
- Use `cargo test --release -- --nocapture` for the FRI integration test to validate the full round trip with realistic sizes.

## Guiding Principles
- Prefer small, composable helpers with explicit types over long imperative blocks.
- Keep the hot paths allocation-aware; add comments only where they clarify non-obvious arithmetic.
- Fail early with context, but avoid verbose error hierarchies; `anyhow` should stay lightweight here.
