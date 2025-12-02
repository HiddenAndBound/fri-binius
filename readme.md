# FRI-Binius

Rust implementation of the FRI protocol over the Binius tower fields, tracking the construction in the paper “FRI meets Binius” (ePrint 2024/504). The codebase focuses on a clean prover/verifier split, efficient SIMD-friendly encodings, and traceable Fiat–Shamir interaction.

## Highlights
- End-to-end demo in `src/main.rs` that commits to random multilinear polynomials, produces evaluation proofs, and verifies them for code lengths up to `2^{l+6}` for `l ∈ [10, 39]`.
- `prover` module builds vector commitments from packed MLEs, drives the folding rounds, and records every Merkle authentication path needed by the verifier.
- `verifier` module replays the Fiat–Shamir challenges, checks the algebraic reductions, and walks the Merkle paths ensuring consistency with the published root.
- Utility modules (`utils/channel`, `utils/code`, `utils/merkle`, `utils/mle`, …) encapsulate domain-specific helpers so the hot loops stay compact and allocation-aware.

## Repository Layout
- `src/lib.rs` keeps the public API small by re-exporting the prover, verifier, and shared utilities alongside a crate-wide `Result` alias.
- `src/prover.rs` and `src/verifier.rs` host the interactive logic with `tracing` instrumentation to ease profiling.
- `src/utils/` contains reusable building blocks (Merkle trees, evaluation bases, Fiat–Shamir channel, etc.).
- `docs/plan.md` documents the current improvement roadmap and coding guidelines.

## Prerequisites
- Rust toolchain with Edition 2024 support (Rust 1.84+ or the matching nightly via `rustup default nightly` as of November 2025).
- Standard Cargo tools such as `cargo fmt` and `cargo clippy` (`rustup component add rustfmt clippy`).

## Quick Start
```bash
# Clone the repo and pull the git-based Binius dependencies
git clone https://github.com/HiddenAndBound/fri-binius.git
cd fri-binius

# Run the end-to-end driver with spans enabled
RUST_LOG=info cargo run --release
```
The demo iterates over multiple polynomial sizes, prints each `2^{ℓ+6}` batch heading, and executes commit → prove → verify for randomly sampled evaluation points.

## Testing & Quality Gates
```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test --release -- --nocapture
```
- `cargo fmt` and `cargo clippy` enforce the style/error-handling conventions introduced in the recent refactor.
- The integration test (triggered via `cargo test --release`) mirrors the driver in `main.rs`, ensuring the prover/verifier stay in lockstep for realistic parameters.

## Observability
- Spans such as `commit`, `prove`, `commit_fri_oracle`, and `verify` expose timing data through `tracing`/`tracing-profile`; set `RUST_LOG=debug` for detailed progress, or use `tracing_profile`’s exported flamegraph tooling if installed locally.
- `Channel` observations are deterministic once seeded, so recording the transcript in a test harness makes failing states reproducible.

## Roadmap & Contributions
The near-term backlog (API re-exports, richer error context, benchmark harness, documentation pass) lives in `docs/plan.md`. Contributions that follow those guidelines—small composable helpers, explicit error paths over `assert!`, and Criterion-based benchmarks—will slot in smoothly. Feel free to open an issue or draft PR describing which checklist item you are addressing.

## References
- Original construction: https://eprint.iacr.org/2024/504.pdf
- Binius crates used here: https://gitlab.com/IrreducibleOSS/binius
