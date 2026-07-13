# Conformance and test vectors

This chapter describes how an independent implementation demonstrates conformance:
the fixtures it can check against, and the executable gates that establish
mutual acceptance with the shipped prover and verifier. It closes the
specification by making "the verifier accepts" concrete and reproducible.

## What conformance means here

As established in [Scope and the conformance contract](scope.md), an honest proof
is randomized, so conformance is **not** byte equality of proofs. It is verifier
acceptance. Two layers of test make this precise:

1. **Deterministic vectors.** Some artifacts *are* fixed functions of their
   inputs and can be checked byte-for-byte: the parse of a given proof, the
   Fiat–Shamir challenges derived from a fixed transcript, and the verifier-key
   digest. An implementation that disagrees on any of these is non-conformant.
2. **Acceptance gates.** The proof as a whole is checked by running the *other*
   side's verifier and asserting it accepts.

## Deterministic fixtures

The directory
[`reference/fixtures/cubic`](https://github.com/Microsoft/vega-prover/tree/main/reference/fixtures/cubic)
holds vectors for the canonical cubic circuit. No fixture in this directory is
committed; every fixture -- including `meta.json` -- is git-ignored and
regenerated on demand, because the keys are large and the proofs are randomized
(zero-knowledge). Regenerate the fixtures with the commands below before running
the tests on a fresh checkout.

| File | Contents | Regenerate with |
| --- | --- | --- |
| `meta.json` | circuit metadata + expected public values (`[15]`) | `cargo test --lib export_cubic_fixtures -- --ignored` |
| `proof.bin` | a real Rust-produced proof (bincode) | `cargo test --lib export_cubic_fixtures -- --ignored` |
| `vk_digest.bin` | the 32-byte verifier-key digest | `cargo test --lib export_cubic_fixtures -- --ignored` |
| `vk.bin` | the Rust verifier key (large) | `cargo test --lib export_cubic_fixtures -- --ignored` |

The reference implementation checks each deterministic layer against these:

- **Proof parsing** — `test_proof_parse.py` parses `proof.bin` and asserts the
  cursor consumes every byte, confirming the [proof object](proof-object.md)
  layout.
- **Transcript** — `test_transcript.py` reproduces the recorded challenges
  byte-for-byte, confirming the [transcript schedule](transcript-schedule.md).
- **Verifier-key digest** — `test_vk_digest.py` recomputes
  \\(\mathrm{SHA\text{-}256}(D)\\) over the digest stream and matches
  `vk_digest.bin`, confirming the [verifier-key](verifier-key.md) encoding.

## Acceptance gates

Acceptance is checked in both directions.

**Rust prover → reference verifier.** `test_verify.py` runs the reference
verifier on the Rust-produced `proof.bin` and asserts acceptance, recovering the
expected public values. This exercises the entire acceptance predicate — instance
validation, both folds, both sum-checks, relaxed Spartan, the pinned public
values, and the final commitment opening — against bytes the reference did not
produce.

**Reference prover → Rust verifier.** The reference prover emits a proof that the
*production* verifier accepts. Two `#[ignore]` harnesses in the Rust test-suite
(`tests/reference_conformance.rs`) deserialize the Python-produced artifacts and
call the real `verify`:

- `verify_python_proof` — the reference proof against the Rust-exported verifier
  key; and
- `verify_python_standalone` — the reference proof against a verifier key that the
  Python `setup.py` itself generated, so the production library plays no part in
  setup, proving, *or* key generation. Only verification is Rust.

Both recover the public value \\(15\\), closing the loop.

## Reproducing the gates

```sh
# deterministic vectors (byte-exact)
sage -python reference/tests/test_proof_parse.py
sage -python reference/tests/test_transcript.py
python3    reference/tests/test_vk_digest.py

# Rust prover -> reference verifier
sage -python reference/tests/test_verify.py

# reference prover self-check, and write the stand-alone fixtures
sage -python reference/tests/test_prove_finish.py
sage -python reference/tests/test_standalone.py

# reference prover -> Rust verifier (both directions of acceptance)
cargo test --test reference_conformance verify_python_proof      -- --ignored --nocapture
cargo test --test reference_conformance verify_python_standalone -- --ignored --nocapture
```

An independent prover conforms when its proofs pass the same acceptance gate: the
production verifier, run on the prover's serialized proof and verifier key,
returns success and the expected public values.

## A note on Sage

The reference uses [Sage](https://www.sagemath.org/) only for group arithmetic and
point (de)compression, isolated behind `curve.py`; the field, polynomial,
encoding, and transcript logic is plain Python. This boundary keeps the
cryptographic core portable and lets the arithmetic be reimplemented against any
curve backend without touching the protocol logic.
