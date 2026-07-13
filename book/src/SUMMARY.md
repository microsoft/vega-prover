# Summary

[Introduction](introduction.md)

# Overview

- [Design goals and threat model](overview/design-goals.md)
- [System architecture](overview/architecture.md)
- [The proving lifecycle](overview/lifecycle.md)
- [Notation and conventions](overview/notation.md)

# Building blocks

- [Fields, groups, and the engine](building-blocks/fields-and-groups.md)
- [Byte encodings and serialization](building-blocks/encodings.md)
- [The Fiat--Shamir transcript](building-blocks/transcript.md)
- [Multilinear polynomials](building-blocks/multilinear.md)
- [The sum-check protocol](building-blocks/sumcheck.md)
- [R1CS and its variants](building-blocks/r1cs.md)
- [Polynomial commitments and the ZK opening](building-blocks/pcs.md)
- [The Spartan argument](building-blocks/spartan.md)
- [NeutronNova folding](building-blocks/neutronnova.md)
- [Nova folding for zero-knowledge](building-blocks/nova-zk.md)
- [Relaxed Spartan](building-blocks/relaxed-spartan.md)
- [The in-circuit verifier](building-blocks/in-circuit-verifier.md)

# The Vega-MC prover

- [Protocol overview](mc/overview.md)
- [Setup](mc/setup.md)
- [Rerandomizable precomputation](mc/prep.md)
- [Proving](mc/prove.md)
- [Verification](mc/verify.md)
- [Zero-knowledge](mc/zero-knowledge.md)

# Implementable specification

- [Scope and the conformance contract](spec/scope.md)
- [Serialization and encodings](spec/serialization.md)
- [The transcript schedule](spec/transcript-schedule.md)
- [Verifier key and digest](spec/verifier-key.md)
- [The proof object](spec/proof-object.md)
- [A simple reference prover](spec/reference-prover.md)
- [Conformance and test vectors](spec/test-vectors.md)

# Appendices

- [Primer: multilinear extensions](appendix/mle-primer.md)
- [Primer: the sum-check protocol](appendix/sumcheck-primer.md)
- [Primer: polynomial commitments](appendix/pcs-primer.md)
- [Primer: folding schemes](appendix/folding-primer.md)
- [Glossary](appendix/glossary.md)
- [Bibliography](appendix/bibliography.md)
