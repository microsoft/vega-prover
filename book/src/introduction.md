# Introduction

Vega is a zero-knowledge proof system for proving statements about existing signed credentials. A prover can hold signed data, such as a mobile driver's license, and convince a verifier that a statement about that data is true without revealing anything beyond the statement. The implementation targets client devices, low proving latency, repeated presentations over the same signed data, and transparent setup.

## What this book specifies

This book has two purposes.

1. It explains the design of Vega and the building blocks used by \\(\mathrm{Vega}\_{\mathrm{MC}}\\), the multi-circuit prover: fields and groups, R1CS, Fiat--Shamir transcripts, sum-check, folding, polynomial commitments, and the verifier circuit.
2. It gives a specification precise enough for an independent team to build a prover whose proof bytes are accepted by the reference Vega verifier.

The focus is the multi-circuit, or MC, variant of the prover. The single-circuit variants exist in the repository, but they are outside the main line of this book.

## The problem Vega addresses

Digital credentials are often already issued and signed by systems that were not designed for zero knowledge. Vega works over those existing credentials rather than requiring a new credential format or a trusted setup ceremony. Its proving strategy is built around repeated structure: the same signed data may be presented many times, and many hashing or parsing steps have uniform circuit shape.

Vega uses fold-and-reuse proving. Repeated work across presentations is pushed into a rerandomizable precomputation. Uniform step-circuit instances are folded into one instance. For zero knowledge, the public-coin verifier-circuit instance is folded with a fresh random satisfying instance in every proof. Credential-specific arithmetization, such as lookup-heavy extraction from signed bytes, motivates the system but is outside this book; the proof system treats circuits abstractly.

## Byte-equivalence contract

The specification chapters pin down the deterministic wire format and the Fiat--Shamir transcript schedule. An independent prover that follows those chapters produces proof bytes that the real Vega verifier accepts. When the independent prover and the reference prover are driven with identical inputs and identical randomness, their proofs are byte-identical. Across two honest zero-knowledge proofs for the same statement, the expected differences come from the fresh randomness used for hiding and rerandomization.

## How to read this book

Readers learning the system should start with the [design goals](overview/design-goals.md), [architecture](overview/architecture.md), and [proving lifecycle](overview/lifecycle.md), then read the building-block chapters as needed. Implementers should read the same overview, then proceed to the MC chapters and the [specification scope](spec/scope.md), which explains where byte-level requirements begin.
