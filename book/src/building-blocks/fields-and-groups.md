# Fields, groups, and the engine

This chapter explains the algebraic choices collected by a Vega engine. An engine fixes the field and group in which the proof system operates, and it also selects the transcript and polynomial-commitment machinery used by the surrounding protocols.

## What an engine fixes

Vega writes \\(\mathbb{G}\\) for the prime-order elliptic-curve group and \\(\mathbb{F}\\) for its scalar field. The engine binds these objects together as associated types:

- a base field for the curve coordinates;
- the scalar field \\(\mathbb{F}\\), used for proof-system arithmetic;
- the group \\(\mathbb{G}\\), whose scalar field is \\(\mathbb{F}\\);
- a Fiat--Shamir transcript engine;
- a polynomial-commitment scheme over scalars in \\(\mathbb{G}\\).

The base field is the field in which curve coordinates live. The scalar field is the field used by witnesses, constraints, multilinear polynomials, sum-check messages, and verifier challenges. These two fields need not be the same field, even when they have the same size scale.

The transcript engine is described in [The transcript](../building-blocks/transcript.md). The commitment scheme is described in [Polynomial commitments and the ZK opening](../building-blocks/pcs.md). Byte encodings are intentionally separate from the algebraic engine definition; see [Encodings](../building-blocks/encodings.md) and [Serialization and encodings](../spec/serialization.md).

## The canonical instantiation

The canonical engine is `T256HyraxEngine`. It fixes

\\[
\mathbb{G} = \text{T256},
\\]

uses the T256 scalar field as \\(\mathbb{F}\\), uses a Keccak256-based Fiat--Shamir transcript, and uses the Hyrax polynomial-commitment scheme.

T256 is an elliptic curve in short-Weierstrass form, with group parameters exposed as coefficients \\(A\\), \\(B\\), the group order, and the base-field order. Its scalar field — the field \\(\mathbb{F}\\) used for all proof-system arithmetic — has prime order equal to the group order,

\\[
p\_{\mathrm{scal}} = \mathtt{0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff},
\\]

and its base field, in which curve coordinates live, has prime order

\\[
p\_{\mathrm{base}} = \mathtt{0xffffffff0000000100000000000000017e72b42b30e7317793135661b1c4b117}.
\\]

Thus \\(\mathbb{F} = \mathbb{F}\_{p\_{\mathrm{scal}}}\\) for the canonical engine. Field elements are roughly 256 bits; the exact byte representation is specified in [Serialization and encodings](../spec/serialization.md), not in this chapter.

The curve equation is \\(y^2 = x^3 + Ax + B\\) over \\(\mathbb{F}\_{p\_{\mathrm{base}}}\\), with

\\[
A = p\_{\mathrm{base}} - 3, \qquad
B = \mathtt{0xb441071b12f4a0366fb552f8e21ed4ac36b06aceeb354224863e60f20219fc56},
\\]

and cofactor one, so \\(|\mathbb{G}| = p\_{\mathrm{scal}}\\). Its standard base point is

\\[
G = \big(\,3,\ \mathtt{0x5a6dd32df58708e64e97345cbe66600decd9d538a351bb3c30b4954925b1f02d}\,\big).
\\]

The commitment generators the prover uses are not this base point; they are derived by hashing, as [Setup](../mc/setup.md) describes.

## Relationship to P256

The scalar field of T256 coincides with the base field of the standard P256 curve:

\\[
\text{T256 scalar field} = \text{P256 base field} = \mathbb{F}\_{p\_{\mathrm{scal}}}.
\\]

This shared prime is the only field the two curves have in common. The base field of T256 is a distinct prime and does not equal the scalar field of P256, so the pair form a one-directional chain rather than a full two-curve cycle. All proof-system arithmetic in the canonical engine takes place in \\(\mathbb{F} = \mathbb{F}\_{p\_{\mathrm{scal}}}\\); the group \\(\mathbb{G} = \text{T256}\\) supplies the commitment generators described below.

## Group operations used by commitments

Vega uses \\(\mathbb{G}\\) additively. A scalar multiplication is written \\(aG\\), and a multi-scalar multiplication has the form

\\[
\sum\_i a\_i G\_i.
\\]

Hyrax commitments use this operation to commit to rows of a vector. For a row \\((v\_0,
\dots,v\_{m-1})\\) and commitment-key generators \\(G\_0,
\dots,G\_{m-1}\\), the non-hiding part is the MSM

\\[
\sum\_{i=0}^{m-1} v\_i G\_i.
\\]

The commitment key also contains a dedicated hiding generator \\(h\\). A blind \\(\rho \in \mathbb{F}\\) contributes the term \\(\rho h\\), so a row commitment has the conceptual form

\\[
\sum\_{i=0}^{m-1} v\_i G\_i + \rho h.
\\]

This additive form is the algebraic reason that commitments can be combined homomorphically, as summarized in [Notation and conventions](../overview/notation.md) and used by [Polynomial commitments and the ZK opening](../building-blocks/pcs.md).
