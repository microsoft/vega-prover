# Notation and conventions

This chapter fixes the notation and conventions used throughout the book. Every later
chapter assumes them. The pedagogical chapters use this notation to build intuition; the
[specification](../spec/scope.md) chapters additionally fix exact byte encodings, on top of
the mathematical objects defined here.

## Fields and the group

Vega operates over a prime-order elliptic-curve group \\(\mathbb{G}\\) with scalar field
\\(\mathbb{F}\\). Unless stated otherwise, *all* arithmetic of the proof system — witnesses,
constraints, sum-check messages, and Fiat–Shamir challenges — happens in \\(\mathbb{F}\\).

- \\(\mathbb{F}\\): the scalar field of \\(\mathbb{G}\\), a prime field of order roughly
  \\(2^{256}\\). Field elements are written with lowercase italic letters, e.g.
  \\(a, b, r, \tau, \rho\\). Every element serializes to a fixed 32 bytes (the exact encoding
  is fixed in [Serialization and encodings](../spec/serialization.md)).
- \\(\mathbb{G}\\): the group, written *additively*. Group elements (curve points) are written
  with uppercase italic letters, e.g. \\(P, Q\\). Fixed public generators are written
  \\(G, H, G\_1, G\_2, \dots\\).
- The canonical instantiation fixes \\(\mathbb{G}\\) to the T256 curve and \\(\mathbb{F}\\) to
  its scalar field; see [Fields, groups, and the engine](../building-blocks/fields-and-groups.md).

## Prover variants

Vega has two prover variants. \\(\mathrm{Vega}\_{\mathrm{MC}}\\) is the **multi-circuit** prover: it
folds many uniform *step* circuits together before proving, and it is the focus of this book.
\\(\mathrm{Vega}\_{\mathrm{SC}}\\) is the **single-circuit** prover it builds on. Unqualified, "the
prover" means \\(\mathrm{Vega}\_{\mathrm{MC}}\\).

## Vectors, matrices, and indexing

- Vectors are bold lowercase, e.g. \\(\mathbf{z}\\); their entries are \\(z\_0, z\_1, \dots\\).
  All indices are 0-based.
- Matrices are uppercase, e.g. \\(A, B, C\\); the entry in row \\(i\\), column \\(j\\) is
  \\(A\_{i,j}\\).
- \\(\mathbf{a} \circ \mathbf{b}\\) is the entrywise (Hadamard) product, and
  \\(\langle \mathbf{a}, \mathbf{b}\rangle = \sum\_i a\_i b\_i\\) is the inner product.
- Bit-strings live in \\(\\{0,1\\}^\ell\\). An integer
  \\(i \in \\{0, \dots, 2^\ell - 1\\}\\) is identified with its bits **least-significant-bit
  first**: \\(i = \sum\_{k=0}^{\ell-1} i\_k\\, 2^k\\), so \\(i\_0\\) is the low bit. This LSB-first
  convention is used consistently for hypercube indexing.

## The Boolean hypercube and multilinear extensions

A function \\(f : \\{0,1\\}^\ell \to \mathbb{F}\\) on the Boolean hypercube has a unique
*multilinear extension* (MLE) \\(\tilde{f} : \mathbb{F}^\ell \to \mathbb{F}\\) that agrees with
\\(f\\) on \\(\\{0,1\\}^\ell\\) and has degree at most one in each variable:
\\[
\tilde{f}(\mathbf{r}) \\;=\\; \sum\_{\mathbf{x} \in \\{0,1\\}^\ell} \widetilde{\mathrm{eq}}(\mathbf{r}, \mathbf{x})\\; f(\mathbf{x}).
\\]
A vector \\(\mathbf{v} \in \mathbb{F}^{2^\ell}\\) is routinely identified with the function
\\(i \mapsto v\_i\\) and hence with its MLE \\(\tilde{v}\\). See the
[multilinear extensions primer](../appendix/mle-primer.md) for background.

## Special polynomials

- **Equality polynomial.** For \\(\mathbf{r}, \mathbf{x} \in \mathbb{F}^\ell\\),
  \\[
\widetilde{\mathrm{eq}}(\mathbf{r}, \mathbf{x}) \\;=\\; \prod\_{k=0}^{\ell-1}\bigl(r\_k x\_k + (1 - r\_k)(1 - x\_k)\bigr).
\\]
  It is the MLE of the indicator \\([\mathbf{r} = \mathbf{x}]\\) on the hypercube.
- **Powers polynomial.** A challenge \\(\tau \in \mathbb{F}\\) induces the weights
  \\(\bigl(\tau^{2^0}, \tau^{2^1}, \dots, \tau^{2^{\ell-1}}\bigr)\\), used to batch many
  constraints into one. Its precise definition and role are given in
  [The sum-check protocol](../building-blocks/sumcheck.md).

## R1CS objects

A *rank-1 constraint system* (R1CS) over \\(\mathbb{F}\\) is given by matrices
\\(A, B, C \in \mathbb{F}^{m \times n}\\). A vector \\(\mathbf{z} \in \mathbb{F}^{n}\\)
*satisfies* it when
\\[
(A\mathbf{z}) \circ (B\mathbf{z}) \\;=\\; C\mathbf{z}.
\\]
The vector \\(\mathbf{z}\\) is partitioned into a constant \\(1\\), the public input/output
\\(\mathbf{x}\\), and the private witness \\(\mathbf{w}\\); the exact layout, together with the
*split*, *multi-round*, and *relaxed* variants that Vega uses, is fixed in
[R1CS and its variants](../building-blocks/r1cs.md).

## Commitments

\\(\mathrm{Com}(\mathbf{v}; \rho)\\) denotes a *hiding* commitment to a vector
\\(\mathbf{v}\\) under blind \\(\rho \in \mathbb{F}\\). Vega uses an additively homomorphic
Pedersen/Hyrax commitment, so
\\(\mathrm{Com}(\mathbf{a}; \rho\_a) + \mathrm{Com}(\mathbf{b}; \rho\_b)
= \mathrm{Com}(\mathbf{a} + \mathbf{b}; \rho\_a + \rho\_b)\\).
The commitment key, the reshaping of long vectors, and the opening argument are described in
[Polynomial commitments and the ZK opening](../building-blocks/pcs.md).

## The transcript and challenges

Interactive protocols are made non-interactive with the Fiat–Shamir transform against a single
running *transcript*. Two operations act on it:

- `absorb(label, value)` folds labeled data into the transcript;
- `squeeze(label)` derives the next verifier challenge in \\(\mathbb{F}\\).

Challenges are written with Greek letters — \\(\tau\\) (constraint batching), \\(\rho\\)
(instance folding), and \\(r\\) with subscripts \\(r\_b, r\_x, r\_y\\) (sum-check round
challenges). The transcript's exact byte-level behavior and the full ordered schedule of
`absorb`/`squeeze` calls are fixed in
[The transcript schedule](../spec/transcript-schedule.md); reproducing that schedule exactly is
what lets an independent prover derive identical challenges.

## Logarithms, padding, and powers of two

- \\(\log\\) is base 2; we write \\(\ell = \log\_2 n\\).
- Where an object's size is not already a power of two, it is padded to the next power of two.
  The specific padding rule (what value is used and where) is stated in each place it matters,
  because it affects the committed and hashed bytes.
