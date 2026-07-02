# Multilinear polynomials

This chapter describes Vega's concrete representation of multilinear polynomials. The general existence, uniqueness, and Lagrange-basis formulas for multilinear extensions are reviewed in the [multilinear extensions primer](../appendix/mle-primer.md).

## Evaluation tables as polynomials

A length-\\(2^\ell\\) vector is viewed as the evaluation table of a function

\\[
f : \\{0,1\\}^\ell \to \mathbb{F}.
\\]

By the convention in [Notation and conventions](../overview/notation.md), an index \\(i\\) corresponds to bits \\((i\_0,
\dots,i\_{\ell-1})\\) satisfying

\\[
i = \sum\_{k=0}^{\ell-1} i\_k 2^k.
\\]

The table entry \\(v\_i\\) is therefore the value at the Boolean point \\((i\_0,
\dots,i\_{\ell-1})\\). Vega's `MultilinearPolynomial` stores this table directly as the vector \\(Z\\), and the polynomial it represents is the MLE \\(\widetilde{Z}\\).

This representation is dense: all \\(2^\ell\\) Boolean evaluations are present. It is the natural form for sum-check rounds, because fixing one variable halves the table while preserving the remaining evaluation table.

## Evaluation by equality weights

For a point \\(\mathbf{r} \in \mathbb{F}^\ell\\), Vega evaluates an MLE by weighting the Boolean table with the equality polynomial

\\[
\widetilde{\mathrm{eq}}(\mathbf{r}, \mathbf{x}) = \prod\_{k=0}^{\ell-1}\bigl(r\_k x\_k + (1-r\_k)(1-x\_k)\bigr).
\\]

Conceptually,

\\[
\widetilde{Z}(\mathbf{r}) = \sum\_{\mathbf{x} \in \\{0,1\\}^\ell} \widetilde{\mathrm{eq}}(\mathbf{r}, \mathbf{x})\\; Z(\mathbf{x}).
\\]

The reusable `EqPolynomial` type stores the point and can evaluate a single equality polynomial or materialize the full vector of \\(2^\ell\\) equality weights. Polynomial commitments also reuse these weights when reducing an opening claim to inner products.

Vega also uses a powers polynomial as another reusable family of weights. For a challenge \\(\tau\\), its weights are derived from \\(\tau^{2^0}, \tau^{2^1}, \dots, \tau^{2^{\ell-1}}\\) and are used to batch constraint terms in sum-check. The concrete role of these weights belongs to [The sum-check protocol](../building-blocks/sumcheck.md).

## Binding the top variable

Vega's table-halving operation binds the top variable first. For a table \\(Z\\) of length \\(2^\ell\\), the top variable is the bit with index \\(\ell-1\\), the most-significant bit in the LSB-first integer labeling. The low half of the table contains entries with that bit equal to \\(0\\); the high half contains the corresponding entries with that bit equal to \\(1\\).

Given a challenge \\(r \in \mathbb{F}\\), `bind_poly_var_top` splits

\\[
Z = (Z\_{\mathrm{lo}} \\;\\|\\; Z\_{\mathrm{hi}})
\\]

into two equal halves. For each position in the low half, let

\\[
a = Z\_{\mathrm{lo}}[j], \qquad b = Z\_{\mathrm{hi}}[j].
\\]

The bound table replaces \\(a\\) by

\\[
(1-r)a + rb,
\\]

then discards the high half. The result has length \\(2^{\ell-1}\\) and is the evaluation table obtained by fixing the variable \\(x\_{\ell-1}=r\\).

This is consistent with 0-based LSB-first indexing: bit \\(0\\) is the lowest-index variable, but the table operation starts from bit \\(\ell-1\\). After one binding, the next top variable is bit \\(\ell-2\\), and so on. Consequently, Vega's sum-check challenge sequence binds variables high-to-low.

## Interaction with sum-check

A sum-check prover repeatedly sends a univariate round polynomial, receives a verifier challenge, and restricts the remaining multilinear tables to that challenge. In Vega, that restriction is exactly the table-halving step above. Each round halves the live tables for the participating MLEs, so after \\(j\\) rounds a table that began with \\(2^\ell\\) entries has \\(2^{\ell-j}\\) entries.

The same high-to-low order is used for the tables that represent witness-derived values and for the auxiliary weights that participate in batching. Equality weights and powers weights are therefore not separate conventions; they are reusable polynomial families arranged to agree with the table order used by the sum-check implementation.

When these evaluations are checked inside a circuit, the same algebraic objects become circuit variables and constraints. That use is described in [The in-circuit verifier](../building-blocks/in-circuit-verifier.md). Exact serialized representations of tables, points, and proofs are specified in [Serialization and encodings](../spec/serialization.md).
