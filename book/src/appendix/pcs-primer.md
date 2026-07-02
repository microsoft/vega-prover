# Primer: polynomial commitments

This appendix gives a self-contained account of vector and polynomial commitments. It treats commitments as a general finite-field and group tool; concrete uses in this book are described in [Polynomial commitments and the ZK opening](../building-blocks/pcs.md).

## Commitment goals

A vector commitment lets a prover publish a short group element that stands for a vector \\(\mathbf{v} \in \mathbb{F}^n\\). Later, the prover can open the commitment to a value derived from \\(\mathbf{v}\\), such as an entry, an inner product, or an evaluation of the multilinear extension \\(\tilde{v}\\). The verifier checks the opening against the original commitment without receiving the whole vector.

The basic security and algebraic properties are the following.

- **Binding.** After a commitment is fixed, the prover cannot open it as two different vectors except with negligible probability. For discrete-log commitments, binding follows from the hardness of finding linear relations among independently generated group bases.
- **Hiding.** The commitment does not reveal the committed vector. A fresh blinding term makes commitments to different vectors indistinguishable to a verifier that does not know the blind.
- **Additive homomorphism.** Commitments can be added. If
  \\[
C\_a = \mathrm{Com}(\mathbf{a};\rho\_a), \qquad C\_b = \mathrm{Com}(\mathbf{b};\rho\_b),
\\]
  then
  \\[
C\_a + C\_b = \mathrm{Com}(\mathbf{a}+\mathbf{b};\rho\_a+\rho\_b).
\\]
  The same identity extends to arbitrary linear combinations.

A polynomial commitment in this setting is a vector commitment applied to the evaluation table of a polynomial. For a length-\\(2^\ell\\) vector \\(\mathbf{v}\\), the committed polynomial is the multilinear extension \\(\tilde{v} : \mathbb{F}^\ell \to \mathbb{F}\\) of that table.

## Pedersen vector commitments

Fix public group elements \\(G\_0,
G\_1,
\dots,
G\_{n-1}, H \in \mathbb{G}\\). A Pedersen commitment to \\(\mathbf{v}=(v\_0,\dots,v\_{n-1})\\) with blind \\(\rho \in \mathbb{F}\\) is
\\[
\mathrm{Com}(\mathbf{v};\rho) = \sum\_{i=0}^{n-1} v\_i G\_i + \rho H.
\\]
The group \\(\mathbb{G}\\) is written additively, so the expression is a multi-scalar multiplication plus a blinding generator.

Binding comes from the discrete-log assumption. If a prover can open the same commitment as both \\((\mathbf{v},\rho)\\) and \\((\mathbf{v}',\rho')\\), then
\\[
\sum\_i (v\_i-v\_i')G\_i + (\rho-\rho')H = 0.
\\]
For distinct openings this gives a nontrivial linear relation among the public bases. The commitment setup chooses the bases so that such a relation is computationally infeasible to find.

Hiding comes from the term \\(\rho H\\). For a fresh blind, the commitment is shifted by an unknown multiple of \\(H\\), so the visible group element does not determine the vector. The blind also participates in the homomorphism:
\\[
\alpha\\,\mathrm{Com}(\mathbf{a};\rho\_a) + \beta\\,\mathrm{Com}(\mathbf{b};\rho\_b) = \mathrm{Com}(\alpha\mathbf{a}+\beta\mathbf{b}; \alpha\rho\_a+\beta\rho\_b).
\\]
This identity is why later protocols can fold many committed vectors into one committed vector.

## Matrix view of an MLE opening

Let \\(\mathbf{v}\in\mathbb{F}^{2^\ell}\\) be the evaluation table of \\(\tilde{v}\\). Choose integers \\(s,c\ge 0\\) with \\(s+c=\ell\\), and reshape the table as a matrix \\(Z\in\mathbb{F}^{2^s\times 2^c}\\). The row index uses one block of Boolean variables, and the column index uses the remaining block. A point \\(\mathbf{r}\in\mathbb{F}^\ell\\) splits accordingly as
\\[
\mathbf{r}=(\mathbf{r}\_{\mathrm{row}},\mathbf{r}\_{\mathrm{col}}).
\\]

Let
\\[
L\_i = \widetilde{\mathrm{eq}}(\mathbf{r}\_{\mathrm{row}}, i), \qquad R\_j = \widetilde{\mathrm{eq}}(\mathbf{r}\_{\mathrm{col}}, j)
\\]
be the equality weights for row and column Boolean indices. The MLE evaluation factors as
\\[
\tilde{v}(\mathbf{r}) = \sum\_i\sum\_j L\_i Z\_{i,j} R\_j = \left\langle \sum\_i L\_i Z\_{i,\*},\\; \mathbf{R}\right\rangle.
\\]
Here \\(Z\_{i,\*}\\) is row \\(i\\), and \\(\mathbf{R}=(R\_j)\_j\\).

This factorization turns one evaluation opening into two smaller tasks. First, the prover forms the row-combined vector
\\[
\mathbf{a} = \sum\_i L\_i Z\_{i,\*}.
\\]
Second, the claimed evaluation becomes the inner product
\\[
\tilde{v}(\mathbf{r}) = \langle \mathbf{a}, \mathbf{R}\rangle.
\\]
If the original commitment is row-wise and additively homomorphic, the verifier can form the matching row-combined commitment from the public row commitments. The remaining proof is an inner-product proof for the committed vector \\(\mathbf{a}\\) against the public vector \\(\mathbf{R}\\).

## A linear inner-product argument

The inner-product statement has the following form. The verifier knows a commitment \\(C\_a=\mathrm{Com}(\mathbf{a};\rho\_a)\\), a public vector \\(\mathbf{b}\\), and a commitment \\(C\_c=\mathrm{Com}(c;\rho\_c)\\). The prover claims
\\[
c = \langle \mathbf{a},\mathbf{b}\rangle.
\\]

A sigma-protocol proof masks the witness before the verifier's challenge is known. The prover samples a fresh random vector \\(\mathbf{d}\\) and fresh blinds. It sends one commitment to \\(\mathbf{d}\\) and one commitment to \\(\langle \mathbf{d},\mathbf{b}\rangle\\). The verifier samples a challenge \\(r\\). The prover answers with the masked linear response
\\[
\mathbf{z} = r\mathbf{a}+\mathbf{d},
\\]
plus the corresponding masked blinds. The verifier checks that \\(\mathbf{z}\\) opens the challenged combination of the vector commitments and that \\(\langle \mathbf{z},\mathbf{b}\rangle\\) opens the challenged combination of the scalar commitments.

The mask gives zero knowledge: before the challenge, \\(\mathbf{d}\\) is fresh and hides \\(\mathbf{a}\\), and the response is distributed as a masked vector. Soundness comes from the same linearity. A prover that can answer two different challenges for the same first messages can be rewound algebraically to recover two incompatible openings unless the committed inner-product relation is true. The proof is linear-size because the response contains a vector of the same length as \\(\mathbf{a}\\); no logarithmic recursive folding is involved.

This primer supplies the background for the concrete [polynomial commitment opening](../building-blocks/pcs.md), which combines the matrix factorization above with a linear inner-product argument.
