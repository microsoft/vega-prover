# Polynomial commitments and the ZK opening

This chapter describes Vega's Hyrax polynomial commitment and the zero-knowledge opening used for the final committed-witness evaluation. The algebraic background is reviewed in the [polynomial-commitment primer](../appendix/pcs-primer.md); the field and group conventions come from [Fields, groups, and the engine](../building-blocks/fields-and-groups.md) and [Notation and conventions](../overview/notation.md).

## Commitment key and row-wise commitments

Vega commits to vectors over \\(\mathbb{F}\\) using a Pedersen-style commitment over \\(\mathbb{G}\\). The commitment key contains a row width, a vector of commitment bases, and a dedicated hiding generator. In the canonical monolithic setting the width is
\\[
\mathtt{num\_cols}=2048.
\\]
The key therefore contains bases
\\[
G\_0, G\_1, \dots, G\_{2047}\in\mathbb{G}
\\]
and a hiding generator \\(H\in\mathbb{G}\\).

A length-\\(n\\) vector \\(\mathbf{v}\\) is reshaped row-wise into
\\[
\left\lceil \frac{n}{2048}\right\rceil
\\]
rows. Each row uses the same generator vector. For row \\(i\\), with entries \\(v\_{i,0},\dots,v\_{i,m\_i-1}\\) and blind \\(\rho\_i\\), the row commitment is
\\[
C\_i = \sum\_{j=0}^{m\_i-1} v\_{i,j}G\_j + \rho\_i H.
\\]
The full commitment is the vector of row commitments
\\[
\mathbf{C}=(C\_0, C\_1, \dots).
\\]
The blind is likewise a vector, one scalar per committed row.

The commitment is hiding because each row has its own fresh multiple of \\(H\\). It is binding under the discrete-log relation assumption for the public bases, as in the Pedersen commitment described in the primer. Its additive form makes it homomorphic row by row:
\\[
\alpha\mathbf{C}^{(1)}+\beta\mathbf{C}^{(2)}
\\]
is the commitment to the same linear combination of the underlying row vectors, with blinds combined by the same coefficients.

The prover-side `is_small` parameter used by commitment routines is only a speed hint for choosing a multi-scalar-multiplication path. Verification does not receive or trust this hint; it checks the resulting group commitments and opening equations.

## Combining and folding commitments

Vega's folding protocols repeatedly replace several committed witnesses by one committed linear combination. Hyrax supports this directly. Given commitments \\(\mathbf{C}^{(k)}\\) and folding weights \\(\lambda\_k\\), the folded commitment has rows
\\[
C\_i^\* = \sum\_k \lambda\_k C\_i^{(k)}.
\\]
The folded blind has rows
\\[
\rho\_i^\* = \sum\_k \lambda\_k \rho\_i^{(k)}.
\\]
By homomorphism,
\\[
C\_i^\* = \sum\_j \left(\sum\_k \lambda\_k v\_{i,j}^{(k)}\right)G\_j + \rho\_i^\* H.
\\]
Thus the folded commitment opens as a commitment to the folded witness. This property is what lets [NeutronNova folding](../building-blocks/neutronnova.md) keep a single committed witness opening after many instances have been combined.

Vega also supports concatenating commitments and blinds when separate committed pieces become one longer committed vector. Concatenation preserves the row-wise interpretation: the combined commitment is the sequence of all row commitments, and the combined blind is the matching sequence of row blinds.

## Evaluation opening by matrix factorization

The final opening treats the committed vector as the evaluation table of an MLE, described in [Multilinear polynomials](../building-blocks/multilinear.md). For a point \\(\mathbf{r}\\), the Hyrax evaluation argument splits the point into a row part and a column part:
\\[
\mathbf{r}=(\mathbf{r}\_{\mathrm{row}},\mathbf{r}\_{\mathrm{col}}).
\\]
The split is determined by the number of committed rows and the fixed row width. The row part has one coordinate for each row variable; the column part has one coordinate for each column variable.

Let \\(Z\\) be the row-wise matrix represented by the committed vector. The prover and verifier derive equality-weight vectors
\\[
\mathbf{L} = \bigl(\widetilde{\mathrm{eq}}(\mathbf{r}\_{\mathrm{row}},i)\bigr)\_i, \qquad \mathbf{R} = \bigl(\widetilde{\mathrm{eq}}(\mathbf{r}\_{\mathrm{col}},j)\bigr)\_j.
\\]
The prover computes the row-combined vector
\\[
\mathbf{a} = \sum\_i L\_i Z\_{i,\*}
\\]
and the matching row-combined blind
\\[
\rho\_a = \sum\_i L\_i \rho\_i.
\\]
The verifier computes the corresponding row-combined commitment directly from the public commitment rows:
\\[
C\_a = \sum\_i L\_i C\_i.
\\]
By homomorphism, \\(C\_a=\mathrm{Com}(\mathbf{a};\rho\_a)\\).

The MLE evaluation then reduces to one inner product:
\\[
\tilde{v}(\mathbf{r}) = \langle \mathbf{a},\mathbf{R}\rangle.
\\]
The claimed scalar evaluation is itself committed with the evaluation commitment supplied by the surrounding protocol. The Hyrax opening proves that the row-combined commitment and the evaluation commitment are consistent with the inner product against the public vector \\(\mathbf{R}\\).

## Linear inner-product opening

Vega's inner-product argument is the linear-size Pedersen sigma protocol described in the primer. The statement contains a commitment to \\(\mathbf{a}\\), a public vector \\(\mathbf{b}\\), and a commitment to \\(c\\), with the relation
\\[
c=\langle \mathbf{a},\mathbf{b}\rangle.
\\]
In the Hyrax opening, \\(\mathbf{b}\\) is the equality-weight vector \\(\mathbf{R}\\) for the column half of the evaluation point.

The prover samples a fresh mask vector \\(\mathbf{d}\\) and fresh blinds. It sends two cross-commitments: `delta`, a commitment to \\(\mathbf{d}\\), and `beta`, a commitment to \\(\langle\mathbf{b},\mathbf{d}\rangle\\). Fiat--Shamir produces one challenge. The prover answers with the masked response vector `z_vec` and the corresponding masked blinds:
\\[
\mathbf{z}=r\mathbf{a}+\mathbf{d}.
\\]
The verifier checks two Pedersen equations: one for the vector commitment and one for the scalar inner-product commitment.

This argument is linear-size. It sends a response vector of column width, and it is not a logarithmic recursive folding argument. Its zero-knowledge property comes from the fresh mask \\(\mathbf{d}\\) and fresh blinding terms, which hide the row-combined witness vector while still allowing the verifier to check the linear equations. Its soundness comes from binding and from the fact that two valid responses to different challenges determine the committed vector and scalar relation.

The public vector \\(\mathbf{b}\\) is omitted from the inner-product transcript representation by design. In this use, \\(\mathbf{b}\\) is recomputed by the verifier from the transcript-derived evaluation point, so omitting it from the bytes absorbed for the inner-product instance does not weaken the binding of the checked relation.

## Role in the MC proof

In \\(\mathrm{Vega}\_{\mathrm{MC}}\\), the sum-check and folding layers produce one committed witness evaluation that remains to be opened. Vega first folds the relevant witness commitments and their evaluation commitments using the same challenge, then invokes the Hyrax evaluation argument on the folded commitment. Verification recomputes the same folded commitments and checks the Hyrax opening against the derived evaluation point.

This opening is the mechanism that hides the committed witness evaluation while proving that it is the value required by the verifier's algebraic checks. The surrounding zero-knowledge construction is described in [Zero-knowledge](../mc/zero-knowledge.md). Exact transcript scheduling and proof-object layout are specified separately in [The transcript schedule](../spec/transcript-schedule.md) and [Proof object](../spec/proof-object.md); this chapter intentionally describes only the algebraic protocol.
