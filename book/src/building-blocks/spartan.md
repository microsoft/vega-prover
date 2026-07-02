# The Spartan argument

This chapter describes the R1CS-specific reduction that sits above the generic [sum-check protocol](sumcheck.md). Spartan turns satisfaction of an [R1CS](r1cs.md) instance into an outer zero-check over constraint rows, an inner matrix-evaluation sum-check over columns, and one final [polynomial commitment opening](pcs.md) for the committed witness.

## From R1CS rows to a zero-check

For R1CS matrices \\(A,B,C\\) and assignment \\(\mathbf{z}\\), satisfaction means
\\[
(A\mathbf{z}) \circ (B\mathbf{z}) = C\mathbf{z}.
\\]
Equivalently, the row error
\\[
\mathbf{e} = (A\mathbf{z}) \circ (B\mathbf{z}) - C\mathbf{z}
\\]
vanishes at every constraint row. The vectors \\(A\mathbf{z}\\), \\(B\mathbf{z}\\), and \\(C\mathbf{z}\\) are viewed as tables on the padded row hypercube. If \\(\ell\_x=\log\_2 m\\), the outer layer batches all row equations into one weighted sum:
\\[
\sum\_{\mathbf{x} \in \\{0,1\\}^{\ell\_x}} W\_\tau(\mathbf{x})\bigl(\widetilde{Az}(\mathbf{x})\widetilde{Bz}(\mathbf{x}) - \widetilde{Cz}(\mathbf{x})\bigr) = 0.
\\]
The sum-check reduces this claim to one random row point \\(r\_x\\). The summand contains a multilinear weight multiplied by the product \\(\widetilde{Az}\\,\widetilde{Bz}\\), so an outer round polynomial can have degree three. Vega therefore uses cubic-with-additive-term prover routines for this layer.

## Row weights in the two Vega variants

\\(\mathrm{Vega}\_{\mathrm{SC}}\\) uses the equality-polynomial weight
\\[
W\_\tau(\mathbf{x}) = \widetilde{\mathrm{eq}}(\tau,\mathbf{x}).
\\]
In the reference implementation, the outer zero-check routine `prove_cubic_with_additive_term_zk` returns \\(r\_x\\). The verifier recomputes \\(\widetilde{\mathrm{eq}}(\tau,r\_x)\\) as `tau_at_rx` and checks the public values
\\[
\bigl[\tau\_{r\_x},\\,\widetilde{X}(r\_y'),\\,\widetilde{A}(r\_x,r\_y)+r\widetilde{B}(r\_x,r\_y)+r^2\widetilde{C}(r\_x,r\_y)\bigr].
\\]

\\(\mathrm{Vega}\_{\mathrm{MC}}\\), the focus of this book, uses a powers-polynomial weight derived from \\(\tau\\). The powers polynomial is the multilinear extension whose Boolean evaluations are \\(1,\tau,\tau^2,\dots\\), with the bit ordering fixed by [Notation and conventions](../overview/notation.md). The folded step branch and the core branch pass through one batched outer round schedule rather than two independent schedules.

For \\(\mathrm{Vega}\_{\mathrm{MC}}\\), the verifier recomputes the powers weight at \\(r\_x\\) as `tau_at_rx`. It also recomputes \\(\widetilde{\mathrm{eq}}(r\_b,\rho)\\), written `eq_rho_at_rb`, from the [NeutronNova fold](neutronnova.md). The six public values pinned by the verifier circuit are, in order,
\\[
\tau\_{r\_x},\quad \widetilde{X}\_{\mathrm{step}}(r\_y'),\quad \widetilde{X}\_{\mathrm{core}}(r\_y'),\quad \widetilde{\mathrm{eq}}(r\_b,\rho),\quad Q\_{\mathrm{step}},\quad Q\_{\mathrm{core}},
\\]
where
\\[
Q\_\star = \widetilde{A}\_\star(r\_x,r\_y)+r\widetilde{B}\_\star(r\_x,r\_y)+r^2\widetilde{C}\_\star(r\_x,r\_y)
\\]
for \\(\star\in\\{\mathrm{step},\mathrm{core}\\}\\). The public-input polynomials \\(\widetilde{X}\_\star\\) are evaluated at \\(r\_y' = r\_y[1..]\\) — the inner point with its leading selector coordinate \\(r\_{y,0}\\) dropped — while the matrix evaluations \\(Q\_\star\\) use the full \\(r\_y\\).

## The inner matrix-evaluation sum-check

After the outer layer fixes \\(r\_x\\), the matrix row variables are bound and the three matrices are combined with a fresh challenge \\(r\\):
\\[
\widetilde{ABC}\_{r\_x,r}(\mathbf{y}) = \widetilde{A}(r\_x,\mathbf{y}) + r\\,\widetilde{B}(r\_x,\mathbf{y}) + r^2\\,\widetilde{C}(r\_x,\mathbf{y}).
\\]
The inner sum-check proves the column-side claim
\\[
\sum\_{\mathbf{y}} \widetilde{ABC}\_{r\_x,r}(\mathbf{y})\\,\widetilde{z}(\mathbf{y})
\\]
over the padded column hypercube and reduces it to one column point \\(r\_y\\). The summand is a product of two multilinear tables, so each inner round polynomial is quadratic. The generic routines for this shape are `prove_quad` and the batched zero-knowledge variant `prove_quad_batched_zk`; \\(\mathrm{Vega}\_{\mathrm{MC}}\\) uses the batched variant to carry the folded step and core branches through the same challenge schedule.

## Hand-off to the witness opening

The terminal inner claim contains \\(\widetilde{z}(r\_y)\\). The verifier recomputes the public-input contribution directly from the public vector and the equality weights induced by \\(r\_y\\). The witness contribution is obtained from the polynomial-commitment opening at \\(r\_y[1..]\\). In \\(\mathrm{Vega}\_{\mathrm{MC}}\\), this opening is the zero-knowledge linear inner-product argument stored as `eval_arg`.

The verifier then compares the resulting value of \\(\widetilde{z}(r\_y)\\), together with the terminal matrix evaluation, against the final inner sum-check claim. This is the boundary between Spartan and [Polynomial commitments and the ZK opening](pcs.md).

## Challenge ownership

The challenges \\(\tau\\), \\(r\_x\\), and \\(r\_y\\) belong to Spartan's row batching and sum-check reductions. The challenges \\(\rho\\) and \\(r\_b\\) belong to [NeutronNova folding](neutronnova.md), which produces the folded step instance that Spartan proves.

For background, see [Multilinear polynomials](multilinear.md), the [sum-check primer](../appendix/sumcheck-primer.md), [R1CS and its variants](r1cs.md), and the generic [sum-check protocol](sumcheck.md). The surrounding \\(\mathrm{Vega}\_{\mathrm{MC}}\\) proving flow is described in [Proving](../mc/prove.md), and byte-exact transcript ordering is fixed in [the transcript schedule](../spec/transcript-schedule.md).
