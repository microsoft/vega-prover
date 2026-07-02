# Primer: folding schemes

This appendix gives a self-contained account of folding for relaxed R1CS. Folding is the accumulation step that replaces several constraint instances with one instance, so a verifier can carry a small state and defer the expensive opening or satisfiability proof to the end.

## Relaxed R1CS

For matrices \\(A,B,C \in \mathbb{F}^{m \times n}\\), a strict R1CS instance is satisfied by \\(\mathbf{z} \in \mathbb{F}^n\\) when
\\[
(A\mathbf{z}) \circ (B\mathbf{z}) \\;=\\; C\mathbf{z}.
\\]
A relaxed instance adds a scalar \\(u \in \mathbb{F}\\) and an error vector \\(E \in \mathbb{F}^m\\):
\\[
(A\mathbf{z}) \circ (B\mathbf{z}) \\;=\\; u\\,C\mathbf{z} + E.
\\]
A strict instance is the special case \\(u=1\\) and \\(E=\mathbf{0}\\). Relaxation is what makes folding closed: after combining two witnesses linearly, the quadratic terms that appear can be recorded in the new error vector instead of requiring another full R1CS instance.

## Folding two instances

Consider two relaxed instances over the same matrices \\(A,B,C\\). Their satisfying assignments are
\\[
(A\mathbf{z}\_1) \circ (B\mathbf{z}\_1) = u\_1 C\mathbf{z}\_1 + E\_1
\\]
and
\\[
(A\mathbf{z}\_2) \circ (B\mathbf{z}\_2) = u\_2 C\mathbf{z}\_2 + E\_2.
\\]
The verifier samples a challenge \\(r \in \mathbb{F}\\), and the folded assignment uses the linear combinations
\\[
\mathbf{z} = \mathbf{z}\_1 + r\mathbf{z}\_2, \qquad u = u\_1 + r u\_2.
\\]
Expanding the left side gives
\\[
(A\mathbf{z}) \circ (B\mathbf{z}) = (A\mathbf{z}\_1)\circ(B\mathbf{z}\_1) + r\bigl((A\mathbf{z}\_1)\circ(B\mathbf{z}\_2) + (A\mathbf{z}\_2)\circ(B\mathbf{z}\_1)\bigr) + r^2 (A\mathbf{z}\_2)\circ(B\mathbf{z}\_2).
\\]
The mixed degree-one part is the cross-term. The prover commits to the corresponding vector as \\(\overline{T}\\), and the folded error is chosen so the expanded equation has the relaxed form
\\[
(A\mathbf{z}) \circ (B\mathbf{z}) \\;=\\; u\\,C\mathbf{z} + E.
\\]
For a general relaxed-relaxed fold, \\(E\\) contains the linear combination of the input errors together with the committed cross-term, with coefficients determined by \\(r\\). For the common relaxed-with-strict case \\((u\_2=1, E\_2=\mathbf{0})\\), this specializes to an error of the form
\\[
E = E\_1 + rT.
\\]

The commitment \\(\overline{T}\\) binds the prover to the mixed term before the challenge is used to form the folded instance. If both input instances are satisfied and \\(T\\) is computed honestly, the folded instance is satisfied. Conversely, if an input equation is false or the cross-term is inconsistent, the folded relaxed equation becomes a nonzero low-degree polynomial in the verifier challenge; it can vanish at a random \\(r\\) only with small probability.

## Folding a batch

A batch of \\(k\\) uniform instances can be folded by a random linear combination. The verifier derives challenges that determine weights \\(\lambda\_0,
\dots,
\lambda\_{k-1} \in \mathbb{F}\\), and the folded witness and public data are formed coordinate-wise:
\\[
\mathbf{z}\_\star = \sum\_{i=0}^{k-1} \lambda\_i \mathbf{z}\_i, \qquad u\_\star = \sum\_{i=0}^{k-1} \lambda\_i u\_i.
\\]
The error vector collects the corresponding linear combination of input errors and the cross-terms introduced by the quadratic constraint. Commitments are folded with the same weights, using the additively homomorphic property of the commitment scheme.

Soundness follows the same principle as the sum-check protocol. If at least one input equation is false, the folded check represents a nonzero polynomial in the verifier's random choices. Its degree is linear in the number of folded instances, so the Schwartz--Zippel lemma bounds the probability that the false batch survives by
\\[
O\\!\left(\frac{k}{|\mathbb{F}|}\right).
\\]
The bound is small when \\(|\mathbb{F}|\\) is large and the challenges are sampled after the prover has committed to the data being folded.

## Verifier shape

Folding turns many satisfiability claims into one accumulated claim. During the batch, the verifier keeps only the folded public data, folded commitments, and the transcript challenges that define the random linear combination. The verifier does not open every witness commitment immediately. Instead, it checks the folding messages as they arrive and defers the single remaining satisfiability proof or commitment opening to the end.

This is the same economy exploited by sum-check: many local equalities are compressed into one random claim, and the final proof opens only the value needed to finish verification. See the [sum-check primer](sumcheck-primer.md), [R1CS](../building-blocks/r1cs.md), and [polynomial commitments](../building-blocks/pcs.md) for the surrounding building blocks.
