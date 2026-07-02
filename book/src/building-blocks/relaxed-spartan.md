# Relaxed Spartan

This chapter describes the Spartan argument used for a relaxed R1CS instance. In Vega this crate-internal argument proves the single folded relaxed instance produced by the [Nova zero-knowledge fold](nova-zk.md).

## Relaxed relation

A relaxed R1CS instance uses the same matrices \\(A,B,C\\) as strict [R1CS](r1cs.md), but carries a relaxation scalar \\(u\\) and an error vector \\(\mathbf{E}\\). For assignment \\(\mathbf{z}\\), satisfaction is
\\[
(A\mathbf{z}) \circ (B\mathbf{z}) = u\\,C\mathbf{z} + \mathbf{E}.
\\]
Strict R1CS is the special case \\(u=1\\) and \\(\mathbf{E}=\mathbf{0}\\). The relaxed relation is closed under the linear combinations used by folding, with accumulated discrepancy represented by \\(\mathbf{E}\\).

## Prover transcript and outer check

`RelaxedR1CSSpartanProof` contains an outer sum-check proof, the three outer terminal claims, an inner sum-check proof, and direct openings for the committed witness \\(\mathbf{W}\\) and error vector \\(\mathbf{E}\\). Its transcript intentionally absorbs only the relaxed scalar and public vector:

- `b"u_relaxed"` absorbs \\(u\\);
- `b"X_relaxed"` absorbs \\(\mathbf{X}\\).

The commitments `comm_W` and `comm_E` are not absorbed by this argument. They must already be bound by the enclosing folding protocol.

The prover squeezes \\(\ell\_x\\) row challenges with label `b"t"` and uses them as the equality-polynomial weight in the relaxed outer zero-check:
\\[
\sum\_{\mathbf{x}} \widetilde{\mathrm{eq}}(t,\mathbf{x})\bigl(\widetilde{Az}(\mathbf{x})\widetilde{Bz}(\mathbf{x}) - u\\,\widetilde{Cz}(\mathbf{x}) - \widetilde{E}(\mathbf{x})\bigr)=0.
\\]
The routine `prove_cubic_with_three_inputs` returns the row point \\(r\_x\\) and the terminal claims for \\(\widetilde{Az}(r\_x)\\), \\(\widetilde{Bz}(r\_x)\\), and \\(u\widetilde{Cz}(r\_x)+\widetilde{E}(r\_x)\\). The prover absorbs these three values under `b"claims_outer"`.

## Inner check and openings

The prover squeezes the combining challenge `b"r"`, binds the matrix row variables at \\(r\_x\\), and forms the row-bound combination for \\(A\\), \\(B\\), and \\(uC\\). It then runs the quadratic inner sum-check with `prove_quad`, reducing the column-side claim to \\(r\_y\\).

The witness and error openings are the zero-knowledge-relevant distinction from the main Spartan opening. Relaxed Spartan uses direct, non-hiding reduced-linear-combination openings:

- `prove_direct` opens \\(\mathbf{W}\\) at \\(r\_y[1..]\\), producing \\((v\_W,\mathrm{blind}\_W)\\);
- `prove_direct` opens \\(\mathbf{E}\\) at \\(r\_x\\), producing \\((v\_E,\mathrm{blind}\_E)\\).

The prover absorbs `b"v_W"` and `b"v_E"` after producing those direct opening vectors. It does not use the zero-knowledge linear IPA described in [Polynomial commitments and the ZK opening](pcs.md).

## Why direct openings are safe in this use

A direct opening would not be zero-knowledge for an arbitrary relaxed instance. Vega runs this argument only on the fold output of the [Nova zero-knowledge fold](nova-zk.md). That fold combines the real verifier-circuit instance with a fresh full-length random satisfying relaxed instance. The random mask supplies the zero-knowledge property; the direct opening is made on the masked folded object, not on the unmasked verifier-circuit witness.

This composition is described from the user-facing proof perspective in [Zero-knowledge](../mc/zero-knowledge.md). Relaxed Spartan relies on that composition rather than adding another hiding opening inside this crate-internal argument.

## Verification

Verification mirrors the prover transcript. It absorbs `b"u_relaxed"` and `b"X_relaxed"`, squeezes the `b"t"` challenges for the outer equality weight, verifies the cubic outer sum-check, absorbs the three outer claims, squeezes `b"r"`, and verifies the quadratic inner sum-check. It then verifies the direct opening of \\(\mathbf{E}\\) at \\(r\_x\\) and of \\(\mathbf{W}\\) at \\(r\_y[1..]\\), recomputes the terminal matrix and assignment evaluations, and absorbs `b"v_W"` and `b"v_E"`.

The byte-exact labels and ordering are fixed in [the transcript schedule](../spec/transcript-schedule.md). This chapter records the algebraic role of the messages and why the direct openings are valid only in the Nova zero-knowledge composition.
