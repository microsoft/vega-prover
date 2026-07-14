# Nova folding for zero-knowledge

This chapter describes the second fold in \\(\mathrm{Vega}\_{\mathrm{MC}}\\): the verifier-circuit instance is folded with a fresh random satisfying relaxed instance before the final relaxed-R1CS proof is produced.

## From verifier circuit to ordinary R1CS

The [in-circuit verifier](in-circuit-verifier.md) is first materialized as a multi-round split R1CS instance. Its round commitments and transcript-derived challenges are validated through the transcript, then the instance is converted to an ordinary R1CS instance. This ordinary instance carries one combined witness commitment and one public vector containing the flattened verifier-circuit challenges followed by the verifier-circuit public values.

This real verifier-circuit instance is not proved directly. Instead, \\(\mathrm{Vega}\_{\mathrm{MC}}\\) samples an independent relaxed [R1CS](r1cs.md) instance and witness of the same shape.

## Sampling the masking instance

The sampled relaxed pair is satisfying by construction. The sampler chooses a full random assignment \\(\mathbf{z}\\), takes the relaxation scalar \\(u\\) from the assignment slot used for the constant/relaxation coordinate, and defines the error vector by

\\[ E = (A\mathbf{z})\circ(B\mathbf{z}) - u\\,C\mathbf{z}. \\]

This makes the relaxed equation true for the sampled pair. The sampler also commits to the random witness and to the computed error vector, so the sampled instance has the same public shape as any other relaxed instance used by [Relaxed Spartan](relaxed-spartan.md).

## The Nova NIFS fold

Nova NIFS folds the random satisfying relaxed instance with the real verifier-circuit R1CS instance. Let the random relaxed pair be \\((U\_1,W\_1)\\), and let the real verifier-circuit pair be \\((U\_2,W\_2)\\). The prover computes the cross-term vector \\(T\\), commits to it as `comm_T`, absorbs \\(U\_1\\), \\(U\_2\\), and `comm_T` into the transcript, and squeezes the folding challenge \\(r\\).

The folded witness is

\\[ \mathbf{W}\_\star = \mathbf{W}\_1 + r\mathbf{W}\_2, \qquad E\_\star = E\_1 + rT, \\]

and the folded public scalar and public vector are

\\[ u\_\star = u\_1 + r, \qquad \mathbf{X}\_\star = \mathbf{X}\_1 + r\mathbf{X}\_2. \\]

The folded instance commitments use the same linear combination: the witness commitment is the random witness commitment plus \\(r\\) times the real witness commitment, and the error commitment is the random error commitment plus \\(r\\) times `comm_T`.

## Why this gives zero knowledge

This fold is the zero-knowledge mechanism. The real verifier-circuit witness appears only with transcript challenge coefficient \\(r\\), while an independently sampled full-length random satisfying relaxed witness appears with coefficient \\(1\\). The subsequent relaxed proof runs on the masked folded instance, not on the original verifier-circuit instance. Since the mask is sampled fresh for each proof and has the full relaxed shape, the later non-ZK relaxed proof leaks no information about the real verifier-circuit witness beyond the public statement.

Verification repeats the same transcript step: it absorbs the random relaxed instance, the real verifier-circuit instance, and `comm_T`, then squeezes \\(r\\) and folds only the instance commitments and public data. The verifier does not know \\(\mathbf{W}\_\star\\). Satisfiability of the folded relaxed instance is supplied later by [Relaxed Spartan](relaxed-spartan.md), and the surrounding \\(\mathrm{Vega}\_{\mathrm{MC}}\\) privacy flow is summarized in [Zero-knowledge](../mc/zero-knowledge.md). For the general folding algebra, see the [folding primer](../appendix/folding-primer.md).
