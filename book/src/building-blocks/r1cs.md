# R1CS and its variants

This chapter defines the rank-1 constraint systems used by Vega and the split forms that let the prover commit to different parts of a witness at different times. It is a building-block chapter: commitment schemes, transcripts, folding, and byte-exact proof objects are treated in their own chapters.

## Standard R1CS

An R1CS shape over \\(\mathbb{F}\\) consists of three sparse matrices
\\[
A, B, C \in \mathbb{F}^{m \times n}.
\\]
For an assignment vector \\(\mathbf{z} \in \mathbb{F}^n\\), the constraint system is satisfied when
\\[
(A\mathbf{z}) \circ (B\mathbf{z}) \\;=\\; C\mathbf{z},
\\]
where \\(\circ\\) is the entrywise product. Equivalently, for every row \\(i \in \\{0,
\dots, m-1\\}\\),
\\[
\Bigl(\sum\_j A\_{i,j} z\_j\Bigr) \Bigl(\sum\_j B\_{i,j} z\_j\Bigr) \\;=\\; \sum\_j C\_{i,j} z\_j.
\\]

The conceptual assignment is partitioned into public input/output \\(\mathbf{x}\\), the constant \\(1\\), and private witness \\(\mathbf{w}\\). Vega's concrete R1CS matrices are laid out with the witness columns first, then the constant column, then the public columns, so the vector multiplied by the matrices is
\\[
\mathbf{z} \\;=\\; (\mathbf{w}, 1, \mathbf{x}).
\\]
The standard instance carries the public vector \\(\mathbf{x}\\) and a commitment to \\(\mathbf{w}\\); the witness carries \\(\mathbf{w}\\) and the blind opening that commitment.

The matrices are stored sparsely. A sparse matrix records the nonzero entries in `data`, their column positions in `indices`, row boundaries in `indptr`, and the total column count in `cols`. This is enough to compute each row of \\(A\mathbf{z}\\), \\(B\mathbf{z}\\), and \\(C\mathbf{z}\\) without materializing dense matrices. Byte-level encodings of verifier keys and proofs are specified in [Verifier key](../spec/verifier-key.md) and [Proof object](../spec/proof-object.md), not in this chapter.

R1CS is the algebraic layer beneath the polynomial protocols. The row products become claims about multilinear extensions in [Multilinear extensions](../building-blocks/multilinear.md), and those claims are checked with [the sum-check protocol](../building-blocks/sumcheck.md). Witness commitments and openings are handled by [Polynomial commitments and the ZK opening](../building-blocks/pcs.md).

## Relaxed R1CS

A relaxed R1CS instance uses the same matrices \\((A,B,C)\\), but changes the satisfaction equation to
\\[
(A\mathbf{z}) \circ (B\mathbf{z}) \\;=\\; u\\,C\mathbf{z} + \mathbf{E}.
\\]
Here \\(u \in \mathbb{F}\\) is a scalar carried by the instance, and \\(\mathbf{E} \in \mathbb{F}^m\\) is an error vector carried by the witness. In the concrete matrix layout,
\\[
\mathbf{z} \\;=\\; (\mathbf{w}, u, \mathbf{x}).
\\]
The standard R1CS condition is the strict case \\(u=1\\) and \\(\mathbf{E}=\mathbf{0}\\).

The scalar \\(u\\) and error vector \\(\mathbf{E}\\) make the relation closed under the linear combinations used by folding. Instead of requiring every folded object to remain a strict R1CS instance at each intermediate step, the accumulated discrepancy is represented explicitly as \\(\mathbf{E}\\). The relaxed instance commits to both \\(\mathbf{w}\\) and \\(\mathbf{E}\\), while the relaxed witness contains the two vectors and their blinds.

Vega also samples random relaxed instance/witness pairs by choosing a full random assignment, taking \\(u\\) from the assignment's constant-position slot, and defining
\\[
\mathbf{E} \\;=\\; (A\mathbf{z}) \circ (B\mathbf{z}) - u\\,C\mathbf{z}.
\\]
This construction makes the sampled pair satisfy the relaxed equation by definition.

## Folding many standard objects

Standard witnesses and instances provide `fold_multiple` operations. Given challenges \\(r\_b\\), Vega derives folding weights and forms linear combinations of witness vectors, public vectors, witness blinds, and witness commitments. The result is a single witness/instance pair whose components represent the weighted combination of many inputs. The algebra that justifies these combinations belongs to [NeutronNova folding](../building-blocks/neutronnova.md); this chapter only records the R1CS objects that participate in that interface.

## Split R1CS

A split R1CS shape keeps the same matrices \\((A,B,C)\\) but partitions the private witness columns into named segments:
\\[
\mathbf{w} \\;=\\; (\mathbf{w}\_{\mathrm{shared}}, \mathbf{w}\_{\mathrm{precommitted}}, \mathbf{w}\_{\mathrm{rest}}).
\\]
The shape records the number of constraints, the padded and unpadded sizes of the three witness segments, the number of public values, and the number of transcript-derived challenges. Its `sizes()` method returns a fixed ten-entry summary: unpadded constraint count; unpadded shared, precommitted, and rest sizes; padded constraint count; padded shared, precommitted, and rest sizes; public-value count; and challenge count.

A split instance contains separate commitments for the shared, precommitted, and rest witness segments, together with public values and derived challenges. The shared and precommitted commitments are optional exactly when their corresponding segments are empty; the rest commitment is always present for the rest segment, including padding. The split assignment uses
\\[
\mathbf{z} \\;=\\; (\mathbf{w}\_{\mathrm{shared}}, \mathbf{w}\_{\mathrm{precommitted}}, \mathbf{w}\_{\mathrm{rest}}, 1, \mathbf{x}, \boldsymbol{\chi}),
\\]
where \\(\boldsymbol{\chi}\\) denotes the challenge values stored with the instance.

The split exists so that different pieces of the witness can be committed at the time they become determined. Shared witness variables are synthesized and committed once, then reused across step instances and the core instance. Precommitted variables are synthesized during preparation, before challenge-dependent rest variables are available. The rest segment is committed after the transcript has produced the challenges needed to finish the witness. This arrangement lets many step instances share one witness commitment component while preserving the single committed witness expected by the ordinary R1CS interface.

The bridge back to ordinary R1CS is explicit. `to_regular_shape()` adds the three witness segment sizes into one `num_vars` value and treats public values plus challenges as the regular public vector. `to_regular_instance()` combines the present segment commitments into one witness commitment and sets
\\[
\mathbf{X} \\;=\\; (\mathbf{x}, \boldsymbol{\chi}).
\\]
After this conversion, the usual R1CS satisfaction and commitment checks apply.

The step circuit and the core circuit can have different split layouts. `equalize()` pads both shapes to the same constraint count and the same total witness-column count, extending the rest segment when necessary and moving public/challenge columns consistently. This gives later folding and verification code uniform dimensions to work with.

## Transcript-derived challenges in a split instance

A split instance validates its challenge vector through the [transcript](../building-blocks/transcript.md). Validation absorbs the shared witness commitment if present, absorbs the precommitted witness commitment if present, squeezes the required number of field challenges, compares them with the instance's stored challenges, and then absorbs the rest commitment. This means the rest commitment is bound after the challenges that determine the rest witness, while the earlier commitments are bound before those challenges are derived.

The exact transcript byte schedule belongs to [the transcript schedule](../spec/transcript-schedule.md). This chapter only uses the mathematical consequence: the challenge vector stored in a split instance must be the vector obtained from the commitments and public context that precede it.

## Multi-round split R1CS

The in-circuit verifier uses a multi-round split form. A `SplitMultiRoundR1CSShape` partitions the witness by rounds rather than by shared/precommitted/rest segments. It records the number of rounds, the padded and unpadded witness size for each round, the number of challenges produced after each round, the number of public values, a commitment width, and the matrices \\((A,B,C)\\).

A `SplitMultiRoundR1CSInstance` contains one witness commitment per round, the public values, and the challenges grouped by round. During witness generation, a round synthesizes that round's variables, commits to the padded round segment, absorbs that commitment into the transcript, and squeezes the next round's challenges. Validation performs the same transcript reconstruction: for each round it checks the commitment length, absorbs the round commitment, squeezes the configured number of challenges, and compares the result with the stored round challenges.

The ordinary R1CS view is again obtained by combining commitments and flattening public data. `to_regular_shape()` sums all per-round witness sizes and all per-round challenge counts. `to_regular_instance()` combines all round commitments into one witness commitment and sets the public input vector to
\\[
\mathbf{X} \\;=\\; (\boldsymbol{\chi}, \mathbf{x}),
\\]
with the flattened challenges before the public values. This ordering matches the way the multi-round circuit inputizes its challenges.

The multi-round split is the shape used by the verifier circuit described in [In-circuit verifier](../building-blocks/in-circuit-verifier.md). It relies on the same transcript abstraction as the rest of the proof system, and it feeds the same commitment-opening machinery described in [Polynomial commitments and the ZK opening](../building-blocks/pcs.md). The proof and verifier-key chapters specify how these objects appear in serialized artifacts: [Verifier key](../spec/verifier-key.md) and [Proof object](../spec/proof-object.md).
