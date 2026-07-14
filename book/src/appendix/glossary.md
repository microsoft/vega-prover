# Glossary

Terms as they are used in this book.

**Absorb / squeeze.** `absorb(label, value)` appends labeled encoded data to the running Fiat--Shamir transcript. `squeeze(label)` derives the next scalar challenge in \\(\mathbb{F}\\), updates the transcript state, and resets the running absorb buffer. (see [The Fiat--Shamir transcript](../building-blocks/transcript.md))

**Accept-conforming prover.** A prover that emits proofs accepted by `verify` under the verifier key. (`verify` returns the public values it recomputes; the application checks them against the intended statement.) It need not reproduce the reference prover's exact proof bytes. (see [Scope and the conformance contract](../spec/scope.md))

**Base field.** The field in which coordinates of \\(\mathbb{G}\\) live. In the canonical engine it is distinct from the scalar field used for proof-system arithmetic. (see [Fields, groups, and the engine](../building-blocks/fields-and-groups.md))

**Blind.** A scalar, or row-indexed tuple of scalars, used with the hiding generator when committing to a vector. Blinds combine linearly when Hyrax commitments are folded or concatenated. (see [Polynomial commitments and the ZK opening](../building-blocks/pcs.md))

**Byte-conforming prover.** A prover that, given the same verifier key, public values, witness, and random tape, reproduces the reference prover's exact serialized proof bytes. This requires exact agreement on the transcript schedule, encodings, proof-object fields, and randomness consumption. (see [Scope and the conformance contract](../spec/scope.md))

**Canonical engine.** The concrete engine fixed by the specification: T256 as \\(\mathbb{G}\\), its scalar field as \\(\mathbb{F}\\), a Keccak256-based Fiat--Shamir transcript, and Hyrax polynomial commitments. Other engines are outside the byte-exact specification. (see [Scope and the conformance contract](../spec/scope.md))

**Challenge.** A scalar in \\(\mathbb{F}\\) derived by `squeeze` from the transcript. Challenges such as \\(\tau\\), \\(\rho\\), \\(r\\), \\(r\_b\\), \\(r\_x\\), and \\(r\_y\\) are owned by specific folding or sum-check layers.

**Commitment key.** The public Hyrax generator material used to commit to vectors: a row width, vector bases, and a hiding generator. Setup creates the keys used for step/core witnesses and for the verifier-circuit witness. (see [Polynomial commitments and the ZK opening](../building-blocks/pcs.md))

**Core circuit.** The circuit instance \\(C\_2\\) that joins the step batch in \\(\mathrm{Vega}\_{\mathrm{MC}}\\). The core branch shares the same committed witness prefix where configured and is proved alongside the folded step branch.

**Eq polynomial.** The equality polynomial \\(\widetilde{\mathrm{eq}}(\mathbf{r},\mathbf{x})\\), the MLE of the equality indicator on the Boolean hypercube. Vega uses it for MLE evaluation weights, folding checks such as `eq_rho_at_rb`, and some row-batching computations. (see [Multilinear polynomials](../building-blocks/multilinear.md))

**Fiat--Shamir transcript.** The Keccak-based state machine that turns protocol messages into deterministic verifier challenges. The mechanism defines `new`, `absorb`, `squeeze`, and `dom_sep`; the MC protocol's exact operation order is fixed separately by the transcript schedule. (see [The Fiat--Shamir transcript](../building-blocks/transcript.md))

**Folding / NIFS.** A non-interactive folding step that replaces two or more instances by one linearly combined instance under transcript-derived challenges. In this book, NeutronNova folding accumulates step instances, while Nova NIFS folds a random relaxed instance with the real verifier-circuit instance.

**Group \\(\mathbb{G}\\).** The prime-order elliptic-curve group used additively for commitments. Scalars from \\(\mathbb{F}\\) multiply group generators in Hyrax commitments and multi-scalar multiplications. (see [Fields, groups, and the engine](../building-blocks/fields-and-groups.md))

**Hiding generator.** The dedicated Hyrax commitment-key generator \\(H\\) (or \\(h\\)) multiplied by the blind in each row commitment. It supplies the Pedersen hiding term independent of the vector bases. (see [Polynomial commitments and the ZK opening](../building-blocks/pcs.md))

**Hyrax commitment.** Vega's row-wise Pedersen-style vector commitment over \\(\mathbb{G}\\). A vector is reshaped into fixed-width rows, each row is committed with the same generator vector and a fresh blind, and the resulting commitment is the vector of row commitments. (see [Polynomial commitments and the ZK opening](../building-blocks/pcs.md))

**In-circuit verifier.** The multi-round R1CS circuit that checks Vega's verifier algebra inside the proof. Its public values are later recomputed natively by the outer verifier and compared against the circuit's outputs. (see [The in-circuit verifier](../building-blocks/in-circuit-verifier.md))

**Multilinear extension (MLE).** The unique multilinear polynomial \\(\widetilde{f}:\mathbb{F}^\ell\to\mathbb{F}\\) that agrees with a table \\(f:\{0,1\}^\ell\to\mathbb{F}\\) on the Boolean hypercube. Vega represents MLEs by dense evaluation tables with LSB-first indexing and binds variables high-to-low in sum-check. (see [Multilinear polynomials](../building-blocks/multilinear.md))

**NeutronNova folding.** The first fold in \\(\mathrm{Vega}\_{\mathrm{MC}}\\), which accumulates many strict step R1CS instances into one folded step instance. It uses \\(\tau\\) to compress constraint residuals and \\(\rho\\) / \\(r\_b\\) challenges to fold the padded step batch. (see [NeutronNova folding](../building-blocks/neutronnova.md))

**Nova folding.** The zero-knowledge fold that combines a fresh random satisfying relaxed instance with the real verifier-circuit instance. The resulting masked relaxed instance is the one proved by relaxed Spartan. (see [Nova folding for zero-knowledge](../building-blocks/nova-zk.md))

**num_steps.** The number of step instances fixed at setup and stored in the verifier key. It must be at least two for \\(\mathrm{Vega}\_{\mathrm{MC}}\\), and its raw value is digested into the verifier key even though folding may pad it to a power of two. (see [Setup](../mc/setup.md))

**Polynomial commitment scheme (PCS).** The commitment and opening machinery used for witness vectors viewed as multilinear-polynomial tables. In the canonical engine, this is Hyrax with a zero-knowledge evaluation opening for the final committed-witness evaluation. (see [Polynomial commitments and the ZK opening](../building-blocks/pcs.md))

**Powers polynomial.** The MLE whose Boolean evaluations are \\(1,\tau,\tau^2,\dots\\) under the book's LSB-first hypercube indexing. \\(\mathrm{Vega}\_{\mathrm{MC}}\\) uses it as the Spartan row weight derived from \\(\tau\\). (see [The Spartan argument](../building-blocks/spartan.md))

**Precommitted witness segment.** The split-R1CS witness segment synthesized and committed during preparation, before challenge-dependent rest variables are available. Its commitment is optional exactly when the segment is empty. (see [R1CS](../building-blocks/r1cs.md))

**Public values.** The public scalar vector carried by an R1CS instance or exposed by the verifier circuit. In the MC verifier circuit these include `tau_at_rx`, `eval_X_step`, `eval_X_core`, `eq_rho_at_rb`, `quotient_step`, and `quotient_core`.

**R1CS.** A rank-1 constraint system over \\(\mathbb{F}\\) with matrices \\(A,B,C\\) and satisfaction equation \\((A\mathbf{z})\circ(B\mathbf{z})=C\mathbf{z}\\). Vega's standard assignment layout is \\(\mathbf{z}=(\mathbf{w},1,\mathbf{x})\\). (see [R1CS](../building-blocks/r1cs.md))

**Relaxed R1CS.** The R1CS variant with equation \\((A\mathbf{z})\circ(B\mathbf{z})=u\,C\mathbf{z}+\mathbf{E}\\). The scalar \\(u\\) and error vector \\(\mathbf{E}\\) make the relation closed under folding. (see [R1CS](../building-blocks/r1cs.md))

**Relaxed Spartan.** The Spartan argument Vega uses for a single folded relaxed R1CS instance. In the MC proof it runs after Nova folding and uses direct openings on the masked folded witness and error. (see [Relaxed Spartan](../building-blocks/relaxed-spartan.md))

**Rest witness segment.** The split-R1CS witness segment committed after transcript-derived challenges are available. It contains variables that depend on those challenges and is always represented by a rest commitment, including padding. (see [R1CS](../building-blocks/r1cs.md))

**Round polynomial.** The univariate polynomial sent in one sum-check round. Vega stores it compressed by omitting the linear coefficient, which the verifier reconstructs from the current claim. (see [The sum-check protocol](../building-blocks/sumcheck.md))

**Scalar field \\(\mathbb{F}\\).** The field used for witnesses, constraints, MLEs, sum-check messages, blinds, and Fiat--Shamir challenges. In the canonical engine it is the scalar field of T256. (see [Fields, groups, and the engine](../building-blocks/fields-and-groups.md))

**Shared witness segment.** The split-R1CS witness segment synthesized and committed once, then reused across step instances and the core instance. The proof object may hoist this shared commitment to the top level and set per-instance shared commitments to `None`. (see [R1CS](../building-blocks/r1cs.md))

**Spartan.** The R1CS argument that reduces satisfaction to an outer zero-check over rows, an inner matrix-evaluation sum-check over columns, and a final committed-witness opening. \\(\mathrm{Vega}\_{\mathrm{MC}}\\) runs Spartan jointly on the folded step branch and the core branch. (see [The Spartan argument](../building-blocks/spartan.md))

**Split R1CS shape.** An R1CS shape whose witness columns are partitioned into shared, precommitted, and rest segments, with public values and transcript-derived challenges tracked separately. It lets Vega commit to witness pieces at the time they become determined while converting back to the ordinary R1CS interface. (see [R1CS](../building-blocks/r1cs.md))

**Step circuit.** The repeated circuit \\(C\_1\\) in the MC statement. Setup fixes `num_steps` copies, and NeutronNova folding accumulates those step instances into one folded step branch.

**Sum-check protocol.** The protocol that checks a hypercube-sum claim through a sequence of low-degree univariate round polynomials and transcript challenges. Vega uses specialized quadratic, cubic, and batched zero-knowledge prover shapes. (see [The sum-check protocol](../building-blocks/sumcheck.md))

**T256 curve.** The canonical elliptic curve used as \\(\mathbb{G}\\). Its scalar field is \\(\mathbb{F}\\), and its base field supplies group-coordinate encodings. (see [Fields, groups, and the engine](../building-blocks/fields-and-groups.md))

**Verifier key.** The public setup object consumed by `verify`, containing commitment material, step/core shapes, verifier-circuit shapes and keys, and `num_steps`. It is not serialized into the proof. (see [Verifier key](../spec/verifier-key.md))

**Verifier-key digest.** The 32-byte SHA-256 digest of the verifier-key fields in their specified order and encodings. It is absorbed into the transcript as `vk`, binding every later challenge to the setup material. (see [Verifier key](../spec/verifier-key.md))

**\\(\mathrm{Vega}\_{\mathrm{MC}}\\) vs \\(\mathrm{Vega}\_{\mathrm{SC}}\\).** \\(\mathrm{Vega}\_{\mathrm{MC}}\\) is the multi-circuit prover described by this book: it folds many step circuits and one core circuit. \\(\mathrm{Vega}\_{\mathrm{SC}}\\) is the single-circuit prover that the MC construction builds on; unqualified "the prover" means \\(\mathrm{Vega}\_{\mathrm{MC}}\\). (see [Notation and conventions](../overview/notation.md))

**Witness.** The private assignment data committed by the prover. In standard R1CS it appears as \\(\mathbf{w}\\) in \\(\mathbf{z}=(\mathbf{w},1,\mathbf{x})\\); in split R1CS it is divided into shared, precommitted, and rest segments.

**Zero-knowledge opening (inner-product argument / IPA).** The Hyrax evaluation opening used for the final committed-witness evaluation. It proves an inner-product relation with fresh masks and blinds, sending `delta`, `beta`, `z_vec`, `z_delta`, and `z_beta` while hiding the row-combined witness vector. (see [Polynomial commitments and the ZK opening](../building-blocks/pcs.md))
