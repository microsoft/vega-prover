# System architecture

This chapter describes the static components of \\(\mathrm{Vega}\_{\mathrm{MC}}\\) and how data moves between them. Byte encodings and transcript order are deferred to the specification chapters.

## Component diagram

The MC proof system is organized as the following components.

- **Engine.** The engine fixes the scalar field \\(\mathbb{F}\\), group \\(\mathbb{G}\\), Fiat--Shamir transcript, and polynomial commitment scheme. The canonical instantiation is `T256HyraxEngine`: T256 group, T256 scalar field, Keccak256 transcript, and Hyrax commitments. See [Fields, groups, and the engine](../building-blocks/fields-and-groups.md) and [The Fiat--Shamir transcript](../building-blocks/transcript.md).
- **Step circuit and core circuit.** The application supplies a uniform step circuit \\(C\_1\\), a core circuit \\(C\_2\\), and a number of steps. Setup synthesizes both into split R1CS shapes. The step shape is instantiated many times; the core shape connects the batch. See [R1CS and its variants](../building-blocks/r1cs.md).
- **Sum-check.** Sum-check reduces large multilinear claims about R1CS satisfaction to evaluations at verifier challenges. Vega uses a constraint-batching challenge \\(\tau\\), folding challenges \\(\rho\\), and round challenges such as \\(r\_x\\) and \\(r\_y\\). See [The sum-check protocol](../building-blocks/sumcheck.md).
- **NeutronNova folding.** Folding compresses many regular step R1CS instances into one folded instance and witness. The verifier later recomputes the same folding challenges from the transcript. See [NeutronNova folding](../building-blocks/neutronnova.md).
- **In-circuit verifier.** A second R1CS circuit checks the algebraic work performed by folding and sum-check. Its public input records the challenges and selected final values that the outer verifier recomputes. See [The in-circuit verifier](../building-blocks/in-circuit-verifier.md).
- **Commitment and zero-knowledge opening.** Hyrax commitments bind the witness vectors and support homomorphic folding. A zero-knowledge evaluation argument opens the folded witness commitment at the point required by the verifier. See [Polynomial commitments and the ZK opening](../building-blocks/pcs.md).

## Data flow

Setup starts from \\(C\_1\\), \\(C\_2\\), and `num_steps`. It produces a prover key containing the split R1CS shapes, commitment key, verifier-circuit shape, and digest of the verifier key. It also produces a verifier key containing the corresponding verifier material and the number of step instances.

Preparation evaluates the shared and precommitted parts of the step and core witnesses. The resulting precommitted state is reusable because the online proof phase rerandomizes it before deriving proof objects.

Proof generation turns each prepared step circuit and the core circuit into split R1CS instances. The step instances share one commitment to the shared witness. The prover converts the split instances to regular instances for folding, folds the step batch, runs the outer and inner sum-checks for the folded step branch and the core branch, and records the verifier's algebraic checks inside the verifier circuit.

After the verifier-circuit instance is built, the prover samples a fresh random satisfying relaxed instance and folds it with the real verifier-circuit instance. A single relaxed R1CS proof establishes satisfaction of the folded instance. A Hyrax/IPA evaluation argument opens the folded committed witness value used by the verifier checks.

Verification reconstructs the shared commitment, validates all public instances against the verifier key, recomputes transcript challenges, folds the step instances, verifies the folded verifier-circuit proof, recomputes the matrix-evaluation checks, and verifies the Hyrax opening. If all checks pass, it returns the step and core public values.

The runtime phases are described next in [The proving lifecycle](lifecycle.md). Byte-level proof layout begins in [Specification scope](../spec/scope.md).
