# The proving lifecycle

This chapter follows the runtime path of the MC proof system. It describes what each public API phase consumes and produces, while the byte-level format is left to [Specification scope](../spec/scope.md).

## Phase 1: setup

`setup` consumes a step circuit \\(C\_1\\), a core circuit \\(C\_2\\), and `num_steps`. The number of steps is the number of step instances that the verifier key is bound to; the implementation requires at least two step instances and pads internally where folding needs a power of two.

Setup synthesizes split R1CS shapes for \\(C\_1\\) and \\(C\_2\\), equalizes their split layouts, derives commitment material, constructs the verifier-circuit shape, and computes the verifier-key digest. It produces a prover key and verifier key.

## Phase 2: prep_prove

`prep_prove` consumes the prover key, the concrete step circuits for one presentation batch, the core circuit, and the `is_small` optimization flag. It constructs the shared witness from the first step circuit, then constructs precommitted witness state for each step circuit and for the core circuit.

This is the first place where fold-and-reuse appears. The prepared state can be reused across presentations over the same signed data because the online proof phase rerandomizes the commitments before producing a proof. When the step shape has no challenge-dependent or rest variables, preparation also caches deterministic matrix-vector products for later folding.

## Phase 3: prove

`prove` consumes the prover key, the step circuits, the core circuit, and a prepared state. It rerandomizes the prepared state in place, so each presentation receives fresh hiding randomness while retaining the expensive precomputation.

The prover then creates split R1CS instances and witnesses for every step circuit and for the core circuit. It folds the step instances with NeutronNova-style folding, using transcript-derived \\(\tau\\) and \\(\rho\\) challenges. It runs the outer and inner sum-checks for the folded step branch and the core branch, and it feeds the verifier's algebraic checks into the MC verifier circuit.

For zero knowledge, the prover samples a fresh random satisfying relaxed instance for the verifier circuit and folds the real verifier-circuit instance with it. The prover proves satisfaction of the resulting relaxed instance and produces a zero-knowledge Hyrax/IPA opening for the committed witness evaluation needed by the checks.

The output proof contains the shared witness commitment, per-step split instances, a core instance, the verifier-circuit instance, the random relaxed instance, the folding proof for the verifier-circuit instance, the relaxed R1CS proof, and the Hyrax evaluation argument. The consumed prepared state is returned so the caller can reuse it.

## Phase 4: verify

`verify` consumes the proof, the verifier key, and the expected number of step instances. It rejects if the proof's step count is zero, differs from the supplied count, or differs from the count bound into the verifier key.

The verifier restores the shared commitment into the step and core instances, validates those instances against the verifier key, rebuilds the Fiat--Shamir transcript, recomputes \\(\tau\\), \\(\rho\\), and later sum-check challenges, and folds the step instances. It verifies the folded verifier-circuit proof, recomputes the matrix and public-value checks, and verifies the Hyrax evaluation argument.

If all checks pass, verification returns the per-step public values and the core public values. These values are the public statement exposed by the proof-system layer; application meaning is assigned by the circuits supplied to setup and proving.
