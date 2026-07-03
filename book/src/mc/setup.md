# Setup

This chapter describes the setup phase for \\(\mathrm{Vega}\_{\mathrm{MC}}\\). Setup fixes the step shape, the core shape, the commitment material, the verifier-circuit shape, and the verifier-key digest that later binds the Fiat--Shamir transcript to those choices.

## Inputs and invariant

`setup` consumes a step circuit \\(C\_1\\), a core circuit \\(C\_2\\), and `num_steps`. It returns a prover key and a verifier key. The verifier key is tied to the supplied number of step instances, so proofs for a different step count are outside that key's statement.

`num_steps` must be at least two. The step branch is folded as a batch by the [NeutronNova building block](../building-blocks/neutronnova.md), and that batch has at least two step instances. A single instance belongs to the single-circuit prover \\(\mathrm{Vega}\_{\mathrm{SC}}\\) rather than this multi-circuit path. Where folding needs a power-of-two batch size, the implementation pads the step count internally before deriving round counts.

## Split R1CS shapes

Setup synthesizes a split R1CS shape for the step circuit and another split R1CS shape for the core circuit. A split shape separates the witness columns into shared, precommitted, and rest segments, while retaining the R1CS matrices and public-value layout described in [R1CS and split shapes](../building-blocks/r1cs.md).

After synthesis, setup equalizes the two split shapes. Equalization aligns their split layouts so the step and core branches use compatible segment boundaries under one commitment scheme. The resulting shapes are stored as `S_step` and `S_core` in both keys.

## Commitment material

Setup builds one shared commitment key `ck` sized for both split shapes and derives the PCS verifier key `vk_ee`. The same `ck` is used for the step and core witness commitments. The PCS precompute step is then applied to `ck`, so later commitments can use the prepared commitment material described in [Polynomial commitments](../building-blocks/pcs.md).

The reference implementation sets `DEFAULT_COMMITMENT_WIDTH` to 2048. This width is the commitment-row width used when split witness segments are padded and when the shared commitment key is generated.

## Verifier circuit

Setup also constructs the verifier circuit used later inside the proof. Its shape depends on the public setup data:

- `num_rounds_b` is the base-two logarithm of the padded step count `num_steps.next_power_of_two()`;
- `num_rounds_x` is the base-two logarithm of the number of step constraints;
- `num_rounds_y` is one more than the base-two logarithm of the total number of step witness variables.

Those dimensions parameterize the in-circuit verifier described in [The in-circuit verifier](../building-blocks/in-circuit-verifier.md). Shape synthesis for that verifier circuit produces `vc_shape`, `vc_ck`, and `vc_vk`; setup also derives `vc_shape_regular` from `vc_shape`. The verifier-circuit commitment key is precomputed before the keys are returned.

## Verifier-key digest

The verifier-key digest is the protocol binding for setup. It is computed from the verifier key with `bincode::DefaultOptions::new().with_little_endian().with_fixint_encoding()` except for the two split R1CS shapes, which use their raw shape byte writer. The digest input order is:

1. `ck`, serialized with the bincode options above;
2. `vk_ee`, serialized with the same bincode options;
3. raw shape bytes for `S_step`;
4. raw shape bytes for `S_core`;
5. `vc_shape`, serialized with the same bincode options;
6. `vc_shape_regular`, serialized with the same bincode options;
7. `vc_ck`, serialized with the same bincode options;
8. `vc_vk`, serialized with the same bincode options;
9. `num_steps`, serialized with the same bincode options.

The digest is memoized in the verifier key's `OnceCell`. The prover key stores the same digest as `vk_digest`. During proving and verification, the digest is absorbed into the Keccak Fiat--Shamir transcript before instance validation and before the folding transcript, so the transcript is bound to the fixed setup material. The transcript mechanism is described in [Fiat--Shamir transcript](../building-blocks/transcript.md). The exact byte layout of the verifier key belongs to [Verifier key](../spec/verifier-key.md) and [Serialization](../spec/serialization.md).

## Key contents

| Field | Key | Meaning |
|---|---|---|
| `ck` | prover, verifier | Shared PCS commitment key for the step and core split witnesses. |
| `vk_ee` | verifier | PCS verifier key paired with `ck`. |
| `S_step` | prover, verifier | Equalized split R1CS shape for each step instance. |
| `S_core` | prover, verifier | Equalized split R1CS shape for the core instance. |
| `vk_digest` | prover | Digest of the verifier key, used as the transcript binding. |
| `vc_shape` | prover, verifier | Split multi-round R1CS shape of the verifier circuit. |
| `vc_shape_regular` | prover, verifier | Regular R1CS view of the verifier-circuit shape. |
| `vc_ck` | prover, verifier | PCS commitment key for the verifier-circuit witness. |
| `vc_vk` | verifier | Verifier-circuit verification key. |
| `num_steps` | verifier | Step count bound into this verifier key. |
| `digest` | verifier | Lazily computed, serde-skipped cache of the verifier-key digest. |

Setup hands these fixed objects to [rerandomizable precomputation](./prep.md), which fills in reusable committed witness state for concrete step and core circuits.
