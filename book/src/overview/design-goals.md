# Design goals and threat model

This chapter states the goals Vega optimizes for and the security properties expected from the multi-circuit proof system. It also separates the proof system from application-specific credential circuits.

## Design goals

### Low proving latency on client devices

Vega is designed for a prover running on a client device. The repository targets statements over signed data and optimizes for low zero-knowledge proving latency rather than batch throughput. \\(\mathrm{Vega}\_{\mathrm{MC}}\\) supports this goal by moving reusable work out of the online presentation path and by folding many uniform step instances before proving the verifier's algebraic checks.

### Repeated presentations over the same signed data

A credential holder may present statements about the same signed data many times. Vega exposes a `prep_prove` phase that builds precommitted state for the step and core circuits. Later proofs rerandomize that state, so the prover reuses expensive preparation while producing fresh zero-knowledge proofs.

### No trusted setup

The canonical implementation uses an engine whose polynomial commitment is Hyrax and whose transcript is Fiat--Shamir over Keccak256. The setup phase derives commitment material and circuit shapes from public parameters and the circuits; it does not rely on a toxic-waste setup ceremony.

### Transparent, standard assumptions

Vega's proof-system checks are expressed through R1CS, sum-check, folding, and an inner-product opening for homomorphic commitments. Security is reduced to the soundness of these algebraic protocols, the binding and hiding properties of the commitments, and the random-oracle-style Fiat--Shamir transform used to derive challenges.

### Zero knowledge

A proof should reveal only that the public statement is true. Vega combines hiding commitments, a zero-knowledge opening argument, rerandomized precommitted state, and a final fold with a fresh random satisfying verifier-circuit instance. The random instance is sampled per proof, not once per setup.

## Threat model and properties

### What the proof establishes

Completeness means an honest prover with satisfying witnesses for the step circuits and the core circuit produces a proof accepted by the verifier. Soundness means a prover cannot convince the verifier unless the folded R1CS claims, the in-circuit verifier checks, and the committed witness openings are mutually consistent for the public statement encoded by the circuits.

The verifier learns the public input/output values returned by verification. For \\(\mathrm{Vega}\_{\mathrm{MC}}\\), these include per-step public values and core public values.

### What remains hidden

Zero knowledge hides private witnesses and credential contents that are not part of the public statement. The verifier sees commitments, folded instances, algebraic proof messages, and public values, but the hiding commitments and fresh per-proof randomness prevent these objects from identifying the underlying witness beyond the statement being proven.

### What the verifier must trust

The verifier must use the intended verifier key. The verifier key contains the commitment verifier key, the step and core R1CS shapes, the verifier-circuit shape and key, and the number of step instances. The transcript is bound to a digest of this verifier key, so proofs are tied to the circuit shapes and parameters represented by that key.

### What is out of scope

This book specifies the proof system. It does not specify how a credential format is parsed, how signatures are represented inside an application circuit, or which lookup-centric arithmetization is best for extracting values from credential bytes. Those choices determine the step and core circuits supplied to Vega; the proof system treats them as R1CS-producing circuits.

For the static structure of these components, continue to [System architecture](architecture.md). For the runtime flow, continue to [The proving lifecycle](lifecycle.md).
