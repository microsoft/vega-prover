# The in-circuit verifier

The in-circuit verifier is the R1CS object that turns Vega's verifier algebra into a committed witness. Instead of asking the outer verifier to execute the full folding and sum-check computation natively, the prover supplies a multi-round R1CS instance whose constraints check that computation. The outer verifier then checks one folded relaxed-R1CS instance and one polynomial-commitment opening, while still recomputing a small set of public values outside the circuit.

## The verifier as an R1CS circuit

The MC proof system uses `VegaMcVerifierCircuit` as its verifier circuit. It is a multi-round circuit: each round produces witness variables, commits to that round's witness, and then receives the Fiat--Shamir challenges for later rounds. This is the setting described by `SplitMultiRoundR1CS` in [R1CS and its variants](r1cs.md). The single-circuit sibling, `VegaVerifierCircuit`, has the same role for a single verifier computation; the MC path uses `VegaMcVerifierCircuit` because it must also account for folding the step instances.

The MC circuit has three algebraic regions. First, it checks the folding sum-check over the step batch. Second, it checks the outer sum-checks for the folded step branch and the core branch. Third, it checks the inner sum-checks for the same two branches. These regions are not separate proofs inside the book's proof object; they are rounds of one multi-round R1CS instance. Exact transcript labels and byte layout belong to [The transcript schedule](../spec/transcript-schedule.md) and [The proof object](../spec/proof-object.md).

If the step batch is padded to \\(2^{\ell\_b}\\), the step constraint system has \\(2^{\ell\_x}\\) constraints, and the combined witness/public vector has size \\(2^{\ell\_y-1}\\) before the extra selector coordinate used by the inner check, then the verifier-circuit challenge rounds have the shape

\\[
\ell\_b \\;+ \ell\_x \\;+ 1 \\;+ \ell\_y .
\\]

The first \\(\ell\_b\\) challenges are the folding/sum-check challenges \\(r\_b\\). The next \\(\ell\_x\\) challenges are the outer sum-check challenges \\(r\_x\\). The single bridge challenge batches the three matrix claims. The final \\(\ell\_y\\) challenges are the inner sum-check challenges \\(r\_y\\).

## Sum-check consistency inside the circuit

Each sum-check round sends a univariate polynomial. Inside the circuit, `enforce_sc_claim` checks the local sum-check relation: the polynomial's evaluations at \\(0\\) and \\(1\\) add to the previous claim. With coefficient vectors, this is enforced as the sum of all coefficients plus the constant coefficient equaling the current claim. `eval_poly_horner` evaluates the previous round polynomial at the next challenge, and `alloc_coeffs` allocates the round polynomial coefficients as witness variables.

The folding region ends by enforcing that the final folding claim equals the accumulated equality weight times the folded step target \\(t\_{\mathrm{step}}\\) — the evaluation/error term the NeutronNova fold carries for the step batch. Conceptually, the circuit checks

\\[
\widetilde{\mathrm{eq}}(r\_b, \rho)\\, t\_{\mathrm{step}} = q\_b,
\\]

where \\(q\_b\\) is the final claim obtained from the folding-round polynomials. The equality value \\(\widetilde{\mathrm{eq}}(r\_b,\rho)\\) is later exposed as a public value so the native verifier can compare it with its own transcript-derived computation.

For each of the step and core branches, `enforce_outer_sc_final_check` checks the final outer zero-check relation. In notation matching the R1CS identity \\((A\mathbf{z}) \circ (B\mathbf{z}) = C\mathbf{z}\\), the checked relation is

\\[
q\_x = \widetilde{\mathrm{eq}}(\tau, r\_x)\\, \bigl( \widetilde{A\mathbf{z}}(r\_x)\\, \widetilde{B\mathbf{z}}(r\_x) - \widetilde{C\mathbf{z}}(r\_x) \bigr).
\\]

Here \\(q\_x\\) is the final outer sum-check claim, and \\(\widetilde{\mathrm{eq}}(\tau,r\_x)\\) is represented in the implementation by the public value `tau_at_rx`. The MC circuit applies this relation once to the folded step branch and once to the core branch.

The bridge into the inner sum-check uses `compute_joint_claim`. Given the three final matrix-vector claims, it forms the batched claim

\\[
q\_{ABC} = q\_A + r q\_B + r^2 q\_C,
\\]

where \\(r\\) is the bridge challenge. This is the claim that the inner sum-check reduces to matrix evaluations.

Finally, `enforce_inner_sc_final_check` checks the inner final relation. The first inner coordinate selects between the witness part and the public-input part of \\(\mathbf{z}\\):

\\[
z(r\_y) = (1-r\_{y,0})\\,W(r\_y') + r\_{y,0}\\,X(r\_y'),
\\]

where \\(r\_y'\\) is the remaining suffix of \\(r\_y\\). Let \\(q\_y\\) be the terminal inner sum-check claim, and let

\\[
Q = \widetilde{A}(r\_x,r\_y) + r\\,\widetilde{B}(r\_x,r\_y) + r^2\\,\widetilde{C}(r\_x,r\_y)
\\]

be the combined matrix evaluation at the inner point. The terminal claim factors as \\(q\_y = Q\\, z(r\_y)\\), so the circuit enforces the quotient relation

\\[
Q\\, z(r\_y) = q\_y,
\\]

exposing \\(Q = q\_y / z(r\_y)\\) as a public value. This \\(Q\\) is the terminal matrix evaluation, not the initial batched claim \\(q\_{ABC}\\) formed above. The native verifier independently recomputes \\(Q = \widetilde{A}+r\widetilde{B}+r^2\widetilde{C}\\) at \\((r\_x,r\_y)\\) for the folded step branch and the core branch and compares both values.

## Public values that close the check

The MC verifier circuit exposes six public values:

| Public value | Native meaning |
| --- | --- |
| `tau_at_rx` | the outer equality/powers evaluation at \\(r\_x\\) |
| `eval_X_step` | the folded step public-input polynomial evaluated at \\(r\_y'\\) |
| `eval_X_core` | the core public-input polynomial evaluated at \\(r\_y'\\) |
| `eq_rho_at_rb` | \\(\widetilde{\mathrm{eq}}(r\_b,\rho)\\) for the folding challenges |
| `quotient_step` | the native step matrix quotient \\(A+rB+r^2C\\) evaluated at the inner point |
| `quotient_core` | the native core matrix quotient \\(A+rB+r^2C\\) evaluated at the inner point |

These public values are deliberately small. The circuit constrains the transcript-dependent algebra, but it does not by itself recompute every sparse matrix evaluation or every public-input evaluation natively. The outer verifier recomputes these six values from the verifier key, public instances, and transcript challenges, then requires equality with the verifier-circuit public input. This native comparison closes the quotient relations that would otherwise be under-constrained by the circuit alone. The full verification path is described in [MC verification](../mc/verify.md).

## Fiat--Shamir binding across rounds

The verifier-circuit challenges are not arbitrary witness values. During validation of a `SplitMultiRoundR1CSInstance`, the verifier processes the committed rounds in order. For each round, it checks that the round commitment has the expected size, absorbs the commitment into the running transcript, squeezes that round's challenges, and compares the derived challenges with the challenges stored in the instance. This makes the in-circuit challenges equal to the transcript challenges without placing the transcript hash function inside the R1CS circuit. The transcript abstraction is introduced in [The Fiat--Shamir transcript](transcript.md).

## Why the circuit enables zero knowledge

Once the verifier's algebra has become one committed R1CS instance, Vega can apply the same folding machinery used elsewhere. The prover samples a fresh random satisfying relaxed instance for the verifier-circuit shape and folds it with the real verifier-circuit instance. The proof then opens the folded committed witness at the point needed by the verifier checks, rather than exposing the original verifier-circuit witness directly.

This structural move is what keeps the native verifier small: it checks satisfaction of the folded relaxed-R1CS instance, verifies the single polynomial-commitment opening, and compares the six public values described above. The relevant background is [NeutronNova folding](neutronnova.md) and [Nova folding for zero-knowledge](nova-zk.md), and the \\(\mathrm{Vega}\_{\mathrm{MC}}\\) zero-knowledge flow is described in [Zero knowledge](../mc/zero-knowledge.md). For the sum-check equations used by the circuit, see [The sum-check protocol](sumcheck.md).
