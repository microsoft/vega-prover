# NeutronNova folding

This chapter describes the first fold in \\(\mathrm{Vega}\_{\mathrm{MC}}\\): accumulation of the many uniform step-circuit R1CS instances into one folded step instance before that instance is proved by [The Spartan argument](spartan.md).

## Inputs to the step fold

The step instances all use the same [R1CS](r1cs.md) shape. Each input starts in strict form, so its assignment \\(\mathbf{z}\\) must satisfy

\\[ (A\mathbf{z})\circ(B\mathbf{z}) = C\mathbf{z} \\]

with relaxation scalar \\(u=1\\) and error \\(\mathbf{E}=\mathbf{0}\\). The prover evaluates the three matrix-vector products for every padded step instance. For row \\(i\\), the local residual is

\\[ (A\mathbf{z})\_i (B\mathbf{z})\_i - (C\mathbf{z})\_i. \\]

A satisfied strict instance has zero residual in every row. The fold checks a random compression of these residuals rather than carrying every row of every step instance forward.

## Two independent randomizers

The fold uses two layers of randomization. First, the transcript supplies \\(\tau\\), which combines the constraint coordinates inside one step instance. This turns the residual vector into one scalar target. If any row residual is nonzero, the \\(\tau\\)-weighted scalar is nonzero except with Schwartz--Zippel probability over \\(\mathbb{F}\\).

Second, the transcript supplies one \\(\rho\\) value for each batch-folding round. These values combine the padded list of step instances. If at least one instance has a nonzero \\(\tau\\)-compressed residual, the \\(\rho\\)-layer catches it except with another Schwartz--Zippel error. The byte-level labels and ordering for deriving \\(\tau\\), the \\(\rho\\) values, and the later verifier-circuit challenges are fixed in [the transcript schedule](../spec/transcript-schedule.md), not in this chapter.

## Round polynomials and verifier-circuit state

For a padded batch of \\(2^{\ell\_b}\\) step instances, the fold runs \\(\ell\_b\\) rounds. In each round the prover constructs a cubic univariate polynomial for the current batch axis and writes its four coefficients into the [in-circuit verifier](in-circuit-verifier.md) state. Processing that verifier-circuit round commits the round witness and returns one challenge; the sequence of these challenges is \\(r\_b\\).

After a round challenge is known, the prover folds the adjacent layers of the precomputed \\(A\mathbf{z}\\), \\(B\mathbf{z}\\), and when needed \\(C\mathbf{z}\\) tables by the affine rule determined by that challenge. The next round operates on half as many batch entries. The verifier-circuit state checks that the cubic-polynomial messages maintain the folded claim, in the same folding style explained in the [folding primer](../appendix/folding-primer.md).

## The final step error value

At the end of the \\(r\_b\\) rounds, the prover has one accumulated target. The target is normalized by the equality-polynomial value at the point selected by \\(r\_b\\) and the transcript-derived \\(\rho\\) values. The normalized scalar is stored in the verifier circuit as `t_out_step`, while the equality value is stored as `eq_rho_at_rb`.

The in-circuit verifier later pins this scalar to the folding transcript by enforcing

\\[ \mathrm{eq\_rho\_at\_rb}\\,\mathrm{t\_out\_step} = \mathrm{claim}. \\]

This constraint prevents the prover from choosing an arbitrary folded error after the verifier-circuit challenges are fixed: the claimed folding result must equal the equality-weighted normalized output.

## Folding the committed step instance

The same \\(r\_b\\)-derived weights used by the polynomial fold also combine the concrete step witnesses and instances. `fold_multiple` forms linear combinations of witness entries, public inputs, witness commitments, and commitment blinds under those shared weights. The result is one ordinary step witness/instance pair with the same dimensions as a single step instance.

That folded step instance is then batched with the core branch and proved by [The Spartan argument](spartan.md). The overall \\(\mathrm{Vega}\_{\mathrm{MC}}\\) proving flow is described in [Proving](../mc/prove.md). This chapter only describes the first fold; the second fold, which masks the verifier-circuit instance for zero knowledge, is described in [Nova folding for zero-knowledge](nova-zk.md).
