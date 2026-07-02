# The sum-check protocol

This chapter describes the concrete sum-check building block used by Vega. The general interactive protocol is reviewed in the [sum-check primer](../appendix/sumcheck-primer.md); here the focus is the generic shape of the round polynomials, the challenges derived by Fiat--Shamir, and the terminal claim returned to an enclosing protocol such as [The Spartan argument](spartan.md).

## What the proof object contains

A sum-check proof is a sequence of univariate round polynomials, one per round. In each round, the prover has a current claim \\(C\_{i-1}\\) about a remaining hypercube sum. It sends a low-degree polynomial \\(g\_i(X)\\), the verifier checks that
\\[
g\_i(0) + g\_i(1) = C\_{i-1},
\\]
derives the next challenge from the Fiat--Shamir transcript, and updates the claim to \\(C\_i = g\_i(r\_i)\\).

Vega stores each round polynomial in compressed form. The compression omits the linear coefficient. During verification, the verifier reconstructs that coefficient from the current claim, because the consistency equation determines it. For a quadratic round polynomial, the stored data contains the constant and quadratic coefficients; for a cubic round polynomial, it contains the constant, quadratic, and cubic coefficients. After decompression, the verifier absorbs the reconstructed round polynomial into the transcript, squeezes the next challenge, and evaluates the polynomial at that challenge.

This re-derivation of challenges is the point where the interactive protocol is bound to Fiat--Shamir. A prover cannot choose later round polynomials independently of earlier messages, because every challenge is recomputed from the transcript state that includes the preceding round polynomial. The byte-level transcript labels and serialization of these messages are fixed in [The transcript schedule](../spec/transcript-schedule.md), not in this pedagogical chapter.

The low-level sum-check verifier checks the number of rounds, checks the advertised degree, reconstructs each round polynomial so that the per-round consistency equation holds, and returns the final claim together with the challenge vector. The enclosing verifier then checks that terminal claim against the appropriate final evaluation for the particular sum-check instance.

## Specialized prover shapes

Vega implements specialized sum-check provers by the degree and algebraic shape of the round polynomial they produce.

- The quadratic prover handles sums whose round polynomial comes from multiplying two multilinear tables. It constructs a degree-2 polynomial from three evaluations, compresses it, derives the round challenge from the transcript, and binds both tables to that challenge.
- The cubic-with-additive-term prover handles sums with a product term plus an additive term, optionally multiplied by another multilinear helper factor. It constructs a degree-3 polynomial from four evaluations, compresses it, derives the challenge, and binds the participating tables to that challenge.
- Batched zero-knowledge variants run the same degree shape for two branches under one transcript schedule. \\(\mathrm{Vega}\_{\mathrm{MC}}\\) uses these variants when the folded step branch and core branch must share the same round challenges.
- Equality-polynomial and powers-polynomial helpers carry an extra multilinear factor through the rounds. This helper factor is what turns a product of two multilinear tables into a cubic round shape in applications such as [The Spartan argument](spartan.md).

The names \\(\tau\\), \\(r\_x\\), and \\(r\_y\\) are used by Spartan's row batching and sum-check reductions. The challenges \\(\rho\\) and \\(r\_b\\) belong to the folding layer, where step instances are accumulated before the Spartan checks; see [NeutronNova folding](neutronnova.md).

## Payoff

The generic sum-check verifier does not evaluate a full hypercube sum. It checks a short sequence of low-degree univariate polynomial messages and returns a random evaluation point plus a terminal claim. The enclosing protocol supplies the meaning of that terminal claim and checks it against the relevant final evaluation.

For the R1CS application, [The Spartan argument](spartan.md) uses an outer sum-check to obtain \\(r\_x\\), an inner matrix-evaluation sum-check to obtain \\(r\_y\\), and a commitment opening for the final witness value. For background on the polynomial objects, see [Multilinear polynomials](multilinear.md) and the [sum-check primer](../appendix/sumcheck-primer.md). The \\(\mathrm{Vega}\_{\mathrm{MC}}\\) proving flow is described in [Proving](../mc/prove.md), and byte-exact transcript behavior is specified in [The transcript schedule](../spec/transcript-schedule.md).
