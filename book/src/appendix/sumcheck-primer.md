# Primer: the sum-check protocol

This appendix gives a self-contained account of the sum-check protocol as a
general interactive proof. It is background for Vega's concrete sum-check
building block, not a description of the protocol instances used by the
implementation.

## The sum being checked

Let \\(g(X\_1,\dots,X\_\ell) \in \mathbb{F}[X\_1,\dots,X\_\ell]\\) be a polynomial
whose degree in each variable is bounded. The prover claims that
\\[
S \\;=\\; \sum\_{\mathbf{x} \in \\{0,1\\}^\ell} g(\mathbf{x}).
\\]
The verifier wants to check this claim without evaluating \\(g\\) on all
\\(2^\ell\\) Boolean points. The protocol reduces the exponential-size sum to one
evaluation of \\(g\\) at a random point
\\((r\_1,\dots,r\_\ell) \in \mathbb{F}^\ell\\).

This setting is common when \\(g\\) is built from
[multilinear extensions](../appendix/mle-primer.md). The sum-check protocol does
not require \\(g\\) itself to be multilinear; it only needs a known degree bound
for each univariate round polynomial.

## Round polynomials

The protocol maintains a current claim \\(C\_{i-1}\\). Initially \\(C\_0 = S\\). In
round \\(i \in \\{1,\dots,\ell\\}\\), the variables
\\(X\_1,\dots,X\_{i-1}\\) have already been bound to verifier challenges
\\(r\_1,\dots,r\_{i-1}\\). The prover sends the univariate polynomial
\\[
g\_i(X) \\;=\\; \sum\_{(x\_{i+1},\dots,x\_\ell) \in \\{0,1\\}^{\ell-i}} g(r\_1,\dots,r\_{i-1}, X, x\_{i+1},\dots,x\_\ell).
\\]
For \\(i=\ell\\), the sum is over the empty set, so
\\[
g\_\ell(X) \\;=\\; g(r\_1,\dots,r\_{\ell-1},X).
\\]
The degree of \\(g\_i\\) is at most the degree of \\(g\\) in variable \\(X\_i\\), because
the other variables are either fixed or summed over Boolean values.

The verifier checks the consistency equation
\\[
g\_i(0) + g\_i(1) \\;=\\; C\_{i-1}.
\\]
If it fails, the verifier rejects. If it holds, the verifier samples a fresh
random challenge \\(r\_i \in \mathbb{F}\\) and updates the claim to
\\[
C\_i \\;=\\; g\_i(r\_i).
\\]
After \\(\ell\\) rounds, the claim is \\(C\_\ell\\), a claimed value for
\\[
g(r\_1,\dots,r\_\ell).
\\]

## Final check

The verifier makes one oracle evaluation of \\(g\\) at the random point and accepts
exactly when
\\[
g(r\_1,\dots,r\_\ell) \\;=\\; C\_\ell.
\\]
This final evaluation is the only place where the verifier needs access to
\\(g\\) away from the Boolean hypercube. In deployed proof systems, the verifier
often obtains this value through a polynomial commitment opening rather than by
holding the whole polynomial.

## Completeness and cost

Completeness is immediate from the definition of each round polynomial. If the
initial claim \\(S\\) is correct and the prover sends the honest \\(g\_i\\), then
\\[
g\_i(0) + g\_i(1)
\\]
is exactly the previous sum with \\(X\_i\\) left unbound, split into the two cases
\\(X\_i=0\\) and \\(X\_i=1\\). The final value \\(g\_\ell(r\_\ell)\\) is exactly
\\(g(r\_1,\dots,r\_\ell)\\).

The prover usually does work proportional to the table of values being summed,
while the verifier checks only \\(\ell\\) low-degree univariate polynomials and
one final evaluation. If every round polynomial has degree at most \\(d\\), the
verifier's round work is \\(O(\ell \cdot d)\\), plus the cost of the final oracle
evaluation or commitment opening.

## Soundness intuition

Suppose the claimed sum is false. A cheating prover may send a first polynomial
\\(h\_1\\) that passes
\\[
h\_1(0) + h\_1(1) = S
\\]
even though \\(h\_1\\) is not the honest round polynomial. Once the verifier sends
a random \\(r\_1\\), the next claim becomes \\(h\_1(r\_1)\\). For the prover to stay
consistent with the true restricted sum, the false polynomial must agree with
the honest polynomial at the sampled point.

By the Schwartz--Zippel lemma, two distinct univariate polynomials of degree at
most \\(d\\) agree at a random field point with probability at most \\(d/|\mathbb{F}|\\).
Across \\(\ell\\) rounds, a union bound gives cheating probability at most
\\[
\frac{\ell d}{|\mathbb{F}|}
\\]
when every round has degree at most \\(d\\). More generally, the bound is
\\((d\_1+\cdots+d\_\ell)/|\mathbb{F}|\\), where \\(d\_i\\) is the degree bound in
round \\(i\\).

## A two-round example

Consider the Boolean predicate "exactly one bit is \\(1\\)" on two variables. Its
indicator polynomial over \\(\mathbb{F}\\) is
\\[
g(X\_1,X\_2) \\;=\\; (1-X\_1)X\_2 + X\_1(1-X\_2) \\;=\\; X\_1 + X\_2 - 2X\_1X\_2.
\\]
Over \\(\\{0,1\\}^2\\), it is \\(1\\) on \\((1,0)\\) and \\((0,1)\\), and \\(0\\) on the
other two points. The claimed sum is therefore \\(S=2\\).

In round 1, the prover sends
\\[
g\_1(X) = g(X,0) + g(X,1) = X + (1-X) = 1.
\\]
The verifier checks
\\[
g\_1(0)+g\_1(1)=1+1=2=S
\\]
and samples \\(r\_1\\). The updated claim is \\(C\_1=g\_1(r\_1)=1\\).

In round 2, the prover sends
\\[
g\_2(X) = g(r\_1,X) = r\_1 + X - 2r\_1X = r\_1 + (1-2r\_1)X.
\\]
The verifier checks
\\[
g\_2(0)+g\_2(1)=r\_1 + (1-r\_1)=1=C\_1
\\]
and samples \\(r\_2\\). The updated claim is
\\[
C\_2 = g\_2(r\_2)=r\_1+r\_2-2r\_1r\_2.
\\]
The final oracle evaluation is
\\[
g(r\_1,r\_2)=r\_1+r\_2-2r\_1r\_2,
\\]
which equals \\(C\_2\\), so the verifier accepts.

## Sending the round polynomial

A round polynomial can be transmitted by coefficients or by enough evaluations
to interpolate it. For a degree-\\(d\\) polynomial, \\(d+1\\) field elements are
sufficient. Implementations may compress the message further when the verifier's
consistency check determines one coefficient from the previous claim.

The interactive verifier's random challenges can be made non-interactive with
Fiat--Shamir: the prover absorbs each round polynomial into the transcript, then
derives \\(r\_i\\) by hashing the transcript. Vega's exact transcript schedule is
specified in [The transcript schedule](../spec/transcript-schedule.md).

Vega's concrete sum-check construction is described in
[The sum-check protocol](../building-blocks/sumcheck.md). Its use inside folding
is described in [NeutronNova folding](../building-blocks/neutronnova.md).
