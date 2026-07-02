# Primer: multilinear extensions

This appendix gives the background on multilinear extensions used throughout the book. It treats multilinear extensions as a general finite-field tool; Vega-specific uses are described in [Multilinear polynomials](../building-blocks/multilinear.md).

## Functions on the Boolean hypercube

Fix a field \\(\mathbb{F}\\) and an integer \\(\ell \ge 0\\). A function on the Boolean hypercube is a map
\\[
f : \\{0,1\\}^\ell \to \mathbb{F}.
\\]
Its input is a bit-string \\(\mathbf{x} = (x\_0,\dots,x\_{\ell-1})\\). The book uses 0-based, least-significant-bit-first indexing: an integer \\(i \in \\{0,\dots,2^\ell-1\\}\\) corresponds to the bit-string satisfying
\\[
i = \sum\_{k=0}^{\ell-1} x\_k 2^k.
\\]
Thus a length-\\(2^\ell\\) vector \\(\mathbf{v}\\) defines a hypercube function by \\(f(\mathbf{x}) = v\_i\\) for this index \\(i\\).

An extension of \\(f\\) is a polynomial function
\\[
\tilde{f} : \mathbb{F}^\ell \to \mathbb{F}
\\]
that agrees with \\(f\\) at every Boolean input:
\\[
\tilde{f}(\mathbf{x}) = f(\mathbf{x}) \qquad \text{for all } \mathbf{x} \in \\{0,1\\}^\ell.
\\]
The extension is multilinear when each variable has degree at most one. Equivalently, after all other variables are fixed, the polynomial is affine in the remaining variable.

## Existence and uniqueness

Every hypercube function \\(f : \\{0,1\\}^\ell \to \mathbb{F}\\) has a unique multilinear extension. Existence follows from the Lagrange basis on the hypercube. For each Boolean point \\(\mathbf{x}\\), define the basis polynomial
\\[
L\_{\mathbf{x}}(\mathbf{r}) = \prod\_{k=0}^{\ell-1} \bigl(r\_k x\_k + (1-r\_k)(1-x\_k)\bigr).
\\]
Then \\(L\_{\mathbf{x}}\\) is multilinear, and for every Boolean \\(\mathbf{y} \in \\{0,1\\}^\ell\\),
\\[
L\_{\mathbf{x}}(\mathbf{y}) = \begin{cases} 1 & \text{if } \mathbf{x} = \mathbf{y},\\\\ 0 & \text{otherwise.} \end{cases}
\\]
The polynomial
\\[
\tilde{f}(\mathbf{r}) = \sum\_{\mathbf{x} \in \\{0,1\\}^\ell} L\_{\mathbf{x}}(\mathbf{r}) f(\mathbf{x})
\\]
therefore agrees with \\(f\\) on the hypercube and has degree at most one in each variable.

Uniqueness follows because the vector space of multilinear polynomials in \\(\ell\\) variables has basis
\\[
\prod\_{k \in S} r\_k \qquad \text{for } S \subseteq \\{0,\dots,\ell-1\\},
\\]
so its dimension is \\(2^\ell\\), the same as the number of hypercube evaluations. The Lagrange polynomials above give \\(2^\ell\\) multilinear polynomials whose evaluation table is the identity matrix. Hence the hypercube evaluations determine exactly one multilinear polynomial.

## Evaluation and eq representations

There are two standard representations of the same object.

The evaluation form stores the table
\\[
\bigl(f(\mathbf{x})\bigr)\_{\mathbf{x} \in \\{0,1\\}^\ell},
\\]
ordered by the LSB-first integer associated with \\(\mathbf{x}\\). This is the representation used when a vector is identified with its MLE.

The Lagrange, or eq, form expands the polynomial in the hypercube basis. The equality polynomial is
\\[
\widetilde{\mathrm{eq}}(\mathbf{r}, \mathbf{x}) = \prod\_{k=0}^{\ell-1}\bigl(r\_k x\_k + (1 - r\_k)(1 - x\_k)\bigr).
\\]
For Boolean \\(\mathbf{x}\\), this is the basis polynomial \\(L\_{\mathbf{x}}(\mathbf{r})\\). The MLE evaluation formula is
\\[
\tilde{f}(\mathbf{r}) = \sum\_{\mathbf{x} \in \\{0,1\\}^\ell} \widetilde{\mathrm{eq}}(\mathbf{r}, \mathbf{x})\\; f(\mathbf{x}).
\\]
This formula evaluates the extension at any point \\(\mathbf{r} \in \mathbb{F}^\ell\\), not only at Boolean points.

## Linear-time evaluation from a table

Given the full evaluation table of size \\(2^\ell\\), the formula above evaluates \\(\tilde{f}(\mathbf{r})\\) in \\(O(2^\ell)\\) field operations: compute the \\(2^\ell\\) eq weights for \\(\mathbf{r}\\), multiply each by the corresponding table entry, and sum.

The same computation has a useful partial-evaluation view. To fix the first variable to \\(r\_0\\), pair table entries whose indices differ only in bit \\(0\\). For each remaining Boolean suffix, replace the two values
\\[
a = f(0, x\_1,\dots,x\_{\ell-1}), \qquad b = f(1, x\_1,\dots,x\_{\ell-1})
\\]
by
\\[
(1-r\_0)a + r\_0 b.
\\]
The table size halves. Repeating this for \\(r\_1,r\_2,\dots,r\_{\ell-1}\\) leaves one field element, equal to \\(\tilde{f}(\mathbf{r})\\). This "fix one variable" operation is the basic table-halving step exploited by sum-check protocols.

## Worked example over \\(\mathbb{F}\_7\\)

Let \\(\ell=2\\), and work in the field \\(\mathbb{F}\_7\\). With LSB-first indexing, the table
\\[
\mathbf{v} = (3,5,6,1)
\\]
means
\\[
f(0,0)=3,\quad f(1,0)=5,\quad f(0,1)=6,\quad f(1,1)=1.
\\]
Evaluate the MLE at \\(\mathbf{r}=(2,3)\\).

The eq weights are
\\[
\begin{aligned} \widetilde{\mathrm{eq}}((2,3),(0,0)) &= (1-2)(1-3) = 2,\\\\ \widetilde{\mathrm{eq}}((2,3),(1,0)) &= 2(1-3) = 3,\\\\ \widetilde{\mathrm{eq}}((2,3),(0,1)) &= (1-2)3 = 4,\\\\ \widetilde{\mathrm{eq}}((2,3),(1,1)) &= 2\cdot 3 = 6, \end{aligned}
\\]
where all arithmetic is modulo \\(7\\). Their sum is \\(2+3+4+6=15=1\\), as expected for Lagrange weights.

Therefore
\\[
\tilde{f}(2,3) = 2\cdot 3 + 3\cdot 5 + 4\cdot 6 + 6\cdot 1 = 51 = 2 \pmod 7.
\\]

The table-halving evaluation gives the same answer. First fix \\(x\_0=2\\):
\\[
\begin{aligned} g(0) &= (1-2)f(0,0) + 2f(1,0) = (-1)3 + 2\cdot 5 = 0,\\\\ g(1) &= (1-2)f(0,1) + 2f(1,1) = (-1)6 + 2\cdot 1 = 3. \end{aligned}
\\]
Then fix \\(x\_1=3\\):
\\[
(1-3)g(0) + 3g(1) = (-2)0 + 3\cdot 3 = 2 \pmod 7.
\\]

## Why MLEs matter in proof systems

MLEs let protocols treat a vector \\(\mathbf{v} \in \mathbb{F}^{2^\ell}\\) as a low-degree object \\(\tilde{v} : \mathbb{F}^\ell \to \mathbb{F}\\). This changes statements about many coordinates into statements about one polynomial evaluated at a verifier-chosen point.

For example, suppose two vectors \\(\mathbf{a},\mathbf{b} \in \mathbb{F}^{2^\ell}\\) are claimed to be equal. They are equal exactly when their MLEs \\(\tilde{a}\\) and \\(\tilde{b}\\) are the same polynomial. If \\(\mathbf{a} \ne \mathbf{b}\\), then \\(\tilde{a}-\tilde{b}\\) is a nonzero polynomial of total degree at most \\(\ell\\). By the Schwartz--Zippel lemma, for uniformly random \\(\mathbf{r} \in \mathbb{F}^\ell\\),
\\[
\Pr\bigl[\tilde{a}(\mathbf{r}) = \tilde{b}(\mathbf{r})\bigr] \le \frac{\ell}{|\mathbb{F}|}.
\\]
Thus equality or consistency of long vectors can be checked by comparing MLE evaluations at random points, with soundness error controlled by the field size. Sum-check and polynomial-commitment protocols build on this reduction from vector relations to low-degree polynomial evaluations.
