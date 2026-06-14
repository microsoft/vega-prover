// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Small-value sumcheck implementation for the first ℓ₀ rounds of Spartan's
//! outer cubic sum-check.
//!
//! This module provides optimized sumcheck proving using native integer
//! arithmetic for polynomials with small coefficients. The key optimization
//! is replacing expensive field multiplications with native integer operations
//! during the initial rounds when polynomial values are guaranteed to be small.
//!
//! This implementation is not a generic small-value prover for arbitrary
//! `A(X) * B(X) - C(X)` relations. It relies on the Spartan outer-sumcheck
//! structure:
//! - `A(x) * B(x) - C(x) = 0` on the Boolean hypercube for satisfying witnesses
//! - contributions with an `∞` prefix coordinate only need the highest-degree
//!   term, so the linear `C` term drops out there
//!
//! Based on "Speeding Up Sum-Check Proving" by Suyash Bagad, Quang Dao,
//! Yuval Domb, and Justin Thaler. <https://eprint.iacr.org/2025/1117.pdf>
//!
//! # Overview
//!
//! The main entry point is [`prove_spartan_outer_cubic_small_value`], which implements

use crate::{
  big_num::{DelayedReduction, SmallValue, SmallValueEngine, SmallValueField},
  errors::SpartanError,
  lagrange_accumulator::{
    LagrangeAccumulators, LagrangeBasisFactory, LagrangeCoeff, LagrangeDomainEvals,
    ReducedLagrangeDomainEvals, SMALL_VALUE_T_DEGREE, SmallValueExtensionBoundedPoly,
    build_accumulators_spartan,
  },
  polys::{
    eq::EqPolynomial,
    multilinear::MultilinearPolynomial,
    univariate::{UniPoly, build_linear_times_quadratic_poly_from_claim},
  },
  start_span,
  sumcheck::{SumcheckProof, eq_sumcheck},
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use ff::PrimeField;
use num_traits::Zero;
use rayon::prelude::*;
use tracing::info;

use crate::sumcheck::PAR_THRESHOLD;

/// Tracks the small-value sum-check state for the first ℓ₀ rounds.
///
/// This struct maintains the precomputed accumulators and running state
/// needed to efficiently evaluate round polynomials using native integer
/// arithmetic instead of field operations.
pub(crate) struct SmallValueSumCheck<Scalar: PrimeField, const D: usize> {
  accumulators: LagrangeAccumulators<Scalar, D>,
  coeff: LagrangeCoeff<Scalar, D>,
  eq_alpha: Scalar,
  basis_factory: LagrangeBasisFactory<Scalar, D>,
}

/// Prove Spartan's outer `poly_A * poly_B - poly_C` relation using (EqPoly-SmallValueSC).
///
/// This function combines small-value optimization (Algorithm 4) for the first ℓ₀ rounds
/// with eq-poly optimization (Algorithm 5) for the remaining rounds.
///
/// Field-element polynomials are created internally via batched eq-weighted binding
/// from the small-value inputs, eliminating the need for pre-allocated field polys.
///
/// This path is Spartan-outer-specific. It assumes:
/// - the witness satisfies `A(x) * B(x) - C(x) = 0` on `{0,1}^n`
/// - for evaluation points containing `∞`, only the highest-degree term matters,
///   so `C` does not contribute to the accumulator
///
/// Generic over `SmallValue` to support both i32/i64 and i64/i128 configurations.
///
/// # Type Parameters
///
/// - `LB`: Number of small-value rounds. The optimized path requires
///   `0 < LB < num_rounds`, so at least one suffix variable remains for the
///   standard eq-sumcheck continuation.
///
/// `poly_A_small` and `poly_B_small` must be provided as
/// [`SmallValueExtensionBoundedPoly`] caller assertions. That certificate covers
/// native Lagrange extension and pairwise small-product bounds; the Spartan
/// outer-relation invariant remains the caller's responsibility.
pub(crate) fn prove_spartan_outer_cubic_small_value<E, SV, const LB: usize>(
  claim: &E::Scalar,
  taus: Vec<E::Scalar>,
  poly_A_small: SmallValueExtensionBoundedPoly<'_, SV, LB>,
  poly_B_small: SmallValueExtensionBoundedPoly<'_, SV, LB>,
  poly_C_small: &MultilinearPolynomial<SV>,
  transcript: &mut E::TE,
) -> Result<(SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError>
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  let num_rounds = taus.len();
  if LB == 0 {
    return Err(SpartanError::SmallValueRoundsZero {
      context: "small-value sumcheck requires LB > 0".to_string(),
    });
  }
  if num_rounds <= LB {
    return Err(SpartanError::InvalidInputLength {
      reason: format!(
        "small-value sumcheck requires num_rounds > LB; got num_rounds={} and LB={}",
        num_rounds, LB
      ),
    });
  }

  let poly_A_small_ref = poly_A_small.as_poly();
  let poly_B_small_ref = poly_B_small.as_poly();
  let mut r: Vec<E::Scalar> = Vec::with_capacity(num_rounds);
  let mut polys: Vec<crate::polys::univariate::CompressedUniPoly<E::Scalar>> =
    Vec::with_capacity(num_rounds);
  let mut claim_per_round = *claim;

  let l0 = LB;
  debug_assert!(l0 < num_rounds);

  // ===== Pre-computation Phase =====
  // Build accumulators A_i(v, u) for all i ∈ [ℓ₀] using small-value arithmetic.
  // Also builds eq pyramids for reuse by EqSumCheckInstance.
  // Internally computes eq tables with balanced split and precomputed eq_cache.
  // Uses: small × small → intermediate (for Az·Bz products),
  // then intermediate × field (for eq weighting via DelayedReduction).
  let (accumulators, mut e_in_pyramid, e_xout_pyramid) =
    build_accumulators_spartan::<E::Scalar, SV, LB>(&poly_A_small, &poly_B_small, &taus);

  let mut small_value_sumcheck =
    SmallValueSumCheck::<E::Scalar, SMALL_VALUE_T_DEGREE>::from_accumulators(accumulators);

  // ===== Small-Value Rounds (0 to ℓ₀-1) =====
  // During these rounds, we use the precomputed accumulators. Polynomials are NOT bound
  // during these rounds - that will happen in the transition phase.
  #[allow(clippy::needless_range_loop)]
  for round in 0..l0 {
    let (_round_span, round_t) = start_span!("sumcheck_smallvalue_round", round = round);

    // Build round polynomial s_i(X) = ℓ_i(X) · t_i(X).
    let (poly, li) = generate_univariate_sumcheck_polynomial_from_accumulator(
      &small_value_sumcheck,
      round,
      taus[round],
      claim_per_round,
    )
    .map_err(|err| match err {
      SpartanError::InvalidSumcheckProof => SpartanError::InternalError {
        reason: format!("small-value sumcheck cannot derive t(1) at round {}", round),
      },
      err => err,
    })?;

    // Transcript interaction
    transcript.absorb(b"p", &poly);
    let r_i = transcript.squeeze(b"c")?;
    r.push(r_i);
    polys.push(poly.compress());

    // Update claim
    claim_per_round = poly.evaluate(&r_i);

    // Advance small-value state (updates R_{i+1} and the prefix eq factor)
    small_value_sumcheck.advance(&li, r_i);

    info!(
      elapsed_ms = %round_t.elapsed().as_millis(),
      round = round,
      "sumcheck_smallvalue_round"
    );
  }

  // ===== Transition Phase =====
  // Create bound field-element polynomials via batched eq-weighted small-value accumulation.
  // Binds all completed small-value rounds in a single pass using field×int
  // unreduced accumulation.
  let (_bind_span, bind_t) = start_span!("bind_poly_vars_transition");
  let (mut poly_A, mut poly_B, mut poly_C) = bind_three_polys_batched_small_value(
    poly_A_small_ref,
    poly_B_small_ref,
    poly_C_small,
    &r[..l0],
  );
  info!(
    elapsed_ms = %bind_t.elapsed().as_millis(),
    "bind_poly_vars_transition"
  );

  // ===== Remaining Rounds (ℓ₀ to ℓ-1) =====
  // Reuse the precomputed pyramids when the optimized rounds completed.
  // The first suffix tau is tracked separately in eval_eq_left, so the left pyramid
  // passed to EqSumCheckInstance must exclude that first tau.
  e_in_pyramid.pop();
  let mut eq_instance = eq_sumcheck::EqSumCheckInstance::<E>::from_pyramids(
    e_in_pyramid,
    e_xout_pyramid,
    &taus[l0..],
    small_value_sumcheck.eq_alpha(),
  );

  // Continue with the remaining rounds using the standard eq instance seeded with the
  // accumulated prefix eq factor from the small-value rounds.
  SumcheckProof::<E>::prove_cubic_with_three_inputs_from_eq_instance(
    claim_per_round,
    l0,
    num_rounds - l0,
    &mut eq_instance,
    &mut poly_A,
    &mut poly_B,
    &mut poly_C,
    transcript,
    &mut r,
    &mut polys,
  )?;

  Ok((
    SumcheckProof {
      compressed_polys: polys,
    },
    r,
    vec![poly_A[0], poly_B[0], poly_C[0]],
  ))
}

impl<Scalar: PrimeField, const D: usize> SmallValueSumCheck<Scalar, D> {
  /// Create a new small-value round tracker with precomputed accumulators.
  fn new(
    accumulators: LagrangeAccumulators<Scalar, D>,
    basis_factory: LagrangeBasisFactory<Scalar, D>,
  ) -> Self {
    Self {
      accumulators,
      coeff: LagrangeCoeff::new(),
      eq_alpha: Scalar::ONE,
      basis_factory,
    }
  }

  /// Create from accumulators with the standard Lagrange basis (0, 1, 2, ...).
  pub(crate) fn from_accumulators(accumulators: LagrangeAccumulators<Scalar, D>) -> Self {
    let basis_factory = LagrangeBasisFactory::<Scalar, D>::new(|i| Scalar::from(i as u64));
    Self::new(accumulators, basis_factory)
  }

  /// Evaluate t_i(u) for all u ∈ Û_D in a single pass for round i.
  pub(crate) fn eval_t_all_u(&self, round: usize) -> ReducedLagrangeDomainEvals<Scalar, D> {
    self.accumulators.eval_t_all_u(round, &self.coeff)
  }

  /// Compute ℓ_i values for the provided w_i.
  pub(crate) fn eq_round_values(&self, w_i: Scalar) -> LagrangeDomainEvals<Scalar, 2> {
    let l0 = self.eq_alpha * (Scalar::ONE - w_i);
    let l1 = self.eq_alpha * w_i;
    let linf = self.eq_alpha * (w_i.double() - Scalar::ONE);
    LagrangeDomainEvals::new(linf, [l0, l1])
  }

  /// Advance the round state with the verifier challenge r_i.
  pub(crate) fn advance(&mut self, li: &LagrangeDomainEvals<Scalar, 2>, r_i: Scalar) {
    self.eq_alpha = li.eval_linear_at(r_i);
    self.coeff.extend(&self.basis_factory.basis_at(r_i));
  }

  /// Returns the accumulated eq factor α = eq(τ_{0..i}, r_{0..i}).
  ///
  /// After l0 rounds, this gives the eq factor that must be incorporated
  /// into the remaining sumcheck rounds.
  pub(crate) fn eq_alpha(&self) -> Scalar {
    self.eq_alpha
  }
}

/// Generate the cubic univariate sumcheck polynomial for one small-value round.
pub(crate) fn generate_univariate_sumcheck_polynomial_from_accumulator<F>(
  state: &SmallValueSumCheck<F, 2>,
  round: usize,
  rho: F,
  t_cur: F,
) -> Result<(UniPoly<F>, LagrangeDomainEvals<F, 2>), SpartanError>
where
  F: PrimeField,
{
  let t_all = state.eval_t_all_u(round);
  let t0 = t_all.at_zero();
  let t_inf = t_all.at_infinity();
  let li = state.eq_round_values(rho);
  let poly = build_linear_times_quadratic_poly_from_claim(
    li.at_zero(),
    li.at_one(),
    li.at_infinity(),
    t_cur,
    t0,
    t_inf,
  )
  .ok_or(SpartanError::InvalidSumcheckProof)?;

  Ok((poly, li))
}

/// Batch-bind l0 top variables of three polynomials using eq-weighted small-value accumulation.
///
/// Instead of l0 sequential passes with field×field muls, computes:
///   `poly_out[s] = Σ_{p ∈ {0,1}^l0} eq(challenges, p) · poly_small[p * stride + s]`
/// in one pass using field×int unreduced accumulation with a single final reduction.
fn bind_three_polys_batched_small_value<F, SV>(
  poly_a_small: &MultilinearPolynomial<SV>,
  poly_b_small: &MultilinearPolynomial<SV>,
  poly_c_small: &MultilinearPolynomial<SV>,
  challenges: &[F],
) -> (
  MultilinearPolynomial<F>,
  MultilinearPolynomial<F>,
  MultilinearPolynomial<F>,
)
where
  SV: Copy + Send + Sync,
  F: PrimeField + DelayedReduction<SV>,
{
  let l0 = challenges.len();
  let n = poly_a_small.Z.len();
  debug_assert_eq!(poly_b_small.Z.len(), n);
  debug_assert_eq!(poly_c_small.Z.len(), n);
  debug_assert_eq!(n % (1 << l0), 0);

  let num_prefixes = 1usize << l0;
  let stride = n >> l0;

  // Precompute eq(challenges, p) for all p ∈ {0,1}^l0
  let eq_table = EqPolynomial::evals_from_points(challenges);
  debug_assert_eq!(eq_table.len(), num_prefixes);

  type Acc<F2, SV2> = <F2 as DelayedReduction<SV2>>::Accumulator;

  // Suffix-outer parallel loop: accumulators live on stack per thread
  let compute = |s: usize| -> (F, F, F) {
    let mut acc_a = Acc::<F, SV>::zero();
    let mut acc_b = Acc::<F, SV>::zero();
    let mut acc_c = Acc::<F, SV>::zero();

    for (p, eq_p) in eq_table.iter().enumerate() {
      let idx = p * stride + s;

      // Single-value accumulation: field × small with delayed reduction
      F::unreduced_multiply_accumulate(&mut acc_a, eq_p, &poly_a_small.Z[idx]);
      F::unreduced_multiply_accumulate(&mut acc_b, eq_p, &poly_b_small.Z[idx]);
      F::unreduced_multiply_accumulate(&mut acc_c, eq_p, &poly_c_small.Z[idx]);
    }

    (F::reduce(&acc_a), F::reduce(&acc_b), F::reduce(&acc_c))
  };

  let mut out_a = vec![F::ZERO; stride];
  let mut out_b = vec![F::ZERO; stride];
  let mut out_c = vec![F::ZERO; stride];

  if stride >= PAR_THRESHOLD {
    out_a
      .par_iter_mut()
      .zip(out_b.par_iter_mut())
      .zip(out_c.par_iter_mut())
      .enumerate()
      .for_each(|(s, ((a, b), c))| {
        let (ra, rb, rc) = compute(s);
        *a = ra;
        *b = rb;
        *c = rc;
      });
  } else {
    for s in 0..stride {
      let (a, b, c) = compute(s);
      out_a[s] = a;
      out_b[s] = b;
      out_c[s] = c;
    }
  }

  (
    MultilinearPolynomial::new(out_a),
    MultilinearPolynomial::new(out_b),
    MultilinearPolynomial::new(out_c),
  )
}

/// Convert a small-value polynomial to its field-valued representation.
fn small_poly_to_field<F, SV>(poly: &MultilinearPolynomial<SV>) -> MultilinearPolynomial<F>
where
  F: PrimeField + SmallValueField<SV>,
  SV: Copy,
{
  MultilinearPolynomial::new(poly.Z.iter().copied().map(F::small_to_field).collect())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    big_num::{SmallValue, SmallValueEngine, SmallValueField},
    polys::multilinear::MultilinearPolynomial,
    provider::PallasHyraxEngine,
    sumcheck::eq_sumcheck::EqSumCheckInstance,
    traits::{Engine, transcript::TranscriptEngineTrait},
  };
  use ff::Field;
  use rand::{Rng, SeedableRng, rngs::StdRng};
  use std::{fmt::Debug, ops::Mul};

  type E = PallasHyraxEngine;
  type F = <E as Engine>::Scalar;

  fn eqe_bit(w: F, x: F) -> F {
    F::ONE - w - x + (w * x).double()
  }

  #[test]
  fn test_eq_round_values_matches_formula() {
    let mut small_value = SmallValueSumCheck::<F, SMALL_VALUE_T_DEGREE>::new(
      LagrangeAccumulators::new(1),
      LagrangeBasisFactory::new(|i| F::from(i as u64)),
    );
    small_value.eq_alpha = F::from(13u64);

    let w = F::from(7u64);
    let li = small_value.eq_round_values(w);

    assert_eq!(li.at_zero(), small_value.eq_alpha * (F::ONE - w));
    assert_eq!(li.at_one(), small_value.eq_alpha * w);
    assert_eq!(
      li.at_infinity(),
      small_value.eq_alpha * (w.double() - F::ONE)
    );
  }

  #[test]
  fn test_advance_matches_prefix_eq_product() {
    let mut small_value = SmallValueSumCheck::<F, SMALL_VALUE_T_DEGREE>::new(
      LagrangeAccumulators::new(4),
      LagrangeBasisFactory::new(|i| F::from(i as u64)),
    );
    let taus = [F::from(2u64), F::from(5u64), F::from(8u64)];
    let rs = [F::from(3u64), F::from(4u64), F::from(7u64)];

    let mut expected = F::ONE;
    for (tau, r) in taus.into_iter().zip(rs) {
      let li = small_value.eq_round_values(tau);
      small_value.advance(&li, r);
      expected *= eqe_bit(tau, r);
      assert_eq!(small_value.eq_alpha(), expected);
    }
  }

  #[test]
  fn test_derive_quadratic_eval_at_one_from_claim_returns_value() {
    let l0 = F::from(2u64);
    let l1 = F::from(5u64);
    let t0 = F::from(11u64);
    let claim = F::from(97u64);

    let s0 = l0 * t0;
    let s1 = claim - s0;
    let expected = s1 * l1.invert().unwrap();

    assert_eq!(
      crate::polys::univariate::derive_quadratic_eval_at_one_from_claim(l0, l1, claim, t0),
      Some(expected)
    );
  }

  #[test]
  fn test_derive_quadratic_eval_at_one_from_claim_returns_none_on_zero_l1() {
    let l0 = F::from(3u64);
    let l1 = F::ZERO;
    let t0 = F::from(4u64);
    let claim = F::from(10u64);

    assert_eq!(
      crate::polys::univariate::derive_quadratic_eval_at_one_from_claim(l0, l1, claim, t0),
      None
    );
  }

  /// Generic helper to test that SmallValueSumCheck produces the same polynomial
  /// evaluations as EqSumCheckInstance across multiple rounds.
  fn run_smallvalue_round_test<SV>()
  where
    SV: SmallValue + Mul<Output = SV> + TryFrom<usize>,
    <SV as TryFrom<usize>>::Error: Debug,
    F: SmallValueEngine<SV>,
  {
    const NUM_VARS: usize = 6;
    const SMALL_VALUE_ROUNDS: usize = 3;

    let n = 1usize << NUM_VARS;
    let taus = (0..NUM_VARS)
      .map(|i| F::from((i + 2) as u64))
      .collect::<Vec<_>>();

    // Small-value polynomials for build_accumulators_spartan
    let az_small: Vec<SV> = (0..n).map(|i| SV::try_from(i + 1).unwrap()).collect();
    let bz_small: Vec<SV> = (0..n).map(|i| SV::try_from(i + 3).unwrap()).collect();
    let cz_small: Vec<SV> = az_small
      .iter()
      .zip(bz_small.iter())
      .map(|(&a, &b)| a * b)
      .collect();

    let az_poly = MultilinearPolynomial::new(az_small.clone());
    let bz_poly = MultilinearPolynomial::new(bz_small.clone());

    // Field polynomials for reference computation
    let az_vals: Vec<F> = az_small.iter().map(|&v| F::small_to_field(v)).collect();
    let bz_vals: Vec<F> = bz_small.iter().map(|&v| F::small_to_field(v)).collect();
    let cz_vals: Vec<F> = cz_small.iter().map(|&v| F::small_to_field(v)).collect();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);
    let cz = MultilinearPolynomial::new(cz_vals);

    // Claim = 0 for satisfying witness (Az·Bz = Cz)
    let mut claim = F::ZERO;

    let az_bound = SmallValueExtensionBoundedPoly::<_, SMALL_VALUE_ROUNDS>::new(&az_poly)
      .expect("Az should be extension-bounded");
    let bz_bound = SmallValueExtensionBoundedPoly::<_, SMALL_VALUE_ROUNDS>::new(&bz_poly)
      .expect("Bz should be extension-bounded");

    // Build accumulators using the checked API
    let (accs, _, _) =
      build_accumulators_spartan::<F, SV, SMALL_VALUE_ROUNDS>(&az_bound, &bz_bound, &taus);
    let mut small_value = SmallValueSumCheck::from_accumulators(accs);

    // Full eq_instance for verification against standard sumcheck
    let mut eq_instance = EqSumCheckInstance::<E>::new(&taus);
    let mut poly_A = az.clone();
    let mut poly_B = bz.clone();
    let mut poly_C = cz.clone();

    for (round, &tau_round) in taus.iter().enumerate().take(SMALL_VALUE_ROUNDS) {
      // Get expected evaluations from standard method
      let (expected_eval_0, expected_eval_2, expected_eval_3) =
        eq_instance.evaluation_points_cubic_with_three_inputs(&poly_A, &poly_B, &poly_C, claim);
      let expected_eval_1 = claim - expected_eval_0; // s(0) + s(1) = claim

      // Build small-value polynomial
      let (poly, li) = generate_univariate_sumcheck_polynomial_from_accumulator(
        &small_value,
        round,
        tau_round,
        claim,
      )
      .expect("l1 should be non-zero for chosen taus");

      // Check all 4 evaluation points
      assert_eq!(
        poly.evaluate(&F::ZERO),
        expected_eval_0,
        "s(0) mismatch at round {}",
        round
      );
      assert_eq!(
        poly.evaluate(&F::ONE),
        expected_eval_1,
        "s(1) mismatch at round {}",
        round
      );
      assert_eq!(
        poly.evaluate(&F::from(2u64)),
        expected_eval_2,
        "s(2) mismatch at round {}",
        round
      );
      assert_eq!(
        poly.evaluate(&F::from(3u64)),
        expected_eval_3,
        "s(3) mismatch at round {}",
        round
      );

      // Advance to next round with a fixed challenge
      let r_i = F::from((round + 7) as u64);
      claim = poly.evaluate(&r_i);

      poly_A.bind_poly_var_top(&r_i);
      poly_B.bind_poly_var_top(&r_i);
      poly_C.bind_poly_var_top(&r_i);
      eq_instance.bound(&r_i);
      small_value.advance(&li, r_i);
    }
  }

  #[test]
  fn test_smallvalue_round_matches_eq_instance_evals_i32() {
    run_smallvalue_round_test::<i32>();
  }

  #[test]
  fn test_smallvalue_round_matches_eq_instance_evals_i64() {
    run_smallvalue_round_test::<i64>();
  }

  /// Test that prove_spartan_outer_cubic_small_value produces identical
  /// output to prove_cubic_with_three_inputs using synthetic small-value polynomials.
  ///
  /// Uses synthetic Az, Bz values in a small range and computes Cz = Az * Bz.
  fn run_equivalence_test<SV>(num_vars: usize)
  where
    SV: SmallValue + Mul<Output = SV> + TryFrom<i32>,
    <SV as TryFrom<i32>>::Error: Debug,
    F: SmallValueEngine<SV>,
  {
    const SEED: u64 = 0xDEADBEEF;
    let mut rng = StdRng::seed_from_u64(SEED);
    let n = 1usize << num_vars;

    // Generate synthetic small-value polynomials
    // Use values in [-100, 100] range to ensure products fit in i32/i64
    let az_small: Vec<SV> = (0..n)
      .map(|_| SV::try_from(rng.gen_range(-100i32..=100i32)).unwrap())
      .collect();
    let bz_small: Vec<SV> = (0..n)
      .map(|_| SV::try_from(rng.gen_range(-100i32..=100i32)).unwrap())
      .collect();
    // Cz = Az * Bz computed in the small domain
    let cz_small: Vec<SV> = az_small
      .iter()
      .zip(&bz_small)
      .map(|(&a, &b)| a * b)
      .collect();

    // Random taus
    let taus: Vec<F> = (0..num_vars).map(|_| F::random(&mut rng)).collect();

    run_equivalence_test_with_taus::<SV>(taus, az_small, bz_small, cz_small);
  }

  fn run_equivalence_test_with_taus<SV>(
    taus: Vec<F>,
    az_small: Vec<SV>,
    bz_small: Vec<SV>,
    cz_small: Vec<SV>,
  ) where
    SV: SmallValue + Mul<Output = SV>,
    F: SmallValueEngine<SV>,
  {
    let num_vars = taus.len();
    let az_vals: Vec<F> = az_small.iter().map(|&v| F::small_to_field(v)).collect();
    let bz_vals: Vec<F> = bz_small.iter().map(|&v| F::small_to_field(v)).collect();
    let cz_vals: Vec<F> = cz_small.iter().map(|&v| F::small_to_field(v)).collect();

    // Claim = 0 for satisfying witness (Az·Bz = Cz on {0,1}^n)
    let claim: F = F::ZERO;

    // Small-value polynomials
    let az_small_poly = MultilinearPolynomial::new(az_small);
    let bz_small_poly = MultilinearPolynomial::new(bz_small);
    let cz_small_poly = MultilinearPolynomial::new(cz_small);

    // Polynomials for standard method
    let mut az1 = MultilinearPolynomial::new(az_vals);
    let mut bz1 = MultilinearPolynomial::new(bz_vals);
    let mut cz1 = MultilinearPolynomial::new(cz_vals);

    // Fresh transcripts with same seed
    let mut transcript1 = <E as Engine>::TE::new(b"test");
    let mut transcript2 = <E as Engine>::TE::new(b"test");

    // Run standard method
    let (proof1, r1, evals1) = SumcheckProof::<E>::prove_cubic_with_three_inputs(
      &claim,
      taus.clone(),
      &mut az1,
      &mut bz1,
      &mut cz1,
      &mut transcript1,
    )
    .expect("standard prove should succeed");

    let az_bound = SmallValueExtensionBoundedPoly::<_, 3>::new(&az_small_poly)
      .expect("Az should be extension-bounded");
    let bz_bound = SmallValueExtensionBoundedPoly::<_, 3>::new(&bz_small_poly)
      .expect("Bz should be extension-bounded");

    // Run small-value method
    let (proof2, r2, evals2) = prove_spartan_outer_cubic_small_value::<E, SV, 3>(
      &claim,
      taus.clone(),
      az_bound,
      bz_bound,
      &cz_small_poly,
      &mut transcript2,
    )
    .expect("small-value prove should succeed");

    // Verify all outputs match
    assert_eq!(r1, r2, "challenges must match for num_vars={}", num_vars);
    assert_eq!(
      proof1, proof2,
      "proofs must match for num_vars={}",
      num_vars
    );
    assert_eq!(
      evals1, evals2,
      "final evals must match for num_vars={}",
      num_vars
    );

    // Verify the proof
    let mut transcript_v = <E as Engine>::TE::new(b"test");
    let (final_claim, r_v) = proof1
      .verify(claim, num_vars, 3, &mut transcript_v)
      .expect("verification should succeed");
    assert_eq!(r_v, r1, "verify challenges must match prover");
    let tau_eval = EqPolynomial::new(taus).evaluate(&r_v);
    let expected = tau_eval * (evals1[0] * evals1[1] - evals1[2]);
    assert_eq!(final_claim, expected, "final claim mismatch");
  }

  /// Test small-value sumcheck equivalence with synthetic polynomials.
  ///
  /// Tests multiple sizes to ensure equivalence holds when the optimized path
  /// can run exactly LB rounds and leave at least one suffix variable.
  /// With LB=3:
  /// - num_vars=4: l0=3, suffix_vars=1
  /// - num_vars=6: l0=3, suffix_vars=3
  /// - num_vars=10: l0=3, suffix_vars=7
  #[test]
  fn test_sumcheck_equivalence_with_synthetic_i32() {
    for num_vars in [4, 6, 10] {
      run_equivalence_test::<i32>(num_vars);
    }
  }

  #[test]
  fn test_sumcheck_equivalence_with_synthetic_i64() {
    for num_vars in [4, 6, 10] {
      run_equivalence_test::<i64>(num_vars);
    }
  }

  #[test]
  fn test_small_value_sumcheck_rejects_num_rounds_lte_lb() {
    const NUM_VARS: usize = 3;
    let n = 1usize << NUM_VARS;
    let az_small_poly = MultilinearPolynomial::new(vec![1i32; n]);
    let bz_small_poly = MultilinearPolynomial::new(vec![2i32; n]);
    let cz_small_poly = MultilinearPolynomial::new(vec![2i32; n]);
    let az_bound = SmallValueExtensionBoundedPoly::<_, 3>::new(&az_small_poly)
      .expect("Az should be extension-bounded");
    let bz_bound = SmallValueExtensionBoundedPoly::<_, 3>::new(&bz_small_poly)
      .expect("Bz should be extension-bounded");
    let taus = vec![F::from(2u64), F::from(3u64), F::from(4u64)];
    let mut transcript = <E as Engine>::TE::new(b"test");

    let result = prove_spartan_outer_cubic_small_value::<E, i32, 3>(
      &F::ZERO,
      taus,
      az_bound,
      bz_bound,
      &cz_small_poly,
      &mut transcript,
    );

    assert!(matches!(
      result,
      Err(SpartanError::InvalidInputLength { .. })
    ));
  }

  #[test]
  fn test_small_value_sumcheck_rejects_zero_lb() {
    const NUM_VARS: usize = 1;
    let n = 1usize << NUM_VARS;
    let az_small_poly = MultilinearPolynomial::new(vec![1i32; n]);
    let bz_small_poly = MultilinearPolynomial::new(vec![2i32; n]);
    let cz_small_poly = MultilinearPolynomial::new(vec![2i32; n]);
    let az_bound = SmallValueExtensionBoundedPoly::<_, 0>::new(&az_small_poly)
      .expect("Az should be extension-bounded");
    let bz_bound = SmallValueExtensionBoundedPoly::<_, 0>::new(&bz_small_poly)
      .expect("Bz should be extension-bounded");
    let taus = vec![F::from(2u64)];
    let mut transcript = <E as Engine>::TE::new(b"test");

    let result = prove_spartan_outer_cubic_small_value::<E, i32, 0>(
      &F::ZERO,
      taus,
      az_bound,
      bz_bound,
      &cz_small_poly,
      &mut transcript,
    );

    assert!(matches!(
      result,
      Err(SpartanError::SmallValueRoundsZero { .. })
    ));
  }

  #[test]
  fn test_small_value_sumcheck_rejects_when_first_tau_is_zero() {
    const NUM_VARS: usize = 6;
    const SEED: u64 = 0xDEADBEEF;
    let mut rng = StdRng::seed_from_u64(SEED);
    let n = 1usize << NUM_VARS;

    let az_small: Vec<i32> = (0..n).map(|_| rng.gen_range(-100i32..=100i32)).collect();
    let bz_small: Vec<i32> = (0..n).map(|_| rng.gen_range(-100i32..=100i32)).collect();
    let cz_small: Vec<i32> = az_small
      .iter()
      .zip(&bz_small)
      .map(|(&a, &b)| a * b)
      .collect();

    let mut taus: Vec<F> = (0..NUM_VARS).map(|_| F::random(&mut rng)).collect();
    taus[0] = F::ZERO;

    let claim = F::ZERO;
    let az_small_poly = MultilinearPolynomial::new(az_small);
    let bz_small_poly = MultilinearPolynomial::new(bz_small);
    let cz_small_poly = MultilinearPolynomial::new(cz_small);
    let az_bound = SmallValueExtensionBoundedPoly::<_, 3>::new(&az_small_poly)
      .expect("Az should be extension-bounded");
    let bz_bound = SmallValueExtensionBoundedPoly::<_, 3>::new(&bz_small_poly)
      .expect("Bz should be extension-bounded");
    let mut transcript = <E as Engine>::TE::new(b"test");

    let result = prove_spartan_outer_cubic_small_value::<E, i32, 3>(
      &claim,
      taus,
      az_bound,
      bz_bound,
      &cz_small_poly,
      &mut transcript,
    );

    assert!(matches!(result, Err(SpartanError::InternalError { .. })));
  }
}

#[cfg(test)]
mod perf_tests {
  use super::*;
  use crate::{
    big_num::SmallValueEngine, polys::multilinear::MultilinearPolynomial, start_span,
    traits::Engine,
  };
  use ff::Field;
  use rand::{Rng, SeedableRng, rngs::StdRng};
  use tracing::info;
  use tracing_subscriber::EnvFilter;

  // Test sizes: smaller for debug builds, full range for release
  #[cfg(debug_assertions)]
  const TEST_SIZES: &[usize] = &[16, 18];

  #[cfg(not(debug_assertions))]
  const TEST_SIZES: &[usize] = &[16, 18, 20, 22, 24];

  fn test_small_value_sumcheck_with<E: Engine>()
  where
    E::Scalar: SmallValueEngine<i64>,
  {
    const SEED: u64 = 0xDEADBEEF;
    let field_name = std::any::type_name::<E::Scalar>()
      .split("::")
      .last()
      .unwrap_or("unknown");

    for &num_vars in TEST_SIZES {
      let len = 1 << num_vars;
      let mut rng = StdRng::seed_from_u64(SEED);

      // Generate synthetic small-value polynomials
      let az_small: Vec<i64> = (0..len).map(|_| rng.gen_range(-100i64..=100i64)).collect();
      let bz_small: Vec<i64> = (0..len).map(|_| rng.gen_range(-100i64..=100i64)).collect();
      let cz_small: Vec<i64> = az_small
        .iter()
        .zip(&bz_small)
        .map(|(&a, &b)| a * b)
        .collect();

      let az_poly = MultilinearPolynomial::new(az_small);
      let bz_poly = MultilinearPolynomial::new(bz_small);
      let cz_poly = MultilinearPolynomial::new(cz_small);
      let az_bound = SmallValueExtensionBoundedPoly::<_, 3>::new(&az_poly)
        .expect("Az should be extension-bounded");
      let bz_bound = SmallValueExtensionBoundedPoly::<_, 3>::new(&bz_poly)
        .expect("Bz should be extension-bounded");

      let taus: Vec<E::Scalar> = (0..num_vars).map(|_| E::Scalar::random(&mut rng)).collect();
      let mut transcript = E::TE::new(b"perf_test");

      let (_span, t) = start_span!(
        "small_value_sumcheck_prove",
        field = field_name,
        num_vars = num_vars
      );

      let (proof, _r, _evals) = prove_spartan_outer_cubic_small_value::<E, _, 3>(
        &E::Scalar::ZERO,
        taus.clone(),
        az_bound,
        bz_bound,
        &cz_poly,
        &mut transcript,
      )
      .expect("proof generation should succeed");

      info!(field = field_name, num_vars, n = len, ms = ?t.elapsed().as_millis(), "completed");

      // Verify proof with fresh transcript
      let mut verifier_transcript = E::TE::new(b"perf_test");
      proof
        .verify(E::Scalar::ZERO, num_vars, 3, &mut verifier_transcript)
        .expect("proof verification should succeed");
    }
  }

  #[test]
  fn test_small_value_sumcheck_perf() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true)
      .with_env_filter(EnvFilter::from_default_env())
      .try_init();

    use crate::provider::Bn254Engine;

    // Always test with BN254
    test_small_value_sumcheck_with::<Bn254Engine>();

    // Additional engines only in release builds
    #[cfg(not(debug_assertions))]
    {
      use crate::provider::{PallasHyraxEngine, T256HyraxEngine};
      test_small_value_sumcheck_with::<PallasHyraxEngine>();
      test_small_value_sumcheck_with::<T256HyraxEngine>();
    }
  }
}
