// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module implements NeutronNova's folding scheme for folding together a batch of R1CS instances
//! This implementation focuses on a non-recursive version of NeutronNova and targets the case where the batch size is moderately large.
//! Since we are in the non-recursive setting, we simply fold a batch of instances into one (all at once, via multi-folding)
//! and then use Spartan to prove that folded instance.
//! The proof system implemented here provides zero-knowledge via Nova's folding scheme.
use crate::start_span;
use crate::{
  Commitment, CommitmentKey, DEFAULT_COMMITMENT_WIDTH, VerifierKey,
  bellpepper::{
    r1cs::{
      MultiRoundSpartanShape, MultiRoundSpartanWitness, PrecommittedState, SpartanShape,
      SpartanWitness,
    },
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  big_num::{
    DelayedReduction, SmallValueEngine, SmallValueField,
    small_value_conversion::to_small_vec_or_zero,
  },
  digest::DigestComputer,
  errors::SpartanError,
  lagrange_accumulator::field_to_i64_or_zero_for_l0,
  math::Math,
  nifs::NovaNIFS,
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
    power::PowPolynomial,
    univariate::{UniPoly, build_linear_times_quadratic_poly_from_claim},
  },
  r1cs::{
    R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, SplitMultiRoundR1CSInstance,
    SplitMultiRoundR1CSShape, SplitR1CSInstance, SplitR1CSShape, weights_from_r,
  },
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    circuit::SpartanCircuit,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    snark::{DigestHelperTrait, SpartanDigest},
    transcript::TranscriptEngineTrait,
  },
  zk::NeutronNovaVerifierCircuit,
};
use ff::{Field, PrimeField};
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use tracing::{debug, info};

pub(crate) fn compute_tensor_decomp(n: usize) -> (usize, usize, usize) {
  let ell = n.next_power_of_two().log_2();
  // we split ell into ell1 and ell2 such that ell1 + ell2 = ell and ell1 >= ell2
  let ell1 = ell.div_ceil(2); // This ensures ell1 >= ell2
  let ell2 = ell / 2;
  let left = 1 << ell1;
  let right = 1 << ell2;

  (ell, left, right)
}

/// A type that holds the NeutronNova NIFS (Non-Interactive Folding Scheme)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaNIFS<E: Engine> {
  _p: PhantomData<E>,
}

/// A small-value NeutronNova NIFS backend.
pub struct SmallValueNeutronNovaNIFS<E: Engine> {
  _p: PhantomData<E>,
}

/// Full field-valued step-circuit Az/Bz/Cz tables.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FieldStepMatvecs<E: Engine> {
  /// Field-valued Az tables in original instance order.
  pub az: Vec<Vec<E::Scalar>>,
  /// Field-valued Bz tables in original instance order.
  pub bz: Vec<Vec<E::Scalar>>,
  /// Field-valued Cz tables in original instance order.
  pub cz: Vec<Vec<E::Scalar>>,
}

impl<E: Engine> FieldStepMatvecs<E> {
  fn from_triples(matvec: Vec<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>)>) -> Self {
    let mut az = Vec::with_capacity(matvec.len());
    let mut bz = Vec::with_capacity(matvec.len());
    let mut cz = Vec::with_capacity(matvec.len());
    for (az_layer, bz_layer, cz_layer) in matvec {
      az.push(az_layer);
      bz.push(bz_layer);
      cz.push(cz_layer);
    }

    Self { az, bz, cz }
  }

  fn len(&self) -> usize {
    self.az.len()
  }
}

/// Small-value Az/Bz tables and the global positions that must use field corrections.
///
/// Invariant: for every index in `large_positions`, every layer's small Az/Bz
/// table stores zero at that index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmallAbStepMatvecs {
  /// Small Az tables; global large positions are zeroed.
  pub az_small: Vec<Vec<i64>>,
  /// Small Bz tables; global large positions are zeroed.
  pub bz_small: Vec<Vec<i64>>,
  /// Positions where any cached value did not fit in the small representation.
  pub large_positions: Vec<usize>,
  /// Number of small-value Lagrange accumulator rounds used to bound Az/Bz.
  pub l0: usize,
}

/// Small-value Az/Bz/Cz tables and the global positions that must use field corrections.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmallAbcStepMatvecs {
  /// Small Az/Bz tables and their large-position metadata.
  pub ab: SmallAbStepMatvecs,
  /// Small Cz tables; global large positions are zeroed.
  pub cz_small: Vec<Vec<i64>>,
  /// Positions where any cached C value did not fit in the small representation.
  pub c_large_positions: Vec<usize>,
}

/// Regular NeutronNova step matvecs.
///
/// `small_abc = None` is the full-field path. `small_abc = Some(..)` enables
/// the old `has_i64` optimization path: small A/B for round claims plus small C
/// for scalar C claims and final C folding.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaStepMatvecs<E: Engine> {
  /// Full field-valued step matvec tables.
  pub field: FieldStepMatvecs<E>,
  /// Optional small Az/Bz/Cz tables for the regular small-value optimization.
  pub small_abc: Option<SmallAbcStepMatvecs>,
}

/// Small-value accumulator NeutronNova step matvecs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SmallValueNeutronNovaStepMatvecs<E: Engine> {
  /// Full field-valued step matvec tables used for large-position corrections.
  pub field: FieldStepMatvecs<E>,
  /// Small Az/Bz/Cz tables used by the small-value accumulator.
  pub small_abc: SmallAbcStepMatvecs,
}

/// Output of a NeutronNova NIFS backend.
///
/// Contains split equality evaluations, folded Az/Bz/Cz row tables, and the folded
/// step witness/instance.
pub type NeutronNovaNifsOutput<E> = (
  Vec<<E as Engine>::Scalar>,
  Vec<<E as Engine>::Scalar>,
  Vec<<E as Engine>::Scalar>,
  Vec<<E as Engine>::Scalar>,
  R1CSWitness<E>,
  R1CSInstance<E>,
);

/// Backend trait for the NeutronNova NIFS step.
///
/// Implementations receive precomputed step `Az/Bz/Cz` row tables and are responsible
/// for producing the folded row tables and folded R1CS witness/instance.
pub trait NeutronNovaNifsStrategy<E: Engine>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Backend-specific proving/preprocessing input.
  type Input: Clone + Send + Sync + PartialEq;

  /// Backend-specific representation of the step `Az/Bz/Cz` layers.
  type StepMatvecs: Send + Sync;

  /// Returns whether witness synthesis should use the small-value assignment path.
  fn is_small(input: &Self::Input) -> bool;

  /// Builds backend-specific step matvecs from regular instances and witnesses.
  fn build_step_matvecs(
    shape: &SplitR1CSShape<E>,
    instances: &[R1CSInstance<E>],
    witnesses: &[R1CSWitness<E>],
    input: &Self::Input,
  ) -> Result<Self::StepMatvecs, SpartanError> {
    if instances.len() != witnesses.len() {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "cannot build step matvecs with {} instances and {} witnesses",
          instances.len(),
          witnesses.len()
        ),
      });
    }

    let matvec: Vec<_> = (0..instances.len())
      .into_par_iter()
      .map(|i| {
        let mut z = Vec::with_capacity(witnesses[i].W.len() + 1 + instances[i].X.len());
        z.extend_from_slice(&witnesses[i].W);
        z.push(E::Scalar::ONE);
        z.extend_from_slice(&instances[i].X);
        shape.multiply_vec(&z)
      })
      .collect::<Result<Vec<_>, _>>()?;

    Self::build_step_matvecs_from_field(matvec, input)
  }

  /// Builds backend-specific step matvecs from already-computed field `Az/Bz/Cz` tables.
  fn build_step_matvecs_from_field(
    field_matvecs: Vec<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>)>,
    input: &Self::Input,
  ) -> Result<Self::StepMatvecs, SpartanError>;

  /// Prove the NIFS folding step using the provided step matvec layers.
  #[allow(clippy::too_many_arguments)]
  fn prove(
    s: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    us: Vec<R1CSInstance<E>>,
    ws: Vec<R1CSWitness<E>>,
    step_matvecs: Self::StepMatvecs,
    nifs_input: &Self::Input,
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<NeutronNovaNifsOutput<E>, SpartanError>;
}

#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn suffix_weight_full<F: Field>(t: usize, ell_b: usize, pair_idx: usize, rhos: &[F]) -> F {
  let mut w = F::ONE;
  let mut k = pair_idx;
  for s in (t + 1)..ell_b {
    let bit = (k & 1) as u8; // LSB-first
    w *= if bit == 0 { F::ONE - rhos[s] } else { rhos[s] };
    k >>= 1;
  }
  w
}

fn extend_with_first_clones<T: Clone>(values: &mut Vec<T>, additional: usize) {
  if additional > 0 {
    values.extend(std::iter::repeat_n(values[0].clone(), additional));
  }
}

pub(crate) fn padded_map_by_repeating_first<'a, T, U>(
  values: &'a [T],
  num_values: usize,
  padded_len: usize,
  mut map_value: impl FnMut(&'a T) -> U,
) -> Vec<U> {
  (0..padded_len)
    .map(|idx| {
      let row = if idx < num_values { idx } else { 0 };
      map_value(&values[row])
    })
    .collect()
}

pub(crate) fn padded_layer_slices<T>(
  layers: &[Vec<T>],
  num_layers: usize,
  n_padded: usize,
) -> Vec<&[T]> {
  padded_map_by_repeating_first(layers, num_layers, n_padded, |layer| layer.as_slice())
}

impl<E: Engine> NeutronNovaNIFS<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Computes the evaluations of the sum-check polynomial at 0, 2, and 3
  /// Uses two-level delayed modular reduction (inner + middle levels).
  /// Note: Outer level (over pairs) uses regular field arithmetic since there are few pairs.
  #[inline(always)]
  #[allow(clippy::needless_range_loop)]
  pub(crate) fn prove_helper(
    round: usize,
    (left, right): (usize, usize),
    e: &[E::Scalar],
    Az1: &[E::Scalar],
    Bz1: &[E::Scalar],
    Cz1: &[E::Scalar],
    Az2: &[E::Scalar],
    Bz2: &[E::Scalar],
  ) -> (E::Scalar, E::Scalar) {
    type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

    // sanity check sizes
    assert_eq!(e.len(), left + right);
    assert_eq!(Az1.len(), left * right);

    let f = &e[left..];
    let e_left = &e[..left];
    let compute_e0 = round != 0;

    let mut acc_e0 = Acc::<E::Scalar>::default();
    let mut acc_quad = Acc::<E::Scalar>::default();

    for i in 0..right {
      let base = i * left;
      let mut inner_e0 = Acc::<E::Scalar>::default();
      let mut inner_quad = Acc::<E::Scalar>::default();

      if compute_e0 {
        for j in 0..left {
          let k = base + j;
          let inner_val = Az1[k] * Bz1[k] - Cz1[k];
          <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
            &mut inner_e0,
            &e_left[j],
            &inner_val,
          );
          let az_diff = Az2[k] - Az1[k];
          let bz_diff = Bz2[k] - Bz1[k];
          let quad_val = az_diff * bz_diff;
          <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
            &mut inner_quad,
            &e_left[j],
            &quad_val,
          );
        }
      } else {
        for j in 0..left {
          let k = base + j;
          let az_diff = Az2[k] - Az1[k];
          let bz_diff = Bz2[k] - Bz1[k];
          let quad_val = az_diff * bz_diff;
          <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
            &mut inner_quad,
            &e_left[j],
            &quad_val,
          );
        }
      }

      let inner_e0_red = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_e0);
      let inner_quad_red = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_quad);

      let f_i = &f[i];
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_e0,
        f_i,
        &inner_e0_red,
      );
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_quad,
        f_i,
        &inner_quad_red,
      );
    }

    (
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_e0),
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_quad),
    )
  }

  /// Small-value variant of `prove_helper` for round 0.
  ///
  /// The round-0 claim has `e0 = 0`, so this returns only the quadratic
  /// coefficient using small integer products plus field corrections at large
  /// positions.
  #[inline(always)]
  #[allow(clippy::needless_range_loop)]
  fn prove_helper_small(
    (left, right): (usize, usize),
    e: &[E::Scalar],
    az1: &[E::Scalar],
    bz1: &[E::Scalar],
    az2: &[E::Scalar],
    bz2: &[E::Scalar],
    az1_small: &[i64],
    bz1_small: &[i64],
    az2_small: &[i64],
    bz2_small: &[i64],
    large_positions: &[usize],
  ) -> E::Scalar
  where
    E::Scalar: DelayedReduction<i128>,
  {
    type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

    let f = &e[left..];
    let e_left = &e[..left];
    let total = left * right;

    let mut acc_quad = Acc::<E::Scalar>::default();

    for i in 0..right {
      let base = i * left;
      let mut inner_acc = <E::Scalar as DelayedReduction<i128>>::Accumulator::default();

      for j in 0..left {
        let k = base + j;
        let az_diff = az2_small[k] as i128 - az1_small[k] as i128;
        let bz_diff = bz2_small[k] as i128 - bz1_small[k] as i128;
        let quad_val = az_diff * bz_diff;
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut inner_acc,
          &e_left[j],
          &quad_val,
        );
      }

      let inner_quad_red = <E::Scalar as DelayedReduction<i128>>::reduce(&inner_acc);
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_quad,
        &f[i],
        &inner_quad_red,
      );
    }

    let mut quad = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_quad);

    // Correction for large-value positions: add field arithmetic for positions
    // where the i64 path contributed 0 instead of the correct value.
    for &k in large_positions {
      if k >= total {
        continue;
      }
      let i = k / left;
      let j = k % left;
      let az_diff = az2[k] - az1[k];
      let bz_diff = bz2[k] - bz1[k];
      quad += f[i] * e_left[j] * az_diff * bz_diff;
    }

    quad
  }

  /// Small-value helper for round 1 after a small round-0 challenge.
  ///
  /// For each group of four original layers, this computes the AB contribution
  /// that would arise after first folding layers `(0, 1)` and `(2, 3)` by `r0`,
  /// without materializing small folded layers. Large positions are corrected
  /// from the field-valued A/B layers.
  #[inline(always)]
  #[allow(clippy::needless_range_loop)]
  fn prove_helper_ab_cross(
    (left, right): (usize, usize),
    e: &[E::Scalar],
    a_small: [&[i64]; 4],
    b_small: [&[i64]; 4],
    a_field: [&[E::Scalar]; 4],
    b_field: [&[E::Scalar]; 4],
    c00: &E::Scalar,
    c01: &E::Scalar,
    c11: &E::Scalar,
    r0: &E::Scalar,
    large_positions: &[usize],
  ) -> (E::Scalar, E::Scalar)
  where
    E::Scalar: DelayedReduction<i128>,
  {
    type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

    let f = &e[left..];
    let e_left = &e[..left];
    let total = left * right;

    let mut acc_e0 = Acc::<E::Scalar>::default();
    let mut acc_quad = Acc::<E::Scalar>::default();

    for i in 0..right {
      let base = i * left;

      let mut e0_00 = <E::Scalar as DelayedReduction<i128>>::Accumulator::default();
      let mut e0_01 = <E::Scalar as DelayedReduction<i128>>::Accumulator::default();
      let mut e0_11 = <E::Scalar as DelayedReduction<i128>>::Accumulator::default();

      for j in 0..left {
        let k = base + j;
        let weight = &e_left[j];
        let (a0, a1) = (a_small[0][k] as i128, a_small[1][k] as i128);
        let (b0, b1) = (b_small[0][k] as i128, b_small[1][k] as i128);
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut e0_00,
          weight,
          &(a0 * b0),
        );
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut e0_01,
          weight,
          &(a0 * b1 + a1 * b0),
        );
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut e0_11,
          weight,
          &(a1 * b1),
        );
      }

      let e0_inner = *c00 * <E::Scalar as DelayedReduction<i128>>::reduce(&e0_00)
        + *c01 * <E::Scalar as DelayedReduction<i128>>::reduce(&e0_01)
        + *c11 * <E::Scalar as DelayedReduction<i128>>::reduce(&e0_11);
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_e0,
        &f[i],
        &e0_inner,
      );

      let mut q_00 = <E::Scalar as DelayedReduction<i128>>::Accumulator::default();
      let mut q_01 = <E::Scalar as DelayedReduction<i128>>::Accumulator::default();
      let mut q_11 = <E::Scalar as DelayedReduction<i128>>::Accumulator::default();

      for j in 0..left {
        let k = base + j;
        let weight = &e_left[j];
        let da0 = a_small[2][k] as i128 - a_small[0][k] as i128;
        let da1 = a_small[3][k] as i128 - a_small[1][k] as i128;
        let db0 = b_small[2][k] as i128 - b_small[0][k] as i128;
        let db1 = b_small[3][k] as i128 - b_small[1][k] as i128;
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut q_00,
          weight,
          &(da0 * db0),
        );
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut q_01,
          weight,
          &(da0 * db1 + da1 * db0),
        );
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut q_11,
          weight,
          &(da1 * db1),
        );
      }

      let quad_inner = *c00 * <E::Scalar as DelayedReduction<i128>>::reduce(&q_00)
        + *c01 * <E::Scalar as DelayedReduction<i128>>::reduce(&q_01)
        + *c11 * <E::Scalar as DelayedReduction<i128>>::reduce(&q_11);
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_quad,
        &f[i],
        &quad_inner,
      );
    }

    let mut e0 = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_e0);
    let mut quad = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_quad);

    if !large_positions.is_empty() {
      let one_minus_r0 = E::Scalar::ONE - *r0;
      for &k in large_positions {
        if k >= total {
          continue;
        }
        let i = k / left;
        let j = k % left;
        let weight = e_left[j] * f[i];

        let az_lo = one_minus_r0 * a_field[0][k] + *r0 * a_field[1][k];
        let bz_lo = one_minus_r0 * b_field[0][k] + *r0 * b_field[1][k];
        let az_hi = one_minus_r0 * a_field[2][k] + *r0 * a_field[3][k];
        let bz_hi = one_minus_r0 * b_field[2][k] + *r0 * b_field[3][k];

        e0 += weight * az_lo * bz_lo;
        quad += weight * (az_hi - az_lo) * (bz_hi - bz_lo);
      }
    }

    (e0, quad)
  }

  /// AB-only variant of prove_helper: computes sum E[k]*Az_lo*Bz_lo (without Cz subtraction)
  /// and the quad term sum E[k]*(Az_hi-Az_lo)*(Bz_hi-Bz_lo).
  /// The caller subtracts the precomputed C_val contribution from e0_ab externally.
  #[inline(always)]
  #[allow(clippy::needless_range_loop)]
  pub(crate) fn prove_helper_ab_only(
    (left, right): (usize, usize),
    e: &[E::Scalar],
    Az1: &[E::Scalar],
    Bz1: &[E::Scalar],
    Az2: &[E::Scalar],
    Bz2: &[E::Scalar],
  ) -> (E::Scalar, E::Scalar) {
    type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

    let f = &e[left..];
    let e_left = &e[..left];

    let mut acc_e0_ab = Acc::<E::Scalar>::default();
    let mut acc_quad = Acc::<E::Scalar>::default();

    for i in 0..right {
      let base = i * left;
      let mut inner_e0 = Acc::<E::Scalar>::default();
      let mut inner_quad = Acc::<E::Scalar>::default();

      for j in 0..left {
        let k = base + j;
        let ab_val = Az1[k] * Bz1[k];
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut inner_e0,
          &e_left[j],
          &ab_val,
        );
        let az_diff = Az2[k] - Az1[k];
        let bz_diff = Bz2[k] - Bz1[k];
        let quad_val = az_diff * bz_diff;
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut inner_quad,
          &e_left[j],
          &quad_val,
        );
      }

      let inner_e0_red = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_e0);
      let inner_quad_red = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner_quad);

      let f_i = &f[i];
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_e0_ab,
        f_i,
        &inner_e0_red,
      );
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut acc_quad,
        f_i,
        &inner_quad_red,
      );
    }

    (
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_e0_ab),
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc_quad),
    )
  }
}

impl<E> NeutronNovaNifsStrategy<E> for NeutronNovaNIFS<E>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: DelayedReduction<i128>,
{
  type Input = bool;
  type StepMatvecs = NeutronNovaStepMatvecs<E>;

  fn is_small(input: &Self::Input) -> bool {
    *input
  }

  fn build_step_matvecs_from_field(
    field_matvecs: Vec<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>)>,
    input: &Self::Input,
  ) -> Result<Self::StepMatvecs, SpartanError> {
    let field = FieldStepMatvecs::from_triples(field_matvecs);
    let small_abc = if *input {
      Some(build_small_abc_from_field(&field, 1)?)
    } else {
      None
    };

    Ok(NeutronNovaStepMatvecs { field, small_abc })
  }

  #[allow(clippy::too_many_arguments)]
  fn prove(
    s: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    us: Vec<R1CSInstance<E>>,
    ws: Vec<R1CSWitness<E>>,
    step_matvecs: Self::StepMatvecs,
    nifs_input: &Self::Input,
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<NeutronNovaNifsOutput<E>, SpartanError> {
    // Determine padding and NIFS rounds
    let n = us.len();
    let transcript_state = prepare_nifs_transcript(s, &us, transcript)?;
    let NifsTranscriptState {
      n_padded,
      ell_b,
      e_eq: E_eq,
      left,
      right,
      rhos,
    } = transcript_state;

    info!(
      "NeutronNova NIFS prove for {} instances and padded to {} instances",
      us.len(),
      n_padded
    );

    let NeutronNovaStepMatvecs { field, small_abc } = step_matvecs;
    let FieldStepMatvecs {
      az: mut A_layers,
      bz: mut B_layers,
      cz: mut C_layers,
    } = field;

    let use_round0_small_optimization = *nifs_input;
    let (
      mut A_small_layers,
      mut B_small_layers,
      mut C_small_layers,
      ab_large_positions,
      c_large_positions,
    ) = if use_round0_small_optimization {
      let small_abc = small_abc.ok_or_else(|| SpartanError::InvalidInputLength {
        reason: "regular small optimization requested but small A/B/C matvecs are missing"
          .to_string(),
      })?;
      if small_abc.ab.l0 != 1 {
        return Err(SpartanError::InvalidInputLength {
          reason: format!(
            "regular NeutronNova small path requires l0=1 but received {}",
            small_abc.ab.l0
          ),
        });
      }
      (
        small_abc.ab.az_small,
        small_abc.ab.bz_small,
        small_abc.cz_small,
        small_abc.ab.large_positions,
        small_abc.c_large_positions,
      )
    } else {
      (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };

    if A_layers.len() != n || B_layers.len() != n || C_layers.len() != n {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "NIFS received step matvec layer lengths A={}, B={}, C={} but {} instances",
          A_layers.len(),
          B_layers.len(),
          C_layers.len(),
          n
        ),
      });
    }
    if use_round0_small_optimization {
      if A_small_layers.len() != n || B_small_layers.len() != n || C_small_layers.len() != n {
        return Err(SpartanError::InvalidInputLength {
          reason: format!(
            "regular small path received small layer lengths A={}, B={}, C={} but {} instances",
            A_small_layers.len(),
            B_small_layers.len(),
            C_small_layers.len(),
            n
          ),
        });
      }
      let expected_len = left * right;
      if !A_small_layers
        .iter()
        .chain(B_small_layers.iter())
        .chain(C_small_layers.iter())
        .all(|layer| layer.len() == expected_len)
      {
        return Err(SpartanError::InvalidInputLength {
          reason: format!(
            "regular small path requires small layers of length {}",
            expected_len
          ),
        });
      }
    }

    let mut us = us;
    let mut ws = ws;
    if us.len() < n_padded {
      let additional = n_padded - n;
      extend_with_first_clones(&mut us, additional);
      extend_with_first_clones(&mut ws, additional);
    }

    if A_layers.len() < n_padded {
      let additional = n_padded - A_layers.len();
      extend_with_first_clones(&mut A_layers, additional);
      extend_with_first_clones(&mut B_layers, additional);
      extend_with_first_clones(&mut C_layers, additional);
    }
    if use_round0_small_optimization && A_small_layers.len() < n_padded {
      let additional = n_padded - A_small_layers.len();
      extend_with_first_clones(&mut A_small_layers, additional);
      extend_with_first_clones(&mut B_small_layers, additional);
      extend_with_first_clones(&mut C_small_layers, additional);
    }

    let mut c_claims = if use_round0_small_optimization {
      compute_small_c_claims_with_field_corrections::<E>(
        &C_small_layers,
        &C_layers,
        &c_large_positions,
        &E_eq,
        left,
        right,
      )?
    } else {
      Vec::new()
    };

    let mut r_bs: Vec<E::Scalar> = Vec::with_capacity(ell_b);
    let mut T_cur = E::Scalar::ZERO;
    let mut acc_eq = E::Scalar::ONE;

    {
      let pairs = n_padded / 2;
      let (e0, quad_coeff) = if use_round0_small_optimization {
        let quad_coeff = A_layers
          .par_chunks(2)
          .zip(B_layers.par_chunks(2))
          .zip(A_small_layers.par_chunks(2))
          .zip(B_small_layers.par_chunks(2))
          .enumerate()
          .map(
            |(pair_idx, (((pair_a, pair_b), pair_a_small), pair_b_small))| {
              let quad_coeff = Self::prove_helper_small(
                (left, right),
                &E_eq,
                &pair_a[0],
                &pair_b[0],
                &pair_a[1],
                &pair_b[1],
                &pair_a_small[0],
                &pair_b_small[0],
                &pair_a_small[1],
                &pair_b_small[1],
                &ab_large_positions,
              );
              let weight = suffix_weight_full::<E::Scalar>(0, ell_b, pair_idx, &rhos);
              quad_coeff * weight
            },
          )
          .reduce(|| E::Scalar::ZERO, |a, b| a + b);
        (E::Scalar::ZERO, quad_coeff)
      } else {
        compute_field_round_claim::<E>(
          &A_layers, &B_layers, &C_layers, &E_eq, left, right, &rhos, 0,
        )?
      };
      let r_b = finish_nifs_field_round(
        &rhos,
        0,
        e0,
        quad_coeff,
        vc,
        vc_state,
        vc_shape,
        vc_ck,
        transcript,
        &mut r_bs,
        &mut T_cur,
        &mut acc_eq,
      )?;

      if use_round0_small_optimization {
        if ell_b == 1 {
          fold_ab_c_claim_pairs::<E>(&mut A_layers, &mut B_layers, &mut c_claims, pairs, r_b);
          A_layers.truncate(pairs);
          B_layers.truncate(pairs);
          c_claims.truncate(pairs);
        } else {
          let t = 1;
          let prev_r_b = r_b;
          let fold_pairs = A_layers.len() / 2;
          let prove_pairs = fold_pairs / 2;
          let one_minus_r0 = E::Scalar::ONE - prev_r_b;
          let c00 = one_minus_r0 * one_minus_r0;
          let c01 = one_minus_r0 * prev_r_b;
          let c11 = prev_r_b * prev_r_b;

          let e_eq_ref = &E_eq;
          let rhos_ref = &rhos;
          let a_layers_ref = &A_layers;
          let b_layers_ref = &B_layers;
          let a_small_ref = &A_small_layers;
          let b_small_ref = &B_small_layers;
          let c_head = &mut c_claims[..4 * prove_pairs];
          let (e0_acc, quad_acc) = c_head
            .par_chunks_mut(4)
            .enumerate()
            .map(|(j, c_chunk)| {
              fold_quad_c_claim_chunk(c_chunk, prev_r_b);
              let (e0_ab, quad_coeff) = Self::prove_helper_ab_cross(
                (left, right),
                e_eq_ref,
                [
                  &a_small_ref[4 * j],
                  &a_small_ref[4 * j + 1],
                  &a_small_ref[4 * j + 2],
                  &a_small_ref[4 * j + 3],
                ],
                [
                  &b_small_ref[4 * j],
                  &b_small_ref[4 * j + 1],
                  &b_small_ref[4 * j + 2],
                  &b_small_ref[4 * j + 3],
                ],
                [
                  &a_layers_ref[4 * j],
                  &a_layers_ref[4 * j + 1],
                  &a_layers_ref[4 * j + 2],
                  &a_layers_ref[4 * j + 3],
                ],
                [
                  &b_layers_ref[4 * j],
                  &b_layers_ref[4 * j + 1],
                  &b_layers_ref[4 * j + 2],
                  &b_layers_ref[4 * j + 3],
                ],
                &c00,
                &c01,
                &c11,
                &prev_r_b,
                &ab_large_positions,
              );
              let e0 = e0_ab - c_chunk[0];
              let weight = suffix_weight_full::<E::Scalar>(t, ell_b, j, rhos_ref);
              (e0 * weight, quad_coeff * weight)
            })
            .reduce(
              || (E::Scalar::ZERO, E::Scalar::ZERO),
              |a, b| (a.0 + b.0, a.1 + b.1),
            );

          if prove_pairs > 0 {
            A_layers[..4 * prove_pairs]
              .par_chunks_mut(4)
              .zip(B_layers[..4 * prove_pairs].par_chunks_mut(4))
              .for_each(|(a_chunk, b_chunk)| {
                fold_quad_chunk(a_chunk, prev_r_b);
                fold_quad_chunk(b_chunk, prev_r_b);
              });
            compact_folded_layers(&mut A_layers, prove_pairs);
            compact_folded_layers(&mut B_layers, prove_pairs);
            compact_c_claims(&mut c_claims, prove_pairs);
          }
          fold_ab_c_claim_pairs_in_range::<E>(
            &mut A_layers,
            &mut B_layers,
            &mut c_claims,
            (2 * prove_pairs)..fold_pairs,
            prev_r_b,
          );
          A_layers.truncate(fold_pairs);
          B_layers.truncate(fold_pairs);
          c_claims.truncate(fold_pairs);

          let pending_r_b = finish_nifs_field_round(
            &rhos,
            t,
            e0_acc,
            quad_acc,
            vc,
            vc_state,
            vc_shape,
            vc_ck,
            transcript,
            &mut r_bs,
            &mut T_cur,
            &mut acc_eq,
          )?;

          continue_ab_suffix_with_c_claims_from_pending(
            &mut A_layers,
            &mut B_layers,
            &mut c_claims,
            &E_eq,
            left,
            right,
            &rhos,
            vc,
            vc_state,
            vc_shape,
            vc_ck,
            transcript,
            2,
            pending_r_b,
            &mut r_bs,
            &mut T_cur,
            &mut acc_eq,
          )?;
        }
      } else {
        if ell_b == 1 {
          fold_abc_pairs(&mut A_layers, &mut B_layers, &mut C_layers, pairs, r_b);
          A_layers.truncate(pairs);
          B_layers.truncate(pairs);
          C_layers.truncate(pairs);
        } else {
          let mut prev_r_b = r_b;
          let mut m = n_padded;

          for t in 1..ell_b {
            let fold_pairs = m / 2;
            let prove_pairs = fold_pairs / 2;
            let e_eq_ref = &E_eq;
            let rhos_ref = &rhos;

            let (a_head, _) = A_layers.split_at_mut(4 * prove_pairs);
            let (b_head, _) = B_layers.split_at_mut(4 * prove_pairs);
            let (c_head, _) = C_layers.split_at_mut(4 * prove_pairs);

            let (e0_acc, quad_acc) = a_head
              .par_chunks_mut(4)
              .zip(b_head.par_chunks_mut(4))
              .zip(c_head.par_chunks_mut(4))
              .enumerate()
              .map(|(j, ((a_chunk, b_chunk), c_chunk))| {
                fold_quad_chunk(a_chunk, prev_r_b);
                fold_quad_chunk(b_chunk, prev_r_b);
                fold_quad_chunk(c_chunk, prev_r_b);
                let (e0, quad_coeff) = Self::prove_helper(
                  t,
                  (left, right),
                  e_eq_ref,
                  &a_chunk[0],
                  &b_chunk[0],
                  &c_chunk[0],
                  &a_chunk[2],
                  &b_chunk[2],
                );
                let weight = suffix_weight_full::<E::Scalar>(t, ell_b, j, rhos_ref);
                (e0 * weight, quad_coeff * weight)
              })
              .reduce(
                || (E::Scalar::ZERO, E::Scalar::ZERO),
                |a, b| (a.0 + b.0, a.1 + b.1),
              );

            compact_folded_layers(&mut A_layers, prove_pairs);
            compact_folded_layers(&mut B_layers, prove_pairs);
            compact_folded_layers(&mut C_layers, prove_pairs);

            fold_abc_pairs_in_range(
              &mut A_layers,
              &mut B_layers,
              &mut C_layers,
              (2 * prove_pairs)..fold_pairs,
              prev_r_b,
            );

            A_layers.truncate(fold_pairs);
            B_layers.truncate(fold_pairs);
            C_layers.truncate(fold_pairs);
            m = fold_pairs;
            prev_r_b = finish_nifs_field_round(
              &rhos,
              t,
              e0_acc,
              quad_acc,
              vc,
              vc_state,
              vc_shape,
              vc_ck,
              transcript,
              &mut r_bs,
              &mut T_cur,
              &mut acc_eq,
            )?;
          }

          let final_pairs = m / 2;
          fold_abc_pairs(
            &mut A_layers,
            &mut B_layers,
            &mut C_layers,
            final_pairs,
            prev_r_b,
          );
          A_layers.truncate(final_pairs);
          B_layers.truncate(final_pairs);
          C_layers.truncate(final_pairs);
        }
      }
    }

    if use_round0_small_optimization {
      let final_weights = weights_from_r::<E::Scalar>(&r_bs, n_padded);
      let c_folded = fold_small_c_with_field_corrections::<E>(
        &C_small_layers,
        &C_layers,
        &c_large_positions,
        &final_weights,
      )?;
      C_layers = vec![c_folded];
    }

    finalize_nifs_step_claim(
      vc, vc_state, vc_shape, vc_ck, transcript, ell_b, T_cur, acc_eq,
    )?;

    let (folded_W, folded_U) = fold_witness_and_instance(s, ck, us, ws, n_padded, n_padded, &r_bs)?;

    Ok((
      E_eq,
      std::mem::take(&mut A_layers[0]),
      std::mem::take(&mut B_layers[0]),
      std::mem::take(&mut C_layers[0]),
      folded_W,
      folded_U,
    ))
  }
}

fn build_small_ab_from_field<E>(
  field: &FieldStepMatvecs<E>,
  l0: usize,
) -> Result<SmallAbStepMatvecs, SpartanError>
where
  E: Engine,
{
  if l0 == 0 {
    return Err(SpartanError::InvalidInputLength {
      reason: "small A/B matvec conversion requires l0 > 0".to_string(),
    });
  }

  let mut az_small = Vec::with_capacity(field.len());
  let mut bz_small = Vec::with_capacity(field.len());
  let mut large_pos_set = std::collections::BTreeSet::new();

  for (az_layer, bz_layer) in field.az.iter().zip(&field.bz) {
    let (az_small_layer, az_large_indices) = field_to_i64_or_zero_for_l0(az_layer, l0);
    let (bz_small_layer, bz_large_indices) = field_to_i64_or_zero_for_l0(bz_layer, l0);
    large_pos_set.extend(az_large_indices);
    large_pos_set.extend(bz_large_indices);
    az_small.push(az_small_layer);
    bz_small.push(bz_small_layer);
  }

  let large_positions: Vec<usize> = large_pos_set.into_iter().collect();
  if !large_positions.is_empty() {
    for (az_small, bz_small) in az_small.iter_mut().zip(bz_small.iter_mut()) {
      for &pos in &large_positions {
        az_small[pos] = 0;
        bz_small[pos] = 0;
      }
    }
  }

  Ok(SmallAbStepMatvecs {
    az_small,
    bz_small,
    large_positions,
    l0,
  })
}

fn build_small_abc_from_field<E>(
  field: &FieldStepMatvecs<E>,
  l0: usize,
) -> Result<SmallAbcStepMatvecs, SpartanError>
where
  E: Engine,
  E::Scalar: SmallValueField<i64>,
{
  let ab = build_small_ab_from_field(field, l0)?;
  let mut cz_small = Vec::with_capacity(field.len());
  let mut c_large_pos_set = std::collections::BTreeSet::new();

  for cz_layer in &field.cz {
    let (cz_small_layer, cz_large_indices) = to_small_vec_or_zero(cz_layer);
    c_large_pos_set.extend(cz_large_indices);
    cz_small.push(cz_small_layer);
  }

  let c_large_positions = c_large_pos_set.into_iter().collect::<Vec<_>>();
  if !c_large_positions.is_empty() {
    for cz_small in &mut cz_small {
      for &pos in &c_large_positions {
        cz_small[pos] = 0;
      }
    }
  }

  let small_abc = SmallAbcStepMatvecs {
    ab,
    cz_small,
    c_large_positions,
  };
  #[cfg(debug_assertions)]
  debug_validate_small_abc_cache::<E>(field, &small_abc);
  Ok(small_abc)
}

#[cfg(debug_assertions)]
pub(crate) fn debug_validate_small_value_step_matvecs<E>(
  step_matvecs: &SmallValueNeutronNovaStepMatvecs<E>,
) where
  E: Engine,
  E::Scalar: SmallValueField<i64>,
{
  debug_validate_small_abc_cache::<E>(&step_matvecs.field, &step_matvecs.small_abc);
}

#[cfg(debug_assertions)]
fn debug_validate_small_abc_cache<E>(field: &FieldStepMatvecs<E>, small_abc: &SmallAbcStepMatvecs)
where
  E: Engine,
  E::Scalar: SmallValueField<i64>,
{
  debug_assert_eq!(
    field.az.len(),
    field.bz.len(),
    "field Az/Bz layer count mismatch"
  );
  debug_assert_eq!(
    field.az.len(),
    field.cz.len(),
    "field Az/Cz layer count mismatch"
  );
  debug_assert_eq!(
    field.az.len(),
    small_abc.ab.az_small.len(),
    "small Az layer count mismatch"
  );
  debug_assert_eq!(
    field.az.len(),
    small_abc.ab.bz_small.len(),
    "small Bz layer count mismatch"
  );
  debug_assert_eq!(
    field.az.len(),
    small_abc.cz_small.len(),
    "small Cz layer count mismatch"
  );

  let Some(first_layer) = field.az.first() else {
    return;
  };
  let layer_len = first_layer.len();
  debug_assert!(
    small_abc
      .ab
      .large_positions
      .windows(2)
      .all(|pair| pair[0] < pair[1]),
    "A/B large positions must be sorted and unique"
  );
  for &pos in &small_abc.ab.large_positions {
    debug_assert!(
      pos < layer_len,
      "A/B large position {} is out of range for layer length {}",
      pos,
      layer_len
    );
  }
  debug_assert!(
    small_abc
      .c_large_positions
      .windows(2)
      .all(|pair| pair[0] < pair[1]),
    "C large positions must be sorted and unique"
  );
  for &pos in &small_abc.c_large_positions {
    debug_assert!(
      pos < layer_len,
      "C large position {} is out of range for layer length {}",
      pos,
      layer_len
    );
  }

  let ab_large_positions = small_abc
    .ab
    .large_positions
    .iter()
    .copied()
    .collect::<std::collections::BTreeSet<_>>();
  let c_large_positions = small_abc
    .c_large_positions
    .iter()
    .copied()
    .collect::<std::collections::BTreeSet<_>>();

  for layer_idx in 0..field.az.len() {
    let az_field = &field.az[layer_idx];
    let bz_field = &field.bz[layer_idx];
    let cz_field = &field.cz[layer_idx];
    let az_small = &small_abc.ab.az_small[layer_idx];
    let bz_small = &small_abc.ab.bz_small[layer_idx];
    let cz_small = &small_abc.cz_small[layer_idx];

    debug_assert_eq!(
      az_field.len(),
      layer_len,
      "field Az layer {} length mismatch",
      layer_idx
    );
    debug_assert_eq!(
      bz_field.len(),
      layer_len,
      "field Bz layer {} length mismatch",
      layer_idx
    );
    debug_assert_eq!(
      cz_field.len(),
      layer_len,
      "field Cz layer {} length mismatch",
      layer_idx
    );
    debug_assert_eq!(
      az_small.len(),
      layer_len,
      "small Az layer {} length mismatch",
      layer_idx
    );
    debug_assert_eq!(
      bz_small.len(),
      layer_len,
      "small Bz layer {} length mismatch",
      layer_idx
    );
    debug_assert_eq!(
      cz_small.len(),
      layer_len,
      "small Cz layer {} length mismatch",
      layer_idx
    );

    for k in 0..layer_len {
      if ab_large_positions.contains(&k) {
        debug_assert_eq!(
          az_small[k], 0,
          "A/B large-position small Az entry must be zero at layer {} index {}",
          layer_idx, k
        );
        debug_assert_eq!(
          bz_small[k], 0,
          "A/B large-position small Bz entry must be zero at layer {} index {}",
          layer_idx, k
        );
      } else {
        debug_assert_eq!(
          <E::Scalar as SmallValueField<i64>>::small_to_field(az_small[k]),
          az_field[k],
          "small Az cache mismatch at layer {} index {}",
          layer_idx,
          k
        );
        debug_assert_eq!(
          <E::Scalar as SmallValueField<i64>>::small_to_field(bz_small[k]),
          bz_field[k],
          "small Bz cache mismatch at layer {} index {}",
          layer_idx,
          k
        );
      }

      if c_large_positions.contains(&k) {
        debug_assert_eq!(
          cz_small[k], 0,
          "C large-position small Cz entry must be zero at layer {} index {}",
          layer_idx, k
        );
      } else {
        debug_assert_eq!(
          <E::Scalar as SmallValueField<i64>>::small_to_field(cz_small[k]),
          cz_field[k],
          "small Cz cache mismatch at layer {} index {}",
          layer_idx,
          k
        );
      }

      debug_assert_eq!(
        az_field[k] * bz_field[k],
        cz_field[k],
        "field Az*Bz != Cz at layer {} index {}",
        layer_idx,
        k
      );
    }
  }
}

impl<E> NeutronNovaNifsStrategy<E> for SmallValueNeutronNovaNIFS<E>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueEngine<i64> + Default,
{
  type Input = usize;
  type StepMatvecs = SmallValueNeutronNovaStepMatvecs<E>;

  fn is_small(_: &Self::Input) -> bool {
    true
  }

  fn build_step_matvecs_from_field(
    field_matvecs: Vec<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>)>,
    input: &Self::Input,
  ) -> Result<Self::StepMatvecs, SpartanError> {
    if *input == 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: "small-value NeutronNova NIFS requires l0 > 0".to_string(),
      });
    }

    let field = FieldStepMatvecs::from_triples(field_matvecs);
    let small_abc = build_small_abc_from_field(&field, *input)?;
    Ok(SmallValueNeutronNovaStepMatvecs { field, small_abc })
  }

  #[allow(clippy::too_many_arguments)]
  fn prove(
    s: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    us: Vec<R1CSInstance<E>>,
    ws: Vec<R1CSWitness<E>>,
    step_matvecs: Self::StepMatvecs,
    nifs_input: &Self::Input,
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<NeutronNovaNifsOutput<E>, SpartanError> {
    crate::small_neutronnova::prove::<E>(
      s,
      ck,
      us,
      ws,
      &step_matvecs,
      *nifs_input,
      vc,
      vc_state,
      vc_shape,
      vc_ck,
      transcript,
    )
  }
}

pub(crate) struct NifsTranscriptState<E: Engine> {
  pub n_padded: usize,
  pub ell_b: usize,
  pub e_eq: Vec<E::Scalar>,
  pub left: usize,
  pub right: usize,
  pub rhos: Vec<E::Scalar>,
}

pub(crate) fn prepare_nifs_transcript<E>(
  s: &SplitR1CSShape<E>,
  us: &[R1CSInstance<E>],
  transcript: &mut E::TE,
) -> Result<NifsTranscriptState<E>, SpartanError>
where
  E: Engine,
{
  if us.is_empty() {
    return Err(invalid_input(
      "NeutronNova NIFS transcript requires at least one instance",
    ));
  }

  let n_padded = us.len().next_power_of_two();
  for idx in 0..n_padded {
    let u = if idx < us.len() { &us[idx] } else { &us[0] };
    transcript.absorb(b"U", u);
  }
  let t = E::Scalar::ZERO;
  transcript.absorb(b"T", &t);

  let (ell_cons, left, right) = compute_tensor_decomp(s.num_cons);
  let tau = transcript.squeeze(b"tau")?;
  let e_eq = PowPolynomial::split_evals(tau, ell_cons, left, right);
  let ell_b = n_padded.log_2();
  let mut rhos = Vec::with_capacity(ell_b);
  for _ in 0..ell_b {
    rhos.push(transcript.squeeze(b"rho")?);
  }

  Ok(NifsTranscriptState {
    n_padded,
    ell_b,
    e_eq,
    left,
    right,
    rhos,
  })
}
/// Build the cubic NIFS round polynomial from the current claim and round claims.
pub(crate) fn generate_nifs_field_round_polynomial<F>(
  rho: F,
  acc_eq: F,
  t_cur: F,
  e0: F,
  quad_coeff: F,
) -> Result<UniPoly<F>, SpartanError>
where
  F: PrimeField,
{
  let linear_at_zero = F::ONE - rho;
  let linear_at_one = rho;
  let linear_at_infinity = rho - linear_at_zero;
  // Scale the pairwise round claims by the eq accumulator from previous rounds.
  let quadratic_at_zero = e0 * acc_eq;
  let quadratic_at_infinity = quad_coeff * acc_eq;

  build_linear_times_quadratic_poly_from_claim(
    linear_at_zero,
    linear_at_one,
    linear_at_infinity,
    t_cur,
    quadratic_at_zero,
    quadratic_at_infinity,
  )
  .ok_or(SpartanError::DivisionByZero)
}

pub(crate) fn fold_layer_pair_into<F: Field>(
  layers: &mut [Vec<F>],
  src_even: usize,
  src_odd: usize,
  dest: usize,
  r: F,
) {
  let even = std::mem::take(&mut layers[src_even]);
  let odd = &layers[src_odd];
  let mut folded = even;
  folded
    .iter_mut()
    .zip(odd.iter())
    .for_each(|(lo, hi)| *lo += r * (*hi - *lo));
  layers[dest] = folded;
}

pub(crate) fn fold_abc_pair_into<F: Field>(
  a_layers: &mut [Vec<F>],
  b_layers: &mut [Vec<F>],
  c_layers: &mut [Vec<F>],
  src_even: usize,
  src_odd: usize,
  dest: usize,
  r: F,
) {
  fold_layer_pair_into(a_layers, src_even, src_odd, dest, r);
  fold_layer_pair_into(b_layers, src_even, src_odd, dest, r);
  fold_layer_pair_into(c_layers, src_even, src_odd, dest, r);
}

fn fold_abc_pairs<F: Field>(
  a_layers: &mut [Vec<F>],
  b_layers: &mut [Vec<F>],
  c_layers: &mut [Vec<F>],
  pairs: usize,
  r: F,
) {
  fold_abc_pairs_in_range(a_layers, b_layers, c_layers, 0..pairs, r);
}

fn fold_abc_pairs_in_range<F: Field>(
  a_layers: &mut [Vec<F>],
  b_layers: &mut [Vec<F>],
  c_layers: &mut [Vec<F>],
  pair_range: std::ops::Range<usize>,
  r: F,
) {
  for i in pair_range {
    fold_abc_pair_into(a_layers, b_layers, c_layers, 2 * i, 2 * i + 1, i, r);
  }
}

pub(crate) fn fold_quad_chunk<F: Field>(chunk: &mut [Vec<F>], r: F) {
  {
    let (lo, hi) = chunk.split_at_mut(1);
    lo[0]
      .iter_mut()
      .zip(hi[0].iter())
      .for_each(|(l, h)| *l += r * (*h - *l));
  }
  {
    let (lo, hi) = chunk.split_at_mut(3);
    lo[2]
      .iter_mut()
      .zip(hi[0].iter())
      .for_each(|(l, h)| *l += r * (*h - *l));
  }
}

pub(crate) fn compact_folded_layers<F>(layers: &mut [Vec<F>], prove_pairs: usize) {
  for j in 0..prove_pairs {
    layers.swap(2 * j, 4 * j);
    layers.swap(2 * j + 1, 4 * j + 2);
  }
}

pub(crate) fn invalid_input(reason: impl Into<String>) -> SpartanError {
  SpartanError::InvalidInputLength {
    reason: reason.into(),
  }
}

pub(crate) fn validate_instance_witness_counts<E>(
  num_instances: usize,
  us: &[R1CSInstance<E>],
  ws: &[R1CSWitness<E>],
) -> Result<(), SpartanError>
where
  E: Engine,
{
  if us.len() != num_instances {
    return Err(invalid_input(format!(
      "instance count {} does not match num_instances {}",
      us.len(),
      num_instances
    )));
  }
  if ws.len() != num_instances {
    return Err(invalid_input(format!(
      "witness count {} does not match num_instances {}",
      ws.len(),
      num_instances
    )));
  }
  Ok(())
}

/// Record one NIFS polynomial in the verifier circuit and return its challenge.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_nifs_round<E>(
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
  round: usize,
  poly: &UniPoly<E::Scalar>,
) -> Result<E::Scalar, SpartanError>
where
  E: Engine,
{
  let coeffs = &poly.coeffs;
  // The verifier circuit stores dense cubic coefficients as public round state.
  vc.nifs_polys[round] = [coeffs[0], coeffs[1], coeffs[2], coeffs[3]];

  let chals =
    SatisfyingAssignment::<E>::process_round(vc_state, vc_shape, vc_ck, vc, round, transcript)?;
  chals
    .first()
    .copied()
    .ok_or_else(|| SpartanError::InternalError {
      reason: format!("NeutronNova NIFS round {} produced no challenge", round),
    })
}

/// Complete one ordinary field-backed NIFS round and advance running claims.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_nifs_field_round<E>(
  rhos: &[E::Scalar],
  round: usize,
  e0: E::Scalar,
  quad_coeff: E::Scalar,
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
  r_bs: &mut Vec<E::Scalar>,
  t_cur: &mut E::Scalar,
  acc_eq: &mut E::Scalar,
) -> Result<E::Scalar, SpartanError>
where
  E: Engine,
{
  let rho = rhos[round];
  let poly = generate_nifs_field_round_polynomial(rho, *acc_eq, *t_cur, e0, quad_coeff)?;
  let r_b = process_nifs_round(vc, vc_state, vc_shape, vc_ck, transcript, round, &poly)?;
  // Carry the sumcheck claim and eq(rho, r_b) accumulator into the next round.
  *t_cur = poly.evaluate(&r_b);
  *acc_eq *= (E::Scalar::ONE - r_b) * (E::Scalar::ONE - rho) + r_b * rho;
  r_bs.push(r_b);
  Ok(r_b)
}

/// Publish the final normalized step claim and consume the final verifier round.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_nifs_step_claim<E>(
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
  final_round: usize,
  t_cur: E::Scalar,
  acc_eq: E::Scalar,
) -> Result<(), SpartanError>
where
  E: Engine,
{
  let acc_eq_inv: Option<E::Scalar> = acc_eq.invert().into();
  // t_cur includes the accumulated eq factor; the verifier circuit stores T_out.
  vc.t_out_step = t_cur * acc_eq_inv.ok_or(SpartanError::DivisionByZero)?;
  vc.eq_rho_at_rb = acc_eq;
  let _ = SatisfyingAssignment::<E>::process_round(
    vc_state,
    vc_shape,
    vc_ck,
    vc,
    final_round,
    transcript,
  )?;
  Ok(())
}

/// Compute the weighted field claims for one NIFS round from materialized A/B/C layers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_field_round_claim<E>(
  a_layers: &[Vec<E::Scalar>],
  b_layers: &[Vec<E::Scalar>],
  c_layers: &[Vec<E::Scalar>],
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
  rhos: &[E::Scalar],
  round: usize,
) -> Result<(E::Scalar, E::Scalar), SpartanError>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: DelayedReduction<E::Scalar>,
{
  let shape = NifsRoundShape {
    e_eq,
    left,
    right,
    rhos,
    round,
  };
  validate_materialized_c_round_inputs(shape, a_layers, b_layers, c_layers)?;
  Ok(compute_weighted_round_claims::<E, _, _>(
    a_layers,
    b_layers,
    c_layers,
    rhos,
    round,
    |pair_a, pair_b, pair_c| {
      let (e0, quad_coeff) = NeutronNovaNIFS::<E>::prove_helper(
        round,
        (left, right),
        e_eq,
        &pair_a[0],
        &pair_b[0],
        &pair_c[0],
        &pair_a[1],
        &pair_b[1],
      );
      (e0, quad_coeff)
    },
  ))
}

#[derive(Clone, Copy)]
struct NifsRoundShape<'a, F> {
  e_eq: &'a [F],
  left: usize,
  right: usize,
  rhos: &'a [F],
  round: usize,
}

impl<F> NifsRoundShape<'_, F>
where
  F: Field,
{
  fn expected_layer_len(self) -> Result<usize, SpartanError> {
    if self.round >= self.rhos.len() {
      return Err(invalid_input(format!(
        "round {} is out of range for {} rho challenges",
        self.round,
        self.rhos.len()
      )));
    }

    let expected_len = self
      .left
      .checked_mul(self.right)
      .ok_or_else(|| invalid_input("left * right overflows"))?;
    let expected_eq_len = self
      .left
      .checked_add(self.right)
      .ok_or_else(|| invalid_input("left + right overflows"))?;
    if self.e_eq.len() != expected_eq_len {
      return Err(invalid_input(format!(
        "E_eq length {} does not match left + right {}",
        self.e_eq.len(),
        expected_eq_len
      )));
    }

    Ok(expected_len)
  }
}

fn validate_layer_family<F>(
  label: &str,
  layers: &[Vec<F>],
  expected_len: usize,
) -> Result<(), SpartanError> {
  if !layers.iter().all(|layer| layer.len() == expected_len) {
    return Err(invalid_input(format!(
      "all {} layers must have length {}",
      label, expected_len
    )));
  }
  Ok(())
}

fn validate_materialized_c_round_inputs<F>(
  shape: NifsRoundShape<'_, F>,
  a_layers: &[Vec<F>],
  b_layers: &[Vec<F>],
  c_layers: &[Vec<F>],
) -> Result<(), SpartanError>
where
  F: Field,
{
  if a_layers.len() != b_layers.len() || a_layers.len() != c_layers.len() {
    return Err(invalid_input("A/B/C layer counts do not match"));
  }
  if a_layers.is_empty() || !a_layers.len().is_multiple_of(2) {
    return Err(invalid_input(
      "round claim layer count must be non-empty and even",
    ));
  }
  let expected_len = shape.expected_layer_len()?;
  validate_layer_family("A", a_layers, expected_len)?;
  validate_layer_family("B", b_layers, expected_len)?;
  validate_layer_family("C", c_layers, expected_len)?;
  Ok(())
}

fn validate_scalar_c_fold_inputs<F>(
  a_layers: &[Vec<F>],
  b_layers: &[Vec<F>],
  c_claims: &[F],
) -> Result<(), SpartanError> {
  if a_layers.len() != b_layers.len() || a_layers.len() != c_claims.len() {
    return Err(invalid_input("A/B layer and C-claim counts do not match"));
  }
  if a_layers.is_empty() || !a_layers.len().is_multiple_of(2) {
    return Err(invalid_input(
      "suffix layer count must be non-empty and even",
    ));
  }
  Ok(())
}

fn validate_scalar_c_round_inputs<F>(
  shape: NifsRoundShape<'_, F>,
  a_layers: &[Vec<F>],
  b_layers: &[Vec<F>],
  c_claims: &[F],
) -> Result<(), SpartanError>
where
  F: Field,
{
  validate_scalar_c_fold_inputs(a_layers, b_layers, c_claims)?;
  let expected_len = shape.expected_layer_len()?;
  validate_layer_family("A", a_layers, expected_len)?;
  validate_layer_family("B", b_layers, expected_len)?;
  Ok(())
}

fn compute_weighted_round_claims<E, C, ClaimFn>(
  a_layers: &[Vec<E::Scalar>],
  b_layers: &[Vec<E::Scalar>],
  c_data: &[C],
  rhos: &[E::Scalar],
  round: usize,
  claim_for_pair: ClaimFn,
) -> (E::Scalar, E::Scalar)
where
  E: Engine,
  C: Sync,
  ClaimFn: Fn(&[Vec<E::Scalar>], &[Vec<E::Scalar>], &[C]) -> (E::Scalar, E::Scalar) + Sync,
{
  let ell_b = rhos.len();
  a_layers
    .par_chunks(2)
    .zip(b_layers.par_chunks(2))
    .zip(c_data.par_chunks(2))
    .enumerate()
    .map(|(pair_idx, ((pair_a, pair_b), pair_c))| {
      let (e0, quad_coeff) = claim_for_pair(pair_a, pair_b, pair_c);
      let w = suffix_weight_full::<E::Scalar>(round, ell_b, pair_idx, rhos);
      (e0 * w, quad_coeff * w)
    })
    .reduce(
      || (E::Scalar::ZERO, E::Scalar::ZERO),
      |a, b| (a.0 + b.0, a.1 + b.1),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_ab_c_claim_round<E>(
  a_layers: &[Vec<E::Scalar>],
  b_layers: &[Vec<E::Scalar>],
  c_claims: &[E::Scalar],
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
  rhos: &[E::Scalar],
  round: usize,
) -> Result<(E::Scalar, E::Scalar), SpartanError>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: DelayedReduction<E::Scalar>,
{
  let shape = NifsRoundShape {
    e_eq,
    left,
    right,
    rhos,
    round,
  };
  validate_scalar_c_round_inputs(shape, a_layers, b_layers, c_claims)?;

  Ok(compute_weighted_round_claims::<E, _, _>(
    a_layers,
    b_layers,
    c_claims,
    rhos,
    round,
    |pair_a, pair_b, pair_c| {
      let (e0_ab, quad_coeff) = NeutronNovaNIFS::<E>::prove_helper_ab_only(
        (left, right),
        e_eq,
        &pair_a[0],
        &pair_b[0],
        &pair_a[1],
        &pair_b[1],
      );
      let e0 = e0_ab - pair_c[0];
      (e0, quad_coeff)
    },
  ))
}

#[cfg(test)]
fn dot_field_layer_with_split_eq<E>(
  layer: &[E::Scalar],
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
) -> Result<E::Scalar, SpartanError>
where
  E: Engine,
  E::Scalar: DelayedReduction<E::Scalar>,
{
  if e_eq.len() != left + right {
    return Err(invalid_input("split equality table has wrong length"));
  }
  if layer.len() != left * right {
    return Err(invalid_input("field layer has wrong length"));
  }

  type Acc<S> = <S as DelayedReduction<S>>::Accumulator;

  let e_left = &e_eq[..left];
  let e_right = &e_eq[left..];
  let mut acc = Acc::<E::Scalar>::default();

  for (i, e_right_i) in e_right.iter().enumerate().take(right) {
    let base = i * left;
    let mut inner = Acc::<E::Scalar>::default();
    for j in 0..left {
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut inner,
        &e_left[j],
        &layer[base + j],
      );
    }
    let inner_red = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&inner);
    <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
      &mut acc, e_right_i, &inner_red,
    );
  }

  Ok(<E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc))
}

#[cfg(test)]
fn compute_field_c_claims<E>(
  c_layers: &[Vec<E::Scalar>],
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
) -> Result<Vec<E::Scalar>, SpartanError>
where
  E: Engine,
  E::Scalar: DelayedReduction<E::Scalar>,
{
  c_layers
    .par_iter()
    .map(|layer| dot_field_layer_with_split_eq::<E>(layer, e_eq, left, right))
    .collect()
}

fn compute_small_c_claims_with_field_corrections<E>(
  c_small_layers: &[Vec<i64>],
  c_field_layers: &[Vec<E::Scalar>],
  large_positions: &[usize],
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
) -> Result<Vec<E::Scalar>, SpartanError>
where
  E: Engine,
  E::Scalar: DelayedReduction<i128> + DelayedReduction<E::Scalar>,
{
  if c_small_layers.is_empty() {
    return Err(invalid_input(
      "cannot compute C claims for empty layer list",
    ));
  }
  if c_small_layers.len() != c_field_layers.len() {
    return Err(invalid_input(
      "small C and field C layer counts do not match",
    ));
  }
  let expected_len = left
    .checked_mul(right)
    .ok_or_else(|| invalid_input("left * right overflows"))?;
  let expected_eq_len = left
    .checked_add(right)
    .ok_or_else(|| invalid_input("left + right overflows"))?;
  if e_eq.len() != expected_eq_len {
    return Err(invalid_input(format!(
      "E_eq length {} does not match left + right {}",
      e_eq.len(),
      expected_eq_len
    )));
  }
  validate_layer_family("small C", c_small_layers, expected_len)?;
  validate_layer_family("field C", c_field_layers, expected_len)?;

  let e_left = &e_eq[..left];
  let e_right = &e_eq[left..];
  let mut vals: Vec<E::Scalar> = c_small_layers
    .par_iter()
    .map(|c_small| {
      type Acc<S> = <S as DelayedReduction<S>>::Accumulator;
      let mut acc = Acc::<E::Scalar>::default();

      for (i, e_right_i) in e_right.iter().enumerate().take(right) {
        let base = i * left;
        let mut inner = <E::Scalar as DelayedReduction<i128>>::Accumulator::default();
        for j in 0..left {
          <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
            &mut inner,
            &e_left[j],
            &(c_small[base + j] as i128),
          );
        }
        let inner_red = <E::Scalar as DelayedReduction<i128>>::reduce(&inner);
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut acc, e_right_i, &inner_red,
        );
      }

      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc)
    })
    .collect();

  if !large_positions.is_empty() {
    for &k in large_positions {
      if k >= expected_len {
        continue;
      }
      let i = k / left;
      let j = k % left;
      let eq_at_k = e_left[j] * e_right[i];
      for (claim, c_field) in vals.iter_mut().zip(c_field_layers.iter()) {
        *claim += eq_at_k * c_field[k];
      }
    }
  }

  Ok(vals)
}

#[cfg(test)]
fn fold_field_layers_with_weights<E>(
  layers: &[Vec<E::Scalar>],
  weights: &[E::Scalar],
) -> Result<Vec<E::Scalar>, SpartanError>
where
  E: Engine,
  E::Scalar: DelayedReduction<E::Scalar>,
{
  if layers.is_empty() {
    return Err(invalid_input("cannot fold empty field layer list"));
  }
  if layers.len() != weights.len() {
    return Err(invalid_input("field layer and weight counts do not match"));
  }
  let layer_len = layers[0].len();
  validate_layer_family("field", layers, layer_len)?;

  let folded = (0..layer_len)
    .into_par_iter()
    .map(|k| {
      type Acc<S> = <S as DelayedReduction<S>>::Accumulator;
      let mut acc = Acc::<E::Scalar>::default();
      for (weight, layer) in weights.iter().zip(layers.iter()) {
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut acc, weight, &layer[k],
        );
      }
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc)
    })
    .collect();

  Ok(folded)
}

fn fold_small_c_with_field_corrections<E>(
  c_small_layers: &[Vec<i64>],
  c_field_layers: &[Vec<E::Scalar>],
  large_positions: &[usize],
  final_weights: &[E::Scalar],
) -> Result<Vec<E::Scalar>, SpartanError>
where
  E: Engine,
  E::Scalar: DelayedReduction<i128>,
{
  if c_small_layers.is_empty() {
    return Err(invalid_input("cannot fold empty small C layer list"));
  }
  if c_small_layers.len() != c_field_layers.len() || c_small_layers.len() != final_weights.len() {
    return Err(invalid_input(
      "small C, field C, and weight counts do not match",
    ));
  }
  let layer_len = c_small_layers[0].len();
  validate_layer_family("small C", c_small_layers, layer_len)?;
  validate_layer_family("field C", c_field_layers, layer_len)?;

  let mut folded: Vec<E::Scalar> = (0..layer_len)
    .into_par_iter()
    .map(|k| {
      let mut acc = <E::Scalar as DelayedReduction<i128>>::Accumulator::default();
      for (weight, c_small) in final_weights.iter().zip(c_small_layers.iter()) {
        <E::Scalar as DelayedReduction<i128>>::unreduced_multiply_accumulate(
          &mut acc,
          weight,
          &(c_small[k] as i128),
        );
      }
      <E::Scalar as DelayedReduction<i128>>::reduce(&acc)
    })
    .collect();

  if !large_positions.is_empty() {
    for &k in large_positions {
      if k >= layer_len {
        continue;
      }
      let mut val = E::Scalar::ZERO;
      for (weight, c_field) in final_weights.iter().zip(c_field_layers.iter()) {
        val += *weight * c_field[k];
      }
      folded[k] = val;
    }
  }

  Ok(folded)
}

fn fold_ab_c_claim_pairs<E>(
  a_layers: &mut [Vec<E::Scalar>],
  b_layers: &mut [Vec<E::Scalar>],
  c_claims: &mut [E::Scalar],
  pairs: usize,
  r: E::Scalar,
) where
  E: Engine,
{
  fold_ab_c_claim_pairs_in_range::<E>(a_layers, b_layers, c_claims, 0..pairs, r);
}

fn fold_ab_c_claim_pairs_in_range<E>(
  a_layers: &mut [Vec<E::Scalar>],
  b_layers: &mut [Vec<E::Scalar>],
  c_claims: &mut [E::Scalar],
  pair_range: std::ops::Range<usize>,
  r: E::Scalar,
) where
  E: Engine,
{
  for i in pair_range {
    fold_layer_pair_into(a_layers, 2 * i, 2 * i + 1, i, r);
    fold_layer_pair_into(b_layers, 2 * i, 2 * i + 1, i, r);
    fold_c_claim_pair_into(c_claims, 2 * i, 2 * i + 1, i, r);
  }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn continue_ab_suffix_with_c_claims<E>(
  a_layers: &mut Vec<Vec<E::Scalar>>,
  b_layers: &mut Vec<Vec<E::Scalar>>,
  c_claims: &mut Vec<E::Scalar>,
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
  rhos: &[E::Scalar],
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
  start_round: usize,
  r_bs: &mut Vec<E::Scalar>,
  t_cur: &mut E::Scalar,
  acc_eq: &mut E::Scalar,
) -> Result<(), SpartanError>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: DelayedReduction<E::Scalar>,
{
  if start_round >= rhos.len() {
    return Ok(());
  }

  let (e0, quad_coeff) = compute_ab_c_claim_round::<E>(
    a_layers,
    b_layers,
    c_claims,
    e_eq,
    left,
    right,
    rhos,
    start_round,
  )?;
  let pending_r_b = finish_nifs_field_round(
    rhos,
    start_round,
    e0,
    quad_coeff,
    vc,
    vc_state,
    vc_shape,
    vc_ck,
    transcript,
    r_bs,
    t_cur,
    acc_eq,
  )?;

  continue_ab_suffix_with_c_claims_from_pending(
    a_layers,
    b_layers,
    c_claims,
    e_eq,
    left,
    right,
    rhos,
    vc,
    vc_state,
    vc_shape,
    vc_ck,
    transcript,
    start_round + 1,
    pending_r_b,
    r_bs,
    t_cur,
    acc_eq,
  )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn continue_ab_suffix_with_c_claims_from_pending<E>(
  a_layers: &mut Vec<Vec<E::Scalar>>,
  b_layers: &mut Vec<Vec<E::Scalar>>,
  c_claims: &mut Vec<E::Scalar>,
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
  rhos: &[E::Scalar],
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
  start_round: usize,
  mut pending_r_b: E::Scalar,
  r_bs: &mut Vec<E::Scalar>,
  t_cur: &mut E::Scalar,
  acc_eq: &mut E::Scalar,
) -> Result<(), SpartanError>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: DelayedReduction<E::Scalar>,
{
  let mut layer_count = a_layers.len();
  for round in start_round..rhos.len() {
    let shape = NifsRoundShape {
      e_eq,
      left,
      right,
      rhos,
      round,
    };
    validate_scalar_c_round_inputs(shape, a_layers, b_layers, c_claims)?;
    let fold_pairs = layer_count / 2;
    let prove_pairs = fold_pairs / 2;
    if prove_pairs == 0 || 4 * prove_pairs != layer_count {
      return Err(invalid_input(
        "merged suffix round requires layer count divisible by 4",
      ));
    }

    let (a_head, _) = a_layers.split_at_mut(4 * prove_pairs);
    let (b_head, _) = b_layers.split_at_mut(4 * prove_pairs);
    let c_head = &mut c_claims[..4 * prove_pairs];

    let (e0, quad_coeff) = a_head
      .par_chunks_mut(4)
      .zip(b_head.par_chunks_mut(4))
      .zip(c_head.par_chunks_mut(4))
      .enumerate()
      .map(|(pair_idx, ((a_chunk, b_chunk), c_chunk))| {
        fold_quad_chunk(a_chunk, pending_r_b);
        fold_quad_chunk(b_chunk, pending_r_b);
        fold_quad_c_claim_chunk(c_chunk, pending_r_b);

        let (e0_ab, quad_coeff) = NeutronNovaNIFS::<E>::prove_helper_ab_only(
          (left, right),
          e_eq,
          &a_chunk[0],
          &b_chunk[0],
          &a_chunk[2],
          &b_chunk[2],
        );
        let e0 = e0_ab - c_chunk[0];
        let w = suffix_weight_full::<E::Scalar>(round, rhos.len(), pair_idx, rhos);
        (e0 * w, quad_coeff * w)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1),
      );

    compact_folded_layers(a_layers, prove_pairs);
    compact_folded_layers(b_layers, prove_pairs);
    compact_c_claims(c_claims, prove_pairs);
    a_layers.truncate(fold_pairs);
    b_layers.truncate(fold_pairs);
    c_claims.truncate(fold_pairs);
    layer_count = fold_pairs;

    pending_r_b = finish_nifs_field_round(
      rhos, round, e0, quad_coeff, vc, vc_state, vc_shape, vc_ck, transcript, r_bs, t_cur, acc_eq,
    )?;
  }

  validate_scalar_c_fold_inputs(a_layers, b_layers, c_claims)?;
  let final_pairs = layer_count / 2;
  fold_ab_c_claim_pairs::<E>(a_layers, b_layers, c_claims, final_pairs, pending_r_b);
  a_layers.truncate(final_pairs);
  b_layers.truncate(final_pairs);
  c_claims.truncate(final_pairs);

  Ok(())
}

fn fold_quad_c_claim_chunk<F: Field>(chunk: &mut [F], r: F) {
  chunk[0] += r * (chunk[1] - chunk[0]);
  chunk[2] += r * (chunk[3] - chunk[2]);
}

fn compact_c_claims<F>(claims: &mut [F], prove_pairs: usize) {
  for j in 0..prove_pairs {
    claims.swap(2 * j, 4 * j);
    claims.swap(2 * j + 1, 4 * j + 2);
  }
}

fn fold_c_claim_pair_into<F: Field>(
  c_claims: &mut [F],
  src_even: usize,
  src_odd: usize,
  dest: usize,
  r: F,
) {
  let even = c_claims[src_even];
  let odd = c_claims[src_odd];
  c_claims[dest] = even + r * (odd - even);
}

pub(crate) fn fold_witness_and_instance<E>(
  s: &SplitR1CSShape<E>,
  ck: &CommitmentKey<E>,
  mut us: Vec<R1CSInstance<E>>,
  mut ws: Vec<R1CSWitness<E>>,
  num_instances: usize,
  n_padded: usize,
  r_bs: &[E::Scalar],
) -> Result<(R1CSWitness<E>, R1CSInstance<E>), SpartanError>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
{
  validate_instance_witness_counts(num_instances, &us, &ws)?;
  if us.is_empty() {
    return Err(invalid_input("cannot fold empty instance list"));
  }

  if us.len() < n_padded {
    let us_additional = n_padded - us.len();
    let ws_additional = n_padded - ws.len();
    extend_with_first_clones(&mut us, us_additional);
    extend_with_first_clones(&mut ws, ws_additional);
  }

  let effective_len = s.num_shared + s.num_precommitted;
  let use_truncated_fold = effective_len > 0;
  if use_truncated_fold {
    for w in ws.iter_mut() {
      w.W.truncate(effective_len);
    }
  }

  let (_fold_final_span, fold_final_t) = start_span!("fold_witnesses");
  let mut folded_w = R1CSWitness::fold_multiple(r_bs, &ws)?;
  if use_truncated_fold {
    let full_dim = s.num_shared + s.num_precommitted + s.num_rest;
    folded_w.W.resize(full_dim, E::Scalar::ZERO);
  }
  info!(elapsed_ms = %fold_final_t.elapsed().as_millis(), "fold_witnesses");

  let (_fold_final_span, fold_final_t) = start_span!("fold_instances");
  let weights = weights_from_r::<E::Scalar>(r_bs, us.len());
  let x_len = us[0].X.len();
  let mut folded_x = vec![E::Scalar::ZERO; x_len];
  for (weight, instance) in weights.iter().zip(us.iter()) {
    for (acc, value) in folded_x.iter_mut().zip(instance.X.iter()) {
      *acc += *weight * *value;
    }
  }

  let comms = us
    .iter()
    .map(|instance| instance.comm_W.clone())
    .collect::<Vec<_>>();
  let folded_comm = if use_truncated_fold {
    let num_data_rows = (s.num_shared + s.num_precommitted).div_ceil(DEFAULT_COMMITMENT_WIDTH);
    <E::PCS as FoldingEngineTrait<E>>::fold_commitments_partial(
      &comms,
      &weights,
      num_data_rows,
      &folded_w.r_W,
      ck,
    )?
  } else {
    <E::PCS as FoldingEngineTrait<E>>::fold_commitments(&comms, &weights)?
  };
  let folded_u = R1CSInstance::<E>::new_unchecked(folded_comm, folded_x)?;
  info!(elapsed_ms = %fold_final_t.elapsed().as_millis(), "fold_instances");

  Ok((folded_w, folded_u))
}

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaProverKey<E: Engine> {
  ck: CommitmentKey<E>,
  S_step: SplitR1CSShape<E>,
  S_core: SplitR1CSShape<E>,
  vk_digest: SpartanDigest, // digest of the verifier's key
  vc_shape: SplitMultiRoundR1CSShape<E>,
  vc_shape_regular: R1CSShape<E>,
  vc_ck: CommitmentKey<E>,
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaVerifierKey<E: Engine> {
  ck: CommitmentKey<E>,
  vk_ee: <E::PCS as PCSEngineTrait<E>>::VerifierKey,
  S_step: SplitR1CSShape<E>,
  S_core: SplitR1CSShape<E>,
  vc_shape: SplitMultiRoundR1CSShape<E>,
  vc_shape_regular: R1CSShape<E>,
  vc_ck: CommitmentKey<E>,
  vc_vk: VerifierKey<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<SpartanDigest>,
}

impl<E: Engine> crate::digest::Digestible for NeutronNovaVerifierKey<E> {
  fn write_bytes<W: Sized + std::io::Write>(&self, w: &mut W) -> Result<(), std::io::Error> {
    use bincode::Options;
    let config = bincode::DefaultOptions::new()
      .with_little_endian()
      .with_fixint_encoding();
    config
      .serialize_into(&mut *w, &self.ck)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    config
      .serialize_into(&mut *w, &self.vk_ee)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    // Use fast raw-byte path for the R1CS shapes
    self.S_step.write_bytes(w)?;
    self.S_core.write_bytes(w)?;
    // Serialize remaining small fields with bincode
    config
      .serialize_into(&mut *w, &self.vc_shape)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    config
      .serialize_into(&mut *w, &self.vc_shape_regular)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    config
      .serialize_into(&mut *w, &self.vc_ck)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    config
      .serialize_into(&mut *w, &self.vc_vk)
      .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(())
  }
}

impl<E: Engine> DigestHelperTrait<E> for NeutronNovaVerifierKey<E> {
  /// Returns the digest of the verifier's key.
  fn digest(&self) -> Result<SpartanDigest, SpartanError> {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::<_>::new(self);
        dc.digest()
      })
      .cloned()
      .map_err(|_| SpartanError::DigestError {
        reason: "Unable to compute digest for SpartanVerifierKey".to_string(),
      })
  }
}

/// A type that holds the pre-processed state for proving
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "M: Serialize, I: Serialize",
  deserialize = "M: Deserialize<'de>, I: Deserialize<'de>"
))]
pub struct NeutronNovaPrepZkSNARK<E: Engine, M, I> {
  ps_step: Vec<PrecommittedState<E>>,
  ps_core: PrecommittedState<E>,
  /// Cached strategy-specific Az/Bz/Cz layers for step circuits.
  cached_step_matvecs: Option<M>,
  /// Public values used when computing cached_step_matvecs, for validation in prove.
  /// Non-empty when the matvec cache is active; prove checks that step circuits produce the same values.
  cached_step_public_values: Vec<Vec<E::Scalar>>,
  /// Backend input used to build this prep state.
  nifs_input: I,
}

/// Holds the proof produced by the NeutronNova folding scheme followed by NeutronNova SNARK
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronNovaZkSNARK<E: Engine> {
  /// Shared commitment stored once (same for all step instances and core).
  comm_W_shared: Option<Commitment<E>>,
  step_instances: Vec<SplitR1CSInstance<E>>,
  core_instance: SplitR1CSInstance<E>,
  eval_arg: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
  U_verifier: SplitMultiRoundR1CSInstance<E>,
  nifs: NovaNIFS<E>,
  random_U: RelaxedR1CSInstance<E>,
  relaxed_snark: crate::spartan_relaxed::RelaxedR1CSSpartanProof<E>,
}

impl<E: Engine> NeutronNovaZkSNARK<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Sets up the NeutronNova SNARK for a batch of circuits of type `C1` and a single circuit of type `C2`
  ///
  /// # Parameters
  /// - `step_circuit`: The circuit to be folded in the batch
  /// - `core_circuit`: The core circuit that connects the batch together
  /// - `num_steps`: The number of step circuits in the batch (will be padded to next power of two internally)
  pub fn setup<C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    step_circuit: &C1,
    core_circuit: &C2,
    num_steps: usize,
  ) -> Result<(NeutronNovaProverKey<E>, NeutronNovaVerifierKey<E>), SpartanError> {
    let (_setup_span, setup_t) = start_span!("neutronnova_setup");

    let (_r1cs_span, r1cs_t) = start_span!("r1cs_shape_generation");
    debug!("Synthesizing step circuit");
    let mut S_step = ShapeCS::r1cs_shape(step_circuit)?;
    debug!("Finished synthesizing step circuit");

    debug!("Synthesizing core circuit");
    let mut S_core = ShapeCS::r1cs_shape(core_circuit)?;
    debug!("Finished synthesizing core circuit");

    SplitR1CSShape::equalize(&mut S_step, &mut S_core);

    info!(
      "Step circuit's witness sizes: shared = {}, precommitted = {}, rest = {}",
      S_step.num_shared, S_step.num_precommitted, S_step.num_rest
    );
    info!(
      "Core circuit's witness sizes: shared = {}, precommitted = {}, rest = {}",
      S_core.num_shared, S_core.num_precommitted, S_core.num_rest
    );
    info!(elapsed_ms = %r1cs_t.elapsed().as_millis(), "r1cs_shape_generation");

    let (_ck_span, ck_t) = start_span!("commitment_key_generation");
    let (ck, vk_ee) = SplitR1CSShape::commitment_key(&[&S_step, &S_core])?;
    E::PCS::precompute_ck(&ck);
    info!(elapsed_ms = %ck_t.elapsed().as_millis(), "commitment_key_generation");

    // Calculate num_rounds_b from num_steps by padding to next power of two
    let (_vc_span, vc_t) = start_span!("verifier_circuit_setup");
    let num_rounds_b = num_steps.next_power_of_two().log_2();

    let num_vars = S_step.num_shared + S_step.num_precommitted + S_step.num_rest;
    let num_rounds_x = usize::try_from(S_step.num_cons.ilog2()).unwrap();
    let num_rounds_y = usize::try_from(num_vars.ilog2()).unwrap() + 1;
    let vc = NeutronNovaVerifierCircuit::<E>::default(num_rounds_b, num_rounds_x, num_rounds_y, 32);
    let (vc_shape, vc_ck, vc_vk) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&vc)?;
    let vc_shape_regular = vc_shape.to_regular_shape();
    info!(elapsed_ms = %vc_t.elapsed().as_millis(), "verifier_circuit_setup");
    // Eagerly init FixedBaseMul table before cloning so both pk/vk get it
    E::PCS::precompute_ck(&vc_ck);
    let vk: NeutronNovaVerifierKey<E> = NeutronNovaVerifierKey {
      ck: ck.clone(),
      S_step: S_step.clone(),
      S_core: S_core.clone(),
      vk_ee,
      vc_shape: vc_shape.clone(),
      vc_shape_regular: vc_shape_regular.clone(),
      vc_ck: vc_ck.clone(),
      vc_vk,
      digest: OnceCell::new(),
    };

    let vk_digest = vk.digest()?;
    let pk = NeutronNovaProverKey {
      ck,
      S_step,
      S_core,
      vc_shape,
      vc_shape_regular,
      vc_ck,
      vk_digest,
    };

    // Eagerly precompute sparse matrix data for the step and core circuits
    pk.S_step.precompute();
    pk.S_core.precompute();
    vk.S_step.precompute();
    vk.S_core.precompute();
    info!(elapsed_ms = %setup_t.elapsed().as_millis(), "neutronnova_setup");
    Ok((pk, vk))
  }

  /// Prepares the pre-processed state for proving
  pub fn prep_prove<C1, C2, Nifs>(
    pk: &NeutronNovaProverKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    nifs_input: &Nifs::Input,
  ) -> Result<NeutronNovaPrepZkSNARK<E, Nifs::StepMatvecs, Nifs::Input>, SpartanError>
  where
    C1: SpartanCircuit<E>,
    C2: SpartanCircuit<E>,
    Nifs: NeutronNovaNifsStrategy<E>,
  {
    let (_prep_span, prep_t) = start_span!("neutronnova_prep_prove");
    let is_small = Nifs::is_small(nifs_input);

    // we synthesize shared witness for the first circuit; every other circuit including the core circuit shares this witness
    let (_shared_span, shared_t) = start_span!("generate_shared_witness");
    let mut ps =
      SatisfyingAssignment::shared_witness(&pk.S_step, &pk.ck, &step_circuits[0], is_small)?;
    info!(elapsed_ms = %shared_t.elapsed().as_millis(), "generate_shared_witness");

    let (_precommit_span, precommit_t) = start_span!(
      "generate_precommitted_witnesses",
      circuits = step_circuits.len() + 1
    );
    let ps_step = (0..step_circuits.len())
      .into_par_iter()
      .map(|i| {
        // copy ps to avoid mutating the original shared witness
        let mut ps_i = ps.clone();
        SatisfyingAssignment::precommitted_witness(
          &mut ps_i,
          &pk.S_step,
          &pk.ck,
          &step_circuits[i],
          is_small,
        )?;
        Ok(ps_i)
      })
      .collect::<Result<Vec<_>, _>>()?;

    // we don't need to make a copy of ps for the core circuit, as it will be used only once
    SatisfyingAssignment::precommitted_witness(
      &mut ps,
      &pk.S_core,
      &pk.ck,
      core_circuit,
      is_small,
    )?;
    info!(elapsed_ms = %precommit_t.elapsed().as_millis(), circuits = step_circuits.len() + 1, "generate_precommitted_witnesses");

    // Precompute full matrix-vector products for step circuits (deterministic).
    // Only valid when step circuits have no rest variables and no challenges,
    // meaning z = [shared_W, precommitted_W, 0..., 1, public_values] is fully known during prep.
    let can_cache_matvec = pk.S_step.num_challenges == 0 && pk.S_step.num_rest_unpadded == 0;

    let (cached_step_matvecs, cached_step_public_values) = if can_cache_matvec {
      // Collect public values for each step circuit so we can validate in prove
      let step_public_values: Vec<Vec<E::Scalar>> = step_circuits
        .iter()
        .map(|c| {
          c.public_values().map_err(|e| SpartanError::SynthesisError {
            reason: format!("Circuit does not provide public IO: {e}"),
          })
        })
        .collect::<Result<Vec<_>, _>>()?;

      let matvec: Vec<_> = (0..ps_step.len())
        .into_par_iter()
        .map(|i| {
          let ps_i = &ps_step[i];
          let public_values = &step_public_values[i];
          let mut z = Vec::with_capacity(ps_i.W.len() + 1 + public_values.len());
          z.extend_from_slice(&ps_i.W);
          z.push(E::Scalar::ONE);
          z.extend_from_slice(public_values);
          pk.S_step.multiply_vec(&z)
        })
        .collect::<Result<Vec<_>, _>>()?;
      let total = matvec.first().map(|(az, _, _)| az.len()).unwrap_or(0);
      let step_matvecs = Nifs::build_step_matvecs_from_field(matvec, nifs_input)?;
      info!(total = total, "cached_step_matvecs_built");
      (Some(step_matvecs), step_public_values)
    } else {
      info!(
        "Step circuit has rest_unpadded={} challenges={}, skipping matvec/i64 caching",
        pk.S_step.num_rest_unpadded, pk.S_step.num_challenges
      );
      (None, Vec::new())
    };

    info!(elapsed_ms = %prep_t.elapsed().as_millis(), "neutronnova_prep_prove");
    Ok(NeutronNovaPrepZkSNARK {
      ps_step,
      ps_core: ps,
      cached_step_matvecs,
      cached_step_public_values,
      nifs_input: nifs_input.clone(),
    })
  }

  /// Prove the folding of a batch of R1CS instances and a core circuit that connects them together.
  /// Takes ownership of `prep_snark` to avoid cloning large witness vectors (~66MB).
  /// Returns the proof and the consumed prep state. Cached step matvecs are one-shot:
  /// rerun `prep_prove` to rebuild the cache before proving again.
  pub fn prove<C1, C2, Nifs>(
    pk: &NeutronNovaProverKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    mut prep_snark: NeutronNovaPrepZkSNARK<E, Nifs::StepMatvecs, Nifs::Input>,
    nifs_input: &Nifs::Input,
  ) -> Result<
    (
      Self,
      NeutronNovaPrepZkSNARK<E, Nifs::StepMatvecs, Nifs::Input>,
    ),
    SpartanError,
  >
  where
    C1: SpartanCircuit<E>,
    C2: SpartanCircuit<E>,
    Nifs: NeutronNovaNifsStrategy<E>,
    E::Scalar: DelayedReduction<i128>,
  {
    let (_prove_span, prove_t) = start_span!("neutronnova_prove");
    let is_small = Nifs::is_small(nifs_input);
    if prep_snark.nifs_input != *nifs_input {
      return Err(SpartanError::InvalidInputLength {
        reason: "prep/prove NIFS inputs do not match".to_string(),
      });
    }

    // rerandomize prep state in-place (we own it, no clone needed)
    let (_rerandomize_span, rerandomize_t) = start_span!("rerandomize_prep_state");
    prep_snark
      .ps_core
      .rerandomize_in_place(&pk.ck, &pk.S_core)?;
    let comm_W_shared = prep_snark.ps_core.comm_W_shared.clone();
    let r_W_shared = prep_snark.ps_core.r_W_shared.clone();
    prep_snark.ps_step.par_iter_mut().try_for_each(|ps_i| {
      ps_i.rerandomize_with_shared_in_place(&pk.ck, &pk.S_step, &comm_W_shared, &r_W_shared)
    })?;
    info!(elapsed_ms = %rerandomize_t.elapsed().as_millis(), "rerandomize_prep_state");

    // Validate that cached matvec matches current step circuit public values.
    // The cache computed in prep_prove includes public_values in the z vector;
    // if the circuits changed, the cache is stale and would produce incorrect proofs.
    match (
      prep_snark.cached_step_matvecs.is_some(),
      prep_snark.cached_step_public_values.is_empty(),
    ) {
      (true, false) => {
        if prep_snark.cached_step_public_values.len() != step_circuits.len() {
          return Err(SpartanError::InternalError {
            reason: format!(
              "Cached matvec was computed for {} step circuits, but prove received {}",
              prep_snark.cached_step_public_values.len(),
              step_circuits.len()
            ),
          });
        }
        for (i, circuit) in step_circuits.iter().enumerate() {
          let current_pv = circuit
            .public_values()
            .map_err(|e| SpartanError::SynthesisError {
              reason: format!("Circuit does not provide public IO: {e}"),
            })?;
          if prep_snark.cached_step_public_values[i] != current_pv {
            return Err(SpartanError::InternalError {
              reason: format!(
                "Step circuit {i} public values changed between prep_prove and prove"
              ),
            });
          }
        }
      }
      (false, false) => {
        return Err(SpartanError::InternalError {
          reason:
            "Cached step matvecs were consumed by a previous prove; rerun prep_prove to rebuild them"
              .to_string(),
        });
      }
      _ => {}
    }

    // Parallel generation of instances and witnesses
    let (_gen_span, gen_t) = start_span!(
      "generate_instances_witnesses",
      step_circuits = step_circuits.len()
    );
    let (res_steps, res_core) = rayon::join(
      || {
        prep_snark
          .ps_step
          .par_iter_mut()
          .zip(step_circuits.par_iter().enumerate())
          .map(|(pre_state, (i, circuit))| {
            let mut transcript = E::TE::new(b"neutronnova_prove");
            transcript.absorb(b"vk", &pk.vk_digest);
            transcript.absorb(
              b"num_circuits",
              &E::Scalar::from(step_circuits.len() as u64),
            );
            transcript.absorb(b"circuit_index", &E::Scalar::from(i as u64));

            let public_values =
              circuit
                .public_values()
                .map_err(|e| SpartanError::SynthesisError {
                  reason: format!("Circuit does not provide public IO: {e}"),
                })?;
            transcript.absorb(b"public_values", &public_values.as_slice());

            SatisfyingAssignment::r1cs_instance_and_witness(
              pre_state,
              &pk.S_step,
              &pk.ck,
              circuit,
              is_small,
              &mut transcript,
            )
          })
          .collect::<Result<Vec<_>, _>>()
          .map(|pairs| {
            let (instances, witnesses): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
            (instances, witnesses)
          })
      },
      || {
        let mut transcript = E::TE::new(b"neutronnova_prove");
        transcript.absorb(b"vk", &pk.vk_digest);
        let public_values_core =
          core_circuit
            .public_values()
            .map_err(|e| SpartanError::SynthesisError {
              reason: format!("Core circuit does not provide public IO: {e}"),
            })?;
        transcript.absorb(b"public_values", &public_values_core.as_slice());
        SatisfyingAssignment::r1cs_instance_and_witness(
          &mut prep_snark.ps_core,
          &pk.S_core,
          &pk.ck,
          core_circuit,
          is_small,
          &mut transcript,
        )
      },
    );

    let ((step_instances, step_witnesses), (core_instance, core_witness)) = (res_steps?, res_core?);
    info!(elapsed_ms = %gen_t.elapsed().as_millis(), step_circuits = step_circuits.len(), "generate_instances_witnesses");

    let (_reg_span, reg_t) = start_span!("convert_to_regular_instances");
    let step_instances_regular = step_instances
      .iter()
      .map(|u| u.to_regular_instance())
      .collect::<Result<Vec<_>, _>>()?;

    let core_instance_regular = core_instance.to_regular_instance()?;
    info!(elapsed_ms = %reg_t.elapsed().as_millis(), "convert_to_regular_instances");
    // We start a new transcript for the NeutronNova NIFS proof
    // All instances will be absorbed into the transcript
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &pk.vk_digest);

    // absorb the core instance; NIFS will absorb the step instances
    transcript.absorb(b"core_instance", &core_instance_regular);

    let n_padded = step_instances_regular.len().next_power_of_two();
    let num_vars = pk.S_step.num_shared + pk.S_step.num_precommitted + pk.S_step.num_rest;
    let num_rounds_b = n_padded.log_2();
    let num_rounds_x = pk.S_step.num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    let mut vc = NeutronNovaVerifierCircuit::<E>::default(
      num_rounds_b,
      num_rounds_x,
      num_rounds_y,
      pk.vc_shape.commitment_width,
    );
    let mut vc_state = SatisfyingAssignment::<E>::initialize_multiround_witness(&pk.vc_shape)?;

    // Perform ZK NIFS prove and collect outputs.
    let (_nifs_span, nifs_t) = start_span!("NIFS");
    let step_matvecs = match prep_snark.cached_step_matvecs.take() {
      Some(cached) => cached,
      None => Nifs::build_step_matvecs(
        &pk.S_step,
        &step_instances_regular,
        &step_witnesses,
        nifs_input,
      )?,
    };
    let (E_eq, Az_step, Bz_step, Cz_step, folded_W, folded_U) = Nifs::prove(
      &pk.S_step,
      &pk.ck,
      step_instances_regular,
      step_witnesses,
      step_matvecs,
      nifs_input,
      &mut vc,
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &mut transcript,
    )?;
    info!(elapsed_ms = %nifs_t.elapsed().as_millis(), "NIFS");

    let (_tensor_span, tensor_t) = start_span!("compute_tensor_and_poly_tau");
    let (_ell, left, _right) = compute_tensor_decomp(pk.S_step.num_cons);
    let mut E1 = E_eq;
    let E2 = E1.split_off(left);

    let mut poly_tau_left = MultilinearPolynomial::new(E1);
    let poly_tau_right = MultilinearPolynomial::new(E2);

    info!(elapsed_ms = %tensor_t.elapsed().as_millis(), "compute_tensor_and_poly_tau");

    // outer sum-check preparation
    let (_mp_span, mp_t) = start_span!("prepare_multilinear_polys");
    let (mut poly_Az_step, mut poly_Bz_step, mut poly_Cz_step) = (
      MultilinearPolynomial::new(Az_step),
      MultilinearPolynomial::new(Bz_step),
      MultilinearPolynomial::new(Cz_step),
    );

    let (mut poly_Az_core, mut poly_Bz_core, mut poly_Cz_core) = {
      let (_core_span, core_t) = start_span!("compute_core_polys");
      let mut z = Vec::with_capacity(
        core_witness.W.len()
          + 1
          + core_instance.public_values.len()
          + core_instance.challenges.len(),
      );
      z.extend_from_slice(&core_witness.W);
      z.push(E::Scalar::ONE);
      z.extend_from_slice(&core_instance.public_values);
      z.extend_from_slice(&core_instance.challenges);

      let (Az, Bz, Cz) = pk.S_core.multiply_vec(&z)?;
      info!(elapsed_ms = %core_t.elapsed().as_millis(), "compute_core_polys");
      (
        MultilinearPolynomial::new(Az),
        MultilinearPolynomial::new(Bz),
        MultilinearPolynomial::new(Cz),
      )
    };

    info!(elapsed_ms = %mp_t.elapsed().as_millis(), "prepare_multilinear_polys");
    let outer_start_index = num_rounds_b + 1;
    // outer sum-check (batched)
    let (_sc_span, sc_t) = start_span!("outer_sumcheck_batched");
    let r_x = SumcheckProof::<E>::prove_cubic_with_additive_term_batched_zk(
      num_rounds_x,
      &mut poly_tau_left,
      &poly_tau_right,
      &mut poly_Az_step,
      &mut poly_Az_core,
      &mut poly_Bz_step,
      &mut poly_Bz_core,
      &mut poly_Cz_step,
      &mut poly_Cz_core,
      &mut vc,
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &mut transcript,
      outer_start_index,
    )?;
    info!(elapsed_ms = %sc_t.elapsed().as_millis(), "outer_sumcheck_batched");
    vc.claim_Az_step = poly_Az_step[0];
    vc.claim_Bz_step = poly_Bz_step[0];
    vc.claim_Cz_step = poly_Cz_step[0];
    vc.claim_Az_core = poly_Az_core[0];
    vc.claim_Bz_core = poly_Bz_core[0];
    vc.claim_Cz_core = poly_Cz_core[0];
    vc.tau_at_rx = poly_tau_left[0];

    let chals = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &vc,
      outer_start_index + num_rounds_x,
      &mut transcript,
    )?;
    let r = chals[0];

    // inner sum-check preparation
    let claim_inner_joint_step = vc.claim_Az_step + r * vc.claim_Bz_step + r * r * vc.claim_Cz_step;
    let claim_inner_joint_core = vc.claim_Az_core + r * vc.claim_Bz_core + r * r * vc.claim_Cz_core;

    let (_eval_rx_span, eval_rx_t) = start_span!("compute_eval_rx");
    let evals_rx = EqPolynomial::evals_from_points(&r_x);
    info!(elapsed_ms = %eval_rx_t.elapsed().as_millis(), "compute_eval_rx");

    let (_sparse_span, sparse_t) = start_span!("compute_eval_table_sparse");
    let (poly_ABC_step, step_lo_eff, step_hi_eff) =
      pk.S_step.bind_and_prepare_poly_ABC_full(&evals_rx, &r);
    let (poly_ABC_core, core_lo_eff, core_hi_eff) =
      pk.S_core.bind_and_prepare_poly_ABC_full(&evals_rx, &r);
    info!(elapsed_ms = %sparse_t.elapsed().as_millis(), "compute_eval_table_sparse");
    // inner sum-check
    let (_sc2_span, sc2_t) = start_span!("inner_sumcheck_batched");

    debug!("Proving inner sum-check with {} rounds", num_rounds_y);
    debug!(
      "Inner sum-check sizes - poly_ABC_step: {}, poly_ABC_core: {}",
      poly_ABC_step.len(),
      poly_ABC_core.len()
    );

    // Build z vectors for the folded and core instances.
    // Non-zero prefix = w_len + 1 + x_len (witness + u + public inputs).
    let (z_folded_vec, z_folded_lo, z_folded_hi) = {
      let mut v = vec![E::Scalar::ZERO; num_vars * 2];
      let w_len = folded_W.W.len();
      v[..w_len].copy_from_slice(&folded_W.W);
      v[w_len] = E::Scalar::ONE;
      let x_len = folded_U.X.len();
      v[w_len + 1..w_len + 1 + x_len].copy_from_slice(&folded_U.X);
      let last_nz = w_len + 1 + x_len;
      (v, last_nz.min(num_vars), last_nz.saturating_sub(num_vars))
    };
    let (z_core_vec, z_core_lo, z_core_hi) = {
      let mut v = vec![E::Scalar::ZERO; num_vars * 2];
      let w_len = core_witness.W.len();
      v[..w_len].copy_from_slice(&core_witness.W);
      v[w_len] = E::Scalar::ONE;
      let x_len = core_instance_regular.X.len();
      v[w_len + 1..w_len + 1 + x_len].copy_from_slice(&core_instance_regular.X);
      let last_nz = w_len + 1 + x_len;
      (v, last_nz.min(num_vars), last_nz.saturating_sub(num_vars))
    };

    // Use actual X length for hi_eff (num_public in SplitR1CSShape may not include shared inputs)
    let step_hi_eff = step_hi_eff.max(z_folded_hi);
    let core_hi_eff = core_hi_eff.max(z_core_hi);

    let (r_y, evals) = SumcheckProof::<E>::prove_quad_batched_zk(
      &[claim_inner_joint_step, claim_inner_joint_core],
      num_rounds_y,
      &mut MultilinearPolynomial::new_with_halves(poly_ABC_step, step_lo_eff, step_hi_eff),
      &mut MultilinearPolynomial::new_with_halves(poly_ABC_core, core_lo_eff, core_hi_eff),
      &mut MultilinearPolynomial::new_with_halves(z_folded_vec, z_folded_lo, z_folded_hi),
      &mut MultilinearPolynomial::new_with_halves(z_core_vec, z_core_lo, z_core_hi),
      &mut vc,
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &mut transcript,
      outer_start_index + num_rounds_x + 1,
    )?;
    info!(elapsed_ms = %sc2_t.elapsed().as_millis(), "inner_sumcheck_batched");

    let eval_Z_step = evals[2];
    let eval_Z_core = evals[3];

    let eval_X_step = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(folded_U.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).unwrap();
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };
    let eval_X_core = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(core_instance_regular.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).unwrap();
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };
    let inv: Option<E::Scalar> = (E::Scalar::ONE - r_y[0]).invert().into();
    let one_minus_ry0_inv = inv.ok_or(SpartanError::DivisionByZero)?;
    let eval_W_step = (eval_Z_step - r_y[0] * eval_X_step) * one_minus_ry0_inv;
    let eval_W_core = (eval_Z_core - r_y[0] * eval_X_core) * one_minus_ry0_inv;

    vc.eval_W_step = eval_W_step;
    vc.eval_W_core = eval_W_core;
    vc.eval_X_step = eval_X_step;
    vc.eval_X_core = eval_X_core;

    // Inner final equality round
    let _ = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &vc,
      outer_start_index + num_rounds_x + 1 + num_rounds_y,
      &mut transcript,
    )?;

    // Commit eval_W_step
    let eval_w_step_commit_round = outer_start_index + num_rounds_x + 1 + num_rounds_y + 1;
    let _ = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &vc,
      eval_w_step_commit_round,
      &mut transcript,
    )?;

    // Commit eval_W_core
    let _ = SatisfyingAssignment::<E>::process_round(
      &mut vc_state,
      &pk.vc_shape,
      &pk.vc_ck,
      &vc,
      eval_w_step_commit_round + 1,
      &mut transcript,
    )?;

    let (U_verifier, W_verifier) =
      SatisfyingAssignment::<E>::finalize_multiround_witness(&mut vc_state, &pk.vc_shape)?;

    let U_verifier_regular = U_verifier.to_regular_instance()?;

    // Sample fresh random instance/witness for ZK (must be done per-prove to preserve zero-knowledge).
    let (random_U, random_W) = pk
      .vc_shape_regular
      .sample_random_instance_witness(&pk.vc_ck)?;
    let (nifs, folded_W_verifier, folded_u, folded_X) = NovaNIFS::<E>::prove(
      &pk.vc_ck,
      &pk.vc_shape_regular,
      &random_U,
      &random_W,
      &U_verifier_regular,
      &W_verifier,
      &mut transcript,
    )?;

    // Prove satisfiability of the folded VC instance via relaxed R1CS Spartan
    let relaxed_snark = crate::spartan_relaxed::RelaxedR1CSSpartanProof::prove(
      &pk.vc_shape_regular,
      &pk.vc_ck,
      &folded_u,
      &folded_X,
      &folded_W_verifier,
      &mut transcript,
    )?;
    // access two claimed commitments to evaluations of W_step and W_core
    let comm_eval_W_step = U_verifier.comm_w_per_round[eval_w_step_commit_round].clone();
    let blind_eval_W_step = vc_state.r_w_per_round[eval_w_step_commit_round].clone();

    let comm_eval_W_core = U_verifier.comm_w_per_round[eval_w_step_commit_round + 1].clone();
    let blind_eval_W_core = vc_state.r_w_per_round[eval_w_step_commit_round + 1].clone();

    // the commitments are already absorbed in the transcript, so we simply squeeze the challenge
    let c_eval = transcript.squeeze(b"c_eval")?;

    // fold evaluation claims into one
    let (_fold_eval_span, fold_eval_t) = start_span!("fold_evaluation_claims");
    let comm = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &[folded_U.comm_W, core_instance_regular.comm_W],
      &[E::Scalar::ONE, c_eval],
    )?;
    let blind = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(
      &[folded_W.r_W.clone(), core_witness.r_W.clone()],
      &[E::Scalar::ONE, c_eval],
    )?;
    let W = folded_W
      .W
      .par_iter()
      .zip(core_witness.W.par_iter())
      .map(|(w1, w2)| *w1 + c_eval * *w2)
      .collect::<Vec<_>>();
    let comm_eval = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &[comm_eval_W_step, comm_eval_W_core],
      &[E::Scalar::ONE, c_eval],
    )?;
    let blind_eval = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(
      &[blind_eval_W_step, blind_eval_W_core],
      &[E::Scalar::ONE, c_eval],
    )?;
    info!(elapsed_ms = %fold_eval_t.elapsed().as_millis(), "fold_evaluation_claims");

    let (_pcs_span, pcs_t) = start_span!("pcs_prove");
    let eval_arg = E::PCS::prove(
      &pk.ck,
      &pk.vc_ck,
      &mut transcript,
      &comm,
      &W,
      &blind,
      &r_y[1..],
      &comm_eval,
      &blind_eval,
    )?;
    info!(elapsed_ms = %pcs_t.elapsed().as_millis(), "pcs_prove");

    // Extract shared commitment (same for all step instances and core) and strip from instances
    let comm_W_shared = step_instances.first().and_then(|u| u.comm_W_shared.clone());
    let step_instances = step_instances
      .into_iter()
      .map(|mut u| {
        u.comm_W_shared = None;
        u
      })
      .collect::<Vec<_>>();
    let mut core_instance = core_instance;
    core_instance.comm_W_shared = None;

    let result = Self {
      comm_W_shared,
      step_instances,
      core_instance,
      eval_arg,
      U_verifier,
      nifs,
      random_U,
      relaxed_snark,
    };

    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "neutronnova_prove");
    Ok((result, prep_snark))
  }

  /// Verifies the NeutronNovaZkSNARK and returns the public IO from the instances
  pub fn verify(
    &self,
    vk: &NeutronNovaVerifierKey<E>,
    num_instances: usize,
  ) -> Result<(Vec<Vec<E::Scalar>>, Vec<E::Scalar>), SpartanError> {
    let (_verify_span, _verify_t) = start_span!("neutronnova_verify");
    if num_instances == 0 || num_instances != self.step_instances.len() {
      return Err(SpartanError::ProofVerifyError {
        reason: format!(
          "Expected {} instances (non-zero), got {}",
          num_instances,
          self.step_instances.len()
        ),
      });
    }

    // Reconstruct step instances and core instance with the shared commitment
    let step_instances: Vec<SplitR1CSInstance<E>> = self
      .step_instances
      .iter()
      .map(|u| {
        let mut u = u.clone();
        u.comm_W_shared = self.comm_W_shared.clone();
        u
      })
      .collect();
    let mut core_instance = self.core_instance.clone();
    core_instance.comm_W_shared = self.comm_W_shared.clone();

    // validate the step instances
    let (_validate_span, validate_t) =
      start_span!("validate_instances", instances = step_instances.len());
    for (i, u) in step_instances.iter().enumerate() {
      let mut transcript = E::TE::new(b"neutronnova_prove");
      transcript.absorb(b"vk", &vk.digest()?);
      transcript.absorb(
        b"num_circuits",
        &E::Scalar::from(step_instances.len() as u64),
      );
      transcript.absorb(b"circuit_index", &E::Scalar::from(i as u64));
      // absorb the public IO into the transcript
      transcript.absorb(b"public_values", &u.public_values.as_slice());

      u.validate(&vk.S_step, &mut transcript)?;
    }

    // validate the core instance
    let mut transcript = E::TE::new(b"neutronnova_prove");
    transcript.absorb(b"vk", &vk.digest()?);
    // absorb the public IO into the transcript
    transcript.absorb(b"public_values", &core_instance.public_values.as_slice());

    core_instance.validate(&vk.S_core, &mut transcript)?;
    info!(elapsed_ms = %validate_t.elapsed().as_millis(), instances = step_instances.len(), "validate_instances");

    // shared commitment consistency was enforced at construction -- all step instances share comm_W_shared
    // also verify it matches the core instance
    for u in &step_instances {
      if u.comm_W_shared != core_instance.comm_W_shared {
        return Err(SpartanError::ProofVerifyError {
          reason: "All instances must have the same shared commitment".to_string(),
        });
      }
    }

    let (_convert_span, convert_t) = start_span!("convert_to_regular_verify");
    let mut step_instances_padded = step_instances.clone();
    if step_instances_padded.len() != step_instances_padded.len().next_power_of_two() {
      let additional =
        step_instances_padded.len().next_power_of_two() - step_instances_padded.len();
      extend_with_first_clones(&mut step_instances_padded, additional);
    }
    let step_instances_regular = step_instances_padded
      .par_iter()
      .map(|u| u.to_regular_instance())
      .collect::<Result<Vec<_>, _>>()?;

    let core_instance_regular = core_instance.to_regular_instance()?;
    info!(elapsed_ms = %convert_t.elapsed().as_millis(), "convert_to_regular_verify");
    // We start a new transcript for the NeutronNova NIFS proof
    let mut transcript = E::TE::new(b"neutronnova_prove");

    // absorb the verifier key and instances
    transcript.absorb(b"vk", &vk.digest()?);
    transcript.absorb(b"core_instance", &core_instance_regular);
    for U in step_instances_regular.iter() {
      transcript.absorb(b"U", U);
    }
    transcript.absorb(b"T", &E::Scalar::ZERO); // we always have T=0 in NeutronNova

    // compute the number of rounds of NIFS, outer sum-check, and inner sum-check
    let num_rounds_b = step_instances_regular.len().log_2();
    let num_vars = vk.S_step.num_shared + vk.S_step.num_precommitted + vk.S_step.num_rest;
    let num_rounds_x = vk.S_step.num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    // we need num_rounds_b challenges for folding the step instances; we also need tau to compress multiple R1CS checks
    let tau = transcript.squeeze(b"tau")?;
    let rhos = (0..num_rounds_b)
      .map(|_| transcript.squeeze(b"rho"))
      .collect::<Result<Vec<_>, _>>()?;

    // validate the provided multi-round verifier instance and advance transcript
    self.U_verifier.validate(&vk.vc_shape, &mut transcript)?;

    let U_verifier_regular = self.U_verifier.to_regular_instance()?;

    // extract challenges and public IO from U_verifier's public IO
    let num_public_values = 6usize;
    let num_challenges = num_rounds_b + num_rounds_x + 1 + num_rounds_y;
    if U_verifier_regular.X.len() != num_challenges + num_public_values {
      return Err(SpartanError::ProofVerifyError {
        reason: format!(
          "Verifier instance has incorrect number of public IO: expected {}, got {}",
          num_challenges + num_public_values,
          U_verifier_regular.X.len()
        ),
      });
    }

    let challenges = &U_verifier_regular.X[0..num_challenges];
    let public_values = &U_verifier_regular.X[num_challenges..num_challenges + 6];

    let r_b = challenges[0..num_rounds_b].to_vec();
    let r_x = challenges[num_rounds_b..num_rounds_b + num_rounds_x].to_vec();
    let r = challenges[num_rounds_b + num_rounds_x]; // r for combining inner claims
    let r_y = challenges[num_rounds_b + num_rounds_x + 1..].to_vec();

    // fold_multiple and nifs.verify are independent: overlap them
    let (folded_U_result, folded_U_verifier_result) = rayon::join(
      || R1CSInstance::fold_multiple(&r_b, &step_instances_regular),
      || {
        self
          .nifs
          .verify(&mut transcript, &self.random_U, &U_verifier_regular)
      },
    );
    let folded_U = folded_U_result?;
    let folded_U_verifier = folded_U_verifier_result?;
    self
      .relaxed_snark
      .verify(
        &vk.vc_shape_regular,
        &vk.vc_vk,
        &folded_U_verifier,
        &mut transcript,
      )
      .map_err(|e| SpartanError::ProofVerifyError {
        reason: format!("Relaxed Spartan verify failed: {e}"),
      })?;
    let (_matrix_eval_span, matrix_eval_t) = start_span!("matrix_evaluations");
    let (eval_A_step, eval_B_step, eval_C_step, eval_A_core, eval_B_core, eval_C_core) = {
      let T_x = EqPolynomial::evals_from_points(&r_x);
      let T_y = EqPolynomial::evals_from_points(&r_y);
      let (eval_A_step, eval_B_step, eval_C_step) = vk.S_step.evaluate_with_tables_fast(&T_x, &T_y);
      let (eval_A_core, eval_B_core, eval_C_core) = vk.S_core.evaluate_with_tables_fast(&T_x, &T_y);

      (
        eval_A_step,
        eval_B_step,
        eval_C_step,
        eval_A_core,
        eval_B_core,
        eval_C_core,
      )
    };
    info!(elapsed_ms = %matrix_eval_t.elapsed().as_millis(), "matrix_evaluations");

    let eval_X_step = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(folded_U.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).unwrap();
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };
    let eval_X_core = {
      let X = vec![E::Scalar::ONE]
        .into_iter()
        .chain(core_instance_regular.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let num_vars_log2 = usize::try_from(num_vars.ilog2()).unwrap();
      SparsePolynomial::new(num_vars_log2, X).evaluate(&r_y[1..])
    };

    // Compute quotient_* = (eval_A + r*eval_B + r^2*eval_C) for both branches
    let quotient_step = eval_A_step + r * eval_B_step + r * r * eval_C_step;
    let quotient_core = eval_A_core + r * eval_B_core + r * r * eval_C_core;
    let tau_at_rx = PowPolynomial::new(&tau, r_x.len()).evaluate(&r_x)?;
    let eq_rho_at_rb = EqPolynomial::new(r_b).evaluate(&rhos);

    if public_values[0] != tau_at_rx
      || public_values[1] != eval_X_step
      || public_values[2] != eval_X_core
      || public_values[3] != eq_rho_at_rb
      || public_values[4] != quotient_step
      || public_values[5] != quotient_core
    {
      return Err(SpartanError::ProofVerifyError {
        reason:
          "Verifier instance public tau_at_rx/eval_X_step/eq_rho_at_rb/eval_X_core/quotients do not match recomputation"
            .to_string(),
      });
    }

    // verify PCS eval
    let c_eval = transcript.squeeze(b"c_eval")?;

    let eval_w_step_commit_round = num_rounds_b + 1 + num_rounds_x + 1 + num_rounds_y + 1;
    let comm_eval_W_step = self.U_verifier.comm_w_per_round[eval_w_step_commit_round].clone();
    let comm_eval_W_core = self.U_verifier.comm_w_per_round[eval_w_step_commit_round + 1].clone();

    let comm = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &[folded_U.comm_W, core_instance_regular.comm_W],
      &[E::Scalar::ONE, c_eval],
    )?;
    let comm_eval = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &[comm_eval_W_step, comm_eval_W_core],
      &[E::Scalar::ONE, c_eval],
    )?;

    let (_pcs_verify_span, pcs_verify_t) = start_span!("pcs_verify");
    E::PCS::verify(
      &vk.vk_ee,
      &vk.vc_ck,
      &mut transcript,
      &comm,
      &r_y[1..],
      &comm_eval,
      &self.eval_arg,
    )?;
    info!(elapsed_ms = %pcs_verify_t.elapsed().as_millis(), "pcs_verify");

    info!(elapsed_ms = %_verify_t.elapsed().as_millis(), "neutronnova_verify");

    let public_values_step = step_instances
      .iter()
      .take(num_instances)
      .map(|u| u.public_values.clone())
      .collect::<Vec<Vec<_>>>();

    let public_values_core = core_instance.public_values.clone();

    // return a vector of public values
    Ok((public_values_step, public_values_core))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::T256HyraxEngine;
  use bellpepper::gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    sha256::sha256,
  };
  use bellpepper_core::{ConstraintSystem, SynthesisError};
  use core::marker::PhantomData;

  #[derive(Clone, Debug)]
  struct Sha256Circuit<E: Engine> {
    preimage: Vec<u8>,
    _p: PhantomData<E>,
  }

  impl<E: Engine> SpartanCircuit<E> for Sha256Circuit<E> {
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
      Ok(vec![E::Scalar::ZERO]) // Placeholder, we don't use public values in this example
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![]) // Placeholder, we don't use shared variables in this example
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
      _: &[AllocatedNum<E::Scalar>],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![]) // Placeholder, we don't use precommitted variables in this example
    }

    fn num_challenges(&self) -> usize {
      0 // Placeholder, we don't use challenges in this example
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _shared: &[AllocatedNum<E::Scalar>],
      _precommitted: &[AllocatedNum<E::Scalar>],
      _challenges: Option<&[E::Scalar]>, // challenges from the verifier
    ) -> Result<(), SynthesisError> {
      // we write a circuit that checks if the input is a SHA256 preimage
      let bit_values: Vec<_> = self
        .preimage
        .clone()
        .into_iter()
        .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
        .map(Some)
        .collect();
      assert_eq!(bit_values.len(), self.preimage.len() * 8);

      let preimage_bits = bit_values
        .into_iter()
        .enumerate()
        .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
        .map(|b| b.map(Boolean::from))
        .collect::<Result<Vec<_>, _>>()?;

      let _ = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(E::Scalar::ZERO))?;
      x.inputize(cs.namespace(|| "inputize x"))?;

      Ok(())
    }
  }

  #[derive(Clone, Debug)]
  struct CacheablePublicCircuit<E: Engine> {
    value: E::Scalar,
  }

  impl<E: Engine> SpartanCircuit<E> for CacheablePublicCircuit<E> {
    fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
      Ok(vec![self.value])
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _: &[AllocatedNum<E::Scalar>],
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(self.value))?;
      x.inputize(cs.namespace(|| "inputize x"))?;
      for i in 0..4 {
        cs.enforce(
          || format!("x equals x {i}"),
          |lc| lc + x.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + x.get_variable(),
        );
      }
      Ok(vec![x])
    }

    fn num_challenges(&self) -> usize {
      0
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
      _: &[AllocatedNum<E::Scalar>],
      _: &[AllocatedNum<E::Scalar>],
      _: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
      Ok(())
    }
  }

  fn generate_sha_r1cs<E: Engine>(
    num_circuits: usize,
    len: usize,
  ) -> (
    NeutronNovaProverKey<E>,
    NeutronNovaVerifierKey<E>,
    Vec<Sha256Circuit<E>>,
  )
  where
    E::PCS: FoldingEngineTrait<E>, // Ensure that the PCS supports folding
  {
    let circuit = Sha256Circuit::<E> {
      preimage: vec![0u8; len],
      _p: Default::default(),
    };

    let (pk, vk) = NeutronNovaZkSNARK::<E>::setup(&circuit, &circuit, num_circuits).unwrap();

    let circuits = (0..num_circuits)
      .map(|i| Sha256Circuit::<E> {
        preimage: vec![i as u8; len],
        _p: Default::default(),
      })
      .collect::<Vec<_>>();

    (pk, vk, circuits)
  }

  fn prove_and_verify_neutron<E, C1, C2, Nifs>(
    pk: &NeutronNovaProverKey<E>,
    vk: &NeutronNovaVerifierKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
    nifs_input: &Nifs::Input,
  ) -> Result<(Vec<Vec<E::Scalar>>, Vec<E::Scalar>), SpartanError>
  where
    E: Engine,
    E::PCS: FoldingEngineTrait<E>,
    E::Scalar: DelayedReduction<i128>,
    C1: SpartanCircuit<E>,
    C2: SpartanCircuit<E>,
    Nifs: NeutronNovaNifsStrategy<E>,
  {
    let ps = NeutronNovaZkSNARK::<E>::prep_prove::<C1, C2, Nifs>(
      pk,
      step_circuits,
      core_circuit,
      nifs_input,
    )?;
    let (snark, _ps) = NeutronNovaZkSNARK::<E>::prove::<C1, C2, Nifs>(
      pk,
      step_circuits,
      core_circuit,
      ps,
      nifs_input,
    )?;
    snark.verify(vk, step_circuits.len())
  }

  fn test_neutron_inner<E: Engine, C1: SpartanCircuit<E>, C2: SpartanCircuit<E>>(
    name: &str,
    pk: &NeutronNovaProverKey<E>,
    vk: &NeutronNovaVerifierKey<E>,
    step_circuits: &[C1],
    core_circuit: &C2,
  ) where
    E::PCS: FoldingEngineTrait<E>,
    E::Scalar: DelayedReduction<i128>,
  {
    println!(
      "[bench_neutron_inner] name: {name}, num_circuits: {}",
      step_circuits.len()
    );

    let nifs_input = true;
    let res = prove_and_verify_neutron::<E, C1, C2, NeutronNovaNIFS<E>>(
      pk,
      vk,
      step_circuits,
      core_circuit,
      &nifs_input,
    );
    println!(
      "[bench_neutron_inner] name: {name}, num_circuits: {}, verify res: {:?}",
      step_circuits.len(),
      res
    );
    assert!(res.is_ok());

    let (public_values_step, _public_values_core) = res.unwrap();
    assert_eq!(public_values_step.len(), step_circuits.len());
  }

  #[test]
  fn test_neutron_sha256() {
    let _ = tracing_subscriber::fmt()
      .with_target(false)
      .with_ansi(true)
      .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
      .try_init();

    type E = T256HyraxEngine;

    for num_circuits in [2, 7, 32, 64] {
      for len in [32, 64].iter() {
        let (pk, vk, circuits) = generate_sha_r1cs::<E>(num_circuits, *len);
        test_neutron_inner(
          &format!("sha256_num_circuits={num_circuits}_len={len}"),
          &pk,
          &vk,
          &circuits,
          &circuits[0], // core circuit is the first one, for test purposes
        );
      }
    }
  }

  #[test]
  fn test_neutron_sha256_small_value_backend_equivalence() -> Result<(), SpartanError> {
    const NUM_CIRCUITS: usize = 7;
    const LEN: usize = 32;
    type E = T256HyraxEngine;

    let (pk, vk, circuits) = generate_sha_r1cs::<E>(NUM_CIRCUITS, LEN);
    let regular_input = false;
    let small_input = 1usize;
    let normal_outputs = prove_and_verify_neutron::<E, _, _, NeutronNovaNIFS<E>>(
      &pk,
      &vk,
      &circuits,
      &circuits[0],
      &regular_input,
    )?;
    let regular_small_input = true;
    let regular_small_outputs = prove_and_verify_neutron::<E, _, _, NeutronNovaNIFS<E>>(
      &pk,
      &vk,
      &circuits,
      &circuits[0],
      &regular_small_input,
    )?;
    let small_outputs = prove_and_verify_neutron::<E, _, _, SmallValueNeutronNovaNIFS<E>>(
      &pk,
      &vk,
      &circuits,
      &circuits[0],
      &small_input,
    )?;

    assert_eq!(normal_outputs, regular_small_outputs);
    assert_eq!(normal_outputs, small_outputs);
    assert_eq!(normal_outputs.0.len(), NUM_CIRCUITS);

    Ok(())
  }

  #[test]
  fn test_prep_prove_stores_nifs_input_when_matvec_cache_skipped() -> Result<(), SpartanError> {
    type E = T256HyraxEngine;
    const NUM_CIRCUITS: usize = 2;
    const LEN: usize = 32;
    const SMALL_VALUE_L0: usize = 3;

    let (pk, _vk, circuits) = generate_sha_r1cs::<E>(NUM_CIRCUITS, LEN);
    let prep = NeutronNovaZkSNARK::<E>::prep_prove::<_, _, SmallValueNeutronNovaNIFS<E>>(
      &pk,
      &circuits,
      &circuits[0],
      &SMALL_VALUE_L0,
    )?;

    assert!(prep.cached_step_matvecs.is_none());
    assert_eq!(prep.nifs_input, SMALL_VALUE_L0);

    Ok(())
  }

  #[test]
  fn test_regular_false_skips_small_conversion() -> Result<(), SpartanError> {
    type E = T256HyraxEngine;
    const NUM_CIRCUITS: usize = 2;

    let step_proto = CacheablePublicCircuit::<E> {
      value: <E as Engine>::Scalar::ZERO,
    };
    let core_proto = step_proto.clone();
    let (pk, vk) = NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, NUM_CIRCUITS)?;
    let step_circuits = (0..NUM_CIRCUITS)
      .map(|i| CacheablePublicCircuit::<E> {
        value: <E as Engine>::Scalar::from(i as u64),
      })
      .collect::<Vec<_>>();
    let core_circuit = CacheablePublicCircuit::<E> {
      value: <E as Engine>::Scalar::ZERO,
    };

    let nifs_input = false;
    let prep = NeutronNovaZkSNARK::<E>::prep_prove::<_, _, NeutronNovaNIFS<E>>(
      &pk,
      &step_circuits,
      &core_circuit,
      &nifs_input,
    )?;
    let cached = prep
      .cached_step_matvecs
      .as_ref()
      .expect("cacheable circuit should produce cached matvecs");
    assert!(cached.small_abc.is_none());
    assert_eq!(cached.field.az.len(), NUM_CIRCUITS);
    assert_eq!(cached.field.bz.len(), NUM_CIRCUITS);
    assert_eq!(cached.field.cz.len(), NUM_CIRCUITS);

    let (snark, _prep) = NeutronNovaZkSNARK::<E>::prove::<_, _, NeutronNovaNIFS<E>>(
      &pk,
      &step_circuits,
      &core_circuit,
      prep,
      &nifs_input,
    )?;
    let (step_public, _core_public) = snark.verify(&vk, NUM_CIRCUITS)?;
    assert_eq!(step_public.len(), NUM_CIRCUITS);

    Ok(())
  }

  #[test]
  fn test_regular_true_builds_small_abc_cache() -> Result<(), SpartanError> {
    type E = T256HyraxEngine;
    const NUM_CIRCUITS: usize = 2;

    let step_proto = CacheablePublicCircuit::<E> {
      value: <E as Engine>::Scalar::ZERO,
    };
    let core_proto = step_proto.clone();
    let (pk, _vk) = NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, NUM_CIRCUITS)?;
    let step_circuits = (0..NUM_CIRCUITS)
      .map(|i| CacheablePublicCircuit::<E> {
        value: <E as Engine>::Scalar::from(i as u64),
      })
      .collect::<Vec<_>>();
    let core_circuit = CacheablePublicCircuit::<E> {
      value: <E as Engine>::Scalar::ZERO,
    };

    let nifs_input = true;
    let prep = NeutronNovaZkSNARK::<E>::prep_prove::<_, _, NeutronNovaNIFS<E>>(
      &pk,
      &step_circuits,
      &core_circuit,
      &nifs_input,
    )?;
    let cached = prep
      .cached_step_matvecs
      .as_ref()
      .expect("cacheable circuit should produce cached matvecs");
    let small_abc = cached
      .small_abc
      .as_ref()
      .expect("regular true should build small A/B/C matvecs");

    assert_eq!(small_abc.ab.l0, 1);
    assert_eq!(small_abc.ab.az_small.len(), NUM_CIRCUITS);
    assert_eq!(small_abc.ab.bz_small.len(), NUM_CIRCUITS);
    assert_eq!(small_abc.cz_small.len(), NUM_CIRCUITS);

    Ok(())
  }

  #[test]
  fn test_regular_round0_small_helper_matches_field_claim() -> Result<(), SpartanError> {
    type E = T256HyraxEngine;
    type Scalar = <E as Engine>::Scalar;

    let left = 2;
    let right = 2;
    let e = vec![
      Scalar::from(2),
      Scalar::from(3),
      Scalar::from(5),
      Scalar::from(7),
    ];
    let rhos = vec![Scalar::from(11)];
    let az = vec![
      vec![
        Scalar::from(1),
        Scalar::from(2),
        Scalar::from(3),
        Scalar::from(4),
      ],
      vec![
        Scalar::from(2),
        Scalar::from(4),
        Scalar::from(6),
        Scalar::from(8),
      ],
    ];
    let bz = vec![
      vec![
        Scalar::from(5),
        Scalar::from(6),
        Scalar::from(7),
        Scalar::from(8),
      ],
      vec![
        Scalar::from(3),
        Scalar::from(6),
        Scalar::from(9),
        Scalar::from(12),
      ],
    ];
    let cz = vec![vec![Scalar::ZERO; left * right]; 2];

    let (_field_e0, field_quad) =
      compute_field_round_claim::<E>(&az, &bz, &cz, &e, left, right, &rhos, 0)?;

    let mut az_small = vec![vec![1, 2, 3, 4], vec![2, 4, 6, 8]];
    let mut bz_small = vec![vec![5, 6, 7, 8], vec![3, 6, 9, 12]];
    let large_positions = vec![2usize];
    for layer in az_small.iter_mut().chain(bz_small.iter_mut()) {
      layer[2] = 0;
    }

    let small_quad = NeutronNovaNIFS::<E>::prove_helper_small(
      (left, right),
      &e,
      &az[0],
      &bz[0],
      &az[1],
      &bz[1],
      &az_small[0],
      &bz_small[0],
      &az_small[1],
      &bz_small[1],
      &large_positions,
    );

    assert_eq!(field_quad, small_quad);

    Ok(())
  }

  #[test]
  fn test_regular_small_c_claims_match_field_with_large_correction() -> Result<(), SpartanError> {
    type E = T256HyraxEngine;
    type Scalar = <E as Engine>::Scalar;

    let left = 2;
    let right = 2;
    let e = vec![
      Scalar::from(2),
      Scalar::from(3),
      Scalar::from(5),
      Scalar::from(7),
    ];
    let c_field = vec![
      vec![
        Scalar::from(1),
        Scalar::from(2),
        Scalar::from(30),
        Scalar::from(4),
      ],
      vec![
        Scalar::from(5),
        Scalar::from(6),
        Scalar::from(70),
        Scalar::from(8),
      ],
    ];
    let mut c_small = vec![vec![1, 2, 30, 4], vec![5, 6, 70, 8]];
    let large_positions = vec![2usize];
    for layer in &mut c_small {
      layer[2] = 0;
    }

    let field_claims = compute_field_c_claims::<E>(&c_field, &e, left, right)?;
    let small_claims = compute_small_c_claims_with_field_corrections::<E>(
      &c_small,
      &c_field,
      &large_positions,
      &e,
      left,
      right,
    )?;

    assert_eq!(field_claims, small_claims);

    Ok(())
  }

  #[test]
  fn test_regular_final_small_c_fold_matches_field_with_large_correction()
  -> Result<(), SpartanError> {
    type E = T256HyraxEngine;
    type Scalar = <E as Engine>::Scalar;

    let c_field = vec![
      vec![Scalar::from(1), Scalar::from(20), Scalar::from(3)],
      vec![Scalar::from(4), Scalar::from(50), Scalar::from(6)],
      vec![Scalar::from(7), Scalar::from(80), Scalar::from(9)],
    ];
    let mut c_small = vec![vec![1, 20, 3], vec![4, 50, 6], vec![7, 80, 9]];
    let large_positions = vec![1usize];
    for layer in &mut c_small {
      layer[1] = 0;
    }
    let weights = vec![Scalar::from(2), Scalar::from(3), Scalar::from(5)];

    let field_fold = fold_field_layers_with_weights::<E>(&c_field, &weights)?;
    let small_fold =
      fold_small_c_with_field_corrections::<E>(&c_small, &c_field, &large_positions, &weights)?;

    assert_eq!(field_fold, small_fold);

    Ok(())
  }

  #[test]
  fn test_cached_step_matvecs_are_one_shot() -> Result<(), SpartanError> {
    type E = T256HyraxEngine;
    const NUM_CIRCUITS: usize = 2;

    let step_proto = CacheablePublicCircuit::<E> {
      value: <E as Engine>::Scalar::ZERO,
    };
    let core_proto = step_proto.clone();
    let (pk, _vk) = NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, NUM_CIRCUITS)?;
    let step_circuits = (0..NUM_CIRCUITS)
      .map(|i| CacheablePublicCircuit::<E> {
        value: <E as Engine>::Scalar::from(i as u64),
      })
      .collect::<Vec<_>>();
    let core_circuit = CacheablePublicCircuit::<E> {
      value: <E as Engine>::Scalar::ZERO,
    };

    let nifs_input = true;
    let prep = NeutronNovaZkSNARK::<E>::prep_prove::<_, _, NeutronNovaNIFS<E>>(
      &pk,
      &step_circuits,
      &core_circuit,
      &nifs_input,
    )?;
    assert!(prep.cached_step_matvecs.is_some());

    let (_snark, consumed_prep) = NeutronNovaZkSNARK::<E>::prove::<_, _, NeutronNovaNIFS<E>>(
      &pk,
      &step_circuits,
      &core_circuit,
      prep,
      &nifs_input,
    )?;
    assert!(consumed_prep.cached_step_matvecs.is_none());
    assert!(!consumed_prep.cached_step_public_values.is_empty());

    let err = NeutronNovaZkSNARK::<E>::prove::<_, _, NeutronNovaNIFS<E>>(
      &pk,
      &step_circuits,
      &core_circuit,
      consumed_prep,
      &nifs_input,
    )
    .expect_err("consumed cached prep should require rerunning prep_prove");

    assert!(matches!(
      err,
      SpartanError::InternalError { reason }
        if reason.contains("rerun prep_prove")
    ));

    Ok(())
  }

  #[test]
  fn test_small_value_neutronnova_nifs_rejects_zero_l0() -> Result<(), SpartanError> {
    type E = T256HyraxEngine;
    const NUM_CIRCUITS: usize = 2;
    const LEN: usize = 32;

    let (pk, _vk, circuits) = generate_sha_r1cs::<E>(NUM_CIRCUITS, LEN);
    let nifs_input = 0usize;
    let prep = NeutronNovaZkSNARK::<E>::prep_prove::<_, _, SmallValueNeutronNovaNIFS<E>>(
      &pk,
      &circuits,
      &circuits[0],
      &nifs_input,
    )?;
    let err = NeutronNovaZkSNARK::<E>::prove::<_, _, SmallValueNeutronNovaNIFS<E>>(
      &pk,
      &circuits,
      &circuits[0],
      prep,
      &nifs_input,
    )
    .expect_err("small-value NIFS should reject l0 == 0");

    assert!(matches!(
      err,
      SpartanError::InvalidInputLength { reason }
        if reason.contains("small-value NeutronNova NIFS requires l0 > 0")
    ));

    Ok(())
  }

  #[test]
  fn test_neutronnova_prove_rejects_input_mismatch() -> Result<(), SpartanError> {
    type E = T256HyraxEngine;
    const NUM_CIRCUITS: usize = 2;
    const LEN: usize = 32;

    let (pk, _vk, circuits) = generate_sha_r1cs::<E>(NUM_CIRCUITS, LEN);
    let prep_input = 2usize;
    let prove_input = 1usize;
    let prep = NeutronNovaZkSNARK::<E>::prep_prove::<_, _, SmallValueNeutronNovaNIFS<E>>(
      &pk,
      &circuits,
      &circuits[0],
      &prep_input,
    )?;
    let err = NeutronNovaZkSNARK::<E>::prove::<_, _, SmallValueNeutronNovaNIFS<E>>(
      &pk,
      &circuits,
      &circuits[0],
      prep,
      &prove_input,
    )
    .expect_err("prove should reject a different NIFS input than prep_prove used");

    assert!(matches!(
      err,
      SpartanError::InvalidInputLength { reason }
        if reason.contains("prep/prove NIFS inputs do not match")
    ));

    Ok(())
  }

  #[test]
  fn test_small_value_neutronnova_nifs_accepts_runtime_small_value_l0() -> Result<(), SpartanError>
  {
    type E = T256HyraxEngine;
    const NUM_CIRCUITS: usize = 4;
    const LEN: usize = 32;

    let (pk, vk, circuits) = generate_sha_r1cs::<E>(NUM_CIRCUITS, LEN);
    let nifs_input = 2usize;
    let prep = NeutronNovaZkSNARK::<E>::prep_prove::<_, _, SmallValueNeutronNovaNIFS<E>>(
      &pk,
      &circuits,
      &circuits[0],
      &nifs_input,
    )?;
    assert_eq!(prep.nifs_input, 2);
    assert!(prep.cached_step_matvecs.is_none());

    let (snark, _prep) = NeutronNovaZkSNARK::<E>::prove::<_, _, SmallValueNeutronNovaNIFS<E>>(
      &pk,
      &circuits,
      &circuits[0],
      prep,
      &nifs_input,
    )?;
    let (step_public, _core_public) = snark.verify(&vk, NUM_CIRCUITS)?;
    assert_eq!(step_public.len(), NUM_CIRCUITS);

    Ok(())
  }
}
