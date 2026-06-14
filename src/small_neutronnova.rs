// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Small-value NeutronNova NIFS helpers.

use crate::{
  CommitmentKey,
  bellpepper::{r1cs::MultiRoundSpartanWitness, solver::SatisfyingAssignment},
  big_num::{DelayedReduction, SmallValue, SmallValueEngine},
  errors::SpartanError,
  lagrange_accumulator::{
    SMALL_VALUE_T_DEGREE, SmallValueExtensionBoundedPoly, build_accumulators_neutronnova,
  },
  neutronnova_zk::{
    NeutronNovaNIFS, compact_folded_layers, finalize_nifs_step_claim, finish_nifs_field_round,
    fold_layer_pair_into, fold_quad_chunk, fold_witness_and_instance, process_nifs_round,
    suffix_weight_full,
  },
  r1cs::{R1CSInstance, R1CSWitness, SplitMultiRoundR1CSShape, SplitR1CSShape, weights_from_r},
  small_sumcheck::{SmallValueSumCheck, generate_univariate_sumcheck_polynomial_from_accumulator},
  traits::{Engine, pcs::FoldingEngineTrait},
  zk::NeutronNovaVerifierCircuit,
};
use ff::Field;
use num_traits::Zero;
use rayon::prelude::*;

/// Certified small A/B layers for small-value NeutronNova NIFS.
pub(crate) struct SmallNeutronNovaAb<'poly, 'layers, SV, const LB: usize>
where
  SV: SmallValue,
{
  pub(crate) num_instances: usize,
  pub(crate) num_constraints: usize,
  pub(crate) a: &'layers [SmallValueExtensionBoundedPoly<'poly, SV, LB>],
  pub(crate) b: &'layers [SmallValueExtensionBoundedPoly<'poly, SV, LB>],
}

impl<E: Engine> NeutronNovaNIFS<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Prove small-value NeutronNova NIFS, deriving round challenges from the verifier circuit.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn prove_small<SV, const LB: usize>(
    s: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    us: Vec<R1CSInstance<E>>,
    ws: Vec<R1CSWitness<E>>,
    small_ab: &SmallNeutronNovaAb<'_, '_, SV, LB>,
    e_eq: &[E::Scalar],
    left: usize,
    right: usize,
    rhos: &[E::Scalar],
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<
    (
      Vec<E::Scalar>,
      Vec<E::Scalar>,
      Vec<E::Scalar>,
      Vec<E::Scalar>,
      R1CSWitness<E>,
      R1CSInstance<E>,
    ),
    SpartanError,
  >
  where
    SV: SmallValue,
    E::Scalar: SmallValueEngine<SV>,
  {
    let n_padded = validate_small_ab::<E, SV, LB>(small_ab, e_eq, left, right, rhos)?;
    if vc.nifs_polys.len() != rhos.len() {
      return Err(invalid_input(format!(
        "verifier circuit has {} NIFS rounds but rhos has {}",
        vc.nifs_polys.len(),
        rhos.len()
      )));
    }

    let a_small = padded_small_layers(small_ab.a, small_ab.num_instances, n_padded);
    let b_small = padded_small_layers(small_ab.b, small_ab.num_instances, n_padded);

    let accumulators = build_accumulators_neutronnova::<E::Scalar, SV, LB>(
      &a_small, &b_small, e_eq, left, right, rhos,
    )?;
    let mut small_value =
      SmallValueSumCheck::<E::Scalar, SMALL_VALUE_T_DEGREE>::from_accumulators(accumulators);
    let a_small_evals = layer_evals(&a_small);
    let b_small_evals = layer_evals(&b_small);

    let ell_b = rhos.len();
    let mut r_bs = Vec::with_capacity(ell_b);
    let mut t_cur = E::Scalar::ZERO;
    let mut acc_eq = E::Scalar::ONE;

    for round in 0..LB {
      let (poly, li) = generate_univariate_sumcheck_polynomial_from_accumulator(
        &small_value,
        round,
        rhos[round],
        t_cur,
      )?;
      let r_i = process_nifs_round(vc, vc_state, vc_shape, vc_ck, transcript, round, &poly)?;
      t_cur = poly.evaluate(&r_i);
      acc_eq = li.eval_linear_at(r_i);
      small_value.advance(&li, r_i);
      r_bs.push(r_i);
    }
    drop(small_value);

    let (az_step, bz_step, cz_step) = if LB == ell_b {
      let final_weights = weights_from_r::<E::Scalar>(&r_bs, n_padded);
      fold_small_final_abc_with_weights::<E, SV>(&a_small_evals, &b_small_evals, &final_weights)?
    } else {
      let prefix_size = 1usize << LB;
      let prefix_weights = weights_from_r::<E::Scalar>(&r_bs, prefix_size);
      let PrefixAbWithCClaims {
        mut a_layers,
        mut b_layers,
        mut c_claims,
      } = materialize_prefix_ab_with_c_claims::<E, SV>(
        &a_small_evals,
        &b_small_evals,
        &prefix_weights,
        prefix_size,
        e_eq,
        left,
        right,
      )?;

      continue_ab_suffix_with_c_claims(
        &mut a_layers,
        &mut b_layers,
        &mut c_claims,
        e_eq,
        left,
        right,
        rhos,
        vc,
        vc_state,
        vc_shape,
        vc_ck,
        transcript,
        LB,
        &mut r_bs,
        &mut t_cur,
        &mut acc_eq,
      )?;

      let final_weights = weights_from_r::<E::Scalar>(&r_bs, n_padded);
      let mut c_folded = fold_small_c_from_ab_with_weights::<E, SV>(
        &a_small_evals,
        &b_small_evals,
        &final_weights,
        n_padded,
      )?;

      (
        a_layers.pop().ok_or_else(empty_fold_error)?,
        b_layers.pop().ok_or_else(empty_fold_error)?,
        c_folded.pop().ok_or_else(empty_fold_error)?,
      )
    };

    finalize_nifs_step_claim(
      vc, vc_state, vc_shape, vc_ck, transcript, ell_b, t_cur, acc_eq,
    )?;

    let (folded_w, folded_u) =
      fold_witness_and_instance(s, ck, us, ws, small_ab.num_instances, n_padded, &r_bs)?;

    Ok((e_eq.to_vec(), az_step, bz_step, cz_step, folded_w, folded_u))
  }
}

fn invalid_input(reason: impl Into<String>) -> SpartanError {
  SpartanError::InvalidInputLength {
    reason: reason.into(),
  }
}

fn empty_fold_error() -> SpartanError {
  invalid_input("small NeutronNova NIFS produced no folded layer")
}

fn validate_small_ab<E, SV, const LB: usize>(
  small_ab: &SmallNeutronNovaAb<'_, '_, SV, LB>,
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
  rhos: &[E::Scalar],
) -> Result<usize, SpartanError>
where
  E: Engine,
  SV: SmallValue,
{
  if small_ab.num_instances == 0 {
    return Err(invalid_input(
      "small NeutronNova NIFS requires at least one instance",
    ));
  }
  if LB == 0 {
    return Err(invalid_input("small NeutronNova NIFS requires l0 > 0"));
  }
  let ell_b = rhos.len();
  if LB > ell_b {
    return Err(invalid_input(format!(
      "small NeutronNova l0 {} exceeds ell_b {}",
      LB, ell_b
    )));
  }

  let shift =
    u32::try_from(ell_b).map_err(|_| invalid_input("ell_b does not fit in a u32 shift"))?;
  let n_padded = 1usize
    .checked_shl(shift)
    .ok_or_else(|| invalid_input("ell_b is too large for this platform"))?;
  if small_ab.num_instances.next_power_of_two() != n_padded {
    return Err(invalid_input(format!(
      "num_instances {} is incompatible with ell_b {}",
      small_ab.num_instances, ell_b
    )));
  }

  let expected_constraints = left
    .checked_mul(right)
    .ok_or_else(|| invalid_input("left * right overflows"))?;
  if small_ab.num_constraints != expected_constraints {
    return Err(invalid_input(format!(
      "num_constraints {} does not match left * right {}",
      small_ab.num_constraints, expected_constraints
    )));
  }
  let expected_eq_len = left
    .checked_add(right)
    .ok_or_else(|| invalid_input("left + right overflows"))?;
  if e_eq.len() != expected_eq_len {
    return Err(invalid_input(format!(
      "e_eq length {} does not match left + right {}",
      e_eq.len(),
      expected_eq_len
    )));
  }

  if small_ab.a.len() != small_ab.num_instances {
    return Err(invalid_input(format!(
      "A layer count {} does not match num_instances {}",
      small_ab.a.len(),
      small_ab.num_instances
    )));
  }
  if small_ab.b.len() != small_ab.num_instances {
    return Err(invalid_input(format!(
      "B layer count {} does not match num_instances {}",
      small_ab.b.len(),
      small_ab.num_instances
    )));
  }
  if !small_ab
    .a
    .iter()
    .all(|layer| layer.as_poly().Z.len() == small_ab.num_constraints)
  {
    return Err(invalid_input(format!(
      "all A layers must have length num_constraints ({})",
      small_ab.num_constraints
    )));
  }
  if !small_ab
    .b
    .iter()
    .all(|layer| layer.as_poly().Z.len() == small_ab.num_constraints)
  {
    return Err(invalid_input(format!(
      "all B layers must have length num_constraints ({})",
      small_ab.num_constraints
    )));
  }

  Ok(n_padded)
}

fn padded_small_layers<'poly, SV, const LB: usize>(
  layers: &[SmallValueExtensionBoundedPoly<'poly, SV, LB>],
  num_instances: usize,
  n_padded: usize,
) -> Vec<SmallValueExtensionBoundedPoly<'poly, SV, LB>>
where
  SV: SmallValue,
{
  (0..n_padded)
    .map(|idx| {
      let row = if idx < num_instances { idx } else { 0 };
      layers[row]
    })
    .collect()
}

fn layer_evals<'poly, SV, const LB: usize>(
  layers: &[SmallValueExtensionBoundedPoly<'poly, SV, LB>],
) -> Vec<&'poly [SV]>
where
  SV: SmallValue,
{
  layers
    .iter()
    .map(|layer| layer.as_poly().Z.as_slice())
    .collect()
}

struct PrefixAbWithCClaims<F> {
  a_layers: Vec<Vec<F>>,
  b_layers: Vec<Vec<F>>,
  c_claims: Vec<F>,
}

#[allow(clippy::too_many_arguments)]
fn materialize_prefix_ab_with_c_claims<E, SV>(
  a_layers: &[&[SV]],
  b_layers: &[&[SV]],
  prefix_weights: &[E::Scalar],
  prefix_size: usize,
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
) -> Result<PrefixAbWithCClaims<E::Scalar>, SpartanError>
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  validate_small_ab_layers(a_layers, b_layers)?;
  if prefix_size == 0 || a_layers.len() % prefix_size != 0 {
    return Err(invalid_input("invalid small prefix group size"));
  }
  if prefix_weights.len() != prefix_size {
    return Err(invalid_input(format!(
      "prefix weight length {} does not match prefix size {}",
      prefix_weights.len(),
      prefix_size
    )));
  }
  validate_split_eq_shape::<E>(a_layers[0].len(), e_eq, left, right)?;

  let a_folded = fold_small_layers_with_weights::<E, SV>(a_layers, prefix_weights, prefix_size)?;
  let b_folded = fold_small_layers_with_weights::<E, SV>(b_layers, prefix_weights, prefix_size)?;
  let c_claims = a_layers
    .par_chunks(prefix_size)
    .zip(b_layers.par_chunks(prefix_size))
    .map(|(a_group, b_group)| {
      let mut acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
      for ((weight, a_layer), b_layer) in prefix_weights
        .iter()
        .zip(a_group.iter())
        .zip(b_group.iter())
      {
        let c_claim = dot_small_product_with_split_eq::<E, SV>(a_layer, b_layer, e_eq, left, right);
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut acc, weight, &c_claim,
        );
      }
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc)
    })
    .collect();

  Ok(PrefixAbWithCClaims {
    a_layers: a_folded,
    b_layers: b_folded,
    c_claims,
  })
}

fn validate_small_ab_layers<SV>(a_layers: &[&[SV]], b_layers: &[&[SV]]) -> Result<(), SpartanError>
where
  SV: SmallValue,
{
  if a_layers.is_empty() {
    return Err(invalid_input("cannot fold empty small A/B layer list"));
  }
  if a_layers.len() != b_layers.len() {
    return Err(invalid_input("small A/B layer counts do not match"));
  }
  let layer_len = a_layers[0].len();
  if !a_layers
    .iter()
    .zip(b_layers.iter())
    .all(|(a, b)| a.len() == layer_len && b.len() == layer_len)
  {
    return Err(invalid_input(
      "all small A/B layers must have the same length",
    ));
  }
  Ok(())
}

fn validate_split_eq_shape<E>(
  layer_len: usize,
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
) -> Result<(), SpartanError>
where
  E: Engine,
{
  let expected_layer_len = left
    .checked_mul(right)
    .ok_or_else(|| invalid_input("left * right overflows"))?;
  if layer_len != expected_layer_len {
    return Err(invalid_input(format!(
      "layer length {} does not match left * right {}",
      layer_len, expected_layer_len
    )));
  }
  let expected_eq_len = left
    .checked_add(right)
    .ok_or_else(|| invalid_input("left + right overflows"))?;
  if e_eq.len() != expected_eq_len {
    return Err(invalid_input(format!(
      "e_eq length {} does not match left + right {}",
      e_eq.len(),
      expected_eq_len
    )));
  }
  Ok(())
}

fn dot_small_product_with_split_eq<E, SV>(
  a_layer: &[SV],
  b_layer: &[SV],
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
) -> E::Scalar
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  let e_left = &e_eq[..left];
  let e_right = &e_eq[left..];
  let mut acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();

  #[allow(clippy::needless_range_loop)]
  for i in 0..right {
    let base = i * left;
    let mut inner = <E::Scalar as DelayedReduction<SV::Product>>::Accumulator::zero();
    for j in 0..left {
      let product = SV::wide_mul(a_layer[base + j], b_layer[base + j]);
      <E::Scalar as DelayedReduction<SV::Product>>::unreduced_multiply_accumulate(
        &mut inner, &e_left[j], &product,
      );
    }
    let inner_red = <E::Scalar as DelayedReduction<SV::Product>>::reduce(&inner);
    <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
      &mut acc,
      &e_right[i],
      &inner_red,
    );
  }

  <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc)
}

fn validate_ab_c_claim_state<F>(
  a_layers: &[Vec<F>],
  b_layers: &[Vec<F>],
  c_claims: &[F],
) -> Result<(), SpartanError> {
  if a_layers.len() != b_layers.len() || a_layers.len() != c_claims.len() {
    return Err(invalid_input("A/B layer and C-claim counts do not match"));
  }
  if a_layers.len() % 2 != 0 {
    return Err(invalid_input("suffix layer count must be even"));
  }
  Ok(())
}

fn compute_ab_c_claim_round<E>(
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
  let ell_b = rhos.len();
  validate_ab_c_claim_state(a_layers, b_layers, c_claims)?;

  Ok(
    a_layers
      .par_chunks(2)
      .zip(b_layers.par_chunks(2))
      .zip(c_claims.par_chunks(2))
      .enumerate()
      .map(|(pair_idx, ((pair_a, pair_b), pair_c))| {
        let (e0_ab, quad_coeff) = NeutronNovaNIFS::<E>::prove_helper_ab_only(
          (left, right),
          e_eq,
          &pair_a[0],
          &pair_b[0],
          &pair_a[1],
          &pair_b[1],
        );
        let e0 = e0_ab - pair_c[0];
        let w = suffix_weight_full::<E::Scalar>(round, ell_b, pair_idx, rhos);
        (e0 * w, quad_coeff * w)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1),
      ),
  )
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
  for i in 0..pairs {
    fold_layer_pair_into(a_layers, 2 * i, 2 * i + 1, i, r);
    fold_layer_pair_into(b_layers, 2 * i, 2 * i + 1, i, r);
    fold_c_claim_pair_into(c_claims, 2 * i, 2 * i + 1, i, r);
  }
}

#[allow(clippy::too_many_arguments)]
fn continue_ab_suffix_with_c_claims<E>(
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
  let ell_b = rhos.len();
  if start_round >= ell_b {
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
  let mut prev_r_b = finish_nifs_field_round(
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

  let mut layer_count = a_layers.len();
  for round in (start_round + 1)..ell_b {
    validate_ab_c_claim_state(a_layers, b_layers, c_claims)?;
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
        fold_quad_chunk(a_chunk, prev_r_b);
        fold_quad_chunk(b_chunk, prev_r_b);
        fold_quad_c_claim_chunk(c_chunk, prev_r_b);

        let (e0_ab, quad_coeff) = NeutronNovaNIFS::<E>::prove_helper_ab_only(
          (left, right),
          e_eq,
          &a_chunk[0],
          &b_chunk[0],
          &a_chunk[2],
          &b_chunk[2],
        );
        let e0 = e0_ab - c_chunk[0];
        let w = suffix_weight_full::<E::Scalar>(round, ell_b, pair_idx, rhos);
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

    prev_r_b = finish_nifs_field_round(
      rhos, round, e0, quad_coeff, vc, vc_state, vc_shape, vc_ck, transcript, r_bs, t_cur, acc_eq,
    )?;
  }

  validate_ab_c_claim_state(a_layers, b_layers, c_claims)?;
  let final_pairs = layer_count / 2;
  fold_ab_c_claim_pairs::<E>(a_layers, b_layers, c_claims, final_pairs, prev_r_b);
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

fn fold_small_final_abc_with_weights<E, SV>(
  a_layers: &[&[SV]],
  b_layers: &[&[SV]],
  weights: &[E::Scalar],
) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError>
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  validate_small_ab_layers(a_layers, b_layers)?;
  if weights.len() != a_layers.len() {
    return Err(invalid_input(format!(
      "weight length {} does not match layer count {}",
      weights.len(),
      a_layers.len()
    )));
  }

  let layer_len = a_layers[0].len();
  let folded = (0..layer_len)
    .into_par_iter()
    .map(|k| {
      let mut a_acc = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();
      let mut b_acc = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();
      let mut c_acc = <E::Scalar as DelayedReduction<SV::Product>>::Accumulator::zero();

      for ((weight, a_layer), b_layer) in weights.iter().zip(a_layers.iter()).zip(b_layers.iter()) {
        <E::Scalar as DelayedReduction<SV>>::unreduced_multiply_accumulate(
          &mut a_acc,
          weight,
          &a_layer[k],
        );
        <E::Scalar as DelayedReduction<SV>>::unreduced_multiply_accumulate(
          &mut b_acc,
          weight,
          &b_layer[k],
        );
        let product = SV::wide_mul(a_layer[k], b_layer[k]);
        <E::Scalar as DelayedReduction<SV::Product>>::unreduced_multiply_accumulate(
          &mut c_acc, weight, &product,
        );
      }

      (
        <E::Scalar as DelayedReduction<SV>>::reduce(&a_acc),
        <E::Scalar as DelayedReduction<SV>>::reduce(&b_acc),
        <E::Scalar as DelayedReduction<SV::Product>>::reduce(&c_acc),
      )
    })
    .collect::<Vec<_>>();

  let mut a_final = Vec::with_capacity(layer_len);
  let mut b_final = Vec::with_capacity(layer_len);
  let mut c_final = Vec::with_capacity(layer_len);
  for (a, b, c) in folded {
    a_final.push(a);
    b_final.push(b);
    c_final.push(c);
  }

  Ok((a_final, b_final, c_final))
}

fn fold_small_layers_with_weights<E, SV>(
  layers: &[&[SV]],
  weights: &[E::Scalar],
  group_size: usize,
) -> Result<Vec<Vec<E::Scalar>>, SpartanError>
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  if layers.is_empty() {
    return Err(invalid_input("cannot fold empty small layer list"));
  }
  if group_size == 0 || layers.len() % group_size != 0 {
    return Err(invalid_input("invalid small-layer fold group size"));
  }
  if weights.len() != group_size {
    return Err(invalid_input(format!(
      "weight length {} does not match group size {}",
      weights.len(),
      group_size
    )));
  }
  let layer_len = layers[0].len();
  if !layers.iter().all(|layer| layer.len() == layer_len) {
    return Err(invalid_input("all small layers must have the same length"));
  }

  Ok(
    layers
      .par_chunks(group_size)
      .map(|group| {
        (0..layer_len)
          .into_par_iter()
          .map(|k| {
            let mut acc = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();
            for (weight, layer) in weights.iter().zip(group.iter()) {
              <E::Scalar as DelayedReduction<SV>>::unreduced_multiply_accumulate(
                &mut acc,
                weight,
                &(*layer)[k],
              );
            }
            <E::Scalar as DelayedReduction<SV>>::reduce(&acc)
          })
          .collect()
      })
      .collect(),
  )
}

fn fold_small_c_from_ab_with_weights<E, SV>(
  a_layers: &[&[SV]],
  b_layers: &[&[SV]],
  weights: &[E::Scalar],
  group_size: usize,
) -> Result<Vec<Vec<E::Scalar>>, SpartanError>
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  if a_layers.is_empty() {
    return Err(invalid_input("cannot fold empty small A/B layer list"));
  }
  if a_layers.len() != b_layers.len() {
    return Err(invalid_input("small A/B layer counts do not match"));
  }
  if group_size == 0 || a_layers.len() % group_size != 0 {
    return Err(invalid_input("invalid small C fold group size"));
  }
  if weights.len() != group_size {
    return Err(invalid_input(format!(
      "weight length {} does not match group size {}",
      weights.len(),
      group_size
    )));
  }
  let layer_len = a_layers[0].len();
  if !a_layers
    .iter()
    .zip(b_layers.iter())
    .all(|(a, b)| a.len() == layer_len && b.len() == layer_len)
  {
    return Err(invalid_input(
      "all small A/B layers must have the same length",
    ));
  }

  Ok(
    a_layers
      .par_chunks(group_size)
      .zip(b_layers.par_chunks(group_size))
      .map(|(a_group, b_group)| {
        (0..layer_len)
          .into_par_iter()
          .map(|k| {
            let mut acc = <E::Scalar as DelayedReduction<SV::Product>>::Accumulator::zero();
            for ((weight, a_layer), b_layer) in
              weights.iter().zip(a_group.iter()).zip(b_group.iter())
            {
              let product = SV::wide_mul((*a_layer)[k], (*b_layer)[k]);
              <E::Scalar as DelayedReduction<SV::Product>>::unreduced_multiply_accumulate(
                &mut acc, weight, &product,
              );
            }
            <E::Scalar as DelayedReduction<SV::Product>>::reduce(&acc)
          })
          .collect()
      })
      .collect(),
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    PCS,
    bellpepper::{
      r1cs::{MultiRoundSpartanShape, MultiRoundSpartanWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
    },
    big_num::SmallValueField,
    polys::{multilinear::MultilinearPolynomial, power::PowPolynomial},
    provider::{PallasHyraxEngine, pcs::hyrax_pc::with_test_blind_seed},
    r1cs::{R1CSInstance, R1CSWitness, SparseMatrix},
    traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait},
  };
  use once_cell::sync::OnceCell;
  use std::fmt::Debug;

  type E = PallasHyraxEngine;
  type F = <E as Engine>::Scalar;

  fn synthetic_small_ab_polys<SV>(
    num_instances: usize,
    num_constraints: usize,
  ) -> (
    Vec<MultilinearPolynomial<SV>>,
    Vec<MultilinearPolynomial<SV>>,
  )
  where
    SV: SmallValue + TryFrom<i64>,
    <SV as TryFrom<i64>>::Error: Debug,
  {
    let a = (0..num_instances)
      .map(|instance| {
        (0..num_constraints)
          .map(move |k| {
            let value = ((13 * instance as i64 + 5 * k as i64 + 7) % 23) - 11;
            SV::try_from(value).unwrap()
          })
          .collect::<Vec<_>>()
      })
      .map(MultilinearPolynomial::new)
      .collect();
    let b = (0..num_instances)
      .map(|instance| {
        (0..num_constraints)
          .map(move |k| {
            let value = ((17 * instance as i64 + 3 * k as i64 + 2) % 29) - 14;
            SV::try_from(value).unwrap()
          })
          .collect::<Vec<_>>()
      })
      .map(MultilinearPolynomial::new)
      .collect();
    (a, b)
  }

  fn certify_layers<SV, const LB: usize>(
    layers: &[MultilinearPolynomial<SV>],
  ) -> Vec<SmallValueExtensionBoundedPoly<'_, SV, LB>>
  where
    SV: SmallValue,
  {
    layers
      .iter()
      .map(|layer| {
        SmallValueExtensionBoundedPoly::<_, LB>::new(layer)
          .expect("synthetic layer should be extension-bounded")
      })
      .collect()
  }

  fn padded_field_layers<SV, const LB: usize>(
    small_ab: &SmallNeutronNovaAb<'_, '_, SV, LB>,
    n_padded: usize,
  ) -> (Vec<Vec<F>>, Vec<Vec<F>>, Vec<Vec<F>>)
  where
    SV: SmallValue,
    F: SmallValueField<SV> + SmallValueEngine<SV>,
  {
    let a_small = padded_small_layers(small_ab.a, small_ab.num_instances, n_padded);
    let b_small = padded_small_layers(small_ab.b, small_ab.num_instances, n_padded);
    let a_small = layer_evals(&a_small);
    let b_small = layer_evals(&b_small);
    let a_field = a_small
      .iter()
      .map(|layer| {
        layer
          .iter()
          .copied()
          .map(<F as SmallValueField<SV>>::small_to_field)
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();
    let b_field = b_small
      .iter()
      .map(|layer| {
        layer
          .iter()
          .copied()
          .map(<F as SmallValueField<SV>>::small_to_field)
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();
    let c_field = a_field
      .iter()
      .zip(b_field.iter())
      .map(|(a, b)| a.iter().zip(b.iter()).map(|(&a, &b)| a * b).collect())
      .collect::<Vec<Vec<F>>>();
    (a_field, b_field, c_field)
  }

  fn synthetic_split_shape(w_len: usize, num_constraints: usize) -> SplitR1CSShape<E> {
    SplitR1CSShape {
      num_cons: num_constraints,
      num_cons_unpadded: num_constraints,
      num_shared_unpadded: 0,
      num_precommitted_unpadded: 0,
      num_rest_unpadded: w_len,
      num_shared: 0,
      num_precommitted: 0,
      num_rest: w_len,
      num_public: 0,
      num_challenges: 0,
      A: SparseMatrix::empty(),
      B: SparseMatrix::empty(),
      C: SparseMatrix::empty(),
      digest: OnceCell::new(),
      precomp_A: OnceCell::new(),
      precomp_B: OnceCell::new(),
      precomp_C: OnceCell::new(),
      filtered_A: OnceCell::new(),
      filtered_B: OnceCell::new(),
      filtered_C: OnceCell::new(),
    }
  }

  fn synthetic_instances_and_witnesses(
    ck: &CommitmentKey<E>,
    num_instances: usize,
    w_len: usize,
    x_len: usize,
  ) -> (Vec<R1CSInstance<E>>, Vec<R1CSWitness<E>>) {
    let mut instances = Vec::with_capacity(num_instances);
    let mut witnesses = Vec::with_capacity(num_instances);

    for instance in 0..num_instances {
      let w = (0..w_len)
        .map(|j| F::from((instance as u64 + 1) * 17 + j as u64 + 3))
        .collect::<Vec<_>>();
      let r_w = PCS::<E>::blind(ck, w_len);
      let comm_w = PCS::<E>::commit(ck, &w, &r_w, false).expect("commitment should succeed");
      let witness =
        R1CSWitness::<E>::new_unchecked(w, r_w, false).expect("witness should construct");
      let x = (0..x_len)
        .map(|j| F::from((instance as u64 + 5) * 19 + j as u64 + 7))
        .collect::<Vec<_>>();
      let instance =
        R1CSInstance::<E>::new_unchecked(comm_w, x).expect("instance should construct");
      instances.push(instance);
      witnesses.push(witness);
    }

    (instances, witnesses)
  }

  fn tensor_decomp(n: usize) -> (usize, usize, usize) {
    let ell = n.next_power_of_two().trailing_zeros() as usize;
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    (ell, 1usize << ell1, 1usize << ell2)
  }

  fn replay_nifs_prefix(
    transcript: &mut <E as Engine>::TE,
    instances: &[R1CSInstance<E>],
    num_constraints: usize,
  ) -> (Vec<F>, Vec<F>, usize, usize) {
    let n_padded = instances.len().next_power_of_two();
    for idx in 0..n_padded {
      let instance = if idx < instances.len() {
        &instances[idx]
      } else {
        &instances[0]
      };
      transcript.absorb(b"U", instance);
    }
    transcript.absorb(b"T", &F::ZERO);

    let (ell_cons, left, right) = tensor_decomp(num_constraints);
    let tau = transcript.squeeze(b"tau").expect("tau should squeeze");
    let e_eq = PowPolynomial::split_evals(tau, ell_cons, left, right);
    let ell_b = n_padded.trailing_zeros() as usize;
    let rhos = (0..ell_b)
      .map(|_| transcript.squeeze(b"rho").expect("rho should squeeze"))
      .collect::<Vec<_>>();

    (e_eq, rhos, left, right)
  }

  fn blind_seed(num_instances: usize, l0: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&(num_instances as u64).to_le_bytes());
    seed[8..16].copy_from_slice(&(l0 as u64).to_le_bytes());
    seed[16..24].copy_from_slice(&(0x5eed_u64).to_le_bytes());
    seed
  }

  fn run_small_matches_standard_case<SV, const LB: usize>(num_instances: usize)
  where
    SV: SmallValue + TryFrom<i64>,
    <SV as TryFrom<i64>>::Error: Debug,
    F: SmallValueEngine<SV> + SmallValueField<SV>,
  {
    let left = 4;
    let right = 2;
    let num_constraints = left * right;
    let n_padded = num_instances.next_power_of_two();
    let ell_b = n_padded.trailing_zeros() as usize;
    let (a_polys, b_polys) = synthetic_small_ab_polys::<SV>(num_instances, num_constraints);
    let a_certs = certify_layers::<SV, LB>(&a_polys);
    let b_certs = certify_layers::<SV, LB>(&b_polys);
    let small_ab = SmallNeutronNovaAb {
      num_instances,
      num_constraints,
      a: &a_certs,
      b: &b_certs,
    };
    let (a_field, b_field, c_field) = padded_field_layers(&small_ab, n_padded);
    let cached_matvec = a_field
      .clone()
      .into_iter()
      .zip(b_field.clone())
      .zip(c_field.clone())
      .map(|((a, b), c)| (a, b, c))
      .collect::<Vec<_>>();

    let mut vc_standard = NeutronNovaVerifierCircuit::<E>::default(ell_b, 1, 2, 32);
    let mut vc_small = NeutronNovaVerifierCircuit::<E>::default(ell_b, 1, 2, 32);
    let (vc_shape, vc_ck, _) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&vc_standard)
        .expect("verifier circuit shape should synthesize");
    let mut vc_state_standard = SatisfyingAssignment::<E>::initialize_multiround_witness(&vc_shape)
      .expect("verifier circuit witness state should initialize");
    let mut vc_state_small = SatisfyingAssignment::<E>::initialize_multiround_witness(&vc_shape)
      .expect("verifier circuit witness state should initialize");
    let s = synthetic_split_shape(5, num_constraints);
    let (instances, witnesses) = synthetic_instances_and_witnesses(&vc_ck, num_instances, 5, 2);

    let mut transcript_standard = <E as Engine>::TE::new(b"small_neutronnova_standard_equivalence");
    let standard = with_test_blind_seed(blind_seed(num_instances, LB), || {
      NeutronNovaNIFS::<E>::prove(
        &s,
        &vc_ck,
        instances.clone(),
        witnesses.clone(),
        Some(cached_matvec),
        None,
        &[],
        &mut vc_standard,
        &mut vc_state_standard,
        &vc_shape,
        &vc_ck,
        &mut transcript_standard,
      )
    })
    .expect("standard NeutronNova NIFS should prove");

    let mut transcript_small = <E as Engine>::TE::new(b"small_neutronnova_standard_equivalence");
    let (e_eq, rhos, derived_left, derived_right) =
      replay_nifs_prefix(&mut transcript_small, &instances, num_constraints);
    assert_eq!(derived_left, left);
    assert_eq!(derived_right, right);

    let small = with_test_blind_seed(blind_seed(num_instances, LB), || {
      NeutronNovaNIFS::<E>::prove_small::<SV, LB>(
        &s,
        &vc_ck,
        instances.clone(),
        witnesses.clone(),
        &small_ab,
        &e_eq,
        derived_left,
        derived_right,
        &rhos,
        &mut vc_small,
        &mut vc_state_small,
        &vc_shape,
        &vc_ck,
        &mut transcript_small,
      )
    })
    .expect("small NeutronNova NIFS should prove");

    assert_eq!(
      vc_small.nifs_polys, vc_standard.nifs_polys,
      "NIFS polynomial mismatch for num_instances={num_instances}, l0={LB}"
    );
    assert_eq!(
      vc_small.t_out_step, vc_standard.t_out_step,
      "t_out_step mismatch for num_instances={num_instances}, l0={LB}"
    );
    assert_eq!(
      vc_small.eq_rho_at_rb, vc_standard.eq_rho_at_rb,
      "eq_rho_at_rb mismatch for num_instances={num_instances}, l0={LB}"
    );
    assert_eq!(
      small.0, standard.0,
      "E_eq mismatch for num_instances={num_instances}, l0={LB}"
    );
    assert_eq!(
      small.1, standard.1,
      "Az fold mismatch for num_instances={num_instances}, l0={LB}"
    );
    assert_eq!(
      small.2, standard.2,
      "Bz fold mismatch for num_instances={num_instances}, l0={LB}"
    );
    assert_eq!(
      small.3, standard.3,
      "Cz fold mismatch for num_instances={num_instances}, l0={LB}"
    );
    assert_eq!(
      small.4, standard.4,
      "folded witness mismatch for num_instances={num_instances}, l0={LB}"
    );
    assert_eq!(
      small.5, standard.5,
      "folded instance mismatch for num_instances={num_instances}, l0={LB}"
    );
  }

  fn run_small_matches_standard_case_for_l0<SV>(num_instances: usize, l0: usize)
  where
    SV: SmallValue + TryFrom<i64>,
    <SV as TryFrom<i64>>::Error: Debug,
    F: SmallValueEngine<SV> + SmallValueField<SV>,
  {
    match l0 {
      1 => run_small_matches_standard_case::<SV, 1>(num_instances),
      2 => run_small_matches_standard_case::<SV, 2>(num_instances),
      3 => run_small_matches_standard_case::<SV, 3>(num_instances),
      4 => run_small_matches_standard_case::<SV, 4>(num_instances),
      5 => run_small_matches_standard_case::<SV, 5>(num_instances),
      _ => panic!("unsupported test l0 {l0}"),
    }
  }

  fn dot_field_with_split_eq(layer: &[F], e_eq: &[F], left: usize, right: usize) -> F {
    let e_left = &e_eq[..left];
    let e_right = &e_eq[left..];
    let mut acc = F::ZERO;
    for i in 0..right {
      let base = i * left;
      let mut inner = F::ZERO;
      for j in 0..left {
        inner += e_left[j] * layer[base + j];
      }
      acc += e_right[i] * inner;
    }
    acc
  }

  #[allow(clippy::too_many_arguments)]
  fn continue_ab_suffix_with_c_claims_simple_for_test<E>(
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
    for round in start_round..rhos.len() {
      let pairs = a_layers.len() / 2;
      let (e0, quad_coeff) = compute_ab_c_claim_round::<E>(
        a_layers, b_layers, c_claims, e_eq, left, right, rhos, round,
      )?;
      let r_b = finish_nifs_field_round(
        rhos, round, e0, quad_coeff, vc, vc_state, vc_shape, vc_ck, transcript, r_bs, t_cur, acc_eq,
      )?;
      fold_ab_c_claim_pairs::<E>(a_layers, b_layers, c_claims, pairs, r_b);
      a_layers.truncate(pairs);
      b_layers.truncate(pairs);
      c_claims.truncate(pairs);
    }
    Ok(())
  }

  #[test]
  fn test_prefix_c_claims_match_materialized_c_dot_e() {
    const LB: usize = 2;
    let num_instances = 8usize;
    let n_padded = num_instances.next_power_of_two();
    let left = 4;
    let right = 2;
    let num_constraints = left * right;
    let (a_polys, b_polys) = synthetic_small_ab_polys::<i32>(num_instances, num_constraints);
    let a_certs = certify_layers::<i32, LB>(&a_polys);
    let b_certs = certify_layers::<i32, LB>(&b_polys);
    let a_small = padded_small_layers(&a_certs, num_instances, n_padded);
    let b_small = padded_small_layers(&b_certs, num_instances, n_padded);
    let a_small_evals = layer_evals(&a_small);
    let b_small_evals = layer_evals(&b_small);
    let e_eq = (0..left + right)
      .map(|i| F::from((i as u64) + 3))
      .collect::<Vec<_>>();
    let r_bs = vec![F::from(7u64), F::from(11u64)];
    let prefix_size = 1usize << LB;
    let prefix_weights = weights_from_r::<F>(&r_bs, prefix_size);

    let prefix = materialize_prefix_ab_with_c_claims::<E, i32>(
      &a_small_evals,
      &b_small_evals,
      &prefix_weights,
      prefix_size,
      &e_eq,
      left,
      right,
    )
    .expect("prefix materialization should succeed");
    let c_layers = fold_small_c_from_ab_with_weights::<E, i32>(
      &a_small_evals,
      &b_small_evals,
      &prefix_weights,
      prefix_size,
    )
    .expect("materialized C fold should succeed");

    assert_eq!(prefix.c_claims.len(), c_layers.len());
    for (claim, c_layer) in prefix.c_claims.iter().zip(c_layers.iter()) {
      assert_eq!(*claim, dot_field_with_split_eq(c_layer, &e_eq, left, right));
    }
  }

  #[test]
  fn test_c_claim_folding_matches_materialized_c_folding() -> Result<(), SpartanError> {
    const LB: usize = 1;
    let num_instances = 8usize;
    let n_padded = num_instances.next_power_of_two();
    let left = 4;
    let right = 2;
    let num_constraints = left * right;
    let (a_polys, b_polys) = synthetic_small_ab_polys::<i32>(num_instances, num_constraints);
    let a_certs = certify_layers::<i32, LB>(&a_polys);
    let b_certs = certify_layers::<i32, LB>(&b_polys);
    let a_small = padded_small_layers(&a_certs, num_instances, n_padded);
    let b_small = padded_small_layers(&b_certs, num_instances, n_padded);
    let a_small_evals = layer_evals(&a_small);
    let b_small_evals = layer_evals(&b_small);
    let (ell_cons, derived_left, derived_right) = tensor_decomp(num_constraints);
    assert_eq!(derived_left, left);
    assert_eq!(derived_right, right);
    let e_eq = PowPolynomial::split_evals(F::from(13u64), ell_cons, left, right);
    let r_bs = vec![F::from(7u64)];
    let prefix_size = 1usize << LB;
    let prefix_weights = weights_from_r::<F>(&r_bs, prefix_size);

    let mut c_layers_materialized = fold_small_c_from_ab_with_weights::<E, i32>(
      &a_small_evals,
      &b_small_evals,
      &prefix_weights,
      prefix_size,
    )?;
    let PrefixAbWithCClaims {
      a_layers,
      b_layers,
      mut c_claims,
    } = materialize_prefix_ab_with_c_claims::<E, i32>(
      &a_small_evals,
      &b_small_evals,
      &prefix_weights,
      prefix_size,
      &e_eq,
      left,
      right,
    )?;

    assert_eq!(a_layers.len(), c_claims.len());
    assert_eq!(b_layers.len(), c_claims.len());
    for (claim, c_layer) in c_claims.iter().zip(c_layers_materialized.iter()) {
      assert_eq!(*claim, dot_field_with_split_eq(c_layer, &e_eq, left, right));
    }

    for r in [F::from(11u64), F::from(13u64)] {
      let pairs = c_claims.len() / 2;
      for i in 0..pairs {
        fold_layer_pair_into(&mut c_layers_materialized, 2 * i, 2 * i + 1, i, r);
        fold_c_claim_pair_into(&mut c_claims, 2 * i, 2 * i + 1, i, r);
      }
      c_layers_materialized.truncate(pairs);
      c_claims.truncate(pairs);

      for (claim, c_layer) in c_claims.iter().zip(c_layers_materialized.iter()) {
        assert_eq!(*claim, dot_field_with_split_eq(c_layer, &e_eq, left, right));
      }
    }

    Ok(())
  }

  #[test]
  fn test_merged_lazy_c_suffix_matches_simple_lazy_c_suffix() -> Result<(), SpartanError> {
    const LB: usize = 1;
    let num_instances = 8usize;
    let n_padded = num_instances.next_power_of_two();
    let ell_b = n_padded.trailing_zeros() as usize;
    let left = 4;
    let right = 2;
    let num_constraints = left * right;
    let (a_polys, b_polys) = synthetic_small_ab_polys::<i32>(num_instances, num_constraints);
    let a_certs = certify_layers::<i32, LB>(&a_polys);
    let b_certs = certify_layers::<i32, LB>(&b_polys);
    let a_small = padded_small_layers(&a_certs, num_instances, n_padded);
    let b_small = padded_small_layers(&b_certs, num_instances, n_padded);
    let a_small_evals = layer_evals(&a_small);
    let b_small_evals = layer_evals(&b_small);
    let (ell_cons, derived_left, derived_right) = tensor_decomp(num_constraints);
    assert_eq!(derived_left, left);
    assert_eq!(derived_right, right);
    let e_eq = PowPolynomial::split_evals(F::from(19u64), ell_cons, left, right);
    let rhos = vec![F::from(2u64), F::from(3u64), F::from(5u64)];
    let seed = blind_seed(num_instances, LB);

    let run_variant = |merged: bool| {
      with_test_blind_seed(seed, || -> Result<_, SpartanError> {
        let mut vc = NeutronNovaVerifierCircuit::<E>::default(ell_b, 1, 2, 32);
        let (vc_shape, vc_ck, _) =
          <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&vc)
            .expect("verifier circuit shape should synthesize");
        let mut vc_state = SatisfyingAssignment::<E>::initialize_multiround_witness(&vc_shape)
          .expect("verifier circuit witness state should initialize");
        let mut transcript = <E as Engine>::TE::new(b"merged_lazy_c_suffix_equivalence");
        let accumulators = build_accumulators_neutronnova::<F, i32, LB>(
          &a_small, &b_small, &e_eq, left, right, &rhos,
        )?;
        let mut small_value =
          SmallValueSumCheck::<F, SMALL_VALUE_T_DEGREE>::from_accumulators(accumulators);
        let mut r_bs = Vec::with_capacity(ell_b);
        let mut t_cur = F::ZERO;
        let mut acc_eq = F::ONE;

        for round in 0..LB {
          let (poly, li) = generate_univariate_sumcheck_polynomial_from_accumulator(
            &small_value,
            round,
            rhos[round],
            t_cur,
          )?;
          let r_i = process_nifs_round(
            &mut vc,
            &mut vc_state,
            &vc_shape,
            &vc_ck,
            &mut transcript,
            round,
            &poly,
          )?;
          t_cur = poly.evaluate(&r_i);
          acc_eq = li.eval_linear_at(r_i);
          small_value.advance(&li, r_i);
          r_bs.push(r_i);
        }

        let prefix_size = 1usize << LB;
        let prefix_weights = weights_from_r::<F>(&r_bs, prefix_size);
        let PrefixAbWithCClaims {
          mut a_layers,
          mut b_layers,
          mut c_claims,
        } = materialize_prefix_ab_with_c_claims::<E, i32>(
          &a_small_evals,
          &b_small_evals,
          &prefix_weights,
          prefix_size,
          &e_eq,
          left,
          right,
        )?;

        if merged {
          continue_ab_suffix_with_c_claims(
            &mut a_layers,
            &mut b_layers,
            &mut c_claims,
            &e_eq,
            left,
            right,
            &rhos,
            &mut vc,
            &mut vc_state,
            &vc_shape,
            &vc_ck,
            &mut transcript,
            LB,
            &mut r_bs,
            &mut t_cur,
            &mut acc_eq,
          )?;
        } else {
          continue_ab_suffix_with_c_claims_simple_for_test(
            &mut a_layers,
            &mut b_layers,
            &mut c_claims,
            &e_eq,
            left,
            right,
            &rhos,
            &mut vc,
            &mut vc_state,
            &vc_shape,
            &vc_ck,
            &mut transcript,
            LB,
            &mut r_bs,
            &mut t_cur,
            &mut acc_eq,
          )?;
        }

        Ok((
          vc.nifs_polys,
          r_bs,
          t_cur,
          acc_eq,
          a_layers,
          b_layers,
          c_claims,
        ))
      })
    };

    let simple = run_variant(false)?;
    let merged = run_variant(true)?;
    assert_eq!(merged, simple);

    Ok(())
  }

  #[test]
  fn test_small_neutronnova_nifs_matches_standard_nifs_same_transcript() {
    for num_instances in [2usize, 4, 8, 16] {
      let ell_b = num_instances.trailing_zeros() as usize;
      for l0 in 1..=ell_b {
        run_small_matches_standard_case_for_l0::<i32>(num_instances, l0);
      }
    }
  }

  #[test]
  fn test_small_neutronnova_nifs_matches_standard_nifs_same_transcript_i64_smoke() {
    run_small_matches_standard_case::<i64, 2>(4);
  }

  #[test]
  fn test_small_neutronnova_nifs_matches_standard_nifs_with_padding() {
    run_small_matches_standard_case::<i32, 1>(3);
  }

  #[test]
  fn test_small_neutronnova_certificate_rejects_out_of_bound_small_values() {
    const L0: usize = 1;
    let poly = MultilinearPolynomial::new(vec![i32::MAX, 0, 0, 0]);
    assert!(matches!(
      SmallValueExtensionBoundedPoly::<_, L0>::new(&poly),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }
}
