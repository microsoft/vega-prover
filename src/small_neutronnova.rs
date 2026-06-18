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
  lagrange_accumulator::{SMALL_VALUE_T_DEGREE, build_accumulators_neutronnova},
  neutronnova_zk::{
    NeutronNovaNIFS, NeutronNovaNifsOutput, NifsTranscriptState, SmallValueNeutronNovaStepMatvecs,
    continue_ab_suffix_with_c_claims, finalize_nifs_step_claim, fold_witness_and_instance,
    invalid_input, padded_layer_slices, prepare_nifs_transcript, process_nifs_round,
    validate_instance_witness_counts,
  },
  r1cs::{R1CSInstance, R1CSWitness, SplitMultiRoundR1CSShape, SplitR1CSShape, weights_from_r},
  small_sumcheck::{SmallValueSumCheck, generate_univariate_sumcheck_polynomial_from_accumulator},
  traits::{Engine, pcs::FoldingEngineTrait},
  zk::NeutronNovaVerifierCircuit,
};
#[cfg(test)]
use crate::{
  lagrange_accumulator::SmallValueExtensionBoundedPoly,
  neutronnova_zk::padded_map_by_repeating_first, polys::multilinear::MultilinearPolynomial,
};
use ff::Field;
use num_traits::Zero;
use rayon::prelude::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove<E>(
  s: &SplitR1CSShape<E>,
  ck: &CommitmentKey<E>,
  us: Vec<R1CSInstance<E>>,
  ws: Vec<R1CSWitness<E>>,
  step_matvecs: &SmallValueNeutronNovaStepMatvecs<E>,
  l0: usize,
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
) -> Result<NeutronNovaNifsOutput<E>, SpartanError>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  E::Scalar: SmallValueEngine<i64> + Default,
{
  let num_instances = us.len();
  if num_instances == 0 {
    return Err(invalid_input(
      "small NeutronNova NIFS requires at least one instance",
    ));
  }
  if ws.len() != num_instances {
    return Err(invalid_input(format!(
      "witness count {} does not match instance count {}",
      ws.len(),
      num_instances
    )));
  }
  if step_matvecs.field.az.len() != num_instances {
    return Err(invalid_input(format!(
      "NIFS received {} step matvec layers but {} instances",
      step_matvecs.field.az.len(),
      num_instances
    )));
  }
  if l0 == 0 {
    return Err(invalid_input(
      "small-value NeutronNova NIFS requires l0 > 0",
    ));
  }
  if step_matvecs.small_abc.ab.l0 != l0 {
    return Err(invalid_input(format!(
      "small-value NeutronNova NIFS expected l0 {} but cache has {}",
      l0, step_matvecs.small_abc.ab.l0
    )));
  }
  #[cfg(debug_assertions)]
  crate::neutronnova_zk::debug_validate_small_value_step_matvecs::<E>(step_matvecs);

  let NifsTranscriptState {
    n_padded,
    e_eq,
    left,
    right,
    rhos,
    ..
  } = prepare_nifs_transcript(s, &us, transcript)?;

  let a_small = padded_layer_slices(&step_matvecs.small_abc.ab.az_small, num_instances, n_padded);
  let b_small = padded_layer_slices(&step_matvecs.small_abc.ab.bz_small, num_instances, n_padded);
  let c_small = padded_layer_slices(&step_matvecs.small_abc.cz_small, num_instances, n_padded);
  let a_field = padded_layer_slices(&step_matvecs.field.az, num_instances, n_padded);
  let b_field = padded_layer_slices(&step_matvecs.field.bz, num_instances, n_padded);
  let c_field = padded_layer_slices(&step_matvecs.field.cz, num_instances, n_padded);
  let ab_large_positions = &step_matvecs.small_abc.ab.large_positions;
  let c_large_positions = &step_matvecs.small_abc.c_large_positions;

  NeutronNovaNIFS::<E>::prove_small::<i64>(
    s,
    ck,
    us,
    ws,
    &a_small,
    &b_small,
    &c_small,
    &a_field,
    &b_field,
    &c_field,
    ab_large_positions,
    c_large_positions,
    &e_eq,
    left,
    right,
    &rhos,
    l0,
    vc,
    vc_state,
    vc_shape,
    vc_ck,
    transcript,
  )
}

impl<E: Engine> NeutronNovaNIFS<E>
where
  E::PCS: FoldingEngineTrait<E>,
{
  /// Prove small-value NeutronNova NIFS, deriving round challenges from the verifier circuit.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn prove_small<SV>(
    s: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    us: Vec<R1CSInstance<E>>,
    ws: Vec<R1CSWitness<E>>,
    a_small_evals: &[&[SV]],
    b_small_evals: &[&[SV]],
    c_small_evals: &[&[SV]],
    a_field_evals: &[&[E::Scalar]],
    b_field_evals: &[&[E::Scalar]],
    c_field_evals: &[&[E::Scalar]],
    ab_large_positions: &[usize],
    c_large_positions: &[usize],
    e_eq: &[E::Scalar],
    left: usize,
    right: usize,
    rhos: &[E::Scalar],
    l0: usize,
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
    E::Scalar: SmallValueEngine<SV> + Default,
  {
    let n_padded = validate_layer_arrays::<E, SV>(
      a_small_evals,
      b_small_evals,
      c_small_evals,
      a_field_evals,
      b_field_evals,
      c_field_evals,
      ab_large_positions,
      c_large_positions,
      e_eq,
      left,
      right,
      rhos,
      l0,
    )?;
    let num_instances = us.len();
    validate_instance_witness_counts::<E>(num_instances, &us, &ws)?;
    if num_instances.next_power_of_two() != n_padded {
      return Err(invalid_input(format!(
        "instance count {} is incompatible with padded layer count {}",
        num_instances, n_padded
      )));
    }
    if vc.nifs_polys.len() != rhos.len() {
      return Err(invalid_input(format!(
        "verifier circuit has {} NIFS rounds but rhos has {}",
        vc.nifs_polys.len(),
        rhos.len()
      )));
    }

    let accumulators = build_accumulators_neutronnova::<E::Scalar, SV>(
      a_small_evals,
      b_small_evals,
      a_field_evals,
      b_field_evals,
      ab_large_positions,
      e_eq,
      left,
      right,
      rhos,
      l0,
    )?;
    let mut small_value =
      SmallValueSumCheck::<E::Scalar, SMALL_VALUE_T_DEGREE>::from_accumulators(accumulators);

    let ell_b = rhos.len();
    let mut r_bs = Vec::with_capacity(ell_b);
    let mut t_cur = E::Scalar::ZERO;
    let mut acc_eq = E::Scalar::ONE;

    for (round, rho) in rhos.iter().copied().enumerate().take(l0) {
      let (poly, li) =
        generate_univariate_sumcheck_polynomial_from_accumulator(&small_value, round, rho, t_cur)?;
      let r_i = process_nifs_round(vc, vc_state, vc_shape, vc_ck, transcript, round, &poly)?;
      t_cur = poly.evaluate(&r_i);
      acc_eq = li.eval_linear_at(r_i);
      small_value.advance(&li, r_i);
      r_bs.push(r_i);
    }
    drop(small_value);

    let (az_step, bz_step, cz_step) = if l0 == ell_b {
      let final_weights = weights_from_r::<E::Scalar>(&r_bs, n_padded);
      fold_final_abc_with_weights::<E, SV>(
        a_small_evals,
        b_small_evals,
        c_small_evals,
        a_field_evals,
        b_field_evals,
        c_field_evals,
        ab_large_positions,
        c_large_positions,
        &final_weights,
      )?
    } else {
      let prefix_size = 1usize << l0;
      let prefix_weights = weights_from_r::<E::Scalar>(&r_bs, prefix_size);

      let PrefixAbWithCClaims {
        mut a_layers,
        mut b_layers,
        mut c_claims,
      } = materialize_prefix_ab_with_c_claims::<E, SV>(
        a_small_evals,
        b_small_evals,
        c_small_evals,
        a_field_evals,
        b_field_evals,
        c_field_evals,
        ab_large_positions,
        c_large_positions,
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
        l0,
        &mut r_bs,
        &mut t_cur,
        &mut acc_eq,
      )?;

      let final_weights = weights_from_r::<E::Scalar>(&r_bs, n_padded);
      let c_folded = fold_final_c_with_weights::<E, SV>(
        c_small_evals,
        c_field_evals,
        c_large_positions,
        &final_weights,
      )?;

      (
        a_layers.pop().ok_or_else(empty_fold_error)?,
        b_layers.pop().ok_or_else(empty_fold_error)?,
        c_folded,
      )
    };

    finalize_nifs_step_claim(
      vc, vc_state, vc_shape, vc_ck, transcript, ell_b, t_cur, acc_eq,
    )?;

    // The NIFS transcript is now fixed; fold the prover witness/instance with
    // the same r_b challenges that the verifier circuit just recorded.
    let (folded_w, folded_u) =
      fold_witness_and_instance(s, ck, us, ws, num_instances, n_padded, &r_bs)?;

    Ok((e_eq.to_vec(), az_step, bz_step, cz_step, folded_w, folded_u))
  }
}

fn empty_fold_error() -> SpartanError {
  invalid_input("small NeutronNova NIFS produced no folded layer")
}

#[allow(clippy::too_many_arguments)]
fn validate_layer_arrays<E, SV>(
  a_small: &[&[SV]],
  b_small: &[&[SV]],
  c_small: &[&[SV]],
  a_field: &[&[E::Scalar]],
  b_field: &[&[E::Scalar]],
  c_field: &[&[E::Scalar]],
  ab_large_positions: &[usize],
  c_large_positions: &[usize],
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
  rhos: &[E::Scalar],
  l0: usize,
) -> Result<usize, SpartanError>
where
  E: Engine,
  SV: SmallValue,
{
  let num_instances = a_small.len();
  if num_instances == 0 {
    return Err(invalid_input(
      "small NeutronNova NIFS requires at least one layer",
    ));
  }
  if b_small.len() != num_instances
    || c_small.len() != num_instances
    || a_field.len() != num_instances
    || b_field.len() != num_instances
    || c_field.len() != num_instances
  {
    return Err(invalid_input("A/B/C layer counts must match"));
  }
  if l0 == 0 {
    return Err(invalid_input("small NeutronNova NIFS requires l0 > 0"));
  }
  let ell_b = rhos.len();
  if l0 > ell_b {
    return Err(invalid_input(format!(
      "small NeutronNova l0 {} exceeds ell_b {}",
      l0, ell_b
    )));
  }

  let shift =
    u32::try_from(ell_b).map_err(|_| invalid_input("ell_b does not fit in a u32 shift"))?;
  let n_padded = 1usize
    .checked_shl(shift)
    .ok_or_else(|| invalid_input("ell_b is too large for this platform"))?;
  if num_instances != n_padded {
    return Err(invalid_input(format!(
      "layer count {} does not match padded count {}",
      num_instances, n_padded
    )));
  }

  let expected_constraints = left
    .checked_mul(right)
    .ok_or_else(|| invalid_input("left * right overflows"))?;
  for (label, positions) in [
    ("A/B large positions", ab_large_positions),
    ("C large positions", c_large_positions),
  ] {
    if !positions.windows(2).all(|pair| pair[0] < pair[1]) {
      return Err(invalid_input(format!("{label} must be sorted and unique")));
    }
    if let Some(pos) = positions
      .iter()
      .copied()
      .find(|&pos| pos >= expected_constraints)
    {
      return Err(invalid_input(format!(
        "{label} contains out-of-range position {} for layer length {}",
        pos, expected_constraints
      )));
    }
  }
  if !a_small
    .iter()
    .chain(b_small.iter())
    .chain(c_small.iter())
    .all(|layer| layer.len() == expected_constraints)
  {
    return Err(invalid_input(format!(
      "all small layers must have length left * right {}",
      expected_constraints
    )));
  }
  if !ab_large_positions.is_empty()
    && !a_field
      .iter()
      .chain(b_field.iter())
      .all(|layer| layer.len() == expected_constraints)
  {
    return Err(invalid_input(format!(
      "all A/B field correction layers must have length left * right {}",
      expected_constraints
    )));
  }
  if !c_large_positions.is_empty()
    && !c_field
      .iter()
      .all(|layer| layer.len() == expected_constraints)
  {
    return Err(invalid_input(format!(
      "all C field correction layers must have length left * right {}",
      expected_constraints
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

  Ok(n_padded)
}

#[cfg(test)]
fn padded_small_layers<'poly, SV, const LB: usize>(
  layers: &[SmallValueExtensionBoundedPoly<'poly, SV, LB>],
  num_instances: usize,
  n_padded: usize,
) -> Vec<SmallValueExtensionBoundedPoly<'poly, SV, LB>>
where
  SV: SmallValue,
{
  padded_map_by_repeating_first(layers, num_instances, n_padded, |layer| *layer)
}

#[cfg(test)]
fn make_layer_evals<'poly, SV, const LB: usize>(
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

#[cfg(test)]
fn padded_plain_layer_evals<SV>(
  layers: &[MultilinearPolynomial<SV>],
  num_instances: usize,
  n_padded: usize,
) -> Vec<&[SV]>
where
  SV: SmallValue,
{
  padded_map_by_repeating_first(layers, num_instances, n_padded, |layer| layer.Z.as_slice())
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
  c_layers: &[&[SV]],
  a_field_layers: &[&[E::Scalar]],
  b_field_layers: &[&[E::Scalar]],
  c_field_layers: &[&[E::Scalar]],
  ab_large_positions: &[usize],
  c_large_positions: &[usize],
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
  if prefix_size == 0 || !a_layers.len().is_multiple_of(prefix_size) {
    return Err(invalid_input("invalid small prefix group size"));
  }
  if prefix_weights.len() != prefix_size {
    return Err(invalid_input(format!(
      "prefix weight length {} does not match prefix size {}",
      prefix_weights.len(),
      prefix_size
    )));
  }
  let a_folded = fold_small_layers_with_weights::<E, SV>(a_layers, prefix_weights, prefix_size)?;
  let b_folded = fold_small_layers_with_weights::<E, SV>(b_layers, prefix_weights, prefix_size)?;
  let mut c_claims: Vec<E::Scalar> = c_layers
    .par_chunks(prefix_size)
    .map(|c_group| {
      let mut acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
      for (weight, c_layer) in prefix_weights.iter().zip(c_group.iter()) {
        let c_claim = dot_small_layer_with_split_eq::<E, SV>(c_layer, e_eq, left, right);
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut acc, weight, &c_claim,
        );
      }
      <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc)
    })
    .collect();
  let mut a_folded = a_folded;
  let mut b_folded = b_folded;

  apply_large_prefix_corrections::<E>(
    &mut a_folded,
    &mut b_folded,
    &mut c_claims,
    a_field_layers,
    b_field_layers,
    c_field_layers,
    ab_large_positions,
    c_large_positions,
    prefix_weights,
    prefix_size,
    e_eq,
    left,
    right,
  );

  Ok(PrefixAbWithCClaims {
    a_layers: a_folded,
    b_layers: b_folded,
    c_claims,
  })
}

#[allow(clippy::too_many_arguments)]
fn apply_large_prefix_corrections<E>(
  a_folded: &mut [Vec<E::Scalar>],
  b_folded: &mut [Vec<E::Scalar>],
  c_claims: &mut [E::Scalar],
  a_field_layers: &[&[E::Scalar]],
  b_field_layers: &[&[E::Scalar]],
  c_field_layers: &[&[E::Scalar]],
  ab_large_positions: &[usize],
  c_large_positions: &[usize],
  prefix_weights: &[E::Scalar],
  prefix_size: usize,
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
) where
  E: Engine,
{
  if ab_large_positions.is_empty() && c_large_positions.is_empty() {
    return;
  }
  let e_left = &e_eq[..left];
  let e_right = &e_eq[left..];
  let total = left * right;

  for (suffix_idx, ((a_out, b_out), c_claim)) in a_folded
    .iter_mut()
    .zip(b_folded.iter_mut())
    .zip(c_claims.iter_mut())
    .enumerate()
  {
    for &k in ab_large_positions {
      if k >= total {
        debug_assert!(
          k < total,
          "A/B large position {} is out of range for layer length {}",
          k,
          total
        );
        continue;
      }
      let mut a_val = E::Scalar::ZERO;
      let mut b_val = E::Scalar::ZERO;
      for (prefix_idx, weight) in prefix_weights.iter().enumerate().take(prefix_size) {
        let layer_idx = suffix_idx * prefix_size + prefix_idx;
        a_val += *weight * a_field_layers[layer_idx][k];
        b_val += *weight * b_field_layers[layer_idx][k];
      }
      a_out[k] = a_val;
      b_out[k] = b_val;
    }

    for &k in c_large_positions {
      if k >= total {
        debug_assert!(
          k < total,
          "C large position {} is out of range for layer length {}",
          k,
          total
        );
        continue;
      }
      let row = k / left;
      let col = k % left;
      let eq_k = e_right[row] * e_left[col];
      let mut c_val = E::Scalar::ZERO;
      for (prefix_idx, weight) in prefix_weights.iter().enumerate().take(prefix_size) {
        let layer_idx = suffix_idx * prefix_size + prefix_idx;
        c_val += *weight * c_field_layers[layer_idx][k];
      }
      *c_claim += eq_k * c_val;
    }
  }
}

fn dot_small_layer_with_split_eq<E, SV>(
  layer: &[SV],
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
    let mut inner = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();
    for j in 0..left {
      <E::Scalar as DelayedReduction<SV>>::unreduced_multiply_accumulate(
        &mut inner,
        &e_left[j],
        &layer[base + j],
      );
    }
    let inner_red = <E::Scalar as DelayedReduction<SV>>::reduce(&inner);
    <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
      &mut acc,
      &e_right[i],
      &inner_red,
    );
  }

  <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&acc)
}

fn fold_final_abc_with_weights<E, SV>(
  a_layers: &[&[SV]],
  b_layers: &[&[SV]],
  c_layers: &[&[SV]],
  a_field_layers: &[&[E::Scalar]],
  b_field_layers: &[&[E::Scalar]],
  c_field_layers: &[&[E::Scalar]],
  ab_large_positions: &[usize],
  c_large_positions: &[usize],
  weights: &[E::Scalar],
) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError>
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  let (mut a_final, mut b_final, mut c_final) =
    fold_small_final_abc_with_weights::<E, SV>(a_layers, b_layers, c_layers, weights)?;
  apply_large_final_corrections::<E>(
    Some(&mut a_final),
    Some(&mut b_final),
    &mut c_final,
    a_field_layers,
    b_field_layers,
    c_field_layers,
    ab_large_positions,
    c_large_positions,
    weights,
  );
  Ok((a_final, b_final, c_final))
}

fn fold_final_c_with_weights<E, SV>(
  c_layers: &[&[SV]],
  c_field_layers: &[&[E::Scalar]],
  c_large_positions: &[usize],
  weights: &[E::Scalar],
) -> Result<Vec<E::Scalar>, SpartanError>
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  let mut c_final = fold_small_layers_with_weights::<E, SV>(c_layers, weights, weights.len())?
    .pop()
    .ok_or_else(empty_fold_error)?;
  apply_large_final_corrections::<E>(
    None,
    None,
    &mut c_final,
    &[],
    &[],
    c_field_layers,
    &[],
    c_large_positions,
    weights,
  );
  Ok(c_final)
}

fn apply_large_final_corrections<E>(
  mut a_final: Option<&mut [E::Scalar]>,
  mut b_final: Option<&mut [E::Scalar]>,
  c_final: &mut [E::Scalar],
  a_field_layers: &[&[E::Scalar]],
  b_field_layers: &[&[E::Scalar]],
  c_field_layers: &[&[E::Scalar]],
  ab_large_positions: &[usize],
  c_large_positions: &[usize],
  weights: &[E::Scalar],
) where
  E: Engine,
{
  let need_a = a_final.is_some();
  let need_b = b_final.is_some();
  let total = c_final.len();
  for &k in ab_large_positions {
    if k >= total {
      debug_assert!(
        k < total,
        "A/B large position {} is out of range for layer length {}",
        k,
        total
      );
      continue;
    }
    let mut a_val = E::Scalar::ZERO;
    let mut b_val = E::Scalar::ZERO;
    for (layer_idx, weight) in weights.iter().enumerate() {
      if need_a {
        a_val += *weight * a_field_layers[layer_idx][k];
      }
      if need_b {
        b_val += *weight * b_field_layers[layer_idx][k];
      }
    }
    if let Some(a_final) = a_final.as_deref_mut() {
      a_final[k] = a_val;
    }
    if let Some(b_final) = b_final.as_deref_mut() {
      b_final[k] = b_val;
    }
  }

  for &k in c_large_positions {
    if k >= total {
      debug_assert!(
        k < total,
        "C large position {} is out of range for layer length {}",
        k,
        total
      );
      continue;
    }
    let mut c_val = E::Scalar::ZERO;
    for (layer_idx, weight) in weights.iter().enumerate() {
      c_val += *weight * c_field_layers[layer_idx][k];
    }
    c_final[k] = c_val;
  }
}

fn fold_small_final_abc_with_weights<E, SV>(
  a_layers: &[&[SV]],
  b_layers: &[&[SV]],
  c_layers: &[&[SV]],
  weights: &[E::Scalar],
) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError>
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  if weights.len() != a_layers.len() {
    return Err(invalid_input(format!(
      "weight length {} does not match layer count {}",
      weights.len(),
      a_layers.len()
    )));
  }

  let layer_len = a_layers[0].len();
  let mut a_final = vec![E::Scalar::ZERO; layer_len];
  let mut b_final = vec![E::Scalar::ZERO; layer_len];
  let mut c_final = vec![E::Scalar::ZERO; layer_len];

  a_final
    .par_iter_mut()
    .zip(b_final.par_iter_mut())
    .zip(c_final.par_iter_mut())
    .enumerate()
    .for_each(|(k, ((a_out, b_out), c_out))| {
      let mut a_acc = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();
      let mut b_acc = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();
      let mut c_acc = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();

      for (((weight, a_layer), b_layer), c_layer) in weights
        .iter()
        .zip(a_layers.iter())
        .zip(b_layers.iter())
        .zip(c_layers.iter())
      {
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
        <E::Scalar as DelayedReduction<SV>>::unreduced_multiply_accumulate(
          &mut c_acc,
          weight,
          &c_layer[k],
        );
      }

      *a_out = <E::Scalar as DelayedReduction<SV>>::reduce(&a_acc);
      *b_out = <E::Scalar as DelayedReduction<SV>>::reduce(&b_acc);
      *c_out = <E::Scalar as DelayedReduction<SV>>::reduce(&c_acc);
    });

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
  if group_size == 0 || !layers.len().is_multiple_of(group_size) {
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
    big_num::{SmallValueField, small_value_conversion::to_small_vec_or_zero},
    lagrange_accumulator::field_to_i64_or_zero_for_l0,
    neutronnova_zk::{
      FieldStepMatvecs, SmallAbStepMatvecs, SmallAbcStepMatvecs, SmallValueNeutronNovaStepMatvecs,
      compute_ab_c_claim_round, compute_field_round_claim, fold_layer_pair_into,
      generate_nifs_field_round_polynomial,
    },
    polys::{multilinear::MultilinearPolynomial, power::PowPolynomial},
    provider::PallasHyraxEngine,
    r1cs::{R1CSInstance, R1CSWitness, SparseMatrix},
    traits::{Engine, pcs::PCSEngineTrait, transcript::TranscriptEngineTrait},
  };
  use once_cell::sync::OnceCell;
  use std::fmt::Debug;

  type E = PallasHyraxEngine;
  type F = <E as Engine>::Scalar;

  fn synthetic_small_abc_polys<SV>(
    num_instances: usize,
    num_constraints: usize,
  ) -> (
    Vec<MultilinearPolynomial<SV>>,
    Vec<MultilinearPolynomial<SV>>,
    Vec<MultilinearPolynomial<SV>>,
  )
  where
    SV: SmallValue + TryFrom<i64>,
    <SV as TryFrom<i64>>::Error: Debug,
  {
    let mut a = Vec::with_capacity(num_instances);
    let mut b = Vec::with_capacity(num_instances);
    let mut c = Vec::with_capacity(num_instances);
    for instance in 0..num_instances {
      let mut a_values = Vec::with_capacity(num_constraints);
      let mut b_values = Vec::with_capacity(num_constraints);
      let mut c_values = Vec::with_capacity(num_constraints);
      for k in 0..num_constraints {
        let a_value = ((13 * instance as i64 + 5 * k as i64 + 7) % 23) - 11;
        let b_value = ((17 * instance as i64 + 3 * k as i64 + 2) % 29) - 14;
        a_values.push(SV::try_from(a_value).unwrap());
        b_values.push(SV::try_from(b_value).unwrap());
        c_values.push(SV::try_from(a_value * b_value).unwrap());
      }
      a.push(MultilinearPolynomial::new(a_values));
      b.push(MultilinearPolynomial::new(b_values));
      c.push(MultilinearPolynomial::new(c_values));
    }
    (a, b, c)
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

  fn field_layers_from_small_evals<SV>(
    a_small: &[&[SV]],
    b_small: &[&[SV]],
    c_small: &[&[SV]],
  ) -> (Vec<Vec<F>>, Vec<Vec<F>>, Vec<Vec<F>>)
  where
    SV: SmallValue,
    F: SmallValueField<SV>,
  {
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
    let c_field = c_small
      .iter()
      .map(|layer| {
        layer
          .iter()
          .copied()
          .map(<F as SmallValueField<SV>>::small_to_field)
          .collect::<Vec<_>>()
      })
      .collect::<Vec<Vec<F>>>();
    (a_field, b_field, c_field)
  }

  fn layer_refs<T>(layers: &[Vec<T>]) -> Vec<&[T]> {
    layers.iter().map(Vec::as_slice).collect()
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
    let (a_polys, b_polys, c_polys) =
      synthetic_small_abc_polys::<SV>(num_instances, num_constraints);
    let a_certs = certify_layers::<SV, LB>(&a_polys);
    let b_certs = certify_layers::<SV, LB>(&b_polys);
    let a_small = padded_small_layers(&a_certs, num_instances, n_padded);
    let b_small = padded_small_layers(&b_certs, num_instances, n_padded);
    let a_small_evals = make_layer_evals(&a_small);
    let b_small_evals = make_layer_evals(&b_small);
    let c_small_evals = padded_plain_layer_evals(&c_polys, num_instances, n_padded);
    let (mut a_layers, mut b_layers, mut c_layers) =
      field_layers_from_small_evals(&a_small_evals, &b_small_evals, &c_small_evals);
    let a_field_layers = a_layers.clone();
    let b_field_layers = b_layers.clone();
    let c_field_layers = c_layers.clone();
    let a_field_refs = layer_refs(&a_field_layers);
    let b_field_refs = layer_refs(&b_field_layers);
    let c_field_refs = layer_refs(&c_field_layers);
    let (ell_cons, derived_left, derived_right) = tensor_decomp(num_constraints);
    assert_eq!(derived_left, left);
    assert_eq!(derived_right, right);
    let e_eq = PowPolynomial::split_evals(F::from(19u64), ell_cons, left, right);
    let rhos = (0..ell_b)
      .map(|round| F::from((2 * round + 3) as u64))
      .collect::<Vec<_>>();
    let r_bs = (0..ell_b)
      .map(|round| F::from((2 * round + 7) as u64))
      .collect::<Vec<_>>();

    let accumulators = build_accumulators_neutronnova::<F, SV>(
      &a_small_evals,
      &b_small_evals,
      &a_field_refs,
      &b_field_refs,
      &[],
      &e_eq,
      left,
      right,
      &rhos,
      LB,
    )
    .expect("small accumulator construction should succeed");
    let mut small_value =
      SmallValueSumCheck::<F, SMALL_VALUE_T_DEGREE>::from_accumulators(accumulators);
    let mut t_cur = F::ZERO;
    let mut acc_eq = F::ONE;

    for (round, (&rho, &r_b)) in rhos.iter().zip(&r_bs).enumerate().take(LB) {
      let (standard_e0, standard_quad_coeff) = compute_field_round_claim::<E>(
        &a_layers, &b_layers, &c_layers, &e_eq, left, right, &rhos, round,
      )
      .expect("standard round claim should compute");
      let standard_poly = generate_nifs_field_round_polynomial::<F>(
        rho,
        acc_eq,
        t_cur,
        standard_e0,
        standard_quad_coeff,
      )
      .expect("standard NIFS polynomial should interpolate");
      let (small_poly, li) =
        generate_univariate_sumcheck_polynomial_from_accumulator(&small_value, round, rho, t_cur)
          .expect("small accumulator polynomial should interpolate");
      assert_eq!(
        small_poly, standard_poly,
        "NIFS polynomial mismatch for num_instances={num_instances}, l0={LB}, round={round}"
      );

      t_cur = small_poly.evaluate(&r_b);
      let standard_acc_eq = acc_eq * ((F::ONE - r_b) * (F::ONE - rho) + r_b * rho);
      acc_eq = li.eval_linear_at(r_b);
      assert_eq!(
        acc_eq, standard_acc_eq,
        "eq accumulator mismatch for num_instances={num_instances}, l0={LB}, round={round}"
      );
      small_value.advance(&li, r_b);
      fold_materialized_layers(&mut a_layers, r_b);
      fold_materialized_layers(&mut b_layers, r_b);
      fold_materialized_layers(&mut c_layers, r_b);
    }
    drop(small_value);

    if LB < ell_b {
      let prefix_size = 1usize << LB;
      let prefix_weights = weights_from_r::<F>(&r_bs[..LB], prefix_size);
      let PrefixAbWithCClaims {
        a_layers: mut small_a_layers,
        b_layers: mut small_b_layers,
        mut c_claims,
      } = materialize_prefix_ab_with_c_claims::<E, SV>(
        &a_small_evals,
        &b_small_evals,
        &c_small_evals,
        &a_field_refs,
        &b_field_refs,
        &c_field_refs,
        &[],
        &[],
        &prefix_weights,
        prefix_size,
        &e_eq,
        left,
        right,
      )
      .expect("prefix materialization should succeed");

      assert_eq!(small_a_layers, a_layers);
      assert_eq!(small_b_layers, b_layers);
      assert_c_claims_match_layers(&c_claims, &c_layers, &e_eq, left, right);

      for (round, (&rho, &r_b)) in rhos.iter().zip(&r_bs).enumerate().skip(LB) {
        let (standard_e0, standard_quad_coeff) = compute_field_round_claim::<E>(
          &a_layers, &b_layers, &c_layers, &e_eq, left, right, &rhos, round,
        )
        .expect("standard suffix claim should compute");
        let (small_e0, small_quad_coeff) = compute_ab_c_claim_round::<E>(
          &small_a_layers,
          &small_b_layers,
          &c_claims,
          &e_eq,
          left,
          right,
          &rhos,
          round,
        )
        .expect("small suffix claim should compute");
        assert_eq!(
          (small_e0, small_quad_coeff),
          (standard_e0, standard_quad_coeff)
        );
        let standard_poly = generate_nifs_field_round_polynomial::<F>(
          rho,
          acc_eq,
          t_cur,
          standard_e0,
          standard_quad_coeff,
        )
        .expect("standard suffix polynomial should interpolate");
        let small_poly =
          generate_nifs_field_round_polynomial::<F>(rho, acc_eq, t_cur, small_e0, small_quad_coeff)
            .expect("small suffix polynomial should interpolate");
        assert_eq!(small_poly, standard_poly);

        t_cur = small_poly.evaluate(&r_b);
        acc_eq *= (F::ONE - r_b) * (F::ONE - rho) + r_b * rho;
        fold_materialized_layers(&mut a_layers, r_b);
        fold_materialized_layers(&mut b_layers, r_b);
        fold_materialized_layers(&mut c_layers, r_b);
        fold_materialized_layers(&mut small_a_layers, r_b);
        fold_materialized_layers(&mut small_b_layers, r_b);
        fold_c_claims(&mut c_claims, r_b);
        assert_c_claims_match_layers(&c_claims, &c_layers, &e_eq, left, right);
      }

      assert_eq!(small_a_layers, a_layers);
      assert_eq!(small_b_layers, b_layers);
    }

    let final_weights = weights_from_r::<F>(&r_bs, n_padded);
    let (small_az, small_bz, small_cz) = fold_small_final_abc_with_weights::<E, SV>(
      &a_small_evals,
      &b_small_evals,
      &c_small_evals,
      &final_weights,
    )
    .expect("small final fold should succeed");
    assert_eq!(
      small_az, a_layers[0],
      "Az fold mismatch for num_instances={num_instances}, l0={LB}"
    );
    assert_eq!(
      small_bz, b_layers[0],
      "Bz fold mismatch for num_instances={num_instances}, l0={LB}"
    );
    assert_eq!(
      small_cz, c_layers[0],
      "Cz fold mismatch for num_instances={num_instances}, l0={LB}"
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

  #[derive(Clone, Copy)]
  enum LargeCase {
    Ab,
    COnly,
  }

  fn synthetic_step_matvecs_with_large_position(
    num_instances: usize,
    num_constraints: usize,
    large_case: LargeCase,
    small_value_l0: usize,
  ) -> SmallValueNeutronNovaStepMatvecs<E> {
    let large_k = 3usize;
    let mut a_field = Vec::with_capacity(num_instances);
    let mut b_field = Vec::with_capacity(num_instances);
    let mut c_field = Vec::with_capacity(num_instances);

    for instance in 0..num_instances {
      let mut a = Vec::with_capacity(num_constraints);
      let mut b = Vec::with_capacity(num_constraints);
      let mut c = Vec::with_capacity(num_constraints);
      for k in 0..num_constraints {
        let mut a_value = F::from((((instance + 3) * (k + 5)) % 17 + 1) as u64);
        let mut b_value = F::from((((instance + 7) * (k + 2)) % 19 + 1) as u64);
        if k == large_k {
          match large_case {
            LargeCase::Ab => {
              a_value = F::from(1u64 << 35) * F::from(1u64 << 35);
              b_value = F::from(3u64);
            }
            LargeCase::COnly => {
              a_value = F::from(1u64 << 31);
              b_value = F::from(1u64 << 31);
            }
          }
        }
        a.push(a_value);
        b.push(b_value);
        c.push(a_value * b_value);
      }
      a_field.push(a);
      b_field.push(b);
      c_field.push(c);
    }

    let mut small = Vec::with_capacity(num_instances);
    let mut ab_large_positions = std::collections::BTreeSet::new();
    let mut c_large_positions = std::collections::BTreeSet::new();
    for ((a, b), c) in a_field.iter().zip(&b_field).zip(&c_field) {
      let (a_small, a_large) = field_to_i64_or_zero_for_l0(a, small_value_l0);
      let (b_small, b_large) = field_to_i64_or_zero_for_l0(b, small_value_l0);
      let (c_small, c_large) = to_small_vec_or_zero(c);
      ab_large_positions.extend(a_large);
      ab_large_positions.extend(b_large);
      c_large_positions.extend(c_large);
      small.push((a_small, b_small, c_small));
    }

    let ab_large_positions = ab_large_positions.into_iter().collect::<Vec<_>>();
    let c_large_positions = c_large_positions.into_iter().collect::<Vec<_>>();
    for (a_small, b_small, c_small) in &mut small {
      for &k in &ab_large_positions {
        a_small[k] = 0;
        b_small[k] = 0;
      }
      for &k in &c_large_positions {
        c_small[k] = 0;
      }
    }

    let (az_small, bz_small, cz_small) = small.into_iter().fold(
      (
        Vec::with_capacity(num_instances),
        Vec::with_capacity(num_instances),
        Vec::with_capacity(num_instances),
      ),
      |(mut az_acc, mut bz_acc, mut cz_acc), (az, bz, cz)| {
        az_acc.push(az);
        bz_acc.push(bz);
        cz_acc.push(cz);
        (az_acc, bz_acc, cz_acc)
      },
    );

    SmallValueNeutronNovaStepMatvecs {
      field: FieldStepMatvecs {
        az: a_field,
        bz: b_field,
        cz: c_field,
      },
      small_abc: SmallAbcStepMatvecs {
        ab: SmallAbStepMatvecs {
          az_small,
          bz_small,
          large_positions: ab_large_positions,
          l0: small_value_l0,
        },
        cz_small,
        c_large_positions,
      },
    }
  }

  fn run_large_position_accumulator_case<const LB: usize>(large_case: LargeCase) {
    let num_instances = 4usize;
    let left = 4usize;
    let right = 2usize;
    let num_constraints = left * right;
    let step_matvecs =
      synthetic_step_matvecs_with_large_position(num_instances, num_constraints, large_case, LB);
    match large_case {
      LargeCase::Ab => {
        assert_eq!(step_matvecs.small_abc.ab.large_positions, vec![3]);
        assert_eq!(step_matvecs.small_abc.c_large_positions, vec![3]);
      }
      LargeCase::COnly => {
        assert!(step_matvecs.small_abc.ab.large_positions.is_empty());
        assert_eq!(step_matvecs.small_abc.c_large_positions, vec![3]);
      }
    }
    let a_small = layer_refs(&step_matvecs.small_abc.ab.az_small);
    let b_small = layer_refs(&step_matvecs.small_abc.ab.bz_small);
    let c_small = layer_refs(&step_matvecs.small_abc.cz_small);
    let a_field_refs = layer_refs(&step_matvecs.field.az);
    let b_field_refs = layer_refs(&step_matvecs.field.bz);
    let c_field_refs = layer_refs(&step_matvecs.field.cz);
    let mut a_layers = step_matvecs.field.az.clone();
    let mut b_layers = step_matvecs.field.bz.clone();
    let mut c_layers = step_matvecs.field.cz.clone();
    let ell_b = num_instances.trailing_zeros() as usize;
    let (ell_cons, derived_left, derived_right) = tensor_decomp(num_constraints);
    assert_eq!((derived_left, derived_right), (left, right));
    let e_eq = PowPolynomial::split_evals(F::from(23u64), ell_cons, left, right);
    let rhos = (0..ell_b)
      .map(|round| F::from((5 * round + 11) as u64))
      .collect::<Vec<_>>();
    let r_bs = (0..ell_b)
      .map(|round| F::from((7 * round + 13) as u64))
      .collect::<Vec<_>>();

    let accumulators = build_accumulators_neutronnova::<F, i64>(
      &a_small,
      &b_small,
      &a_field_refs,
      &b_field_refs,
      &step_matvecs.small_abc.ab.large_positions,
      &e_eq,
      left,
      right,
      &rhos,
      LB,
    )
    .expect("mixed accumulator construction should succeed");
    let mut small_value =
      SmallValueSumCheck::<F, SMALL_VALUE_T_DEGREE>::from_accumulators(accumulators);
    let mut t_cur = F::ZERO;
    let mut acc_eq = F::ONE;

    for (round, (&rho, &r_b)) in rhos.iter().zip(&r_bs).enumerate().take(LB) {
      let (standard_e0, standard_quad_coeff) = compute_field_round_claim::<E>(
        &a_layers, &b_layers, &c_layers, &e_eq, left, right, &rhos, round,
      )
      .expect("standard field round should compute");
      let standard_poly = generate_nifs_field_round_polynomial::<F>(
        rho,
        acc_eq,
        t_cur,
        standard_e0,
        standard_quad_coeff,
      )
      .expect("standard polynomial should interpolate");
      let (small_poly, li) =
        generate_univariate_sumcheck_polynomial_from_accumulator(&small_value, round, rho, t_cur)
          .expect("mixed accumulator polynomial should interpolate");
      assert_eq!(small_poly, standard_poly);

      t_cur = small_poly.evaluate(&r_b);
      acc_eq = li.eval_linear_at(r_b);
      small_value.advance(&li, r_b);
      fold_materialized_layers(&mut a_layers, r_b);
      fold_materialized_layers(&mut b_layers, r_b);
      fold_materialized_layers(&mut c_layers, r_b);
    }

    for &r_b in r_bs.iter().take(ell_b).skip(LB) {
      fold_materialized_layers(&mut a_layers, r_b);
      fold_materialized_layers(&mut b_layers, r_b);
      fold_materialized_layers(&mut c_layers, r_b);
    }

    let final_weights = weights_from_r::<F>(&r_bs, num_instances);
    let (az, bz, cz) = fold_final_abc_with_weights::<E, i64>(
      &a_small,
      &b_small,
      &c_small,
      &a_field_refs,
      &b_field_refs,
      &c_field_refs,
      &step_matvecs.small_abc.ab.large_positions,
      &step_matvecs.small_abc.c_large_positions,
      &final_weights,
    )
    .expect("mixed final fold should succeed");
    assert_eq!(az, a_layers[0]);
    assert_eq!(bz, b_layers[0]);
    assert_eq!(cz, c_layers[0]);
  }

  fn dot_field_with_split_eq(layer: &[F], e_eq: &[F], left: usize, right: usize) -> F {
    let e_left = &e_eq[..left];
    let e_right = &e_eq[left..];
    let mut acc = F::ZERO;
    for (i, e_right_i) in e_right.iter().enumerate().take(right) {
      let base = i * left;
      let mut inner = F::ZERO;
      for j in 0..left {
        inner += e_left[j] * layer[base + j];
      }
      acc += *e_right_i * inner;
    }
    acc
  }

  fn fold_materialized_layers(layers: &mut Vec<Vec<F>>, r: F) {
    let pairs = layers.len() / 2;
    for i in 0..pairs {
      fold_layer_pair_into(layers, 2 * i, 2 * i + 1, i, r);
    }
    layers.truncate(pairs);
  }

  fn fold_c_claims(c_claims: &mut Vec<F>, r: F) {
    let pairs = c_claims.len() / 2;
    for i in 0..pairs {
      let even = c_claims[2 * i];
      let odd = c_claims[2 * i + 1];
      c_claims[i] = even + r * (odd - even);
    }
    c_claims.truncate(pairs);
  }

  fn assert_c_claims_match_layers(
    c_claims: &[F],
    c_layers: &[Vec<F>],
    e_eq: &[F],
    left: usize,
    right: usize,
  ) {
    assert_eq!(c_claims.len(), c_layers.len());
    for (claim, c_layer) in c_claims.iter().zip(c_layers.iter()) {
      assert_eq!(*claim, dot_field_with_split_eq(c_layer, e_eq, left, right));
    }
  }

  #[test]
  fn test_prefix_c_claims_match_materialized_c_dot_e() {
    const LB: usize = 2;
    let num_instances = 8usize;
    let n_padded = num_instances.next_power_of_two();
    let left = 4;
    let right = 2;
    let num_constraints = left * right;
    let (a_polys, b_polys, c_polys) =
      synthetic_small_abc_polys::<i32>(num_instances, num_constraints);
    let a_certs = certify_layers::<i32, LB>(&a_polys);
    let b_certs = certify_layers::<i32, LB>(&b_polys);
    let a_small = padded_small_layers(&a_certs, num_instances, n_padded);
    let b_small = padded_small_layers(&b_certs, num_instances, n_padded);
    let a_small_evals = make_layer_evals(&a_small);
    let b_small_evals = make_layer_evals(&b_small);
    let c_small_evals = padded_plain_layer_evals(&c_polys, num_instances, n_padded);
    let (a_field_layers, b_field_layers, c_field_layers) =
      field_layers_from_small_evals(&a_small_evals, &b_small_evals, &c_small_evals);
    let a_field_refs = layer_refs(&a_field_layers);
    let b_field_refs = layer_refs(&b_field_layers);
    let c_field_refs = layer_refs(&c_field_layers);
    let e_eq = (0..left + right)
      .map(|i| F::from((i as u64) + 3))
      .collect::<Vec<_>>();
    let r_bs = vec![F::from(7u64), F::from(11u64)];
    let prefix_size = 1usize << LB;
    let prefix_weights = weights_from_r::<F>(&r_bs, prefix_size);

    let prefix = materialize_prefix_ab_with_c_claims::<E, i32>(
      &a_small_evals,
      &b_small_evals,
      &c_small_evals,
      &a_field_refs,
      &b_field_refs,
      &c_field_refs,
      &[],
      &[],
      &prefix_weights,
      prefix_size,
      &e_eq,
      left,
      right,
    )
    .expect("prefix materialization should succeed");
    let c_layers =
      fold_small_layers_with_weights::<E, i32>(&c_small_evals, &prefix_weights, prefix_size)
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
    let (a_polys, b_polys, c_polys) =
      synthetic_small_abc_polys::<i32>(num_instances, num_constraints);
    let a_certs = certify_layers::<i32, LB>(&a_polys);
    let b_certs = certify_layers::<i32, LB>(&b_polys);
    let a_small = padded_small_layers(&a_certs, num_instances, n_padded);
    let b_small = padded_small_layers(&b_certs, num_instances, n_padded);
    let a_small_evals = make_layer_evals(&a_small);
    let b_small_evals = make_layer_evals(&b_small);
    let c_small_evals = padded_plain_layer_evals(&c_polys, num_instances, n_padded);
    let (a_field_layers, b_field_layers, c_field_layers) =
      field_layers_from_small_evals(&a_small_evals, &b_small_evals, &c_small_evals);
    let a_field_refs = layer_refs(&a_field_layers);
    let b_field_refs = layer_refs(&b_field_layers);
    let c_field_refs = layer_refs(&c_field_layers);
    let (ell_cons, derived_left, derived_right) = tensor_decomp(num_constraints);
    assert_eq!(derived_left, left);
    assert_eq!(derived_right, right);
    let e_eq = PowPolynomial::split_evals(F::from(13u64), ell_cons, left, right);
    let r_bs = vec![F::from(7u64)];
    let prefix_size = 1usize << LB;
    let prefix_weights = weights_from_r::<F>(&r_bs, prefix_size);

    let mut c_layers_materialized =
      fold_small_layers_with_weights::<E, i32>(&c_small_evals, &prefix_weights, prefix_size)?;
    let PrefixAbWithCClaims {
      a_layers,
      b_layers,
      mut c_claims,
    } = materialize_prefix_ab_with_c_claims::<E, i32>(
      &a_small_evals,
      &b_small_evals,
      &c_small_evals,
      &a_field_refs,
      &b_field_refs,
      &c_field_refs,
      &[],
      &[],
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
        let even = c_claims[2 * i];
        let odd = c_claims[2 * i + 1];
        c_claims[i] = even + r * (odd - even);
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
  fn test_lazy_c_suffix_claims_match_materialized_suffix_fixed_challenges() {
    run_small_matches_standard_case::<i32, 1>(8);
  }

  #[test]
  fn test_small_neutronnova_prove_small_smoke_random_blinds() {
    const LB: usize = 1;
    let num_instances = 4usize;
    let left = 4;
    let right = 2;
    let num_constraints = left * right;
    let n_padded = num_instances.next_power_of_two();
    let ell_b = n_padded.trailing_zeros() as usize;
    let (a_polys, b_polys, c_polys) =
      synthetic_small_abc_polys::<i32>(num_instances, num_constraints);
    let a_certs = certify_layers::<i32, LB>(&a_polys);
    let b_certs = certify_layers::<i32, LB>(&b_polys);
    let a_small = padded_small_layers(&a_certs, num_instances, n_padded);
    let b_small = padded_small_layers(&b_certs, num_instances, n_padded);
    let a_small_evals = make_layer_evals(&a_small);
    let b_small_evals = make_layer_evals(&b_small);
    let c_small_evals = padded_plain_layer_evals(&c_polys, num_instances, n_padded);
    let (a_field_layers, b_field_layers, c_field_layers) =
      field_layers_from_small_evals(&a_small_evals, &b_small_evals, &c_small_evals);
    let a_field_refs = layer_refs(&a_field_layers);
    let b_field_refs = layer_refs(&b_field_layers);
    let c_field_refs = layer_refs(&c_field_layers);
    let mut vc = NeutronNovaVerifierCircuit::<E>::default(ell_b, 1, 2, 32);
    let (vc_shape, vc_ck, _) =
      <ShapeCS<E> as MultiRoundSpartanShape<E>>::multiround_r1cs_shape(&vc)
        .expect("verifier circuit shape should synthesize");
    let mut vc_state = SatisfyingAssignment::<E>::initialize_multiround_witness(&vc_shape)
      .expect("verifier circuit witness state should initialize");
    let s = synthetic_split_shape(5, num_constraints);
    let (instances, witnesses) = synthetic_instances_and_witnesses(&vc_ck, num_instances, 5, 2);
    let mut transcript = <E as Engine>::TE::new(b"small_neutronnova_smoke");
    let (e_eq, rhos, derived_left, derived_right) =
      replay_nifs_prefix(&mut transcript, &instances, num_constraints);

    assert_eq!(derived_left, left);
    assert_eq!(derived_right, right);
    NeutronNovaNIFS::<E>::prove_small::<i32>(
      &s,
      &vc_ck,
      instances,
      witnesses,
      &a_small_evals,
      &b_small_evals,
      &c_small_evals,
      &a_field_refs,
      &b_field_refs,
      &c_field_refs,
      &[],
      &[],
      &e_eq,
      derived_left,
      derived_right,
      &rhos,
      LB,
      &mut vc,
      &mut vc_state,
      &vc_shape,
      &vc_ck,
      &mut transcript,
    )
    .expect("small NeutronNova NIFS should prove with ordinary random blinds");
  }

  #[test]
  fn test_small_neutronnova_nifs_matches_standard_nifs_fixed_challenges() {
    for num_instances in [2usize, 4, 8, 16] {
      let ell_b = num_instances.trailing_zeros() as usize;
      for l0 in 1..=ell_b {
        run_small_matches_standard_case_for_l0::<i32>(num_instances, l0);
      }
    }
  }

  #[test]
  fn test_small_neutronnova_nifs_matches_standard_nifs_fixed_challenges_i64_smoke() {
    run_small_matches_standard_case::<i64, 2>(4);
  }

  #[test]
  fn test_small_neutronnova_nifs_matches_standard_nifs_with_padding() {
    run_small_matches_standard_case::<i32, 1>(3);
  }

  #[test]
  fn test_mixed_accumulator_corrects_large_ab_positions() {
    run_large_position_accumulator_case::<1>(LargeCase::Ab);
    run_large_position_accumulator_case::<2>(LargeCase::Ab);
  }

  #[test]
  fn test_mixed_accumulator_corrects_c_only_large_positions() {
    run_large_position_accumulator_case::<1>(LargeCase::COnly);
    run_large_position_accumulator_case::<2>(LargeCase::COnly);
  }

  #[test]
  fn test_mixed_accumulator_rejects_out_of_range_large_position() {
    const L0: usize = 1;
    let num_instances = 4usize;
    let left = 4usize;
    let right = 2usize;
    let num_constraints = left * right;
    let step_matvecs =
      synthetic_step_matvecs_with_large_position(num_instances, num_constraints, LargeCase::Ab, L0);
    let a_small = layer_refs(&step_matvecs.small_abc.ab.az_small);
    let b_small = layer_refs(&step_matvecs.small_abc.ab.bz_small);
    let a_field_refs = layer_refs(&step_matvecs.field.az);
    let b_field_refs = layer_refs(&step_matvecs.field.bz);
    let ell_b = num_instances.trailing_zeros() as usize;
    let e_eq = (0..left + right)
      .map(|i| F::from((i as u64) + 3))
      .collect::<Vec<_>>();
    let rhos = (0..ell_b)
      .map(|round| F::from((5 * round + 11) as u64))
      .collect::<Vec<_>>();

    let err = match build_accumulators_neutronnova::<F, i64>(
      &a_small,
      &b_small,
      &a_field_refs,
      &b_field_refs,
      &[num_constraints],
      &e_eq,
      left,
      right,
      &rhos,
      L0,
    ) {
      Ok(_) => panic!("out-of-range large position should be rejected"),
      Err(err) => err,
    };
    assert!(matches!(
      err,
      SpartanError::InvalidInputLength { reason } if reason.contains("out of range")
    ));
  }

  #[cfg(debug_assertions)]
  #[test]
  #[should_panic(expected = "small Az cache mismatch")]
  fn test_debug_cache_validation_rejects_non_large_small_entry_corruption() {
    let mut step_matvecs = synthetic_step_matvecs_with_large_position(4, 8, LargeCase::Ab, 1);
    assert!(!step_matvecs.small_abc.ab.large_positions.contains(&0));
    step_matvecs.small_abc.ab.az_small[0][0] += 1;

    crate::neutronnova_zk::debug_validate_small_value_step_matvecs::<E>(&step_matvecs);
  }

  #[cfg(debug_assertions)]
  #[test]
  #[should_panic(expected = "large-position small Az entry must be zero")]
  fn test_debug_cache_validation_rejects_large_position_small_entry_corruption() {
    let mut step_matvecs = synthetic_step_matvecs_with_large_position(4, 8, LargeCase::Ab, 1);
    let large_position = step_matvecs.small_abc.ab.large_positions[0];
    step_matvecs.small_abc.ab.az_small[0][large_position] = 1;

    crate::neutronnova_zk::debug_validate_small_value_step_matvecs::<E>(&step_matvecs);
  }

  #[cfg(debug_assertions)]
  #[test]
  #[should_panic(expected = "field Az*Bz != Cz")]
  fn test_debug_cache_validation_rejects_field_cz_relation_corruption() {
    let mut step_matvecs = synthetic_step_matvecs_with_large_position(4, 8, LargeCase::Ab, 1);
    assert!(!step_matvecs.small_abc.ab.large_positions.contains(&0));
    step_matvecs.field.cz[0][0] += F::ONE;
    step_matvecs.small_abc.cz_small[0][0] += 1;

    crate::neutronnova_zk::debug_validate_small_value_step_matvecs::<E>(&step_matvecs);
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
