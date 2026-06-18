// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Small-value NeutronNova NIFS helpers.

use std::{collections::BTreeSet, marker::PhantomData};

use crate::{
  CommitmentKey,
  bellpepper::{r1cs::MultiRoundSpartanWitness, solver::SatisfyingAssignment},
  big_num::{DelayedReduction, WideMul},
  errors::SpartanError,
  lagrange_accumulator::{
    ExtensionBoundProduct, ExtensionBoundedPoly, SMALL_VALUE_T_DEGREE, SmallValue,
    SmallValueEngine, SmallValueField, build_accumulators_neutronnova,
  },
  neutronnova_zk::{
    FieldStepMLEs, NeutronNovaNifsOutput, NeutronNovaNifsStrategy, NifsTranscriptState,
    SmallAbStepMLEs, SmallAbcStepMLEs, continue_ab_suffix_with_c_claims, finalize_nifs_step_claim,
    fold_witness_and_instance, invalid_input, padded_layer_slices, padded_map_by_repeating_first,
    prepare_nifs_transcript, process_nifs_round, validate_instance_witness_counts,
  },
  r1cs::{R1CSInstance, R1CSWitness, SplitMultiRoundR1CSShape, SplitR1CSShape, weights_from_r},
  small_sumcheck::{SmallValueSumCheck, generate_univariate_sumcheck_polynomial_from_accumulator},
  traits::{Engine, pcs::FoldingEngineTrait},
  zk::NeutronNovaVerifierCircuit,
};
#[cfg(test)]
use crate::{
  lagrange_accumulator::SmallValueExtensionBoundedPoly, polys::multilinear::MultilinearPolynomial,
};
use ff::Field;
use num_traits::{Bounded, ToPrimitive, Zero};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A small-value NeutronNova NIFS backend.
pub struct SmallValueNeutronNovaNIFS<E: Engine, SV, const L0: usize> {
  _p: PhantomData<(E, SV)>,
}

impl<E: Engine, SV, const L0: usize> Debug for SmallValueNeutronNovaNIFS<E, SV, L0> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SmallValueNeutronNovaNIFS")
      .field("L0", &L0)
      .finish()
  }
}

/// Small-value accumulator NeutronNova step MLEs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "SV: Serialize", deserialize = "SV: Deserialize<'de>"))]
pub struct SmallValueNeutronNovaStepMLEs<E: Engine, SV, const L0: usize>
where
  SV: WideMul,
{
  /// Full field-valued step MLE tables used for large-position corrections.
  pub field: FieldStepMLEs<E>,
  /// Small Az/Bz/Cz tables used by the small-value accumulator.
  pub small_abc: SmallAbcStepMLEs<
    ExtensionBoundedPoly<SV, <SV as WideMul>::Product, 2, L0>,
    ExtensionBoundedPoly<SV, <SV as WideMul>::Product, 1, L0>,
  >,
}

impl<E, SV, const L0: usize> NeutronNovaNifsStrategy<E> for SmallValueNeutronNovaNIFS<E, SV, L0>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  SV: SmallValue + Bounded + ToPrimitive,
  E::Scalar: SmallValueEngine<SV> + Default,
{
  type Input = ();
  type StepMLEs = SmallValueNeutronNovaStepMLEs<E, SV, L0>;

  fn is_small(_: &Self::Input) -> bool {
    true
  }

  fn build_step_mles_from_field(
    field_mles: Vec<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>)>,
    _input: &Self::Input,
  ) -> Result<Self::StepMLEs, SpartanError> {
    let field = FieldStepMLEs::from_triples(field_mles);
    let small_abc = build_certified_small_abc_from_field::<E, SV, L0>(&field)?;
    Ok(SmallValueNeutronNovaStepMLEs { field, small_abc })
  }

  #[allow(clippy::too_many_arguments)]
  fn prove(
    s: &SplitR1CSShape<E>,
    ck: &CommitmentKey<E>,
    us: Vec<R1CSInstance<E>>,
    ws: Vec<R1CSWitness<E>>,
    step_mles: Self::StepMLEs,
    _nifs_input: &Self::Input,
    vc: &mut NeutronNovaVerifierCircuit<E>,
    vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
    vc_shape: &SplitMultiRoundR1CSShape<E>,
    vc_ck: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<NeutronNovaNifsOutput<E>, SpartanError> {
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
    if step_mles.field.az.len() != num_instances {
      return Err(invalid_input(format!(
        "NIFS received {} step MLE layers but {} instances",
        step_mles.field.az.len(),
        num_instances
      )));
    }
    if L0 == 0 {
      return Err(invalid_input(
        "small-value NeutronNova NIFS requires L0 > 0",
      ));
    }
    #[cfg(debug_assertions)]
    debug_validate_small_value_step_mles::<E, SV, L0>(&step_mles);

    let NifsTranscriptState {
      e_eq,
      left,
      right,
      rhos,
      ..
    } = prepare_nifs_transcript(s, &us, transcript)?;

    prove_small::<E, SV, L0>(
      s,
      ck,
      us,
      ws,
      &step_mles.field,
      &step_mles.small_abc,
      e_eq,
      left,
      right,
      &rhos,
      vc,
      vc_state,
      vc_shape,
      vc_ck,
      transcript,
    )
  }
}

fn build_certified_small_abc_from_field<E, SV, const L0: usize>(
  field: &FieldStepMLEs<E>,
) -> Result<
  SmallAbcStepMLEs<
    ExtensionBoundedPoly<SV, <SV as WideMul>::Product, 2, L0>,
    ExtensionBoundedPoly<SV, <SV as WideMul>::Product, 1, L0>,
  >,
  SpartanError,
>
where
  E: Engine,
  SV: SmallValue + Bounded + ToPrimitive,
  E::Scalar: SmallValueField<SV>,
{
  if L0 == 0 {
    return Err(SpartanError::InvalidInputLength {
      reason: "small-value NeutronNova NIFS requires L0 > 0".to_string(),
    });
  }

  let mut az_small = Vec::with_capacity(field.len());
  let mut bz_small = Vec::with_capacity(field.len());
  let mut cz_small = Vec::with_capacity(field.len());
  let mut ab_field_positions = BTreeSet::new();
  let mut c_field_positions = BTreeSet::new();

  for az_layer in &field.az {
    let (cert, field_positions) =
      ExtensionBoundedPoly::<SV, <SV as WideMul>::Product, 2, L0>::from_field_values_with_product_bound(
        az_layer,
      );
    ab_field_positions.extend(field_positions);
    az_small.push(cert);
  }

  for bz_layer in &field.bz {
    let (cert, field_positions) =
      ExtensionBoundedPoly::<SV, <SV as WideMul>::Product, 2, L0>::from_field_values_with_product_bound(
        bz_layer,
      );
    ab_field_positions.extend(field_positions);
    bz_small.push(cert);
  }

  for cz_layer in &field.cz {
    let (cert, field_positions) =
      ExtensionBoundedPoly::<SV, <SV as WideMul>::Product, 1, L0>::from_field_values_with_extension_bound(
        cz_layer,
      );
    c_field_positions.extend(field_positions);
    cz_small.push(cert);
  }

  if !ab_field_positions.is_empty() {
    for cert in az_small.iter_mut().chain(bz_small.iter_mut()) {
      cert.zero_positions(&ab_field_positions);
    }
  }
  if !c_field_positions.is_empty() {
    for cert in &mut cz_small {
      cert.zero_positions(&c_field_positions);
    }
  }

  let small_abc = SmallAbcStepMLEs {
    ab: SmallAbStepMLEs {
      az_small,
      bz_small,
      field_positions: ab_field_positions.into_iter().collect(),
    },
    cz_small,
    c_field_positions: c_field_positions.into_iter().collect(),
  };
  #[cfg(debug_assertions)]
  debug_validate_certified_small_abc_cache::<E, SV, L0>(field, &small_abc);
  Ok(small_abc)
}

#[cfg(debug_assertions)]
fn debug_validate_small_value_step_mles<E, SV, const L0: usize>(
  step_mles: &SmallValueNeutronNovaStepMLEs<E, SV, L0>,
) where
  E: Engine,
  SV: SmallValue + Bounded + ToPrimitive,
  E::Scalar: SmallValueField<SV>,
{
  debug_validate_certified_small_abc_cache::<E, SV, L0>(&step_mles.field, &step_mles.small_abc);
}

#[cfg(debug_assertions)]
fn debug_validate_certified_small_abc_cache<E, SV, const L0: usize>(
  field: &FieldStepMLEs<E>,
  small_abc: &SmallAbcStepMLEs<
    ExtensionBoundedPoly<SV, <SV as WideMul>::Product, 2, L0>,
    ExtensionBoundedPoly<SV, <SV as WideMul>::Product, 1, L0>,
  >,
) where
  E: Engine,
  SV: SmallValue + Bounded + ToPrimitive,
  E::Scalar: SmallValueField<SV>,
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
  for &pos in &small_abc.ab.field_positions {
    debug_assert!(
      pos < layer_len,
      "A/B field position {} is out of range for layer length {}",
      pos,
      layer_len
    );
  }
  for &pos in &small_abc.c_field_positions {
    debug_assert!(
      pos < layer_len,
      "C field position {} is out of range for layer length {}",
      pos,
      layer_len
    );
  }

  for layer_idx in 0..field.az.len() {
    let az_field = &field.az[layer_idx];
    let bz_field = &field.bz[layer_idx];
    let cz_field = &field.cz[layer_idx];
    let az_small = &small_abc.ab.az_small[layer_idx].as_poly().Z;
    let bz_small = &small_abc.ab.bz_small[layer_idx].as_poly().Z;
    let cz_small = &small_abc.cz_small[layer_idx].as_poly().Z;

    debug_assert_eq!(az_field.len(), layer_len);
    debug_assert_eq!(bz_field.len(), layer_len);
    debug_assert_eq!(cz_field.len(), layer_len);
    debug_assert_eq!(az_small.len(), layer_len);
    debug_assert_eq!(bz_small.len(), layer_len);
    debug_assert_eq!(cz_small.len(), layer_len);

    for k in 0..layer_len {
      if small_abc.ab.field_positions.contains(&k) {
        debug_assert!(
          az_small[k].is_zero(),
          "A/B field-position small Az entry must be zero at layer {} index {}",
          layer_idx,
          k
        );
        debug_assert!(
          bz_small[k].is_zero(),
          "A/B field-position small Bz entry must be zero at layer {} index {}",
          layer_idx,
          k
        );
      } else {
        debug_assert_eq!(
          <E::Scalar as SmallValueField<SV>>::small_to_field(az_small[k]),
          az_field[k],
          "small Az cache mismatch at layer {} index {}",
          layer_idx,
          k
        );
        debug_assert_eq!(
          <E::Scalar as SmallValueField<SV>>::small_to_field(bz_small[k]),
          bz_field[k],
          "small Bz cache mismatch at layer {} index {}",
          layer_idx,
          k
        );
      }

      if small_abc.c_field_positions.contains(&k) {
        debug_assert!(
          cz_small[k].is_zero(),
          "C field-position small Cz entry must be zero at layer {} index {}",
          layer_idx,
          k
        );
      } else {
        debug_assert_eq!(
          <E::Scalar as SmallValueField<SV>>::small_to_field(cz_small[k]),
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

/// Prove small-value NeutronNova NIFS, deriving round challenges from the verifier circuit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_small<E, SV, const L0: usize>(
  s: &SplitR1CSShape<E>,
  ck: &CommitmentKey<E>,
  us: Vec<R1CSInstance<E>>,
  ws: Vec<R1CSWitness<E>>,
  field: &FieldStepMLEs<E>,
  small_abc: &SmallAbcStepMLEs<
    ExtensionBoundedPoly<SV, <SV as WideMul>::Product, 2, L0>,
    ExtensionBoundedPoly<SV, <SV as WideMul>::Product, 1, L0>,
  >,
  e_eq: Vec<E::Scalar>,
  left: usize,
  right: usize,
  rhos: &[E::Scalar],
  vc: &mut NeutronNovaVerifierCircuit<E>,
  vc_state: &mut <SatisfyingAssignment<E> as MultiRoundSpartanWitness<E>>::MultiRoundState,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
) -> Result<NeutronNovaNifsOutput<E>, SpartanError>
where
  E: Engine,
  E::PCS: FoldingEngineTrait<E>,
  SV: SmallValue + Bounded + ToPrimitive,
  E::Scalar: SmallValueEngine<SV> + Default,
{
  let num_instances = us.len();
  if num_instances == 0 {
    return Err(invalid_input(
      "small NeutronNova NIFS requires at least one instance",
    ));
  }
  if field.az.len() != num_instances
    || field.bz.len() != num_instances
    || field.cz.len() != num_instances
    || small_abc.ab.az_small.len() != num_instances
    || small_abc.ab.bz_small.len() != num_instances
    || small_abc.cz_small.len() != num_instances
  {
    return Err(invalid_input(
      "small NeutronNova MLE layer counts must match instance count",
    ));
  }

  let n_padded = num_instances.next_power_of_two();
  let a_bounded = padded_bounded_layers(&small_abc.ab.az_small, num_instances, n_padded);
  let b_bounded = padded_bounded_layers(&small_abc.ab.bz_small, num_instances, n_padded);
  let a_small_evals = bounded_layer_evals(&a_bounded);
  let b_small_evals = bounded_layer_evals(&b_bounded);
  let c_small_evals = padded_bounded_layer_evals(&small_abc.cz_small, num_instances, n_padded);
  let a_field_evals = padded_layer_slices(&field.az, num_instances, n_padded);
  let b_field_evals = padded_layer_slices(&field.bz, num_instances, n_padded);
  let c_field_evals = padded_layer_slices(&field.cz, num_instances, n_padded);

  let l0 = L0;
  let validated_n_padded = validate_layer_arrays::<E, SV>(
    &a_small_evals,
    &b_small_evals,
    &c_small_evals,
    &a_field_evals,
    &b_field_evals,
    &c_field_evals,
    &small_abc.ab.field_positions,
    &small_abc.c_field_positions,
    &e_eq,
    left,
    right,
    rhos,
    l0,
  )?;
  validate_instance_witness_counts::<E>(num_instances, &us, &ws)?;
  if n_padded != validated_n_padded {
    return Err(invalid_input(format!(
      "instance count {} is incompatible with padded layer count {}",
      num_instances, validated_n_padded
    )));
  }
  if vc.nifs_polys.len() != rhos.len() {
    return Err(invalid_input(format!(
      "verifier circuit has {} NIFS rounds but rhos has {}",
      vc.nifs_polys.len(),
      rhos.len()
    )));
  }

  let accumulators = build_accumulators_neutronnova::<E::Scalar, SV, L0>(
    &a_bounded,
    &b_bounded,
    &a_field_evals,
    &b_field_evals,
    &small_abc.ab.field_positions,
    &e_eq,
    left,
    right,
    rhos,
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
      &a_small_evals,
      &b_small_evals,
      &c_small_evals,
      &a_field_evals,
      &b_field_evals,
      &c_field_evals,
      &small_abc.ab.field_positions,
      &small_abc.c_field_positions,
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
      &a_small_evals,
      &b_small_evals,
      &c_small_evals,
      &a_field_evals,
      &b_field_evals,
      &c_field_evals,
      &small_abc.ab.field_positions,
      &small_abc.c_field_positions,
      &prefix_weights,
      prefix_size,
      &e_eq,
      left,
      right,
    )?;

    continue_ab_suffix_with_c_claims(
      &mut a_layers,
      &mut b_layers,
      &mut c_claims,
      &e_eq,
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
      &c_small_evals,
      &c_field_evals,
      &small_abc.c_field_positions,
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

  Ok((e_eq, az_step, bz_step, cz_step, folded_w, folded_u))
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

fn padded_bounded_layers<SV, Product, const D: usize, const L0: usize>(
  layers: &[ExtensionBoundedPoly<SV, Product, D, L0>],
  num_instances: usize,
  n_padded: usize,
) -> Vec<&ExtensionBoundedPoly<SV, Product, D, L0>> {
  padded_map_by_repeating_first(layers, num_instances, n_padded, |layer| layer)
}

fn bounded_layer_evals<'a, SV, Product, const D: usize, const L0: usize>(
  layers: &[&'a ExtensionBoundedPoly<SV, Product, D, L0>],
) -> Vec<&'a [SV]>
where
  SV: SmallValue + Bounded + ToPrimitive,
  Product: ExtensionBoundProduct,
{
  layers
    .iter()
    .map(|layer| layer.as_poly().Z.as_slice())
    .collect()
}

fn padded_bounded_layer_evals<SV, Product, const D: usize, const L0: usize>(
  layers: &[ExtensionBoundedPoly<SV, Product, D, L0>],
  num_instances: usize,
  n_padded: usize,
) -> Vec<&[SV]>
where
  SV: SmallValue + Bounded + ToPrimitive,
  Product: ExtensionBoundProduct,
{
  let bounded_layers = padded_bounded_layers(layers, num_instances, n_padded);
  bounded_layer_evals(&bounded_layers)
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

/// Fold the first `log2(prefix_size)` instance dimensions into materialized A/B suffix layers.
///
/// The small-value accumulator proves the first `L0` NeutronNova rounds without materializing
/// intermediate field layers. When the proof switches back to the ordinary suffix rounds, this
/// helper rebuilds the remaining A/B layers:
/// `A'_s = sum_p prefix_weights[p] * A_{s * prefix_size + p}`, and similarly for B.
///
/// C is not materialized here. The suffix rounds only need scalar C claims, so this helper computes
/// `c_claims[s] = <e_eq, C'_s>` directly from the original C layers and includes the sparse
/// field-backed positions that were zeroed in the small representation.
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
  if a_layers.is_empty() {
    return Err(invalid_input("cannot materialize empty prefix layer list"));
  }
  if b_layers.len() != a_layers.len() || c_layers.len() != a_layers.len() {
    return Err(invalid_input("A/B/C prefix layer counts must match"));
  }
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
  if left == 0 || right == 0 {
    return Err(invalid_input(
      "prefix materialization requires non-empty tensor factors",
    ));
  }
  let layer_len = a_layers[0].len();
  if !a_layers
    .iter()
    .chain(b_layers.iter())
    .chain(c_layers.iter())
    .all(|layer| layer.len() == layer_len)
  {
    return Err(invalid_input("all prefix layers must have the same length"));
  }
  if left.checked_mul(right) != Some(layer_len) {
    return Err(invalid_input(format!(
      "prefix layer length {} does not match left * right {}",
      layer_len,
      left.saturating_mul(right)
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

  let suffix_groups = a_layers.len() / prefix_size;

  // Materialize one folded A/B layer for each remaining suffix assignment:
  // layer'_s[k] = sum_p prefix_weights[p] * layer_{s * prefix_size + p}[k].
  let mut a_folded = vec![vec![E::Scalar::ZERO; layer_len]; suffix_groups];
  let mut b_folded = vec![vec![E::Scalar::ZERO; layer_len]; suffix_groups];

  // Flatten the output layers into mutable row slices so each parallel task owns
  // exactly one (suffix, row) of A'/B'.
  let a_rows = a_folded
    .iter_mut()
    .flat_map(|layer| layer.chunks_mut(left))
    .collect::<Vec<_>>();
  let b_rows = b_folded
    .iter_mut()
    .flat_map(|layer| layer.chunks_mut(left))
    .collect::<Vec<_>>();

  a_rows
    .into_par_iter()
    .zip(b_rows)
    .enumerate()
    .for_each(|(task_idx, (a_row, b_row))| {
      // Recover the output suffix layer and tensor row for this flattened task.
      let suffix_idx = task_idx / right;
      let row = task_idx % right;
      let layer_base = suffix_idx * prefix_size;

      // Fold all prefix layers at each column.
      for col in 0..left {
        let k = row * left + col;
        let mut a_acc = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();
        let mut b_acc = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();
        for (prefix_idx, weight) in prefix_weights.iter().enumerate() {
          let layer_idx = layer_base + prefix_idx;
          <E::Scalar as DelayedReduction<SV>>::unreduced_multiply_accumulate(
            &mut a_acc,
            weight,
            &a_layers[layer_idx][k],
          );
          <E::Scalar as DelayedReduction<SV>>::unreduced_multiply_accumulate(
            &mut b_acc,
            weight,
            &b_layers[layer_idx][k],
          );
        }

        a_row[col] = <E::Scalar as DelayedReduction<SV>>::reduce(&a_acc);
        b_row[col] = <E::Scalar as DelayedReduction<SV>>::reduce(&b_acc);
      }
    });

  // Patch sparse A/B positions that were zeroed in the small-value layers by recomputing their
  // prefix folds with full field values.
  apply_large_prefix_ab_corrections::<E>(
    &mut a_folded,
    &mut b_folded,
    a_field_layers,
    b_field_layers,
    ab_large_positions,
    prefix_weights,
    prefix_size,
  );

  let c_claims = compute_prefix_c_claims::<E, SV>(
    c_layers,
    c_field_layers,
    c_large_positions,
    prefix_weights,
    prefix_size,
    e_eq,
    left,
    right,
  )?;

  Ok(PrefixAbWithCClaims {
    a_layers: a_folded,
    b_layers: b_folded,
    c_claims,
  })
}

#[allow(clippy::too_many_arguments)]
fn apply_large_prefix_ab_corrections<E>(
  a_folded: &mut [Vec<E::Scalar>],
  b_folded: &mut [Vec<E::Scalar>],
  a_field_layers: &[&[E::Scalar]],
  b_field_layers: &[&[E::Scalar]],
  ab_large_positions: &[usize],
  prefix_weights: &[E::Scalar],
  prefix_size: usize,
) where
  E: Engine,
{
  if ab_large_positions.is_empty() {
    return;
  }
  let total = a_folded.first().map_or(0, Vec::len);

  for (suffix_idx, (a_out, b_out)) in a_folded.iter_mut().zip(b_folded.iter_mut()).enumerate() {
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
      let mut a_acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
      let mut b_acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
      for (prefix_idx, weight) in prefix_weights.iter().enumerate().take(prefix_size) {
        let layer_idx = suffix_idx * prefix_size + prefix_idx;
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut a_acc,
          weight,
          &a_field_layers[layer_idx][k],
        );
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut b_acc,
          weight,
          &b_field_layers[layer_idx][k],
        );
      }
      a_out[k] = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&a_acc);
      b_out[k] = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&b_acc);
    }
  }
}

#[allow(clippy::too_many_arguments)]
fn compute_prefix_c_claims<E, SV>(
  c_layers: &[&[SV]],
  c_field_layers: &[&[E::Scalar]],
  c_large_positions: &[usize],
  prefix_weights: &[E::Scalar],
  prefix_size: usize,
  e_eq: &[E::Scalar],
  left: usize,
  right: usize,
) -> Result<Vec<E::Scalar>, SpartanError>
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  if c_layers.is_empty() {
    return Err(invalid_input(
      "cannot compute C claims for empty layer list",
    ));
  }
  if prefix_size == 0 || !c_layers.len().is_multiple_of(prefix_size) {
    return Err(invalid_input("invalid C claim prefix group size"));
  }
  if prefix_weights.len() != prefix_size {
    return Err(invalid_input(format!(
      "prefix weight length {} does not match prefix size {}",
      prefix_weights.len(),
      prefix_size
    )));
  }
  if left == 0 || right == 0 {
    return Err(invalid_input(
      "prefix C claims require non-empty tensor factors",
    ));
  }
  let layer_len = c_layers[0].len();
  if !c_layers.iter().all(|layer| layer.len() == layer_len) {
    return Err(invalid_input("all C layers must have the same length"));
  }
  if left.checked_mul(right) != Some(layer_len) {
    return Err(invalid_input(format!(
      "C layer length {} does not match left * right {}",
      layer_len,
      left.saturating_mul(right)
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
  if !c_large_positions.is_empty()
    && (c_field_layers.len() != c_layers.len()
      || !c_field_layers.iter().all(|layer| layer.len() == layer_len))
  {
    return Err(invalid_input(
      "C field correction layers must match small C layers",
    ));
  }

  let e_left = &e_eq[..left];
  let e_right = &e_eq[left..];
  let large_eq = c_large_positions
    .iter()
    .filter_map(|&k| {
      if k >= layer_len {
        debug_assert!(
          k < layer_len,
          "C large position {} is out of range for layer length {}",
          k,
          layer_len
        );
        None
      } else {
        Some((k, e_right[k / left] * e_left[k % left]))
      }
    })
    .collect::<Vec<_>>();
  let suffix_groups = c_layers.len() / prefix_size;
  Ok(
    (0..suffix_groups)
      .into_par_iter()
      .map(|suffix_idx| {
        let layer_base = suffix_idx * prefix_size;
        prefix_weights
          .iter()
          .enumerate()
          .fold(E::Scalar::ZERO, |acc, (prefix_idx, weight)| {
            let layer_idx = layer_base + prefix_idx;
            let mut layer_claim =
              dot_small_layer_with_split_eq::<E, SV>(c_layers[layer_idx], e_left, e_right, left);

            if !large_eq.is_empty() {
              let c_field = c_field_layers[layer_idx];
              let mut large_acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
              for &(k, eq_k) in &large_eq {
                <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
                  &mut large_acc,
                  &eq_k,
                  &c_field[k],
                );
              }
              layer_claim += <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&large_acc);
            }

            acc + *weight * layer_claim
          })
      })
      .collect(),
  )
}

fn dot_small_layer_with_split_eq<E, SV>(
  layer: &[SV],
  e_left: &[E::Scalar],
  e_right: &[E::Scalar],
  left: usize,
) -> E::Scalar
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  let mut acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
  for (row, e_right_value) in e_right.iter().enumerate() {
    let base = row * left;
    let mut inner_acc = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();
    for (col, e_left_value) in e_left.iter().enumerate() {
      <E::Scalar as DelayedReduction<SV>>::unreduced_multiply_accumulate(
        &mut inner_acc,
        e_left_value,
        &layer[base + col],
      );
    }
    let inner = <E::Scalar as DelayedReduction<SV>>::reduce(&inner_acc);
    <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
      &mut acc,
      e_right_value,
      &inner,
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
  let mut c_final = fold_small_final_c_with_weights::<E, SV>(c_layers, weights)?;
  if !c_large_positions.is_empty()
    && (c_field_layers.len() != c_layers.len()
      || !c_field_layers
        .iter()
        .all(|layer| layer.len() == c_final.len()))
  {
    return Err(invalid_input(
      "C field correction layers must match small C layers",
    ));
  }
  apply_large_final_c_corrections::<E>(&mut c_final, c_field_layers, c_large_positions, weights);
  Ok(c_final)
}

fn apply_large_final_c_corrections<E>(
  c_final: &mut [E::Scalar],
  c_field_layers: &[&[E::Scalar]],
  c_large_positions: &[usize],
  weights: &[E::Scalar],
) where
  E: Engine,
{
  if c_large_positions.is_empty() {
    return;
  }
  let total = c_final.len();
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
    let mut c_acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
    for (layer_idx, weight) in weights.iter().enumerate() {
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut c_acc,
        weight,
        &c_field_layers[layer_idx][k],
      );
    }
    c_final[k] = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&c_acc);
  }
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
    let mut a_acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
    let mut b_acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
    for (layer_idx, weight) in weights.iter().enumerate() {
      if need_a {
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut a_acc,
          weight,
          &a_field_layers[layer_idx][k],
        );
      }
      if need_b {
        <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
          &mut b_acc,
          weight,
          &b_field_layers[layer_idx][k],
        );
      }
    }
    if let Some(a_final) = a_final.as_deref_mut() {
      a_final[k] = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&a_acc);
    }
    if let Some(b_final) = b_final.as_deref_mut() {
      b_final[k] = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&b_acc);
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
    let mut c_acc = <E::Scalar as DelayedReduction<E::Scalar>>::Accumulator::zero();
    for (layer_idx, weight) in weights.iter().enumerate() {
      <E::Scalar as DelayedReduction<E::Scalar>>::unreduced_multiply_accumulate(
        &mut c_acc,
        weight,
        &c_field_layers[layer_idx][k],
      );
    }
    c_final[k] = <E::Scalar as DelayedReduction<E::Scalar>>::reduce(&c_acc);
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

fn fold_small_final_c_with_weights<E, SV>(
  c_layers: &[&[SV]],
  weights: &[E::Scalar],
) -> Result<Vec<E::Scalar>, SpartanError>
where
  E: Engine,
  SV: SmallValue,
  E::Scalar: SmallValueEngine<SV>,
{
  if c_layers.is_empty() {
    return Err(invalid_input("cannot fold empty C layer list"));
  }
  if weights.len() != c_layers.len() {
    return Err(invalid_input(format!(
      "weight length {} does not match C layer count {}",
      weights.len(),
      c_layers.len()
    )));
  }
  let layer_len = c_layers[0].len();
  if !c_layers.iter().all(|layer| layer.len() == layer_len) {
    return Err(invalid_input("all C layers must have the same length"));
  }

  Ok(
    (0..layer_len)
      .into_par_iter()
      .map(|k| {
        let mut c_acc = <E::Scalar as DelayedReduction<SV>>::Accumulator::zero();
        for (weight, c_layer) in weights.iter().zip(c_layers.iter()) {
          <E::Scalar as DelayedReduction<SV>>::unreduced_multiply_accumulate(
            &mut c_acc,
            weight,
            &c_layer[k],
          );
        }
        <E::Scalar as DelayedReduction<SV>>::reduce(&c_acc)
      })
      .collect(),
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    lagrange_accumulator::SmallValueField,
    neutronnova_zk::{
      NeutronNovaNifsStrategy, SmallValueNeutronNovaNIFS, SmallValueNeutronNovaStepMLEs,
      compute_field_round_claim, fold_layer_pair_into, generate_nifs_field_round_polynomial,
    },
    polys::{multilinear::MultilinearPolynomial, power::PowPolynomial},
    provider::PallasHyraxEngine,
    traits::Engine,
  };
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
  ) -> Vec<SmallValueExtensionBoundedPoly<SV, LB>>
  where
    SV: SmallValue + Bounded + ToPrimitive,
  {
    layers
      .iter()
      .map(|layer| {
        SmallValueExtensionBoundedPoly::<_, LB>::new(layer.clone())
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

  fn tensor_decomp(n: usize) -> (usize, usize, usize) {
    let ell = n.next_power_of_two().trailing_zeros() as usize;
    let ell1 = ell.div_ceil(2);
    let ell2 = ell / 2;
    (ell, 1usize << ell1, 1usize << ell2)
  }

  fn synthetic_step_mles_with_large_c_position<const L0: usize>(
    num_instances: usize,
    num_constraints: usize,
  ) -> SmallValueNeutronNovaStepMLEs<E, i64, L0> {
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
          a_value = F::from(1u64 << 31);
          b_value = F::from(1u64 << 31);
        }
        a.push(a_value);
        b.push(b_value);
        c.push(a_value * b_value);
      }
      a_field.push(a);
      b_field.push(b);
      c_field.push(c);
    }

    let field_mles = a_field
      .into_iter()
      .zip(b_field)
      .zip(c_field)
      .map(|((az, bz), cz)| (az, bz, cz))
      .collect::<Vec<_>>();

    SmallValueNeutronNovaNIFS::<E, i64, L0>::build_step_mles_from_field(field_mles, &())
      .expect("synthetic field layers should build certified small MLEs")
  }

  fn run_c_only_large_position_final_fold_case<const LB: usize>() {
    let num_instances = 4usize;
    let left = 4usize;
    let right = 2usize;
    let num_constraints = left * right;
    let step_mles = synthetic_step_mles_with_large_c_position::<LB>(num_instances, num_constraints);
    assert!(step_mles.small_abc.ab.field_positions.is_empty());
    assert_eq!(step_mles.small_abc.c_field_positions, vec![3]);
    let a_bounded = padded_bounded_layers(
      &step_mles.small_abc.ab.az_small,
      num_instances,
      num_instances,
    );
    let b_bounded = padded_bounded_layers(
      &step_mles.small_abc.ab.bz_small,
      num_instances,
      num_instances,
    );
    let a_small = bounded_layer_evals(&a_bounded);
    let b_small = bounded_layer_evals(&b_bounded);
    let c_small =
      padded_bounded_layer_evals(&step_mles.small_abc.cz_small, num_instances, num_instances);
    let a_field_refs = layer_refs(&step_mles.field.az);
    let b_field_refs = layer_refs(&step_mles.field.bz);
    let c_field_refs = layer_refs(&step_mles.field.cz);
    let mut a_layers = step_mles.field.az.clone();
    let mut b_layers = step_mles.field.bz.clone();
    let mut c_layers = step_mles.field.cz.clone();
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

    let accumulators = build_accumulators_neutronnova::<F, i64, LB>(
      &a_bounded,
      &b_bounded,
      &a_field_refs,
      &b_field_refs,
      &step_mles.small_abc.ab.field_positions,
      &e_eq,
      left,
      right,
      &rhos,
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
      &step_mles.small_abc.ab.field_positions,
      &step_mles.small_abc.c_field_positions,
      &final_weights,
    )
    .expect("mixed final fold should succeed");
    assert_eq!(az, a_layers[0]);
    assert_eq!(bz, b_layers[0]);
    assert_eq!(cz, c_layers[0]);
  }

  #[test]
  fn test_prefix_materialization_corrects_c_field_positions_and_suffix_fold()
  -> Result<(), SpartanError> {
    const LB: usize = 1;
    let num_instances = 8usize;
    let left = 4usize;
    let right = 2usize;
    let num_constraints = left * right;
    let step_mles = synthetic_step_mles_with_large_c_position::<LB>(num_instances, num_constraints);
    assert_eq!(step_mles.small_abc.c_field_positions, vec![3]);

    let a_bounded = padded_bounded_layers(
      &step_mles.small_abc.ab.az_small,
      num_instances,
      num_instances,
    );
    let b_bounded = padded_bounded_layers(
      &step_mles.small_abc.ab.bz_small,
      num_instances,
      num_instances,
    );
    let a_small = bounded_layer_evals(&a_bounded);
    let b_small = bounded_layer_evals(&b_bounded);
    let c_small =
      padded_bounded_layer_evals(&step_mles.small_abc.cz_small, num_instances, num_instances);
    let a_field_refs = layer_refs(&step_mles.field.az);
    let b_field_refs = layer_refs(&step_mles.field.bz);
    let c_field_refs = layer_refs(&step_mles.field.cz);
    let ell_b = num_instances.trailing_zeros() as usize;
    let (ell_cons, derived_left, derived_right) = tensor_decomp(num_constraints);
    assert_eq!((derived_left, derived_right), (left, right));
    let e_eq = PowPolynomial::split_evals(F::from(29u64), ell_cons, left, right);
    let r_bs = (0..ell_b)
      .map(|round| F::from((11 * round + 17) as u64))
      .collect::<Vec<_>>();
    let prefix_size = 1usize << LB;
    let prefix_weights = weights_from_r::<F>(&r_bs[..LB], prefix_size);

    let prefix = materialize_prefix_ab_with_c_claims::<E, i64>(
      &a_small,
      &b_small,
      &c_small,
      &a_field_refs,
      &b_field_refs,
      &c_field_refs,
      &[],
      &step_mles.small_abc.c_field_positions,
      &prefix_weights,
      prefix_size,
      &e_eq,
      left,
      right,
    )?;

    let expected_prefix_c =
      materialize_field_prefix_layers(&c_field_refs, &prefix_weights, prefix_size);
    assert_c_claims_match_layers(&prefix.c_claims, &expected_prefix_c, &e_eq, left, right);

    let final_weights = weights_from_r::<F>(&r_bs, num_instances);
    let (_, _, expected_c) = fold_final_abc_with_weights::<E, i64>(
      &a_small,
      &b_small,
      &c_small,
      &a_field_refs,
      &b_field_refs,
      &c_field_refs,
      &[],
      &step_mles.small_abc.c_field_positions,
      &final_weights,
    )?;
    let c_folded = fold_final_c_with_weights::<E, i64>(
      &c_small,
      &c_field_refs,
      &step_mles.small_abc.c_field_positions,
      &final_weights,
    )?;
    assert_eq!(c_folded, expected_c);

    Ok(())
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

  fn materialize_field_prefix_layers(
    layers: &[&[F]],
    prefix_weights: &[F],
    prefix_size: usize,
  ) -> Vec<Vec<F>> {
    let suffix_groups = layers.len() / prefix_size;
    let layer_len = layers[0].len();
    (0..suffix_groups)
      .map(|suffix_idx| {
        let mut folded = vec![F::ZERO; layer_len];
        for (prefix_idx, weight) in prefix_weights.iter().enumerate() {
          let layer_idx = suffix_idx * prefix_size + prefix_idx;
          for (out, value) in folded.iter_mut().zip(layers[layer_idx].iter()) {
            *out += *weight * *value;
          }
        }
        folded
      })
      .collect()
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
    let a_small = padded_bounded_layers(&a_certs, num_instances, n_padded);
    let b_small = padded_bounded_layers(&b_certs, num_instances, n_padded);
    let a_small_evals = bounded_layer_evals(&a_small);
    let b_small_evals = bounded_layer_evals(&b_small);
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

    let expected_prefix_c =
      materialize_field_prefix_layers(&c_field_refs, &prefix_weights, prefix_size);
    assert_c_claims_match_layers(&prefix.c_claims, &expected_prefix_c, &e_eq, left, right);
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
    let a_small = padded_bounded_layers(&a_certs, num_instances, n_padded);
    let b_small = padded_bounded_layers(&b_certs, num_instances, n_padded);
    let a_small_evals = bounded_layer_evals(&a_small);
    let b_small_evals = bounded_layer_evals(&b_small);
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

    let mut c_layers_materialized =
      materialize_field_prefix_layers(&c_field_refs, &prefix_weights, prefix_size);
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
  fn test_final_fold_corrects_c_only_large_positions() {
    run_c_only_large_position_final_fold_case::<1>();
    run_c_only_large_position_final_fold_case::<2>();
  }
}
