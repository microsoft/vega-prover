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
    NeutronNovaNIFS, continue_field_suffix, finalize_nifs_step_claim, fold_witness_and_instance,
    process_nifs_round,
  },
  r1cs::{R1CSInstance, R1CSWitness, SplitMultiRoundR1CSShape, SplitR1CSShape, weights_from_r},
  small_sumcheck::{SmallValueSumCheck, generate_univariate_sumcheck_polynomial},
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
      let (poly, li) =
        generate_univariate_sumcheck_polynomial(&small_value, round, rhos[round], t_cur)?;
      let r_i = process_nifs_round(vc, vc_state, vc_shape, vc_ck, transcript, round, &poly)?;
      t_cur = poly.evaluate(&r_i);
      acc_eq = li.eval_linear_at(r_i);
      small_value.advance(&li, r_i);
      r_bs.push(r_i);
    }

    let (az_step, bz_step, cz_step) = if LB == ell_b {
      let final_weights = weights_from_r::<E::Scalar>(&r_bs, n_padded);
      let mut a_folded =
        fold_small_layers_with_weights::<E, SV>(&a_small_evals, &final_weights, n_padded)?;
      let mut b_folded =
        fold_small_layers_with_weights::<E, SV>(&b_small_evals, &final_weights, n_padded)?;
      let mut c_folded = fold_small_c_from_ab_with_weights::<E, SV>(
        &a_small_evals,
        &b_small_evals,
        &final_weights,
        n_padded,
      )?;
      (
        a_folded.pop().ok_or_else(empty_fold_error)?,
        b_folded.pop().ok_or_else(empty_fold_error)?,
        c_folded.pop().ok_or_else(empty_fold_error)?,
      )
    } else {
      let prefix_size = 1usize << LB;
      let prefix_weights = weights_from_r::<E::Scalar>(&r_bs, prefix_size);
      let mut a_layers =
        fold_small_layers_with_weights::<E, SV>(&a_small_evals, &prefix_weights, prefix_size)?;
      let mut b_layers =
        fold_small_layers_with_weights::<E, SV>(&b_small_evals, &prefix_weights, prefix_size)?;
      let mut c_layers = fold_small_c_from_ab_with_weights::<E, SV>(
        &a_small_evals,
        &b_small_evals,
        &prefix_weights,
        prefix_size,
      )?;

      continue_field_suffix(
        &mut a_layers,
        &mut b_layers,
        &mut c_layers,
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

      (
        a_layers.pop().ok_or_else(empty_fold_error)?,
        b_layers.pop().ok_or_else(empty_fold_error)?,
        c_layers.pop().ok_or_else(empty_fold_error)?,
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
