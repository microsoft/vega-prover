// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Zero-knowledge sum-check drivers for Vega NeutronNova.

use crate::zk::NeutronNovaVerifierCircuit;
use ff::Field;
use vega_core::{
  CommitmentKey,
  bellpepper::{
    r1cs::{MultiRoundSpartanWitness, MultiRoundState},
    solver::SatisfyingAssignment,
  },
  errors::SpartanError,
  polys::{multilinear::MultilinearPolynomial, univariate::UniPoly},
  r1cs::SplitMultiRoundR1CSShape,
  sumcheck::SumcheckProof,
  traits::Engine,
};

/// Executes a **quadratic** batched sum-check in zero-knowledge mode and returns the
/// sequence of verifier challenges used for the inner round.
///
/// Uses delayed modular reduction for improved performance.
pub fn prove_quad_batched_zk<E: Engine>(
  claims: &[E::Scalar; 2],
  num_rounds: usize,
  poly_A_0: &mut MultilinearPolynomial<E::Scalar>,
  poly_A_1: &mut MultilinearPolynomial<E::Scalar>,
  poly_B_0: &mut MultilinearPolynomial<E::Scalar>,
  poly_B_1: &mut MultilinearPolynomial<E::Scalar>,
  verifier_circuit: &mut NeutronNovaVerifierCircuit<E>,
  state: &mut MultiRoundState<E>,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
  start_round: usize,
) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
  let mut r_y: Vec<E::Scalar> = Vec::with_capacity(num_rounds);
  // Maintain separate claims for step and core branches
  let mut claim_step_round = claims[0];
  let mut claim_core_round = claims[1];

  for j in 0..num_rounds {
    // -------- interpolate coeffs --------
    let ((eval0_s, t_inf_s), (eval0_c, t_inf_c)) = rayon::join(
      || SumcheckProof::<E>::compute_eval_points_quad(poly_A_0, poly_B_0),
      || SumcheckProof::<E>::compute_eval_points_quad(poly_A_1, poly_B_1),
    );

    // step branch -- BDDT: eval_2 = 2*claim - 3*eval_0 + 2*t_inf
    let three_eval0_s = eval0_s + eval0_s + eval0_s;
    let eval2_s = claim_step_round + claim_step_round - three_eval0_s + t_inf_s + t_inf_s;
    let evals_s = vec![eval0_s, claim_step_round - eval0_s, eval2_s];
    let poly_s = UniPoly::from_evals(&evals_s)?;
    let coeffs_step = [poly_s.coeffs[0], poly_s.coeffs[1], poly_s.coeffs[2]];

    // core branch -- BDDT: eval_2 = 2*claim - 3*eval_0 + 2*t_inf
    let three_eval0_c = eval0_c + eval0_c + eval0_c;
    let eval2_c = claim_core_round + claim_core_round - three_eval0_c + t_inf_c + t_inf_c;
    let evals_c = vec![eval0_c, claim_core_round - eval0_c, eval2_c];
    let poly_c = UniPoly::from_evals(&evals_c)?;
    let coeffs_core = [poly_c.coeffs[0], poly_c.coeffs[1], poly_c.coeffs[2]];

    verifier_circuit.inner_polys_step[j] = coeffs_step;
    verifier_circuit.inner_polys_core[j] = coeffs_core;

    // -------- transcript / witness handling --------
    let chals = SatisfyingAssignment::<E>::process_round(
      state,
      vc_shape,
      vc_ck,
      verifier_circuit,
      start_round + j,
      transcript,
    )?;
    let r_j = chals[0];
    r_y.push(r_j);

    // -------- bind polys --------
    rayon::join(
      || {
        rayon::join(
          || poly_A_0.bind_poly_var_top(&r_j),
          || poly_B_0.bind_poly_var_top(&r_j),
        );
      },
      || {
        rayon::join(
          || poly_A_1.bind_poly_var_top(&r_j),
          || poly_B_1.bind_poly_var_top(&r_j),
        );
      },
    );

    // -------- advance claim for next round --------
    claim_step_round = poly_s.evaluate(&r_j);
    claim_core_round = poly_c.evaluate(&r_j);
  }

  Ok((
    r_y,
    vec![poly_A_0[0], poly_A_1[0], poly_B_0[0], poly_B_1[0]],
  ))
}

/// Executes a **cubic-with-additive-term** batched outer sum-check in zero-knowledge mode
/// and returns the sequence of verifier challenges.
pub fn prove_cubic_with_additive_term_batched_zk<E: Engine>(
  num_rounds: usize,
  pow_tau_left: &mut MultilinearPolynomial<E::Scalar>,
  pow_tau_right: &MultilinearPolynomial<E::Scalar>,
  poly_A_step: &mut MultilinearPolynomial<E::Scalar>,
  poly_A_core: &mut MultilinearPolynomial<E::Scalar>,
  poly_B_step: &mut MultilinearPolynomial<E::Scalar>,
  poly_B_core: &mut MultilinearPolynomial<E::Scalar>,
  poly_C_step: &mut MultilinearPolynomial<E::Scalar>,
  poly_C_core: &mut MultilinearPolynomial<E::Scalar>,
  verifier_circuit: &mut NeutronNovaVerifierCircuit<E>,
  state: &mut MultiRoundState<E>,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
  start_round: usize,
) -> Result<Vec<E::Scalar>, SpartanError> {
  let mut base_tau = E::Scalar::ONE;
  let mut len_pow_tau = pow_tau_left.Z.len() * pow_tau_right.Z.len();

  let mut r_x: Vec<E::Scalar> = Vec::with_capacity(num_rounds);

  let mut claim_step = verifier_circuit.t_out_step;
  let mut claim_core = E::Scalar::ZERO;

  for i in 0..num_rounds {
    // step branch
    let ((mut eval0_s, mut eval2_s, mut eval3_s), (mut eval0_c, mut eval2_c, mut eval3_c)) =
      rayon::join(
        || {
          SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term_with_outer_pow(
            pow_tau_left,
            pow_tau_right,
            poly_A_step,
            poly_B_step,
            poly_C_step,
          )
        },
        || {
          SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term_with_outer_pow(
            pow_tau_left,
            pow_tau_right,
            poly_A_core,
            poly_B_core,
            poly_C_core,
          )
        },
      );

    eval0_s *= base_tau;
    eval2_s *= base_tau;
    eval3_s *= base_tau;
    eval0_c *= base_tau;
    eval2_c *= base_tau;
    eval3_c *= base_tau;

    let evals_s = vec![eval0_s, claim_step - eval0_s, eval2_s, eval3_s];
    let poly_s = UniPoly::from_evals(&evals_s)?;
    let coeffs_step = [
      poly_s.coeffs[0],
      poly_s.coeffs[1],
      poly_s.coeffs[2],
      poly_s.coeffs[3],
    ];

    let evals_c = vec![eval0_c, claim_core - eval0_c, eval2_c, eval3_c];
    let poly_c = UniPoly::from_evals(&evals_c)?;
    let coeffs_core = [
      poly_c.coeffs[0],
      poly_c.coeffs[1],
      poly_c.coeffs[2],
      poly_c.coeffs[3],
    ];

    verifier_circuit.outer_polys_step[i] = coeffs_step;
    verifier_circuit.outer_polys_core[i] = coeffs_core;

    // -------- transcript / witness handling --------
    let chals = SatisfyingAssignment::<E>::process_round(
      state,
      vc_shape,
      vc_ck,
      verifier_circuit,
      start_round + i,
      transcript,
    )?;
    let r_i = chals[0];
    r_x.push(r_i);

    // -------- advance claim and bind polys --------
    claim_step = poly_s.evaluate(&r_i);
    claim_core = poly_c.evaluate(&r_i);

    // bind polynomials to the verifier's challenge
    rayon::join(
      || {
        rayon::join(
          || poly_A_step.bind_poly_var_top(&r_i),
          || poly_A_core.bind_poly_var_top(&r_i),
        );
      },
      || {
        rayon::join(
          || {
            rayon::join(
              || poly_B_step.bind_poly_var_top(&r_i),
              || poly_B_core.bind_poly_var_top(&r_i),
            );
          },
          || {
            rayon::join(
              || poly_C_step.bind_poly_var_top(&r_i),
              || poly_C_core.bind_poly_var_top(&r_i),
            );
          },
        );
      },
    );

    // bind polynomial power of tau
    // list power of tau (pow_tau) halves effectively
    len_pow_tau >>= 1;
    let one = E::Scalar::ONE;
    let left = pow_tau_left.Z.len();
    let pow = pow_tau_left.Z[len_pow_tau % left] * pow_tau_right.Z[len_pow_tau / left];
    base_tau *= (pow - one) * r_i + one;
  }

  pow_tau_left.Z[0] = base_tau;

  Ok(r_x)
}
