// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Zero-knowledge sum-check drivers for Vega Spartan.

use crate::zk::SpartanVerifierCircuit;
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
  sumcheck::{SumcheckProof, eq_sumcheck},
  traits::Engine,
};

/// Executes the **outer** cubic-with-additive-term sum-check in
/// Zero-knowledge outer sum-check for the cubic-with-additive-term case.
pub fn prove_cubic_with_additive_term_zk<E: Engine>(
  num_rounds: usize,
  taus: &[E::Scalar],
  poly_Az: &mut MultilinearPolynomial<E::Scalar>,
  poly_Bz: &mut MultilinearPolynomial<E::Scalar>,
  poly_Cz: &mut MultilinearPolynomial<E::Scalar>,
  verifier_circuit: &mut SpartanVerifierCircuit<E>,
  state: &mut MultiRoundState<E>,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
) -> Result<Vec<E::Scalar>, SpartanError> {
  let mut r_x: Vec<E::Scalar> = Vec::with_capacity(num_rounds);
  let mut claim_outer_round = E::Scalar::ZERO;
  let mut eq_instance = eq_sumcheck::EqSumCheckInstance::<E>::new(taus.to_vec());

  for i in 0..num_rounds {
    // -------- interpolate coefficients --------
    let (eval0, eval2, eval3) = if i == 0 {
      // Zero-check round 0: t(0) = 0 (binary points -> R1CS satisfied), no Cz needed.
      eq_instance.evaluation_points_zero_check_round0(poly_Az, poly_Bz)
    } else {
      eq_instance.evaluation_points_cubic_with_three_inputs(
        poly_Az,
        poly_Bz,
        poly_Cz,
        claim_outer_round,
      )
    };
    let evals = vec![eval0, claim_outer_round - eval0, eval2, eval3];
    let poly = UniPoly::from_evals(&evals)?;
    verifier_circuit.outer_polys[i] = [
      poly.coeffs[0],
      poly.coeffs[1],
      poly.coeffs[2],
      poly.coeffs[3],
    ];

    // -------- transcript / witness handling --------
    let chals = SatisfyingAssignment::<E>::process_round(
      state,
      vc_shape,
      vc_ck,
      verifier_circuit,
      i,
      transcript,
    )?;
    r_x.push(chals[0]);

    // -------- advance claim and bind polys --------
    claim_outer_round = poly.evaluate(&chals[0]);

    rayon::join(
      || poly_Az.bind_poly_var_top(&chals[0]),
      || {
        rayon::join(
          || poly_Bz.bind_poly_var_top(&chals[0]),
          || poly_Cz.bind_poly_var_top(&chals[0]),
        );
      },
    );
    eq_instance.bound(&chals[0]);
  }

  Ok(r_x)
}

/// Executes a **quadratic** sum-check in zero-knowledge mode and returns the
/// Zero-knowledge quadratic sum-check used for the inner round.
///
/// Uses delayed modular reduction for improved performance.
pub fn prove_quad_zk<E: Engine>(
  claim: &E::Scalar,
  num_rounds: usize,
  poly_ABC: &mut MultilinearPolynomial<E::Scalar>,
  poly_z: &mut MultilinearPolynomial<E::Scalar>,
  verifier_circuit: &mut SpartanVerifierCircuit<E>,
  state: &mut MultiRoundState<E>,
  vc_shape: &SplitMultiRoundR1CSShape<E>,
  vc_ck: &CommitmentKey<E>,
  transcript: &mut E::TE,
  start_round: usize,
  inner_poly_offset: usize,
) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
  let mut r_y: Vec<E::Scalar> = Vec::with_capacity(num_rounds);
  let mut claim_current_round = *claim;

  for j in 0..num_rounds {
    // -------- interpolate coeffs --------
    let (eval0, t_inf) = SumcheckProof::<E>::compute_eval_points_quad(poly_ABC, poly_z);
    // BDDT: eval_2 = 2*claim - 3*eval_0 + 2*t_inf
    let three_eval0 = eval0 + eval0 + eval0;
    let eval2 = claim_current_round + claim_current_round - three_eval0 + t_inf + t_inf;
    let evals = vec![eval0, claim_current_round - eval0, eval2];
    let poly = UniPoly::from_evals(&evals)?;

    verifier_circuit.inner_polys[inner_poly_offset + j] =
      [poly.coeffs[0], poly.coeffs[1], poly.coeffs[2]];

    // -------- transcript / witness handling --------
    let chals = SatisfyingAssignment::<E>::process_round(
      state,
      vc_shape,
      vc_ck,
      verifier_circuit,
      start_round + j,
      transcript,
    )?;
    r_y.push(chals[0]);

    // -------- bind polys --------
    rayon::join(
      || poly_ABC.bind_poly_var_top(&chals[0]),
      || poly_z.bind_poly_var_top(&chals[0]),
    );

    // -------- advance claim for next round --------
    claim_current_round = poly.evaluate(&chals[0]);
  }

  Ok((r_y, vec![poly_ABC[0], poly_z[0]]))
}
