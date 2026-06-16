// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Builder functions for constructing Lagrange accumulators (Procedure 9).
//!
//! This module provides:
//! - [`build_accumulators_spartan`]: Optimized builder for Spartan's cubic relation

#[cfg(test)]
use super::extension::{bit_rev_prefix_table, gather_and_extend_prefix};
use super::{
  accumulator::LagrangeAccumulators, csr::Csr, domain::LagrangeIndex,
  extension::extend_to_lagrange_domain, extension_bound::ExtensionBoundedPoly,
  index::AccumulatorPrefixIndex, thread_state::SpartanThreadState,
};
use crate::{
  big_num::{DelayedReduction, SmallValue, SmallValueEngine, WideMul},
  polys::{eq::build_eq_pyramid, eq::compute_suffix_eq_pyramid},
};
#[cfg(test)]
use crate::{errors::SpartanError, r1cs::weights_from_r};
use ff::PrimeField;
use num_traits::Zero;
use rayon::prelude::*;

use super::index::compute_idx4;

/// Polynomial degree D for small-value sumcheck accumulator tables.
/// For A·B relations, D=2 yields quadratic t_i.
pub(crate) const SMALL_VALUE_T_DEGREE: usize = 2;

/// Extension-bound certificate specialized to pairwise small-value products.
pub(crate) type SmallValueExtensionBoundedPoly<'a, SV, const LB: usize> =
  ExtensionBoundedPoly<'a, SV, <SV as WideMul>::Product, SMALL_VALUE_T_DEGREE, LB>;

/// Builds the table accumulators `A_i(v, u)` used in Spartan's first `l0`
/// outer sumcheck rounds.
///
/// For round `i = 1, ..., l0`, the table is indexed by:
/// - `v ∈ U_2^{i-1}`: the prefix fixed by earlier rounds
/// - `u ∈ Û_2 = {∞, 0}`: the current coordinate
///
/// Each entry stores the suffix-summed contribution
///
/// `A_i(v, u) = Σ_{y ∈ {0,1}^{l0-i}} Σ_{x ∈ {0,1}^{ℓ-l0}}
///     eq((τ_{i+1}, ..., τ_ℓ), (y, x)) · Az(v, u, y, x) · Bz(v, u, y, x)`.
///
/// Here:
/// - `y` is the remaining binary suffix inside the first `l0` variables
/// - `x` is the suffix after the first `l0` variables
/// - `Az(v, u, y, x)` and `Bz(v, u, y, x)` evaluate the first `l0` coordinates
///   on the degree-2 Lagrange domain `U_2 = {∞, 0, 1}`, while the remaining
///   coordinates stay on the Boolean hypercube
///
/// Intuitively, `A_i(v, u)` is a table bucket that collects every contribution
/// compatible with the prefix `v` and current coordinate `u`.
///
/// The table stores only `u ∈ Û_2`, not `u = 1`, because the `u = 1` value is
/// recovered later from the sumcheck relation.
///
/// This builder is Spartan-outer-specific using a table accumulator
/// It relies on the structure of Spartan's first sumcheck:
/// - on `{0,1}^n`, satisfying witnesses obey `Az(x) · Bz(x) - Cz(x) = 0`, so
///   purely binary β points can be skipped
/// - if a prefix coordinate is `∞`, only the highest-degree term contributes,
///   so the linear `Cz` term drops out of the accumulator
///
/// D is the degree bound of `t_i(X)` (not `s_i`); for Spartan, `D = 2`.
///
/// # Type Parameters
///
/// - `F`: Field type with small-value and delayed reduction support
/// - `SV`: Witness value type (like i32 or i64)
///
/// # Returns
///
/// A tuple of (accumulators, e_in_pyramid, e_xout_pyramid) where:
/// - accumulators: The computed table accumulators for the small-value rounds
/// - e_in_pyramid: Full eq pyramid for inner variables τ[l0..l0+in_vars], has in_vars+1 layers
/// - e_xout_pyramid: Full eq pyramid for outer variables τ[l0+in_vars..ℓ], has xout_vars+1 layers
///
/// The pyramids can be reused by EqSumCheckInstance for the remaining sumcheck rounds,
/// avoiding redundant eq polynomial computation.
///
/// # Parallelism strategy
///
/// - Outer parallel loop over x_out values (using Rayon fold-reduce)
/// - Each thread maintains thread-local accumulators
/// - Final reduction merges all thread-local results via element-wise addition
///
/// # Spartan-specific optimizations (D=2)
///
/// - Skip binary betas: for satisfying witnesses, Az·Bz = Cz on {0,1}^n, so Az·Bz - Cz = 0
/// - Only process betas containing ∞: these are exactly the points where the
///   highest-degree `Az·Bz` term can contribute after the `Cz` term drops out
pub(crate) fn build_accumulators_spartan<F, SV, const LB: usize>(
  az: &SmallValueExtensionBoundedPoly<'_, SV, LB>,
  bz: &SmallValueExtensionBoundedPoly<'_, SV, LB>,
  taus: &[F],
) -> (
  LagrangeAccumulators<F, SMALL_VALUE_T_DEGREE>,
  Vec<Vec<F>>,
  Vec<Vec<F>>,
)
where
  F: SmallValueEngine<SV>,
  SV: SmallValue,
{
  let l0 = LB;
  let az = az.as_poly();
  let bz = bz.as_poly();
  let base: usize = SMALL_VALUE_T_DEGREE + 1;
  let l = az.Z.len().trailing_zeros() as usize;
  debug_assert_eq!(az.Z.len(), 1usize << l, "poly size must be power of 2");
  debug_assert_eq!(az.Z.len(), bz.Z.len());
  debug_assert_eq!(taus.len(), l, "taus must have length ℓ");
  debug_assert!(l0 < l, "l0 must be < ℓ");

  let suffix_vars = l - l0;
  let prefix_size = 1usize << l0;

  // Precompute eq pyramids with balanced split
  let (eq_tables, in_vars, xout_vars) = precompute_eq_tables(taus, l0);
  let num_x_out = 1usize << xout_vars;

  // Get top layers (full eq tables) for accumulator computation
  let e_in = eq_tables
    .e_in_pyramid
    .last()
    .expect("e_in_pyramid non-empty");
  let e_xout = eq_tables
    .e_xout_pyramid
    .last()
    .expect("e_xout_pyramid non-empty");
  debug_assert_eq!(e_in.len(), 1 << in_vars);
  debug_assert_eq!(e_xout.len(), 1 << xout_vars);

  // Build beta → prefix index cache
  let BetaPrefixCache {
    cache: beta_prefix_cache,
    num_betas,
  } = build_beta_cache::<SMALL_VALUE_T_DEGREE>(l0);

  // Only betas containing at least one ∞ coordinate contribute non-zero values.
  // On binary inputs {0,1}^n, Az·Bz = Cz (R1CS identity), so Az·Bz - Cz = 0.
  let betas_with_infty: Vec<usize> = (0..num_betas)
    .filter(|&i| (0..l0).any(|d| (i / base.pow(d as u32)).is_multiple_of(base)))
    .collect();

  let ext_size = base.pow(l0 as u32); // (D+1)^l0

  // Build eq_cache: precomputes e_xout[x_out] * e_y[round][y] products.
  // Layout: eq_cache[round][x_out * num_y + y] for cache-friendly access.
  // Each parallel task (fixed x_out) accesses a contiguous block of size num_y.
  let eq_cache: Vec<Vec<F>> = eq_tables
    .e_y
    .iter()
    .map(|round_ey| {
      e_xout
        .iter()
        .flat_map(|ex| round_ey.iter().map(|ey| *ex * *ey))
        .collect()
    })
    .collect();

  // Precompute num_y per round for transposed access
  let num_y_per_round: Vec<usize> = eq_tables.e_y.iter().map(|ey| ey.len()).collect();

  // Parallel over x_out with thread-local state (zero per-iteration allocations)
  type State<F2, SV2> = SpartanThreadState<F2, SV2, SMALL_VALUE_T_DEGREE>;

  let fold_results: Vec<State<F, SV>> = (0..num_x_out)
    .into_par_iter()
    .fold(
      || State::<F, SV>::new(l0, num_betas, prefix_size, ext_size),
      |mut state: State<F, SV>, x_out_bits| {
        // Reset partial sums for this x_out iteration
        state.reset_partial_sums();

        // Inner loop over x_in - accumulate into UNREDUCED form
        for (x_in_bits, e_in_eval) in e_in.iter().enumerate() {
          let suffix = (x_in_bits << xout_vars) | x_out_bits;

          // Fill prefix buffers by index assignment (no allocation)
          #[allow(clippy::needless_range_loop)]
          for prefix in 0..prefix_size {
            let idx = (prefix << suffix_vars) | suffix;
            state.az_prefix_boolean_evals[prefix] = az.Z[idx];
            state.bz_prefix_boolean_evals[prefix] = bz.Z[idx];
          }

          // Extend Az and Bz to Lagrange domain in-place (zero allocation)
          let az_size = extend_to_lagrange_domain::<SV, SMALL_VALUE_T_DEGREE>(
            &state.az_prefix_boolean_evals,
            &mut state.az_extended_evals,
            &mut state.az_extended_scratch,
          );
          let az_ext = &state.az_extended_evals[..az_size];

          let bz_size = extend_to_lagrange_domain::<SV, SMALL_VALUE_T_DEGREE>(
            &state.bz_prefix_boolean_evals,
            &mut state.bz_extended_evals,
            &mut state.bz_extended_scratch,
          );
          let bz_ext = &state.bz_extended_evals[..bz_size];

          // Only process betas with ∞ - binary betas contribute 0 for satisfying witnesses
          // Uses delayed modular reduction: accumulates into unreduced wide-limb form.
          for &beta_idx in &betas_with_infty {
            let prod = SV::wide_mul(az_ext[beta_idx], bz_ext[beta_idx]);
            F::unreduced_multiply_accumulate(&mut state.partial_sums[beta_idx], e_in_eval, &prod);
          }
        }

        // Pre-compute and filter: reduce all non-zero betas upfront
        for &beta_idx in &betas_with_infty {
          if state.partial_sums[beta_idx].is_zero() {
            continue;
          }
          // Reduce partial sum to field element
          let val = <F as DelayedReduction<SV::Product>>::reduce(&state.partial_sums[beta_idx]);
          if val == F::ZERO {
            continue;
          }
          state.beta_values.push((beta_idx, val));
        }

        // Distribute beta values → A_i(v,u) via idx4 using precomputed eq_cache
        // Multiply-accumulate into wide accumulator (Montgomery REDC at end)
        for &(beta_idx, ref val) in &state.beta_values {
          for pref in &beta_prefix_cache[beta_idx] {
            // Transposed layout: eq_cache[round][x_out * num_y + y] for contiguous y access
            let num_y = num_y_per_round[pref.round_0 as usize];
            let eq_eval = eq_cache[pref.round_0 as usize][x_out_bits * num_y + pref.y_idx as usize];
            <F as DelayedReduction<F>>::unreduced_multiply_accumulate(
              &mut state.acc.rounds[pref.round_0 as usize].data_mut()[pref.v_idx as usize]
                [pref.u_idx as usize],
              val,
              &eq_eval,
            );
          }
        }

        state
      },
    )
    .collect();

  // Sequential merge: avoids parallel reduce tree overhead and identity allocations.
  let merged = fold_results
    .into_iter()
    .reduce(|mut a, b| {
      a.acc.merge(&b.acc);
      a
    })
    .expect("num_x_out > 0 guarantees non-empty fold results");

  // Finalize: reduce each bucket from wide 9-limb to field element
  let accumulators = merged
    .acc
    .map(|acc| <F as DelayedReduction<F>>::reduce(acc));

  // Return accumulators along with full pyramids for EqSumCheckInstance reuse
  (
    accumulators,
    eq_tables.e_in_pyramid,
    eq_tables.e_xout_pyramid,
  )
}

/// Builds the table accumulators used by NeutronNova's small-value NIFS.
///
/// Handles both modes uniformly:
/// - full-small (`l0 == ell_b`): every instance-folding bit lives in the
///   small-value prefix;
/// - partial-small (`0 < l0 < ell_b`): the first `l0` instance bits use the
///   accumulator path, while the suffix bits are summed with Boolean equality
///   weights from `rhos[l0..]`.
#[cfg(test)]
pub(crate) fn build_accumulators_neutronnova<'poly, F, SV, const LB: usize>(
  a_layers: &[SmallValueExtensionBoundedPoly<'poly, SV, LB>],
  b_layers: &[SmallValueExtensionBoundedPoly<'poly, SV, LB>],
  e_eq: &[F],
  left: usize,
  right: usize,
  rhos: &[F],
) -> Result<LagrangeAccumulators<F, SMALL_VALUE_T_DEGREE>, SpartanError>
where
  F: PrimeField + DelayedReduction<SV::Product> + DelayedReduction<F> + Send + Sync,
  SV: SmallValue,
{
  let l0 = LB;
  let n = a_layers.len();
  if n == 0 || !n.is_power_of_two() {
    return Err(invalid_input(format!(
      "build_accumulators_neutronnova requires a non-empty power-of-two layer count; got {}",
      n
    )));
  }
  let ell_b = n.trailing_zeros() as usize;

  if l0 == 0 || l0 > ell_b {
    return Err(invalid_input(format!(
      "build_accumulators_neutronnova requires 0 < l0 <= ell_b; got l0={} and ell_b={}",
      l0, ell_b
    )));
  }
  if rhos.len() != ell_b {
    return Err(invalid_input(format!(
      "rhos length {} does not match ell_b {}",
      rhos.len(),
      ell_b
    )));
  }
  if b_layers.len() != n {
    return Err(invalid_input(format!(
      "A/B layer counts do not match: A has {}, B has {}",
      n,
      b_layers.len()
    )));
  }
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
  let expected_layer_len = left
    .checked_mul(right)
    .ok_or_else(|| invalid_input("left * right overflows"))?;
  if expected_layer_len == 0 {
    return Err(invalid_input("left * right must be non-zero"));
  }
  let a_evals = layer_evals(a_layers);
  let b_evals = layer_evals(b_layers);
  if !a_evals
    .iter()
    .all(|layer| layer.len() == expected_layer_len)
  {
    return Err(invalid_input(format!(
      "all A layers must have length left * right ({})",
      expected_layer_len
    )));
  }
  if !b_evals
    .iter()
    .all(|layer| layer.len() == expected_layer_len)
  {
    return Err(invalid_input(format!(
      "all B layers must have length left * right ({})",
      expected_layer_len
    )));
  }

  // Partition the instance bits into the Lagrange prefix handled here and the
  // Boolean suffix folded by equality weights.
  let base: usize = SMALL_VALUE_T_DEGREE + 1;
  let l0_shift = u32::try_from(l0).map_err(|_| invalid_input("l0 does not fit in a u32 shift"))?;
  let prefix_size = 1usize
    .checked_shl(l0_shift)
    .ok_or_else(|| invalid_input("l0 is too large for this platform"))?;
  let suffix_shift = u32::try_from(ell_b - l0)
    .map_err(|_| invalid_input("ell_b - l0 does not fit in a u32 shift"))?;
  let suffix_groups = 1usize
    .checked_shl(suffix_shift)
    .ok_or_else(|| invalid_input("ell_b - l0 is too large for this platform"))?;
  let ext_size = base
    .checked_pow(l0 as u32)
    .ok_or_else(|| invalid_input("small-value extension table size overflows"))?;
  ext_size
    .checked_mul(l0)
    .ok_or_else(|| invalid_input("small-value beta cache size overflows"))?;

  // Equality data for suffix bits beyond LB. In full-small mode this collapses
  // to one suffix group with weight 1.
  let e_b = compute_suffix_eq_pyramid(rhos, l0);
  let suffix_weights = weights_from_r(&rhos[l0..], suffix_groups);

  let e_left = &e_eq[..left];
  let e_right = &e_eq[left..];
  let swap_loops = left > right;
  let outer_dim = if swap_loops { left } else { right };
  let (e_outer, e_inner) = if swap_loops {
    (e_left, e_right)
  } else {
    (e_right, e_left)
  };

  // Cache eq_outer * eq_suffix_round in round-major layout:
  // e_cache[round][x_outer * num_y + y].
  let e_cache: Vec<Vec<F>> = e_b
    .iter()
    .map(|round_ey| {
      e_outer
        .iter()
        .flat_map(|eo| round_ey.iter().map(|ey| *eo * *ey))
        .collect()
    })
    .collect();
  let num_y_per_round: Vec<usize> = e_b.iter().map(|ey| ey.len()).collect();

  // Map each Lagrange beta point to accumulator buckets. Only beta points with
  // an ∞ coordinate can contribute to the stored table.
  let bit_rev = bit_rev_prefix_table(l0);
  let BetaPrefixCache {
    cache: beta_prefix_cache,
    num_betas,
  } = build_beta_cache::<SMALL_VALUE_T_DEGREE>(l0);

  let betas_with_infty: Vec<usize> = (0..num_betas)
    .filter(|&i| (0..l0).any(|d| (i / base.pow(d as u32)).is_multiple_of(base)))
    .collect();

  type State<F2, SV2> = SpartanThreadState<F2, SV2, SMALL_VALUE_T_DEGREE>;
  let process_outer = |state: &mut State<F, SV>, x_outer: usize| {
    // One worker fixes x_outer, extends each LB prefix, and accumulates the
    // suffix-weighted A/B products in unreduced form.
    state.reset_partial_sums();

    for (x_inner, &e_inner_val) in e_inner.iter().enumerate() {
      let idx = if swap_loops {
        x_inner * left + x_outer
      } else {
        x_outer * left + x_inner
      };

      for (suffix_idx, &suffix_weight) in suffix_weights.iter().enumerate() {
        let layer_base = suffix_idx << l0;
        let az_size = gather_and_extend_prefix(
          &a_evals,
          &bit_rev,
          layer_base,
          idx,
          &mut state.az_prefix_boolean_evals,
          &mut state.az_extended_evals,
          &mut state.az_extended_scratch,
        );
        let az_ext = &state.az_extended_evals[..az_size];

        let bz_size = gather_and_extend_prefix(
          &b_evals,
          &bit_rev,
          layer_base,
          idx,
          &mut state.bz_prefix_boolean_evals,
          &mut state.bz_extended_evals,
          &mut state.bz_extended_scratch,
        );
        let bz_ext = &state.bz_extended_evals[..bz_size];
        let weighted_inner = e_inner_val * suffix_weight;

        for &beta_idx in &betas_with_infty {
          let prod = SV::wide_mul(az_ext[beta_idx], bz_ext[beta_idx]);
          F::unreduced_multiply_accumulate(
            &mut state.partial_sums[beta_idx],
            &weighted_inner,
            &prod,
          );
        }
      }
    }

    // Reduce each beta once, then route it into the affected round buckets.
    for &beta_idx in &betas_with_infty {
      let unreduced = &state.partial_sums[beta_idx];
      if unreduced.is_zero() {
        continue;
      }
      let val = <F as DelayedReduction<SV::Product>>::reduce(unreduced);
      if val != F::ZERO {
        state.beta_values.push((beta_idx, val));
      }
    }

    // Apply the cached equality weights while scattering into accumulator cells.
    for &(beta_idx, ref val) in &state.beta_values {
      for pref in &beta_prefix_cache[beta_idx] {
        let round = pref.round_0 as usize;
        let num_y = num_y_per_round[round];
        let e_val = e_cache[round][x_outer * num_y + pref.y_idx as usize];
        <F as DelayedReduction<F>>::unreduced_multiply_accumulate(
          &mut state.acc.rounds[round].data_mut()[pref.v_idx as usize][pref.u_idx as usize],
          val,
          &e_val,
        );
      }
    }
  };

  // Merge the thread-local accumulator buckets produced by the outer loop.
  let merged = (0..outer_dim)
    .into_par_iter()
    .fold(
      || State::<F, SV>::new(l0, num_betas, prefix_size, ext_size),
      |mut state: State<F, SV>, x_outer| {
        process_outer(&mut state, x_outer);
        state
      },
    )
    .reduce_with(|mut a, b| {
      a.acc.merge(&b.acc);
      a
    })
    .unwrap_or_else(|| State::<F, SV>::new(l0, num_betas, prefix_size, ext_size));

  // Convert merged wide accumulators back into field elements.
  Ok(
    merged
      .acc
      .map(|acc| <F as DelayedReduction<F>>::reduce(acc)),
  )
}

#[cfg(test)]
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

/// Precomputed eq polynomial pyramids with balanced split.
struct EqSplitTables<F: PrimeField> {
  /// Full pyramid for inner variables: eq(τ[l0..l0+in_vars], ·)
  /// Layer k has size 2^k, layer in_vars is the full table (size 2^in_vars)
  e_in_pyramid: Vec<Vec<F>>,
  /// Full pyramid for outer variables: eq(τ[l0+in_vars..], ·)
  /// Layer k has size 2^k, layer xout_vars is the full table (size 2^xout_vars)
  e_xout_pyramid: Vec<Vec<F>>,
  /// Suffix eq pyramid for prefix variables, Vec per round
  e_y: Vec<Vec<F>>,
}

/// Cached prefix indices for O(1) scatter access.
pub(crate) struct BetaPrefixCache {
  cache: Csr<AccumulatorPrefixIndex>,
  num_betas: usize,
}

/// Precompute eq polynomial pyramids with balanced split for e_in and e_xout.
///
/// Returns (tables, in_vars, xout_vars) where:
/// - in_vars = ceil((l - l0) / 2) - variables for inner loop (e_in)
/// - xout_vars = floor((l - l0) / 2) - variables for outer loop (e_xout)
///
/// The balanced split reduces precomputation cost by ~33% compared to the
/// asymmetric l/2 split, and enables odd number of rounds.
///
/// Both e_in and e_xout are returned as full pyramids (not just top layers),
/// enabling reuse by EqSumCheckInstance for the remaining sumcheck rounds.
fn precompute_eq_tables<F: PrimeField>(taus: &[F], l0: usize) -> (EqSplitTables<F>, usize, usize) {
  let l = taus.len();
  let suffix_vars = l - l0;
  let in_vars = suffix_vars.div_ceil(2); // ceiling: e_in larger (inner loop, sequential access)
  let xout_vars = suffix_vars - in_vars; // floor: e_xout smaller (outer loop, reused)

  // Build full pyramids (not just top layers) for reuse by EqSumCheckInstance
  let e_in_pyramid = build_eq_pyramid(&taus[l0..l0 + in_vars]); // in_vars+1 layers
  let e_xout_pyramid = build_eq_pyramid(&taus[l0 + in_vars..]); // xout_vars+1 layers
  let e_y = compute_suffix_eq_pyramid(&taus[..l0], l0); // Vec per round, total 2^l0 - 1

  (
    EqSplitTables {
      e_in_pyramid,
      e_xout_pyramid,
      e_y,
    },
    in_vars,
    xout_vars,
  )
}

/// Build beta → prefix index cache for O(1) scatter access.
pub(crate) fn build_beta_cache<const D: usize>(l0: usize) -> BetaPrefixCache {
  let base: usize = D + 1;
  let num_betas = base.pow(l0 as u32);
  let mut cache: Csr<AccumulatorPrefixIndex> = Csr::with_capacity(num_betas, num_betas * l0);
  for b in 0..num_betas {
    let beta = LagrangeIndex::<D>::from_flat_index(b, l0);
    let entries = compute_idx4(&beta);
    cache.push(&entries);
  }

  BetaPrefixCache { cache, num_betas }
}

#[cfg(test)]
fn invalid_input(reason: impl Into<String>) -> SpartanError {
  SpartanError::InvalidInputLength {
    reason: reason.into(),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    lagrange_accumulator::domain::LagrangeHatPoint, polys::multilinear::MultilinearPolynomial,
    provider::pasta::pallas,
  };
  use ff::Field;

  type Scalar = pallas::Scalar;

  // Use the shared constant for polynomial degree in tests
  const D: usize = SMALL_VALUE_T_DEGREE;

  /// Binary-β zero shortcut: Az=Bz=Cz=first variable (x0), so Az·Bz−Cz=0 on binary β.
  /// Non-binary β (∞) should yield non-zero in some bucket.
  #[test]
  fn test_binary_beta_zero_shortcut_behavior() {
    // Use l0=1 so round 0 buckets are fed only by β of length 1 (easy to reason about).
    const L0: usize = 1;
    let l = 2;

    // Az = Bz = top bit x0 (most significant of 2 bits)
    // For satisfying witness, Cz = Az * Bz = Az (since Az ∈ {0,1} and Az = Bz)
    let az_vals: Vec<i32> = (0..(1 << l)).map(|bits| (bits >> (l - 1)) & 1).collect();
    let bz_vals: Vec<i32> = (0..(1 << l)).map(|bits| (bits >> (l - 1)) & 1).collect();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);

    let taus: Vec<Scalar> = vec![Scalar::from(3u64), Scalar::from(5u64)];

    let az =
      SmallValueExtensionBoundedPoly::<_, L0>::new(&az).expect("Az should be extension-bounded");
    let bz =
      SmallValueExtensionBoundedPoly::<_, L0>::new(&bz).expect("Bz should be extension-bounded");

    let (acc, _, _) = build_accumulators_spartan(&az, &bz, &taus);

    // Only round 0 exists (v is empty). β ranges over U_d with binary {0,1} and non-binary {∞}.
    // Buckets for u = 0 should be zero (binary β), bucket for u = ∞ should be non-zero.
    let u_inf = LagrangeHatPoint::<D>::Infinity.to_index(); // 0
    let u_zero = LagrangeHatPoint::<D>::Finite(0).to_index(); // 1

    assert!(
      bool::from(acc.get(0, 0, u_zero).is_zero()),
      "binary β should give zero for u=0"
    );
    assert!(
      !bool::from(acc.get(0, 0, u_inf).is_zero()),
      "non-binary β (∞) should give non-zero"
    );
  }

  /// Test build_accumulators_spartan with i32 witnesses produces consistent results.
  ///
  /// Verifies that running the same computation twice produces the same output.
  #[test]
  fn test_build_accumulators_spartan_small_consistent() {
    const L0: usize = 2;
    let l0 = L0;

    // Define deterministic Az, Bz over {0,1}^4 using small values
    let eval = |bits: usize| -> i32 {
      let x0 = (bits >> 3) & 1;
      let x1 = (bits >> 2) & 1;
      let x2 = (bits >> 1) & 1;
      let x3 = bits & 1;
      (x0 + 2 * x1 + 3 * x2 + 4 * x3 + 5) as i32
    };

    let az_vals: Vec<i32> = (0..16).map(&eval).collect();
    let bz_vals: Vec<i32> = (0..16).map(|b| eval(b) + 7).collect();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);

    // Taus (length ℓ)
    let taus: Vec<Scalar> = vec![
      Scalar::from(5u64),
      Scalar::from(7u64),
      Scalar::from(11u64),
      Scalar::from(13u64),
    ];

    let az =
      SmallValueExtensionBoundedPoly::<_, L0>::new(&az).expect("Az should be extension-bounded");
    let bz =
      SmallValueExtensionBoundedPoly::<_, L0>::new(&bz).expect("Bz should be extension-bounded");

    // Build accumulators twice
    let (acc1, _, _) = build_accumulators_spartan(&az, &bz, &taus);
    let (acc2, _, _) = build_accumulators_spartan(&az, &bz, &taus);

    // Compare all buckets
    for round in 0..l0 {
      let num_v = (D + 1).pow(round as u32);
      for v_idx in 0..num_v {
        for u_idx in 0..D {
          let got = acc1.get(round, v_idx, u_idx);
          let expect = acc2.get(round, v_idx, u_idx);
          assert_eq!(
            got, expect,
            "Mismatch at round {}, v_idx {}, u_idx {}",
            round, v_idx, u_idx
          );
        }
      }
    }
  }

  /// Test build_accumulators_spartan with i32 witnesses using larger inputs to stress test.
  #[test]
  fn test_build_accumulators_spartan_small_larger() {
    const L0: usize = 3;
    let l0 = L0;
    let l = 10;
    let n = 1 << l;

    // Create polynomials with varying small values
    let az_vals: Vec<i32> = (0..n).map(|i| (i % 1000) + 1).collect();
    let bz_vals: Vec<i32> = (0..n).map(|i| ((i * 7) % 1000) + 1).collect();

    let az = MultilinearPolynomial::new(az_vals);
    let bz = MultilinearPolynomial::new(bz_vals);

    // Random-looking taus
    let taus: Vec<Scalar> = (0..l).map(|i| Scalar::from((i * 7 + 3) as u64)).collect();

    let az =
      SmallValueExtensionBoundedPoly::<_, L0>::new(&az).expect("Az should be extension-bounded");
    let bz =
      SmallValueExtensionBoundedPoly::<_, L0>::new(&bz).expect("Bz should be extension-bounded");

    // Build accumulators twice to verify consistency
    let (acc1, _, _) = build_accumulators_spartan(&az, &bz, &taus);
    let (acc2, _, _) = build_accumulators_spartan(&az, &bz, &taus);

    for round in 0..l0 {
      let num_v = (D + 1).pow(round as u32);
      for v_idx in 0..num_v {
        for u_idx in 0..D {
          let got = acc1.get(round, v_idx, u_idx);
          let expect = acc2.get(round, v_idx, u_idx);
          assert_eq!(
            got, expect,
            "Mismatch at round {}, v_idx {}, u_idx {}",
            round, v_idx, u_idx
          );
        }
      }
    }
  }

  #[test]
  fn test_neutronnova_small_layer_certificate_rejects_out_of_bound_a() {
    const L0: usize = 1;
    let poly = MultilinearPolynomial::new(vec![i32::MAX, 0, 0, 0]);

    assert!(matches!(
      SmallValueExtensionBoundedPoly::<_, L0>::new(&poly),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }

  #[test]
  fn test_neutronnova_small_layer_certificate_rejects_out_of_bound_b() {
    const L0: usize = 1;
    let poly = MultilinearPolynomial::new(vec![0, 0, i32::MAX, 0]);

    assert!(matches!(
      SmallValueExtensionBoundedPoly::<_, L0>::new(&poly),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }
}
