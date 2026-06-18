// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Thread-local scratch buffers for accumulator building.
//!
//! These structs eliminate per-iteration heap allocations in the parallel fold loops
//! of `build_accumulators_spartan`. By hoisting buffer allocations to the fold identity
//! closure (called once per Rayon thread subdivision), we reduce allocations from
//! O(num_x_out) to O(num_threads).

use super::accumulator::LagrangeAccumulators;
use super::extension_bound::SmallValue;
use crate::big_num::DelayedReduction;
use num_traits::Zero;

/// Reusable A/B prefix and Lagrange-extension buffers.
pub(crate) struct PrefixExtensionScratch<T> {
  /// Prefix evaluations of A for current suffix. Size: 2^l0.
  pub a_prefix: Vec<T>,
  /// Prefix evaluations of B for current suffix. Size: 2^l0.
  pub b_prefix: Vec<T>,
  /// Result buffer for A Lagrange extension. Size: (D+1)^l0.
  pub a_ext: Vec<T>,
  /// Scratch buffer for A Lagrange extension.
  pub a_scratch: Vec<T>,
  /// Result buffer for B Lagrange extension. Size: (D+1)^l0.
  pub b_ext: Vec<T>,
  /// Scratch buffer for B Lagrange extension.
  pub b_scratch: Vec<T>,
}

impl<T> PrefixExtensionScratch<T>
where
  T: Copy + Default,
{
  pub(crate) fn new(prefix_size: usize, ext_size: usize) -> Self {
    Self {
      a_prefix: vec![T::default(); prefix_size],
      b_prefix: vec![T::default(); prefix_size],
      a_ext: vec![T::default(); ext_size],
      a_scratch: vec![T::default(); ext_size],
      b_ext: vec![T::default(); ext_size],
      b_scratch: vec![T::default(); ext_size],
    }
  }
}

/// Thread-local scratch buffers for `build_accumulators_spartan`.
///
/// # Motivation
///
/// Without this optimization, the fold closure allocates vectors on every x_out iteration.
/// By hoisting these buffers into a struct created once per Rayon thread subdivision
/// (in the fold identity closure), we reduce allocations from O(num_x_out) to O(num_threads).
///
/// # Buffer Layout
///
/// - `partial_sums`: Accumulates products over x_in loop (unreduced SignedWideLimbs)
/// - `acc`: Bucket accumulators for scatter phase (unreduced WideLimbs<9> for F×F)
/// - `az_*/bz_*`: Extension buffers for small-value polynomials
///
/// # Type Parameters
///
/// - `F`: Field type with small-value and delayed reduction support
/// - `SV`: Witness value type (i32 or i64 for small-value witnesses)
/// - `D`: Polynomial degree bound
pub(crate) struct SpartanThreadState<F, SV, const D: usize>
where
  F: Copy + DelayedReduction<SV::Product> + DelayedReduction<F>,
  SV: SmallValue,
{
  /// Partial sums indexed by β, accumulated over the x_in loop.
  /// Uses unreduced wide-limb form for delayed modular reduction.
  /// Reset each x_out iteration.
  pub partial_sums: Vec<<F as DelayedReduction<SV::Product>>::Accumulator>,
  /// Bucket accumulators for scatter phase.
  /// Uses unreduced F×F form (accumulator for field × field products).
  /// Accumulated across all x_out iterations, then merged.
  pub acc: LagrangeAccumulators<<F as DelayedReduction<F>>::Accumulator, D>,
  /// Reusable prefix and extension buffers for A/B.
  pub ext: PrefixExtensionScratch<SV>,
  /// Reusable buffer for filtered (beta_idx, reduced_value) pairs.
  /// Eliminates per-x_out allocation overhead.
  pub beta_values: Vec<(usize, F)>,
}

impl<F, SV, const D: usize> SpartanThreadState<F, SV, D>
where
  F: Copy + DelayedReduction<SV::Product> + DelayedReduction<F>,
  SV: SmallValue,
{
  pub fn new(l0: usize, num_betas: usize, prefix_size: usize, ext_size: usize) -> Self {
    Self {
      partial_sums: vec![<F as DelayedReduction<SV::Product>>::Accumulator::zero(); num_betas],
      acc: LagrangeAccumulators::new(l0),
      ext: PrefixExtensionScratch::new(prefix_size, ext_size),
      beta_values: Vec::with_capacity(num_betas),
    }
  }

  /// Zero out partial sums and beta_values for the next x_out iteration.
  /// This is O(num_betas) but much cheaper than reallocating.
  #[inline]
  pub fn reset_partial_sums(&mut self) {
    for sum in &mut self.partial_sums {
      *sum = <F as DelayedReduction<SV::Product>>::Accumulator::zero();
    }
    self.beta_values.clear();
  }
}
