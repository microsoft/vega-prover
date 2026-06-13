// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Prove-local workspace data structures for the small-value accumulator
//! path: transposed `SmallAbc` tables, the partial-`l0` `PrefixWorkspace`,
//! the full-batch `ExtendedPrefixMleEvals` cache, and the prefix/suffix fold
//! helpers that operate on them.

#[allow(clippy::wildcard_imports)]
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct SmallAbc<SV> {
  pub(super) l0: usize,
  pub(super) num_instances: usize,
  pub(super) num_constraints: usize,
  pub(super) a: Vec<SV>,
  pub(super) b: Vec<SV>,
  pub(super) c: Vec<SV>,
}

impl<SV> SmallAbc<SV> {
  pub(super) fn row<'a>(&'a self, table: &'a [SV], idx: usize) -> &'a [SV] {
    let start = idx * self.num_constraints;
    &table[start..start + self.num_constraints]
  }

  pub(super) fn padded_row_idx(&self, idx: usize) -> Result<usize, SpartanError> {
    if self.num_instances == 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: "accumulator cache has no step-instance rows".into(),
      });
    }
    Ok(if idx < self.num_instances { idx } else { 0 })
  }

  pub(super) fn a_row(&self, idx: usize) -> &[SV] {
    self.row(&self.a, idx)
  }

  pub(super) fn b_row(&self, idx: usize) -> &[SV] {
    self.row(&self.b, idx)
  }

  pub(super) fn c_row(&self, idx: usize) -> &[SV] {
    self.row(&self.c, idx)
  }
}

/// Prove-local transposed view of `SmallAbc` for a partial-`l0` proof.
///
/// Each table uses the layout:
/// `suffix_group -> constraint -> prefix_values[0..2^l0]`.
///
/// This keeps prefix slices contiguous for both accumulator construction and
/// prefix folding without moving transcript-independent partial-`l0` work into
/// prep.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
  serialize = "SV: Serialize, <SV as WideMul>::Output: Serialize",
  deserialize = "SV: Deserialize<'de>, <SV as WideMul>::Output: Deserialize<'de>"
))]
pub(super) struct PrefixWorkspace<SV: WideMul> {
  pub(super) l0: usize,
  pub(super) prefix_size: usize,
  pub(super) ext_size: usize,
  pub(super) suffix_groups: usize,
  pub(super) num_constraints: usize,
  pub(super) beta_indices: Vec<usize>,
  pub(super) a: Vec<SV>,
  pub(super) b: Vec<SV>,
  pub(super) c: Vec<SV>,
  pub(super) ab_ext: Vec<<SV as WideMul>::Output>,
}

impl<SV> std::fmt::Debug for PrefixWorkspace<SV>
where
  SV: WideMul + std::fmt::Debug,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("PrefixWorkspace")
      .field("l0", &self.l0)
      .field("prefix_size", &self.prefix_size)
      .field("ext_size", &self.ext_size)
      .field("suffix_groups", &self.suffix_groups)
      .field("num_constraints", &self.num_constraints)
      .field("beta_indices_len", &self.beta_indices.len())
      .field("a_len", &self.a.len())
      .field("b_len", &self.b.len())
      .field("c_len", &self.c.len())
      .field("ab_ext_len", &self.ab_ext.len())
      .finish()
  }
}

impl<SV> PrefixWorkspace<SV>
where
  SV: SmallValue,
  <SV as WideMul>::Output: Copy + Zero + Send + Sync,
{
  pub(super) fn build(
    mle_inputs: &SmallAbc<SV>,
    l0: usize,
    n_padded: usize,
  ) -> Result<Self, SpartanError> {
    if mle_inputs.num_instances == 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: "cannot build prefix workspace for empty step batch".into(),
      });
    }
    let prefix_size = 1usize << l0;
    if n_padded % prefix_size != 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "prefix workspace instance count {} is not divisible by prefix size {}",
          n_padded, prefix_size
        ),
      });
    }

    let suffix_groups = n_padded / prefix_size;
    let num_constraints = mle_inputs.num_constraints;
    let table_len = suffix_groups * num_constraints * prefix_size;
    let mut a = vec![SV::default(); table_len];
    let mut b = vec![SV::default(); table_len];
    let mut c = vec![SV::default(); table_len];

    rayon::join(
      || {
        Self::transpose_table(
          mle_inputs,
          &mle_inputs.a,
          &mut a,
          prefix_size,
          suffix_groups,
        )
      },
      || {
        rayon::join(
          || {
            Self::transpose_table(
              mle_inputs,
              &mle_inputs.b,
              &mut b,
              prefix_size,
              suffix_groups,
            )
          },
          || {
            Self::transpose_table(
              mle_inputs,
              &mle_inputs.c,
              &mut c,
              prefix_size,
              suffix_groups,
            )
          },
        )
      },
    );

    let ext_size = 3usize.pow(l0 as u32);
    let beta_indices = beta_indices_with_infty(l0);
    let ext_table_len = suffix_groups * num_constraints * beta_indices.len();
    let mut ab_ext = vec![<SV as WideMul>::Output::zero(); ext_table_len];
    Self::extend_prefix_product_table(&a, &b, &mut ab_ext, prefix_size, ext_size, &beta_indices);

    Ok(Self {
      l0,
      prefix_size,
      ext_size,
      suffix_groups,
      num_constraints,
      beta_indices,
      a,
      b,
      c,
      ab_ext,
    })
  }

  pub(super) fn transpose_table(
    mle_inputs: &SmallAbc<SV>,
    source: &[SV],
    dest: &mut [SV],
    prefix_size: usize,
    suffix_groups: usize,
  ) {
    let num_constraints = mle_inputs.num_constraints;
    dest
      .par_chunks_mut(num_constraints * prefix_size)
      .take(suffix_groups)
      .enumerate()
      .for_each(|(suffix_idx, suffix_chunk)| {
        for prefix_idx in 0..prefix_size {
          let layer_idx = suffix_idx * prefix_size + prefix_idx;
          let row_idx = if layer_idx < mle_inputs.num_instances {
            layer_idx
          } else {
            0
          };
          let row_start = row_idx * num_constraints;
          let row = &source[row_start..row_start + num_constraints];
          for (constraint_idx, &value) in row.iter().enumerate() {
            suffix_chunk[constraint_idx * prefix_size + prefix_idx] = value;
          }
        }
      });
  }

  pub(super) fn extend_prefix_product_table(
    a_source: &[SV],
    b_source: &[SV],
    dest: &mut [<SV as WideMul>::Output],
    prefix_size: usize,
    ext_size: usize,
    beta_indices: &[usize],
  ) {
    let bit_rev = bit_rev_prefix_table(prefix_size.log_2());
    dest
      .par_chunks_mut(beta_indices.len())
      .zip(a_source.par_chunks(prefix_size))
      .zip(b_source.par_chunks(prefix_size))
      .for_each_init(
        || {
          (
            vec![SV::default(); prefix_size],
            vec![SV::default(); prefix_size],
            vec![SV::default(); ext_size],
            vec![SV::default(); ext_size],
            vec![SV::default(); ext_size],
            vec![SV::default(); ext_size],
          )
        },
        |(a_prefix, b_prefix, a_ext, a_scratch, b_ext, b_scratch),
         ((dest_chunk, a_source_chunk), b_source_chunk)| {
          for (p, &rev) in bit_rev.iter().enumerate() {
            a_prefix[p] = a_source_chunk[rev];
            b_prefix[p] = b_source_chunk[rev];
          }
          let a_produced = extend_to_lagrange_domain::<SV, 2>(a_prefix, a_ext, a_scratch);
          let b_produced = extend_to_lagrange_domain::<SV, 2>(b_prefix, b_ext, b_scratch);
          debug_assert_eq!(a_produced, ext_size);
          debug_assert_eq!(b_produced, ext_size);
          for (slot, &beta_idx) in beta_indices.iter().enumerate() {
            dest_chunk[slot] = a_ext[beta_idx].wide_mul(b_ext[beta_idx]);
          }
        },
      );
  }
}

pub(super) fn beta_indices_with_infty(l0: usize) -> Vec<usize> {
  let base = 3usize;
  let ext_size = base.pow(l0 as u32);
  (0..ext_size)
    .filter(|&idx| (0..l0).any(|d| (idx / base.pow(d as u32)) % base == 0))
    .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fold_suffix_round_chunk<E: Engine>(
  dims: (usize, usize),
  e_eq: &[E::Scalar],
  round: usize,
  ell_b: usize,
  pair_idx: usize,
  rhos: &[E::Scalar],
  prev_r_b: E::Scalar,
  a_chunk: &mut [Vec<E::Scalar>],
  b_chunk: &mut [Vec<E::Scalar>],
  c_chunk: &mut [E::Scalar],
  c_layer_chunk: Option<&mut [Vec<E::Scalar>]>,
) -> (E::Scalar, E::Scalar)
where
  E::PCS: FoldingEngineTrait<E>,
{
  for chunk in [&mut *a_chunk, &mut *b_chunk] {
    super::super::fold_quad_chunk(chunk, &prev_r_b);
  }

  if let Some(c_layer_chunk) = c_layer_chunk {
    super::super::fold_quad_chunk(c_layer_chunk, &prev_r_b);
  }

  {
    let c0 = c_chunk[0];
    c_chunk[0] += prev_r_b * (c_chunk[1] - c0);
    let c2 = c_chunk[2];
    c_chunk[2] += prev_r_b * (c_chunk[3] - c2);
  }

  let (e0_ab, qc) = NeutronNovaNIFS::<E>::compute_tensor_eq_ab_fold_extension_terms(
    dims,
    e_eq,
    &a_chunk[0],
    &b_chunk[0],
    &a_chunk[2],
    &b_chunk[2],
  );
  let e0 = e0_ab - c_chunk[0];
  let w = suffix_weight_full::<E::Scalar>(round, ell_b, pair_idx, rhos);
  (e0 * w, qc * w)
}

/// `Az` and `Bz` evaluations extended from `{0,1}^l0` to `U_2^l0`.
///
/// The vectors are constraint-major, with each constraint owning one contiguous
/// slice of length `3^l0`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ExtendedPrefixMleEvals<SV> {
  pub(super) num_constraints: usize,
  pub(super) domain_size: usize,
  pub(super) a: Vec<SV>,
  pub(super) b: Vec<SV>,
}

pub(super) fn build_extended_prefix_mle_evals<SV>(
  mle_inputs: &SmallAbc<SV>,
  l0: usize,
) -> Result<ExtendedPrefixMleEvals<SV>, SpartanError>
where
  SV: SmallValue,
{
  let prefix_size = 1usize << l0;
  let ext_size = 3usize.pow(l0 as u32);
  let num_constraints = mle_inputs.num_constraints;
  if mle_inputs.num_instances == 0 {
    return Err(SpartanError::InvalidInputLength {
      reason: "cannot precompute full-batch extension cache for empty step batch".into(),
    });
  }

  let mut a_layers: Vec<&[SV]> = (0..mle_inputs.num_instances)
    .map(|idx| mle_inputs.a_row(idx))
    .collect();
  let mut b_layers: Vec<&[SV]> = (0..mle_inputs.num_instances)
    .map(|idx| mle_inputs.b_row(idx))
    .collect();
  if a_layers.len() < prefix_size {
    let first_a = *a_layers.first().ok_or(SpartanError::InvalidInputLength {
      reason: "cannot pad empty full-batch A cache".into(),
    })?;
    let first_b = *b_layers.first().ok_or(SpartanError::InvalidInputLength {
      reason: "cannot pad empty full-batch B cache".into(),
    })?;
    a_layers.resize(prefix_size, first_a);
    b_layers.resize(prefix_size, first_b);
  }

  let bit_rev = bit_rev_prefix_table(l0);

  let mut a_ext = vec![SV::default(); num_constraints * ext_size];
  let mut b_ext = vec![SV::default(); num_constraints * ext_size];
  if rayon::current_num_threads() <= 1 {
    let mut a_prefix = vec![SV::default(); prefix_size];
    let mut b_prefix = vec![SV::default(); prefix_size];
    let mut a_buf = vec![SV::default(); ext_size];
    let mut a_scratch = vec![SV::default(); ext_size];
    let mut b_buf = vec![SV::default(); ext_size];
    let mut b_scratch = vec![SV::default(); ext_size];
    for idx in 0..num_constraints {
      let a_size = gather_and_extend_prefix(
        &a_layers,
        &bit_rev,
        0,
        idx,
        &mut a_prefix,
        &mut a_buf,
        &mut a_scratch,
      );
      let b_size = gather_and_extend_prefix(
        &b_layers,
        &bit_rev,
        0,
        idx,
        &mut b_prefix,
        &mut b_buf,
        &mut b_scratch,
      );
      debug_assert_eq!(a_size, ext_size);
      debug_assert_eq!(b_size, ext_size);
      let start = idx * ext_size;
      let end = start + ext_size;
      a_ext[start..end].copy_from_slice(&a_buf[..a_size]);
      b_ext[start..end].copy_from_slice(&b_buf[..b_size]);
    }
  } else {
    a_ext
      .par_chunks_mut(ext_size)
      .zip(b_ext.par_chunks_mut(ext_size))
      .enumerate()
      .for_each_init(
        || {
          (
            vec![SV::default(); prefix_size],
            vec![SV::default(); prefix_size],
            vec![SV::default(); ext_size],
            vec![SV::default(); ext_size],
            vec![SV::default(); ext_size],
            vec![SV::default(); ext_size],
          )
        },
        |(a_prefix, b_prefix, a_buf, a_scratch, b_buf, b_scratch), (idx, (a_chunk, b_chunk))| {
          let a_size =
            gather_and_extend_prefix(&a_layers, &bit_rev, 0, idx, a_prefix, a_buf, a_scratch);
          let b_size =
            gather_and_extend_prefix(&b_layers, &bit_rev, 0, idx, b_prefix, b_buf, b_scratch);
          debug_assert_eq!(a_size, ext_size);
          debug_assert_eq!(b_size, ext_size);
          a_chunk.copy_from_slice(&a_buf[..a_size]);
          b_chunk.copy_from_slice(&b_buf[..b_size]);
        },
      );
  }

  Ok(ExtendedPrefixMleEvals {
    num_constraints,
    domain_size: ext_size,
    a: a_ext,
    b: b_ext,
  })
}

pub(crate) fn fold_small_value_vectors<F, SV, V>(weights: &[F], vectors: &[V]) -> Vec<F>
where
  F: Field + DelayedReduction<SV>,
  V: AsRef<[SV]> + Sync,
  SV: Send + Sync,
{
  let dim = vectors[0].as_ref().len();
  (0..dim)
    .into_par_iter()
    .map(|j| {
      let mut acc = <F as DelayedReduction<SV>>::Accumulator::zero();
      for (wi, vector) in weights.iter().zip(vectors.iter()) {
        let vector = vector.as_ref();
        <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc, wi, &vector[j]);
      }
      <F as DelayedReduction<SV>>::reduce(&acc)
    })
    .collect()
}

pub(super) fn multilinear_with_effective_halves<F>(values: Vec<F>) -> MultilinearPolynomial<F>
where
  F: Field,
{
  let half = values.len() / 2;
  let lo_eff = effective_nonzero_prefix(&values[..half]);
  let hi_eff = effective_nonzero_prefix(&values[half..]);
  MultilinearPolynomial::new_with_halves(values, lo_eff, hi_eff)
}

pub(super) fn effective_nonzero_prefix<F>(values: &[F]) -> usize
where
  F: Field,
{
  values
    .iter()
    .rposition(|value| !bool::from(value.is_zero()))
    .map_or(0, |idx| idx + 1)
}

pub(super) fn fold_prefix_workspace_pair_values<F, SV>(
  weights: &[F],
  prefix_workspace: &PrefixWorkspace<SV>,
  lo_suffix: usize,
  hi_suffix: usize,
  constraint_idx: usize,
  num_constraints: usize,
) -> (F, F, F, F, F, F)
where
  F: Field + DelayedReduction<SV>,
  SV: WideMul,
{
  let prefix_size = weights.len();
  debug_assert!(prefix_size > 0);
  debug_assert!(num_constraints > 0);
  debug_assert_eq!(prefix_workspace.num_constraints, num_constraints);
  debug_assert_eq!(prefix_workspace.prefix_size, prefix_size);

  let lo_start = (lo_suffix * num_constraints + constraint_idx) * prefix_size;
  let hi_start = (hi_suffix * num_constraints + constraint_idx) * prefix_size;
  let a_lo = &prefix_workspace.a[lo_start..lo_start + prefix_size];
  let a_hi = &prefix_workspace.a[hi_start..hi_start + prefix_size];
  let b_lo = &prefix_workspace.b[lo_start..lo_start + prefix_size];
  let b_hi = &prefix_workspace.b[hi_start..hi_start + prefix_size];
  let c_lo = &prefix_workspace.c[lo_start..lo_start + prefix_size];
  let c_hi = &prefix_workspace.c[hi_start..hi_start + prefix_size];

  let mut acc_a_lo = <F as DelayedReduction<SV>>::Accumulator::zero();
  let mut acc_a_hi = <F as DelayedReduction<SV>>::Accumulator::zero();
  let mut acc_b_lo = <F as DelayedReduction<SV>>::Accumulator::zero();
  let mut acc_b_hi = <F as DelayedReduction<SV>>::Accumulator::zero();
  let mut acc_c_lo = <F as DelayedReduction<SV>>::Accumulator::zero();
  let mut acc_c_hi = <F as DelayedReduction<SV>>::Accumulator::zero();
  for ((((((weight, a0), a1), b0), b1), c0), c1) in weights
    .iter()
    .zip(a_lo.iter())
    .zip(a_hi.iter())
    .zip(b_lo.iter())
    .zip(b_hi.iter())
    .zip(c_lo.iter())
    .zip(c_hi.iter())
  {
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_a_lo, weight, a0);
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_a_hi, weight, a1);
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_b_lo, weight, b0);
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_b_hi, weight, b1);
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_c_lo, weight, c0);
    <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc_c_hi, weight, c1);
  }
  (
    <F as DelayedReduction<SV>>::reduce(&acc_a_lo),
    <F as DelayedReduction<SV>>::reduce(&acc_a_hi),
    <F as DelayedReduction<SV>>::reduce(&acc_b_lo),
    <F as DelayedReduction<SV>>::reduce(&acc_b_hi),
    <F as DelayedReduction<SV>>::reduce(&acc_c_lo),
    <F as DelayedReduction<SV>>::reduce(&acc_c_hi),
  )
}

#[cfg(test)]
pub(super) fn fold_prefix_workspace_table<F, SV>(
  weights: &[F],
  table: &[SV],
  num_constraints: usize,
) -> Vec<Vec<F>>
where
  F: Field + DelayedReduction<SV> + Send + Sync,
  SV: Send + Sync,
{
  let prefix_size = weights.len();
  debug_assert!(prefix_size > 0);
  debug_assert!(num_constraints > 0);
  debug_assert_eq!(table.len() % (num_constraints * prefix_size), 0);
  let num_prefix_rows = table.len() / prefix_size;
  let mut folded_flat = vec![F::ZERO; num_prefix_rows];

  folded_flat
    .par_iter_mut()
    .enumerate()
    .for_each(|(row_idx, out)| {
      let start = row_idx * prefix_size;
      let prefix = &table[start..start + prefix_size];
      let mut acc = <F as DelayedReduction<SV>>::Accumulator::zero();
      for (weight, value) in weights.iter().zip(prefix.iter()) {
        <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(&mut acc, weight, value);
      }
      *out = <F as DelayedReduction<SV>>::reduce(&acc);
    });

  folded_flat
    .chunks(num_constraints)
    .map(|chunk| chunk.to_vec())
    .collect()
}

#[cfg(test)]
pub(super) fn fold_small_layers_by_prefix<F, SV, V>(
  weights: &[F],
  layers: &[V],
  prefix_size: usize,
) -> Vec<Vec<F>>
where
  F: Field + DelayedReduction<SV>,
  V: AsRef<[SV]> + Sync,
  SV: Send + Sync,
{
  debug_assert!(prefix_size > 0);
  debug_assert_eq!(layers.len() % prefix_size, 0);
  let suffix_groups = layers.len() / prefix_size;

  (0..suffix_groups)
    .into_par_iter()
    .map(|suffix_idx| {
      let start = suffix_idx * prefix_size;
      let end = start + prefix_size;
      fold_small_value_vectors::<F, SV, _>(weights, &layers[start..end])
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;

  type Scalar = pallas::Scalar;

  #[test]
  pub(super) fn test_prefix_workspace_fold_matches_layer_fold_with_padding() {
    let l0 = 2;
    let prefix_size = 1usize << l0;
    let n_padded = 8;
    let num_instances = 6;
    let num_constraints = 7;
    let make_table = |salt: i32| {
      (0..num_instances)
        .flat_map(|layer| {
          (0..num_constraints)
            .map(move |idx| ((layer as i32 * 13 + idx as i32 * 5 + salt) % 31) - 15)
        })
        .collect()
    };
    let small_abc = SmallAbc {
      l0,
      num_instances,
      num_constraints,
      a: make_table(0),
      b: make_table(3),
      c: make_table(7),
    };
    let workspace = PrefixWorkspace::build(&small_abc, l0, n_padded).unwrap();
    let weights = vec![
      Scalar::from(2u64),
      Scalar::from(3u64),
      Scalar::from(5u64),
      Scalar::from(7u64),
    ];

    let mut a_layers = Vec::with_capacity(n_padded);
    for idx in 0..n_padded {
      let row_idx = small_abc.padded_row_idx(idx).unwrap();
      a_layers.push(small_abc.a_row(row_idx));
    }

    let layer_fold =
      fold_small_layers_by_prefix::<Scalar, i32, _>(&weights, &a_layers, prefix_size);
    let workspace_fold =
      fold_prefix_workspace_table::<Scalar, i32>(&weights, &workspace.a, num_constraints);

    assert_eq!(layer_fold, workspace_fold);
  }
}

pub(super) fn fold_ab_pair_into<E: Engine>(
  a_layers: &mut [Vec<E::Scalar>],
  b_layers: &mut [Vec<E::Scalar>],
  src_even: usize,
  src_odd: usize,
  dest: usize,
  r: E::Scalar,
) {
  fold_layer_pair_into(a_layers, src_even, src_odd, dest, r);
  fold_layer_pair_into(b_layers, src_even, src_odd, dest, r);
}

pub(super) fn compact_folded_layers_ab<E: Engine>(
  a_layers: &mut [Vec<E::Scalar>],
  b_layers: &mut [Vec<E::Scalar>],
  prove_pairs: usize,
) {
  compact_folded_layers(a_layers, prove_pairs);
  compact_folded_layers(b_layers, prove_pairs);
}

pub(super) fn compact_folded_layers<F>(layers: &mut [Vec<F>], prove_pairs: usize) {
  super::super::compact_quads(layers, prove_pairs);
}

pub(super) fn fold_scalar_pair_into<F: Field>(
  values: &mut [F],
  src_even: usize,
  src_odd: usize,
  dest: usize,
  r: F,
) {
  let even = values[src_even];
  values[dest] = even + r * (values[src_odd] - even);
}

pub(super) fn compact_folded_scalars<F>(values: &mut [F], prove_pairs: usize) {
  super::super::compact_quads(values, prove_pairs);
}

pub(super) fn fold_final_ab_pairs<E: Engine>(
  a_layers: &mut [Vec<E::Scalar>],
  b_layers: &mut [Vec<E::Scalar>],
  pairs: usize,
  r: E::Scalar,
) {
  fold_final_layer_pairs(a_layers, pairs, r);
  fold_final_layer_pairs(b_layers, pairs, r);
}

pub(super) fn fold_prefix_workspace_final_table<F, SV>(
  weights: &[F],
  table: &[SV],
  num_constraints: usize,
  prefix_size: usize,
) -> Vec<F>
where
  F: Field + DelayedReduction<SV> + Send + Sync,
  SV: Send + Sync,
{
  debug_assert!(num_constraints > 0);
  debug_assert!(prefix_size > 0);
  debug_assert_eq!(weights.len() % prefix_size, 0);
  let suffix_groups = weights.len() / prefix_size;
  debug_assert_eq!(table.len(), suffix_groups * num_constraints * prefix_size);

  (0..num_constraints)
    .into_par_iter()
    .map(|constraint_idx| {
      let mut acc = <F as DelayedReduction<SV>>::Accumulator::zero();
      for suffix_idx in 0..suffix_groups {
        let table_base = (suffix_idx * num_constraints + constraint_idx) * prefix_size;
        let weight_base = suffix_idx * prefix_size;
        for prefix_idx in 0..prefix_size {
          <F as DelayedReduction<SV>>::unreduced_multiply_accumulate(
            &mut acc,
            &weights[weight_base + prefix_idx],
            &table[table_base + prefix_idx],
          );
        }
      }
      <F as DelayedReduction<SV>>::reduce(&acc)
    })
    .collect()
}
