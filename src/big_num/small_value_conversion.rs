// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Conversion helpers for the small-value optimization.
//!
//! The NeutronNova small-value fast path works on signed `i64` views of field
//! elements, with large values zeroed and corrected separately using field
//! arithmetic.

use ff::PrimeField;

/// Maximum absolute value for "small" field elements stored as i64.
///
/// Chosen so that all i128 arithmetic in small-value sumcheck consumers remains
/// overflow-free.
const SMALL_VALUE_MAX: u64 = (1u64 << 62) - 1;
const SMALL_VALUE_MIN_I64: i64 = -(SMALL_VALUE_MAX as i64);

#[allow(dead_code)]
pub(crate) fn i32_to_field<F: PrimeField>(value: i32) -> F {
  i64_to_field::<F>(value as i64)
}

#[allow(dead_code)]
pub(crate) fn i64_to_field<F: PrimeField>(value: i64) -> F {
  if value >= 0 {
    F::from(value as u64)
  } else {
    -F::from(value.wrapping_neg() as u64)
  }
}

#[allow(dead_code)]
pub(crate) fn i128_to_field<F: PrimeField>(value: i128) -> F {
  let two_64 = F::from(u64::MAX) + F::ONE;

  if value >= 0 {
    let value_u128 = value as u128;
    let lo = value_u128 as u64;
    let hi = (value_u128 >> 64) as u64;
    F::from(lo) + F::from(hi) * two_64
  } else {
    let mag = value.wrapping_neg() as u128;
    let lo = mag as u64;
    let hi = (mag >> 64) as u64;
    -(F::from(lo) + F::from(hi) * two_64)
  }
}

#[inline]
fn high_bytes_are_zero(bytes: &[u8], width_bytes: usize) -> bool {
  bytes[width_bytes..].iter().all(|&b| b == 0)
}

#[inline]
fn lower_bytes_to_u128(bytes: &[u8], width_bytes: usize) -> u128 {
  let mut buf = [0u8; 16];
  buf[..width_bytes].copy_from_slice(&bytes[..width_bytes]);
  u128::from_le_bytes(buf)
}

#[inline]
fn try_field_to_i64<F: PrimeField>(val: &F) -> Option<i64> {
  let repr = val.to_repr();
  let bytes = repr.as_ref();

  if high_bytes_are_zero(bytes, 8) {
    let mag = lower_bytes_to_u128(bytes, 8);
    if mag <= i64::MAX as u128 {
      return Some(mag as i64);
    }
  }

  let neg_repr = val.neg().to_repr();
  let neg_bytes = neg_repr.as_ref();
  if high_bytes_are_zero(neg_bytes, 8) {
    let mag = lower_bytes_to_u128(neg_bytes, 8);
    if mag > 0 && mag <= (i64::MAX as u128) + 1 {
      return Some(-(mag as i128) as i64);
    }
  }

  None
}

/// Convert field elements to i64 values, storing 0 for values outside the
/// small-value range and recording those positions for field correction.
#[inline(never)]
pub(crate) fn to_small_vec_or_zero<F: PrimeField>(poly: &[F]) -> (Vec<i64>, Vec<usize>) {
  let mut result = Vec::with_capacity(poly.len());
  let mut large_positions = Vec::new();

  for (idx, f) in poly.iter().enumerate() {
    match try_field_to_i64(f) {
      Some(val) if (SMALL_VALUE_MIN_I64..=SMALL_VALUE_MAX as i64).contains(&val) => {
        result.push(val);
      }
      _ => {
        result.push(0);
        large_positions.push(idx);
      }
    }
  }

  (result, large_positions)
}

/// Generate small-value conversion tests for a field type.
#[cfg(test)]
#[macro_export]
macro_rules! test_small_value_conversion {
  ($name:ident, $field:ty) => {
    mod $name {
      #[test]
      fn small_vec_or_zero() {
        $crate::big_num::small_value_conversion::tests::test_small_vec_or_zero_impl::<$field>();
      }
    }
  };
}

#[cfg(test)]
pub(crate) mod tests {
  use super::*;

  /// Test to_small_vec_or_zero with accepted small values and rejected large values.
  pub(crate) fn test_small_vec_or_zero_impl<F: PrimeField + Copy>() {
    let vals: Vec<F> = vec![
      F::ZERO,
      F::from(1u64),
      F::from(5u64),
      -F::from(3u64),
      F::from(SMALL_VALUE_MAX),
      -F::from(SMALL_VALUE_MAX),
    ];
    let (small, large) = to_small_vec_or_zero(&vals);
    assert!(large.is_empty());
    assert_eq!(
      small,
      vec![0, 1, 5, -3, SMALL_VALUE_MAX as i64, SMALL_VALUE_MIN_I64]
    );

    let above = SMALL_VALUE_MAX + 1;
    let vals = vec![F::from(above), -F::from(above)];
    let (small, large) = to_small_vec_or_zero(&vals);
    assert_eq!(small, vec![0, 0]);
    assert_eq!(large, vec![0, 1]);
  }
}
