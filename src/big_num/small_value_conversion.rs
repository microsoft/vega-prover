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

use super::{DelayedReduction, WideMul};
use ff::PrimeField;
use num_traits::Zero;
use std::ops::{Add, Sub};

/// Small integer type usable in small-value sumcheck.
///
/// Bundles the arithmetic and widening requirements needed by the Lagrange
/// accumulator code.
pub trait SmallValue:
  WideMul + Copy + Default + Zero + Add<Output = Self> + Sub<Output = Self> + Send + Sync
{
}

impl SmallValue for i32 {}
impl SmallValue for i64 {}

/// Field that supports small-value sumcheck with value type `SV`.
pub trait SmallValueEngine<SV: SmallValue>:
  PrimeField
  + SmallValueField<SV>
  + DelayedReduction<SV>
  + DelayedReduction<SV::Product>
  + DelayedReduction<Self>
  + Send
  + Sync
{
}

impl<F, SV> SmallValueEngine<SV> for F
where
  SV: SmallValue,
  F: PrimeField
    + SmallValueField<SV>
    + DelayedReduction<SV>
    + DelayedReduction<SV::Product>
    + DelayedReduction<F>
    + Send
    + Sync,
{
}

/// Trait for fields that support conversion to and from native small values.
#[allow(dead_code)]
pub trait SmallValueField<SmallValue>: PrimeField {
  /// Convert a native small value to a field element.
  fn small_to_field(value: SmallValue) -> Self;

  /// Try to convert a field element to a native small value.
  fn try_field_to_small(value: &Self) -> Option<SmallValue>;
}

/// Maximum absolute value for "small" field elements stored as i64.
///
/// Chosen so that all i128 arithmetic in small-value sumcheck consumers remains
/// overflow-free.
const SMALL_VALUE_MAX: u64 = (1u64 << 62) - 1;
const SMALL_VALUE_MIN_I64: i64 = -(SMALL_VALUE_MAX as i64);

#[derive(Clone, Copy)]
enum SignedMagnitude {
  Positive(u128),
  Negative(u128),
}

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
fn try_field_to_signed_magnitude<F: PrimeField>(
  val: &F,
  width_bytes: usize,
) -> Option<SignedMagnitude> {
  let repr = val.to_repr();
  let bytes = repr.as_ref();

  if high_bytes_are_zero(bytes, width_bytes) {
    return Some(SignedMagnitude::Positive(lower_bytes_to_u128(
      bytes,
      width_bytes,
    )));
  }

  let neg_repr = val.neg().to_repr();
  let neg_bytes = neg_repr.as_ref();
  if high_bytes_are_zero(neg_bytes, width_bytes) {
    let mag = lower_bytes_to_u128(neg_bytes, width_bytes);
    if mag > 0 {
      return Some(SignedMagnitude::Negative(mag));
    }
  }

  None
}

#[inline]
#[allow(dead_code)]
fn try_field_to_i32<F: PrimeField>(val: &F) -> Option<i32> {
  match try_field_to_signed_magnitude(val, 4)? {
    SignedMagnitude::Positive(mag) if mag <= i32::MAX as u128 => Some(mag as i32),
    SignedMagnitude::Negative(mag) if mag <= (i32::MAX as u128) + 1 => Some(-(mag as i64) as i32),
    _ => None,
  }
}

#[inline]
fn try_field_to_i64<F: PrimeField>(val: &F) -> Option<i64> {
  match try_field_to_signed_magnitude(val, 8)? {
    SignedMagnitude::Positive(mag) if mag <= i64::MAX as u128 => Some(mag as i64),
    SignedMagnitude::Negative(mag) if mag <= (i64::MAX as u128) + 1 => Some(-(mag as i128) as i64),
    _ => None,
  }
}

#[inline]
#[allow(dead_code)]
fn try_field_to_i128<F: PrimeField>(val: &F) -> Option<i128> {
  match try_field_to_signed_magnitude(val, 16)? {
    SignedMagnitude::Positive(mag) if mag <= i128::MAX as u128 => Some(mag as i128),
    SignedMagnitude::Negative(mag) if mag <= (i128::MAX as u128) + 1 => {
      Some(mag.wrapping_neg() as i128)
    }
    _ => None,
  }
}

impl<F: PrimeField> SmallValueField<i32> for F {
  #[inline]
  fn small_to_field(value: i32) -> Self {
    i32_to_field(value)
  }

  #[inline]
  fn try_field_to_small(value: &Self) -> Option<i32> {
    try_field_to_i32(value)
  }
}

impl<F: PrimeField> SmallValueField<i64> for F {
  #[inline]
  fn small_to_field(value: i64) -> Self {
    i64_to_field(value)
  }

  #[inline]
  fn try_field_to_small(value: &Self) -> Option<i64> {
    try_field_to_i64(value)
  }
}

impl<F: PrimeField> SmallValueField<i128> for F {
  #[inline]
  fn small_to_field(value: i128) -> Self {
    i128_to_field(value)
  }

  #[inline]
  fn try_field_to_small(value: &Self) -> Option<i128> {
    try_field_to_i128(value)
  }
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
