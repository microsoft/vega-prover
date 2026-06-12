// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Bounds certificate for native small-value Lagrange extension.

use crate::{big_num::SmallValue, errors::SpartanError, polys::multilinear::MultilinearPolynomial};

/// Small integer type with enough metadata to check extension safety.
pub(crate) trait ExtensionSmallValue: SmallValue {
  /// Maximum accepted absolute value before extension.
  const MAX_ABS: u128;

  /// Absolute value as an unsigned integer, including signed MIN values.
  fn abs_as_u128(self) -> u128;
}

macro_rules! impl_extension_small_value {
  ($ty:ty) => {
    impl ExtensionSmallValue for $ty {
      const MAX_ABS: u128 = <$ty>::MAX as u128;

      #[inline]
      fn abs_as_u128(self) -> u128 {
        let value = self as i128;
        if value < 0 {
          (-value) as u128
        } else {
          value as u128
        }
      }
    }
  };
}

impl_extension_small_value!(i32);
impl_extension_small_value!(i64);

/// Maximum absolute input value safe for `LB` rounds of extension over `U_D`.
pub(crate) fn max_extension_input_abs<SV, const D: usize, const LB: usize>() -> u128
where
  SV: ExtensionSmallValue,
{
  let base = D as u128 + 1;
  let mut growth = 1u128;

  for _ in 0..LB {
    let Some(next_growth) = growth.checked_mul(base) else {
      return 0;
    };
    if next_growth > SV::MAX_ABS {
      return 0;
    }
    growth = next_growth;
  }

  SV::MAX_ABS / growth
}

/// Check that all values are safe for `LB` rounds of native Lagrange extension.
pub(crate) fn check_extension_bound<SV, const D: usize, const LB: usize>(
  poly: &MultilinearPolynomial<SV>,
) -> Result<(), SpartanError>
where
  SV: ExtensionSmallValue,
{
  let max_abs = max_extension_input_abs::<SV, D, LB>();

  if let Some(value) = poly
    .Z
    .iter()
    .copied()
    .find(|value| value.abs_as_u128() > max_abs)
  {
    return Err(SpartanError::SmallValueOverflow {
      value: value.abs_as_u128().to_string(),
      context: format!(
        "small-value Lagrange extension bound exceeded: max_abs={}, D={}, LB={}",
        max_abs, D, LB
      ),
    });
  }

  Ok(())
}

/// Polynomial whose values are certified safe for native Lagrange extension.
///
/// In debug builds, construction scans the full polynomial and checks the
/// extension bound. In release builds, this expensive O(n) scan is skipped and
/// construction acts as a caller assertion.
pub(crate) struct ExtensionBoundedPoly<'a, SV, const D: usize, const LB: usize> {
  poly: &'a MultilinearPolynomial<SV>,
}

impl<'a, SV, const D: usize, const LB: usize> ExtensionBoundedPoly<'a, SV, D, LB>
where
  SV: ExtensionSmallValue,
{
  /// Construct a certificate for native Lagrange extension.
  ///
  /// The bound check is intentionally debug-only because scanning the full
  /// polynomial is expensive on the prover hot path.
  pub(crate) fn new(poly: &'a MultilinearPolynomial<SV>) -> Result<Self, SpartanError> {
    #[cfg(debug_assertions)]
    check_extension_bound::<SV, D, LB>(poly)?;
    Ok(Self { poly })
  }

  /// Return the checked polynomial.
  pub(crate) fn as_poly(&self) -> &'a MultilinearPolynomial<SV> {
    self.poly
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_extension_bound_accepts_exact_bound_i32() {
    const D: usize = 2;
    const LB: usize = 3;
    let bound = max_extension_input_abs::<i32, D, LB>();
    let poly = MultilinearPolynomial::new(vec![bound as i32, -(bound as i32), 0, 1]);

    assert!(ExtensionBoundedPoly::<i32, D, LB>::new(&poly).is_ok());
  }

  #[test]
  fn test_check_extension_bound_rejects_above_bound_i32() {
    const D: usize = 2;
    const LB: usize = 3;
    let bound = max_extension_input_abs::<i32, D, LB>();
    let poly = MultilinearPolynomial::new(vec![(bound + 1) as i32]);

    assert!(matches!(
      check_extension_bound::<i32, D, LB>(&poly),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }

  #[test]
  fn test_extension_bound_accepts_exact_bound_i64() {
    const D: usize = 2;
    const LB: usize = 4;
    let bound = max_extension_input_abs::<i64, D, LB>();
    let poly = MultilinearPolynomial::new(vec![bound as i64, -(bound as i64), 0, 1]);

    assert!(ExtensionBoundedPoly::<i64, D, LB>::new(&poly).is_ok());
  }

  #[test]
  fn test_check_extension_bound_rejects_above_bound_i64() {
    const D: usize = 2;
    const LB: usize = 4;
    let bound = max_extension_input_abs::<i64, D, LB>();
    let poly = MultilinearPolynomial::new(vec![(bound + 1) as i64]);

    assert!(matches!(
      check_extension_bound::<i64, D, LB>(&poly),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }

  #[test]
  fn test_check_extension_bound_rejects_i64_min_without_overflowing_abs() {
    let poly = MultilinearPolynomial::new(vec![i64::MIN]);

    assert!(matches!(
      check_extension_bound::<i64, 2, 1>(&poly),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }
}
