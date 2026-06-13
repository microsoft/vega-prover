// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Bounds certificate for native small-value Lagrange extension.

use std::{fmt::Display, marker::PhantomData};

use crate::{big_num::SmallValue, errors::SpartanError, polys::multilinear::MultilinearPolynomial};
use num_integer::Roots;
use num_traits::{
  Bounded, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub, FromPrimitive, NumCast, One, Signed,
  ToPrimitive, Zero,
};

/// Maximum positive magnitude representable by `T`, converted into `Magnitude`.
#[inline]
fn max_abs<T, Magnitude>() -> Option<Magnitude>
where
  T: Bounded + ToPrimitive,
  Magnitude: NumCast,
{
  NumCast::from(T::max_value())
}

/// Absolute value converted into `Magnitude`.
#[inline]
fn abs_as<T, Magnitude>(value: T) -> Option<Magnitude>
where
  T: ToPrimitive,
  Magnitude: CheckedNeg + NumCast + PartialOrd + Signed + Zero,
{
  let value: Magnitude = NumCast::from(value)?;
  if value < Magnitude::zero() {
    value.checked_neg()
  } else {
    Some(value)
  }
}

/// Worst growth from extending one Boolean coordinate to `U_D`.
///
/// Fix every coordinate except the one being extended. Along that coordinate,
/// an MLE is the line determined by its two Boolean endpoint values:
///
/// `p(t) = (1 - t) p(0) + t p(1)`.
///
/// If both endpoint values have absolute value at most `B`, then a finite
/// extension point `t` has
///
/// `|p(t)| <= (|1 - t| + |t|) B`.
///
/// The finite part of `U_D` is `{0, 1, ..., D - 1}`. For `t = D - 1`, this
/// coefficient sum is `|2 - D| + |D - 1| = 2D - 3` when `D >= 2`.
///
/// The special point `∞` stores the leading coefficient of the line:
///
/// `p(∞) = p(1) - p(0)`,
///
/// so `|p(∞)| <= 2B`.
///
/// Therefore one extension round can multiply the input magnitude by at most
/// `H_D = max(2, 2D - 3)`.
#[inline]
fn extension_step_growth<T, const D: usize>() -> Option<T>
where
  T: CheckedMul + CheckedSub + Copy + FromPrimitive,
{
  let two = T::from_usize(2)?;
  if D <= 2 {
    Some(two)
  } else {
    T::from_usize(D)?
      .checked_mul(&two)?
      .checked_sub(&T::from_usize(3)?)
  }
}

#[inline]
fn checked_pow<T>(mut base: T, mut exp: usize) -> Option<T>
where
  T: CheckedMul + Copy + One,
{
  let mut acc = T::one();
  while exp > 0 {
    if exp & 1 == 1 {
      acc = acc.checked_mul(&base)?;
    }
    exp >>= 1;
    if exp > 0 {
      base = base.checked_mul(&base)?;
    }
  }
  Some(acc)
}

/// Maximum absolute MLE evaluation safe for native Lagrange extension.
///
/// A [`MultilinearPolynomial`] is stored in evaluation form: one value for each
/// Boolean point in `{0,1}^n`. This bound applies to those original evaluation
/// table entries before any coordinates are extended to the Lagrange domain
/// `U_D`.
///
/// Let `B` bound the absolute value of every original evaluation. For one
/// small-value factor, the `LB`-coordinate Lagrange extension is a tensor
/// product of one-coordinate interpolation weights, so
///
/// `|p(β)| <= B * H_D^LB`,
///
/// where `H_D = max(2, 2D - 3)` is computed by [`extension_step_growth`].
///
/// The native path certified here extends values into `SV` and then multiplies
/// two extended values into `Product`. Therefore both of these inequalities
/// must hold:
///
/// `B * H_D^LB <= SV_max`
///
/// and
///
/// `(B * H_D^LB)^2 <= Product_max`.
///
/// Equivalently, every original evaluation table entry must satisfy
///
/// `B <= min(SV_max, floor(sqrt(Product_max))) / H_D^LB`.
///
/// The returned magnitude is represented as `Product`, the widened product type
/// used by the caller's native small-value multiplication path. Spartan uses
/// `<SV as WideMul>::Product`.
pub(crate) fn max_extension_input_abs<SV, Product, const D: usize, const LB: usize>() -> Product
where
  SV: SmallValue + Bounded + ToPrimitive,
  Product: Bounded
    + CheckedDiv
    + CheckedMul
    + CheckedSub
    + Copy
    + FromPrimitive
    + NumCast
    + One
    + PartialOrd
    + Roots
    + ToPrimitive
    + Zero,
{
  let Some(growth) = extension_step_growth::<Product, D>().and_then(|base| checked_pow(base, LB))
  else {
    return Product::zero();
  };

  let extension_limit = max_abs::<SV, Product>().unwrap_or_else(Product::max_value);
  let Some(product_limit) = max_abs::<Product, Product>().map(|value| value.sqrt()) else {
    return Product::zero();
  };
  let limit = if extension_limit <= product_limit {
    extension_limit
  } else {
    product_limit
  };
  if growth > limit {
    return Product::zero();
  }

  limit.checked_div(&growth).unwrap_or_else(Product::zero)
}

/// Check that original MLE evaluations are safe for native extension.
///
/// This scans the polynomial's original evaluation table before extension, not
/// already-extended values. Passing means every original evaluation is small
/// enough that `LB` rounds of extension over `U_D` fit in `SV`, and the
/// following pairwise native product fits in `Product`.
pub(crate) fn check_extension_bound<SV, Product, const D: usize, const LB: usize>(
  poly: &MultilinearPolynomial<SV>,
) -> Result<(), SpartanError>
where
  SV: SmallValue + Bounded + ToPrimitive,
  Product: Bounded
    + CheckedDiv
    + CheckedMul
    + CheckedNeg
    + CheckedSub
    + Copy
    + FromPrimitive
    + NumCast
    + One
    + PartialOrd
    + Roots
    + Signed
    + ToPrimitive
    + Zero
    + Display,
{
  let max_abs = max_extension_input_abs::<SV, Product, D, LB>();

  if let Some(value) = poly
    .Z
    .iter()
    .copied()
    .find(|&value| abs_as::<SV, Product>(value).map_or(true, |value_abs| value_abs > max_abs))
  {
    return Err(SpartanError::SmallValueOverflow {
      value: abs_as::<SV, Product>(value).map_or_else(
        || "magnitude exceeds product type".to_string(),
        |value_abs| value_abs.to_string(),
      ),
      context: format!(
        "small-value Lagrange extension/product bound exceeded: max_abs={}, D={}, LB={}",
        max_abs, D, LB
      ),
    });
  }

  Ok(())
}

/// Polynomial whose original MLE evaluations are certified safe for native extension.
///
/// The certificate is parameterized by the input value type `SV`, the caller's
/// widened native product type `Product`, the extension domain degree `D`, and
/// the number of native extension rounds `LB`. It certifies that the
/// polynomial's original evaluation table can be extended into `SV` and that
/// the following pairwise native product fits in `Product`.
///
/// Use [`ExtensionBoundedPoly::new`] to scan and verify the bound, or
/// [`ExtensionBoundedPoly::new_unchecked`] when the caller has established the
/// bound through some other path and wants to avoid the O(n) scan.
pub(crate) struct ExtensionBoundedPoly<'a, SV, Product, const D: usize, const LB: usize> {
  poly: &'a MultilinearPolynomial<SV>,
  _product: PhantomData<fn() -> Product>,
}

impl<'a, SV, Product, const D: usize, const LB: usize> ExtensionBoundedPoly<'a, SV, Product, D, LB>
where
  SV: SmallValue + Bounded + ToPrimitive,
  Product: Bounded
    + CheckedDiv
    + CheckedMul
    + CheckedNeg
    + CheckedSub
    + Copy
    + FromPrimitive
    + NumCast
    + One
    + PartialOrd
    + Roots
    + Signed
    + ToPrimitive
    + Zero
    + Display,
{
  /// Construct a certificate for native Lagrange extension and pairwise product.
  ///
  /// This scans the full polynomial and rejects any original evaluation that
  /// could overflow during native extension or the following pairwise native
  /// product.
  pub(crate) fn new(poly: &'a MultilinearPolynomial<SV>) -> Result<Self, SpartanError> {
    check_extension_bound::<SV, Product, D, LB>(poly)?;
    Ok(Self::new_unchecked(poly))
  }

  /// Construct a certificate without scanning the polynomial.
  ///
  /// This is the hot-path constructor for callers that have already established
  /// `max_extension_input_abs::<SV, Product, D, LB>()` by construction or by an
  /// earlier check. Passing an out-of-bound polynomial invalidates the native
  /// extension/product safety guarantee carried by this certificate.
  pub(crate) fn new_unchecked(poly: &'a MultilinearPolynomial<SV>) -> Self {
    Self {
      poly,
      _product: PhantomData,
    }
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
  fn test_extension_step_growth() {
    assert_eq!(extension_step_growth::<i64, 2>(), Some(2));
    assert_eq!(extension_step_growth::<i64, 3>(), Some(3));
    assert_eq!(extension_step_growth::<i64, 4>(), Some(5));
  }

  #[test]
  fn test_extension_bound_uses_spartan_growth_i32() {
    const D: usize = 2;
    const LB: usize = 3;
    let bound = max_extension_input_abs::<i32, i64, D, LB>();
    let growth = checked_pow(extension_step_growth::<i64, D>().unwrap(), LB).unwrap();
    let limit = max_abs::<i32, i64>()
      .unwrap()
      .min(max_abs::<i64, i64>().unwrap().sqrt());

    assert_eq!(bound, limit / growth);
  }

  #[test]
  fn test_extension_bound_uses_product_limit() {
    const D: usize = 2;
    const LB: usize = 2;
    let product_limit = max_abs::<i16, i16>().unwrap().sqrt();
    let growth = checked_pow(extension_step_growth::<i16, D>().unwrap(), LB).unwrap();

    assert_eq!(product_limit, 181);
    assert_eq!(
      max_extension_input_abs::<i32, i16, D, LB>(),
      product_limit / growth
    );
  }

  #[test]
  fn test_extension_bound_accepts_exact_bound_i32() {
    const D: usize = 2;
    const LB: usize = 3;
    let bound = max_extension_input_abs::<i32, i64, D, LB>();
    let poly = MultilinearPolynomial::new(vec![bound as i32, -(bound as i32), 0, 1]);

    assert!(ExtensionBoundedPoly::<i32, i64, D, LB>::new(&poly).is_ok());
  }

  #[test]
  fn test_check_extension_bound_rejects_above_bound_i32() {
    const D: usize = 2;
    const LB: usize = 3;
    let bound = max_extension_input_abs::<i32, i64, D, LB>();
    let poly = MultilinearPolynomial::new(vec![(bound + 1) as i32]);

    assert!(matches!(
      check_extension_bound::<i32, i64, D, LB>(&poly),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }

  #[test]
  fn test_extension_bounded_poly_new_rejects_above_bound() {
    const D: usize = 2;
    const LB: usize = 3;
    let bound = max_extension_input_abs::<i32, i64, D, LB>();
    let poly = MultilinearPolynomial::new(vec![(bound + 1) as i32]);

    assert!(matches!(
      ExtensionBoundedPoly::<i32, i64, D, LB>::new(&poly),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }

  #[test]
  fn test_extension_bounded_poly_new_unchecked_skips_bound_check() {
    const D: usize = 2;
    const LB: usize = 3;
    let bound = max_extension_input_abs::<i32, i64, D, LB>();
    let poly = MultilinearPolynomial::new(vec![(bound + 1) as i32]);

    let bounded = ExtensionBoundedPoly::<i32, i64, D, LB>::new_unchecked(&poly);
    assert!(std::ptr::eq(bounded.as_poly(), &poly));
  }

  #[test]
  fn test_extension_bound_uses_spartan_growth_i64() {
    const D: usize = 2;
    const LB: usize = 4;
    let bound = max_extension_input_abs::<i64, i128, D, LB>();
    let growth = checked_pow(extension_step_growth::<i128, D>().unwrap(), LB).unwrap();
    let limit = max_abs::<i64, i128>()
      .unwrap()
      .min(max_abs::<i128, i128>().unwrap().sqrt());

    assert_eq!(bound, limit / growth);
  }

  #[test]
  fn test_extension_bound_accepts_exact_bound_i64() {
    const D: usize = 2;
    const LB: usize = 4;
    let bound = max_extension_input_abs::<i64, i128, D, LB>();
    let poly = MultilinearPolynomial::new(vec![bound as i64, -(bound as i64), 0, 1]);

    assert!(ExtensionBoundedPoly::<i64, i128, D, LB>::new(&poly).is_ok());
  }

  #[test]
  fn test_check_extension_bound_rejects_above_bound_i64() {
    const D: usize = 2;
    const LB: usize = 4;
    let bound = max_extension_input_abs::<i64, i128, D, LB>();
    let poly = MultilinearPolynomial::new(vec![(bound + 1) as i64]);

    assert!(matches!(
      check_extension_bound::<i64, i128, D, LB>(&poly),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }

  #[test]
  fn test_check_extension_bound_rejects_i64_min_without_overflowing_abs() {
    let poly = MultilinearPolynomial::new(vec![i64::MIN]);

    assert!(matches!(
      check_extension_bound::<i64, i128, 2, 1>(&poly),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }
}
