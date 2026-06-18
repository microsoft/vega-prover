// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Bounds certificate for native small-value Lagrange extension.

use std::{
  collections::BTreeSet,
  fmt::{self, Debug, Display, Formatter},
  marker::PhantomData,
};

use super::accumulator_builder::SMALL_VALUE_T_DEGREE;
use crate::{
  big_num::{SmallValue, SmallValueField},
  errors::SpartanError,
  polys::multilinear::MultilinearPolynomial,
};
use num_integer::Roots;
use num_traits::{
  Bounded, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub, FromPrimitive, NumCast, One, Signed,
  ToPrimitive, Zero,
};
use serde::{Deserialize, Serialize};

/// Numeric operations needed to compute native extension/product bounds.
pub(crate) trait ExtensionMagnitude:
  Bounded
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
  + Zero
{
}

impl<T> ExtensionMagnitude for T where
  T: Bounded
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
    + Zero
{
}

/// Product type that can report native small-value extension bound failures.
pub(crate) trait ExtensionBoundProduct:
  ExtensionMagnitude + CheckedNeg + Signed + Display
{
}

impl<T> ExtensionBoundProduct for T where T: ExtensionMagnitude + CheckedNeg + Signed + Display {}

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
pub(crate) fn max_extension_input_abs_for_rounds<SV, Product, const D: usize>(
  rounds: usize,
) -> Product
where
  SV: SmallValue + Bounded + ToPrimitive,
  Product: ExtensionMagnitude,
{
  let Some(growth) =
    extension_step_growth::<Product, D>().and_then(|base| checked_pow(base, rounds))
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

/// Maximum absolute MLE evaluation safe for native Lagrange extension only.
///
/// This is the linear-term analogue of [`max_extension_input_abs_for_rounds`]:
/// it guarantees that `LB` rounds of extension over `U_D` fit back into `SV`,
/// but it does not reserve headroom for a following native product.
pub(crate) fn max_extension_fit_input_abs_for_rounds<SV, Magnitude, const D: usize>(
  rounds: usize,
) -> Magnitude
where
  SV: SmallValue + Bounded + ToPrimitive,
  Magnitude: ExtensionMagnitude,
{
  let Some(growth) =
    extension_step_growth::<Magnitude, D>().and_then(|base| checked_pow(base, rounds))
  else {
    return Magnitude::zero();
  };

  let limit = max_abs::<SV, Magnitude>().unwrap_or_else(Magnitude::max_value);
  if growth > limit {
    return Magnitude::zero();
  }

  limit.checked_div(&growth).unwrap_or_else(Magnitude::zero)
}

pub(crate) fn max_extension_input_abs<SV, Product, const D: usize, const LB: usize>() -> Product
where
  SV: SmallValue + Bounded + ToPrimitive,
  Product: ExtensionMagnitude,
{
  max_extension_input_abs_for_rounds::<SV, Product, D>(LB)
}

pub(crate) fn max_extension_fit_input_abs<SV, Magnitude, const D: usize, const LB: usize>()
-> Magnitude
where
  SV: SmallValue + Bounded + ToPrimitive,
  Magnitude: ExtensionMagnitude,
{
  max_extension_fit_input_abs_for_rounds::<SV, Magnitude, D>(LB)
}

/// Check that original MLE evaluations are safe for native extension.
///
/// This scans the polynomial's original evaluation table before extension, not
/// already-extended values. Passing means every original evaluation is small
/// enough that `LB` rounds of extension over `U_D` fit in `SV`, and the
/// following pairwise native product fits in `Product`.
pub(crate) fn check_extension_bound_values_for_rounds<SV, Product, const D: usize>(
  values: impl IntoIterator<Item = SV>,
  rounds: usize,
  context: impl Display,
) -> Result<(), SpartanError>
where
  SV: SmallValue + Bounded + ToPrimitive,
  Product: ExtensionBoundProduct,
{
  let max_abs = max_extension_input_abs_for_rounds::<SV, Product, D>(rounds);
  check_extension_values_against_bound::<SV, Product, D>(
    values,
    max_abs,
    rounds,
    context,
    "extension/product",
  )
}

pub(crate) fn check_extension_fit_bound_values_for_rounds<SV, Magnitude, const D: usize>(
  values: impl IntoIterator<Item = SV>,
  rounds: usize,
  context: impl Display,
) -> Result<(), SpartanError>
where
  SV: SmallValue + Bounded + ToPrimitive,
  Magnitude: ExtensionBoundProduct,
{
  let max_abs = max_extension_fit_input_abs_for_rounds::<SV, Magnitude, D>(rounds);
  check_extension_values_against_bound::<SV, Magnitude, D>(
    values,
    max_abs,
    rounds,
    context,
    "extension",
  )
}

fn check_extension_values_against_bound<SV, Magnitude, const D: usize>(
  values: impl IntoIterator<Item = SV>,
  max_abs: Magnitude,
  rounds: usize,
  context: impl Display,
  bound_name: &str,
) -> Result<(), SpartanError>
where
  SV: SmallValue + Bounded + ToPrimitive,
  Magnitude: ExtensionBoundProduct,
{
  for value in values {
    let Some(value_abs) = abs_as::<SV, Magnitude>(value) else {
      return Err(SpartanError::SmallValueOverflow {
        value: "magnitude exceeds bound type".to_string(),
        context: format!(
          "{}: small-value Lagrange {} bound exceeded: max_abs={}, D={}, rounds={}",
          context, bound_name, max_abs, D, rounds
        ),
      });
    };
    if value_abs > max_abs {
      return Err(SpartanError::SmallValueOverflow {
        value: value_abs.to_string(),
        context: format!(
          "{}: small-value Lagrange {} bound exceeded: max_abs={}, D={}, rounds={}",
          context, bound_name, max_abs, D, rounds
        ),
      });
    }
  }

  Ok(())
}

#[inline]
pub(crate) fn small_value_fits_abs_bound<SV, Magnitude>(value: SV, max_abs: Magnitude) -> bool
where
  SV: ToPrimitive,
  Magnitude: ExtensionBoundProduct,
{
  abs_as::<SV, Magnitude>(value).is_some_and(|value_abs| value_abs <= max_abs)
}

pub(crate) fn check_extension_bound<SV, Product, const D: usize, const LB: usize>(
  poly: &MultilinearPolynomial<SV>,
) -> Result<(), SpartanError>
where
  SV: SmallValue + Bounded + ToPrimitive,
  Product: ExtensionBoundProduct,
{
  check_extension_bound_values_for_rounds::<SV, Product, D>(
    poly.Z.iter().copied(),
    LB,
    "small-value polynomial",
  )
}

pub(crate) fn check_extension_fit_bound<SV, Magnitude, const D: usize, const LB: usize>(
  poly: &MultilinearPolynomial<SV>,
) -> Result<(), SpartanError>
where
  SV: SmallValue + Bounded + ToPrimitive,
  Magnitude: ExtensionBoundProduct,
{
  check_extension_fit_bound_values_for_rounds::<SV, Magnitude, D>(
    poly.Z.iter().copied(),
    LB,
    "small-value polynomial",
  )
}

/// Convert field elements into i64 values that are safe for `l0` rounds of the
/// NeutronNova small-value accumulator. Unsafe values are replaced with zero
/// and returned as large positions for field-arithmetic correction.
pub(crate) fn field_to_i64_or_zero_for_l0<F>(values: &[F], l0: usize) -> (Vec<i64>, Vec<usize>)
where
  F: SmallValueField<i64>,
{
  let max_abs = max_extension_input_abs_for_rounds::<i64, i128, SMALL_VALUE_T_DEGREE>(l0);
  let mut small = Vec::with_capacity(values.len());
  let mut large_positions = Vec::new();

  for (idx, value) in values.iter().enumerate() {
    match F::try_field_to_small(value).and_then(|small_value| {
      abs_as::<i64, i128>(small_value)
        .filter(|value_abs| *value_abs <= max_abs)
        .map(|_| small_value)
    }) {
      Some(small_value) => small.push(small_value),
      None => {
        small.push(0);
        large_positions.push(idx);
      }
    }
  }

  (small, large_positions)
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
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "SV: Serialize", deserialize = "SV: Deserialize<'de>"))]
pub struct ExtensionBoundedPoly<SV, Product, const D: usize, const LB: usize> {
  poly: MultilinearPolynomial<SV>,
  _product: PhantomData<fn() -> Product>,
}

impl<SV, Product, const D: usize, const LB: usize> Debug
  for ExtensionBoundedPoly<SV, Product, D, LB>
where
  SV: Debug,
{
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    f.debug_struct("ExtensionBoundedPoly")
      .field("poly", &self.poly)
      .finish()
  }
}

#[allow(private_bounds)]
impl<SV, Product, const D: usize, const LB: usize> ExtensionBoundedPoly<SV, Product, D, LB>
where
  SV: SmallValue + Bounded + ToPrimitive,
  Product: ExtensionBoundProduct,
{
  /// Construct a certificate for native Lagrange extension and pairwise product.
  ///
  /// This scans the full polynomial and rejects any original evaluation that
  /// could overflow during native extension or the following pairwise native
  /// product.
  pub(crate) fn new(poly: MultilinearPolynomial<SV>) -> Result<Self, SpartanError> {
    check_extension_bound::<SV, Product, D, LB>(&poly)?;
    Ok(Self::new_unchecked(poly))
  }

  /// Construct a certificate for native Lagrange extension only.
  ///
  /// This is used for linear terms such as `Cz`, which must fit during native
  /// extension but are not multiplied by another native small value.
  pub(crate) fn new_extension_only(poly: MultilinearPolynomial<SV>) -> Result<Self, SpartanError> {
    check_extension_fit_bound::<SV, Product, D, LB>(&poly)?;
    Ok(Self::new_unchecked(poly))
  }

  /// Construct a bounded small polynomial plus positions that must use field values.
  ///
  /// Any input value that cannot satisfy the extension/product bound is replaced
  /// with zero in the returned certificate and its index is returned.
  pub(crate) fn try_new_with_field_positions(
    poly: &MultilinearPolynomial<SV>,
  ) -> (Self, BTreeSet<usize>) {
    Self::new_with_positions(
      poly.clone(),
      max_extension_input_abs::<SV, Product, D, LB>(),
    )
  }

  /// Construct a bounded small polynomial from owned evaluations plus field-backed positions.
  ///
  /// This is the allocation-friendly variant for callers that just produced the
  /// small polynomial and do not need to keep the unchecked copy.
  pub(crate) fn try_new_owned_with_field_positions(
    poly: MultilinearPolynomial<SV>,
  ) -> (Self, BTreeSet<usize>) {
    Self::new_with_positions(poly, max_extension_input_abs::<SV, Product, D, LB>())
  }

  /// Construct an extension-bounded small polynomial plus field-backed positions.
  ///
  /// Any input value that cannot satisfy the extension-only bound is replaced
  /// with zero in the returned certificate and its index is returned.
  pub(crate) fn try_new_extension_only_with_field_positions(
    poly: &MultilinearPolynomial<SV>,
  ) -> (Self, BTreeSet<usize>) {
    Self::new_with_positions(
      poly.clone(),
      max_extension_fit_input_abs::<SV, Product, D, LB>(),
    )
  }

  /// Construct an extension-bounded small polynomial from owned evaluations.
  ///
  /// This is used for linear terms such as `Cz` when the caller can transfer
  /// ownership of freshly converted small values.
  pub(crate) fn try_new_owned_extension_only_with_field_positions(
    poly: MultilinearPolynomial<SV>,
  ) -> (Self, BTreeSet<usize>) {
    Self::new_with_positions(poly, max_extension_fit_input_abs::<SV, Product, D, LB>())
  }

  fn new_with_positions(
    mut poly: MultilinearPolynomial<SV>,
    max_abs: Product,
  ) -> (Self, BTreeSet<usize>) {
    let mut field_positions = BTreeSet::new();
    for (idx, value) in poly.Z.iter_mut().enumerate() {
      if !small_value_fits_abs_bound::<SV, Product>(*value, max_abs) {
        *value = SV::zero();
        field_positions.insert(idx);
      }
    }

    (Self::new_unchecked(poly), field_positions)
  }

  /// Construct a certificate without scanning the polynomial.
  ///
  /// This is the hot-path constructor for callers that have already established
  /// `max_extension_input_abs::<SV, Product, D, LB>()` by construction or by an
  /// earlier check. Passing an out-of-bound polynomial invalidates the native
  /// extension/product safety guarantee carried by this certificate.
  pub(crate) fn new_unchecked(poly: MultilinearPolynomial<SV>) -> Self {
    Self {
      poly,
      _product: PhantomData,
    }
  }

  /// Return the checked polynomial.
  pub(crate) fn as_poly(&self) -> &MultilinearPolynomial<SV> {
    &self.poly
  }

  /// Consume and return the checked polynomial.
  pub(crate) fn into_poly(self) -> MultilinearPolynomial<SV> {
    self.poly
  }

  /// Zero field-backed positions in this certified polynomial.
  pub(crate) fn zero_positions(&mut self, positions: &BTreeSet<usize>) {
    for &idx in positions {
      self.poly.Z[idx] = SV::zero();
    }
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
  fn test_runtime_extension_bound_matches_const_i32() {
    const D: usize = 2;
    const LB: usize = 3;

    assert_eq!(
      max_extension_input_abs_for_rounds::<i32, i64, D>(LB),
      max_extension_input_abs::<i32, i64, D, LB>()
    );
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

    assert!(ExtensionBoundedPoly::<i32, i64, D, LB>::new(poly).is_ok());
  }

  #[test]
  fn test_runtime_extension_bound_accepts_exact_bound_i32() {
    const D: usize = 2;
    const LB: usize = 3;
    let bound = max_extension_input_abs_for_rounds::<i32, i64, D>(LB);

    assert!(
      check_extension_bound_values_for_rounds::<i32, i64, D>(
        [bound as i32, -(bound as i32), 0, 1],
        LB,
        "runtime values",
      )
      .is_ok()
    );
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
  fn test_runtime_extension_bound_rejects_above_bound_i32() {
    const D: usize = 2;
    const LB: usize = 3;
    let bound = max_extension_input_abs_for_rounds::<i32, i64, D>(LB);

    assert!(matches!(
      check_extension_bound_values_for_rounds::<i32, i64, D>(
        [(bound + 1) as i32],
        LB,
        "runtime values",
      ),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }

  #[test]
  fn test_field_to_i64_or_zero_for_l0_marks_values_above_runtime_bound() {
    use crate::{provider::PallasHyraxEngine, traits::Engine};

    type F = <PallasHyraxEngine as Engine>::Scalar;
    let l0 = 2;
    let bound = max_extension_input_abs_for_rounds::<i64, i128, SMALL_VALUE_T_DEGREE>(l0);
    let at_bound = F::from(u64::try_from(bound).unwrap());
    let above_bound = F::from(u64::try_from(bound + 1).unwrap());

    let (small, large_positions) =
      field_to_i64_or_zero_for_l0(&[F::from(7u64), at_bound, above_bound], l0);

    assert_eq!(small, vec![7, i64::try_from(bound).unwrap(), 0]);
    assert_eq!(large_positions, vec![2]);
  }

  #[test]
  fn test_extension_bounded_poly_new_rejects_above_bound() {
    const D: usize = 2;
    const LB: usize = 3;
    let bound = max_extension_input_abs::<i32, i64, D, LB>();
    let poly = MultilinearPolynomial::new(vec![(bound + 1) as i32]);

    assert!(matches!(
      ExtensionBoundedPoly::<i32, i64, D, LB>::new(poly),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
  }

  #[test]
  fn test_extension_bounded_poly_new_unchecked_skips_bound_check() {
    const D: usize = 2;
    const LB: usize = 3;
    let bound = max_extension_input_abs::<i32, i64, D, LB>();
    let poly = MultilinearPolynomial::new(vec![(bound + 1) as i32]);

    let bounded = ExtensionBoundedPoly::<i32, i64, D, LB>::new_unchecked(poly.clone());
    assert_eq!(bounded.as_poly().Z, poly.Z);
  }

  #[test]
  fn test_try_new_with_field_positions_keeps_in_bound_values() {
    const D: usize = 2;
    const LB: usize = 1;
    let bound = max_extension_input_abs::<i32, i16, D, LB>() as i32;
    let poly = MultilinearPolynomial::new(vec![bound, -bound, 0, 1]);
    let (bounded, field_positions) =
      ExtensionBoundedPoly::<i32, i16, D, LB>::try_new_with_field_positions(&poly);

    assert!(field_positions.is_empty());
    assert_eq!(bounded.as_poly().Z, poly.Z);
  }

  #[test]
  fn test_try_new_with_field_positions_zeroes_out_of_bound_values() {
    const D: usize = 2;
    const LB: usize = 1;
    let bound = max_extension_input_abs::<i32, i16, D, LB>() as i32;
    let poly = MultilinearPolynomial::new(vec![bound, bound + 1, -(bound + 1), 7]);
    let (bounded, field_positions) =
      ExtensionBoundedPoly::<i32, i16, D, LB>::try_new_with_field_positions(&poly);

    assert_eq!(field_positions.into_iter().collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(bounded.as_poly().Z, vec![bound, 0, 0, 7]);
    assert_eq!(poly.Z, vec![bound, bound + 1, -(bound + 1), 7]);
  }

  #[test]
  fn test_extension_only_bound_is_weaker_than_product_bound() {
    const D: usize = 1;
    const LB: usize = 1;
    let product_bound = max_extension_input_abs::<i32, i16, D, LB>() as i32;
    let extension_only_bound = max_extension_fit_input_abs::<i32, i16, D, LB>() as i32;
    assert!(product_bound < extension_only_bound);

    let poly = MultilinearPolynomial::new(vec![product_bound + 1]);
    assert!(matches!(
      ExtensionBoundedPoly::<i32, i16, D, LB>::new(poly.clone()),
      Err(SpartanError::SmallValueOverflow { .. })
    ));
    assert!(ExtensionBoundedPoly::<i32, i16, D, LB>::new_extension_only(poly).is_ok());
  }

  #[test]
  fn test_try_new_extension_only_with_field_positions_uses_extension_bound() {
    const D: usize = 1;
    const LB: usize = 1;
    let bound = max_extension_fit_input_abs::<i32, i16, D, LB>() as i32;
    let poly = MultilinearPolynomial::new(vec![bound, bound + 1]);
    let (bounded, field_positions) =
      ExtensionBoundedPoly::<i32, i16, D, LB>::try_new_extension_only_with_field_positions(&poly);

    assert_eq!(field_positions.into_iter().collect::<Vec<_>>(), vec![1]);
    assert_eq!(bounded.as_poly().Z, vec![bound, 0]);
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
  fn test_runtime_extension_bound_matches_const_i64() {
    const D: usize = 2;
    const LB: usize = 4;

    assert_eq!(
      max_extension_input_abs_for_rounds::<i64, i128, D>(LB),
      max_extension_input_abs::<i64, i128, D, LB>()
    );
  }

  #[test]
  fn test_extension_bound_accepts_exact_bound_i64() {
    const D: usize = 2;
    const LB: usize = 4;
    let bound = max_extension_input_abs::<i64, i128, D, LB>();
    let poly = MultilinearPolynomial::new(vec![bound as i64, -(bound as i64), 0, 1]);

    assert!(ExtensionBoundedPoly::<i64, i128, D, LB>::new(poly).is_ok());
  }

  #[test]
  fn test_runtime_extension_bound_accepts_exact_bound_i64() {
    const D: usize = 2;
    const LB: usize = 4;
    let bound = max_extension_input_abs_for_rounds::<i64, i128, D>(LB);

    assert!(
      check_extension_bound_values_for_rounds::<i64, i128, D>(
        [bound as i64, -(bound as i64), 0, 1],
        LB,
        "runtime values",
      )
      .is_ok()
    );
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
  fn test_runtime_extension_bound_rejects_above_bound_i64() {
    const D: usize = 2;
    const LB: usize = 4;
    let bound = max_extension_input_abs_for_rounds::<i64, i128, D>(LB);

    assert!(matches!(
      check_extension_bound_values_for_rounds::<i64, i128, D>(
        [(bound + 1) as i64],
        LB,
        "runtime values",
      ),
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
