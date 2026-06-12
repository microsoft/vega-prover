// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SmallMultiEq: Equality constraints with bounded coefficients.
//!
//! This module provides the `SmallMultiEq` trait and its implementation:
//! - [`NoBatchEq`]: Each equality constraint is enforced directly (i32 path)
//!
//! All SHA-256 witnesses are bits (0/1), so NoBatchEq with limbed addition suffices.
//! Max coefficient ~2^18 fits in i32, enabling pure i128-free accumulation.
//!
//! # Why SmallMultiEq?
//!
//! Bellpepper's `MultiEq` batches equality constraints using powers of 2 as
//! coefficients, accumulating up to 2^237 after ~237 batched equalities.
//! This breaks small-value optimization because matrix entries would be huge.
//!
//! `NoBatchEq` keeps coefficients bounded at 2^18 (max from limbed addition).

use super::{addmany, small_uint32::SmallUInt32};
use crate::small_constraint_system::{SmallConstraintSystem, SmallLinearCombination};
use bellpepper_core::SynthesisError;

// ============================================================================
// SmallMultiEq Trait
// ============================================================================

/// Constraint system extension for equality constraints with bounded coefficients.
///
/// This trait extends `SmallConstraintSystem` with methods for enforcing equality
/// constraints and multi-operand addition.
pub trait SmallMultiEq<W, C>: SmallConstraintSystem<W, C> {
  /// Enforce that `lhs` equals `rhs`.
  fn enforce_equal(&mut self, lhs: &SmallLinearCombination<C>, rhs: &SmallLinearCombination<C>);

  /// Add multiple SmallUInt32 values together using limbed addition.
  fn addmany(&mut self, operands: &[SmallUInt32]) -> Result<SmallUInt32, SynthesisError>;
}

// ============================================================================
// NoBatchEq - Direct enforcement, limbed addition
// ============================================================================

/// Constraint system wrapper that enforces equality constraints directly.
///
/// Each call to `enforce_equal` immediately creates a constraint. Uses limbed
/// addition (max coefficient 2^18, fits i32) for multi-operand addition.
pub struct NoBatchEq<'a, W, C, CS: SmallConstraintSystem<W, C>> {
  pub(crate) cs: &'a mut CS,
  _marker: std::marker::PhantomData<(W, C)>,
}

impl<'a, W, C, CS: SmallConstraintSystem<W, C>> NoBatchEq<'a, W, C, CS> {
  /// Create a new NoBatchEq wrapper around a constraint system.
  pub fn new(cs: &'a mut CS) -> Self {
    NoBatchEq {
      cs,
      _marker: std::marker::PhantomData,
    }
  }
}

/// Direct equality for the small-value path with i32 coefficients.
///
/// Shape backends record these constraints; witness backends accept the i32 LCs and no-op.
impl<W: Copy + From<bool>, CS: SmallConstraintSystem<W, i32>> SmallMultiEq<W, i32>
  for NoBatchEq<'_, W, i32, CS>
{
  fn enforce_equal(
    &mut self,
    lhs: &SmallLinearCombination<i32>,
    rhs: &SmallLinearCombination<i32>,
  ) {
    if self.cs.is_witness_generator() {
      return;
    }

    // Enforce: lhs - rhs = 0, expressed as (lhs - rhs) * ONE = 0
    let mut diff = lhs.clone();
    for (var, coeff) in &rhs.terms {
      diff.add_term(*var, -*coeff);
    }
    self.cs.enforce(
      diff,
      SmallLinearCombination::one(1i32), // b = 1 * ONE
      SmallLinearCombination::zero(),    // c = 0
    );
  }

  fn addmany(&mut self, operands: &[SmallUInt32]) -> Result<SmallUInt32, SynthesisError> {
    assert!(operands.len() >= 2);
    assert!(operands.len() <= 10);

    if let Some(sum) = try_constant_sum(operands) {
      return Ok(SmallUInt32::constant(sum));
    }

    addmany::limbed::<W, _>(self, operands)
  }
}

impl<W: Copy, C, CS: SmallConstraintSystem<W, C>> SmallConstraintSystem<W, C>
  for NoBatchEq<'_, W, C, CS>
{
  fn alloc<F>(&mut self, f: F) -> Result<bellpepper_core::Variable, SynthesisError>
  where
    F: FnOnce() -> Result<W, SynthesisError>,
  {
    self.cs.alloc(f)
  }

  fn alloc_input<F>(&mut self, f: F) -> Result<bellpepper_core::Variable, SynthesisError>
  where
    F: FnOnce() -> Result<W, SynthesisError>,
  {
    self.cs.alloc_input(f)
  }

  fn enforce(
    &mut self,
    a: SmallLinearCombination<C>,
    b: SmallLinearCombination<C>,
    c: SmallLinearCombination<C>,
  ) {
    self.cs.enforce(a, b, c);
  }

  fn is_witness_generator(&self) -> bool {
    self.cs.is_witness_generator()
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Try to compute sum as a constant if all operands are constant.
fn try_constant_sum(operands: &[SmallUInt32]) -> Option<u32> {
  if !operands.iter().all(|op| op.get_value().is_some()) {
    return None;
  }

  let all_constant = operands.iter().all(|op| {
    op.bits_le()
      .iter()
      .all(|b| matches!(b, crate::gadgets::small_boolean::SmallBoolean::Constant(_)))
  });

  if all_constant {
    let sum: u32 = operands
      .iter()
      .map(|op| op.get_value().unwrap())
      .fold(0u32, |a, b| a.wrapping_add(b));
    Some(sum)
  } else {
    None
  }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::small_constraint_system::SmallShapeCS;

  #[test]
  fn test_no_batch_eq_basic() {
    let mut cs = SmallShapeCS::<i32>::new();

    let a = cs.alloc(|| Ok(0i8)).unwrap();
    let b = cs.alloc(|| Ok(0i8)).unwrap();

    {
      let mut eq = NoBatchEq::<i8, i32, _>::new(&mut cs);
      let lhs = SmallLinearCombination::from_variable(a, 1i32);
      let rhs = SmallLinearCombination::from_variable(b, 1i32);
      eq.enforce_equal(&lhs, &rhs);
    }

    // 1 constraint from alloc bool a, 1 from alloc bool b, 1 from enforce_equal = 3 total? No —
    // SmallShapeCS::alloc doesn't enforce boolean, just records variable.
    // enforce_equal adds 1 constraint.
    assert_eq!(cs.num_constraints(), 1);
  }

  #[test]
  fn test_no_batch_eq_addmany() {
    let mut cs = SmallShapeCS::<i32>::new();

    let a = SmallUInt32::constant(100);
    let b = SmallUInt32::constant(200);
    let c = SmallUInt32::constant(300);

    let mut eq = NoBatchEq::<i8, i32, _>::new(&mut cs);
    let result = eq.addmany(&[a, b, c]).unwrap();
    assert_eq!(result.get_value(), Some(600));
  }
}
