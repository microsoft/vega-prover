// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! SHA-256 chain circuit using the small SHA-256 gadget.

use super::{
  bytes_to_public_scalars,
  small::{alloc_preimage_small_bits, expose_small_hash_bits_as_public},
};
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError, num::AllocatedNum};
use ff::{PrimeField, PrimeFieldBits};
use sha2::{Digest, Sha256};
use std::marker::PhantomData;

use crate::{
  gadgets::{NoBatchEq, SmallBoolean, small_sha256_int},
  small_constraint_system::{SmallConstraintSystem, SmallToBellpepperCS},
  traits::{Engine, circuit::SpartanCircuit},
};

/// SHA-256 chain circuit using the small SHA-256 gadget.
///
/// Chains `chain_length` SHA-256 hashes starting from a 256-bit input.
/// Hash[0] = SHA-256(input), Hash[i] = SHA-256(Hash[i-1])
#[derive(Debug, Clone)]
pub struct SmallSha256ChainCircuit<Scalar: PrimeField> {
  /// 32-byte (256-bit) input to start the chain
  pub input: [u8; 32],
  /// Number of SHA-256 hashes in the chain
  pub chain_length: usize,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> SmallSha256ChainCircuit<Scalar> {
  /// Create a new SHA-256 chain circuit.
  pub fn new(input: [u8; 32], chain_length: usize) -> Self {
    Self {
      input,
      chain_length,
      _p: PhantomData,
    }
  }

  /// Compute the expected final hash by applying SHA-256 chain_length times.
  pub fn expected_output(&self) -> [u8; 32] {
    let mut current = self.input;
    for _ in 0..self.chain_length {
      let mut hasher = Sha256::new();
      hasher.update(current);
      current = hasher.finalize().into();
    }
    current
  }
}

/// Shared synthesis body: allocate the chain input bits and run the small
/// SHA-256 gadget `chain_length` times, returning the final digest bits.
fn synthesize_sha256_chain_bits<CS>(
  cs: &mut CS,
  input: &[u8; 32],
  chain_length: usize,
) -> Result<Vec<SmallBoolean>, SynthesisError>
where
  CS: SmallConstraintSystem<i8, i32>,
{
  let mut current_bits = alloc_preimage_small_bits::<i8, _>(cs, input)?;
  let mut eq = NoBatchEq::<i8, i32, _>::new(cs);
  for _ in 0..chain_length {
    current_bits = small_sha256_int::<i8, _>(&mut eq, &current_bits)?;
  }
  Ok(current_bits)
}

impl<E: Engine> SpartanCircuit<E> for SmallSha256ChainCircuit<E::Scalar>
where
  E::Scalar: PrimeFieldBits,
{
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    Ok(bytes_to_public_scalars(&self.expected_output()))
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    let mut small_cs = SmallToBellpepperCS::<E::Scalar, CS>::new(cs);
    let current_bits = synthesize_sha256_chain_bits(&mut small_cs, &self.input, self.chain_length)?;

    #[cfg(debug_assertions)]
    super::assert_small_bits_match_bytes(&current_bits, &self.expected_output());

    expose_small_hash_bits_as_public::<i8, _>(&mut small_cs, &current_bits)?;

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<E::Scalar>],
    _: &[AllocatedNum<E::Scalar>],
    _: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

impl<Scalar: PrimeField + PrimeFieldBits> Circuit<Scalar> for SmallSha256ChainCircuit<Scalar> {
  fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let mut small_cs = SmallToBellpepperCS::<Scalar, CS>::new(cs);
    let _ = synthesize_sha256_chain_bits(&mut small_cs, &self.input, self.chain_length)?;
    Ok(())
  }
}
