use bellpepper::gadgets::{sha256::sha256_compression_function, uint32::UInt32};
use bellpepper_core::test_cs::TestConstraintSystem;
use bellpepper_core::{
  ConstraintSystem,
  boolean::{AllocatedBit, Boolean},
};
use halo2curves::t256::Fq as Scalar;

#[test]
fn count_sha256() {
  let mut cs = TestConstraintSystem::<Scalar>::new();
  let mut input_bits = vec![];
  for i in 0..512 {
    input_bits.push(Boolean::Is(
      AllocatedBit::alloc(cs.namespace(|| format!("bit{}", i)), Some(false)).unwrap(),
    ));
  }

  let mut current_hash = vec![];
  for i in 0..8 {
    current_hash.push(UInt32::alloc(cs.namespace(|| format!("h{}", i)), Some(0)).unwrap());
  }

  sha256_compression_function(&mut cs, &input_bits, &current_hash).unwrap();
  println!("SHA256 compression constraints: {}", cs.num_constraints());
}
