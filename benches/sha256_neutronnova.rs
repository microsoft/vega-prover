// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! benches/sha256_neutronnova.rs
//! Criterion benchmarks for NeutronNova {setup, prep_prove, prove, verify}
//! on a batch of SHA-256 single-block compressions.
//!
//! Run with: `RUSTFLAGS="-C target-cpu=native" cargo bench --bench sha256_neutronnova`
//! Override thread counts with `BENCH_THREADS=1,4,8`.
//! The benchmark compares regular field, regular round-0 small, and L0=3 small-value backends.

#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use bellpepper::gadgets::{sha256::sha256_compression_function, uint32::UInt32};
use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use ff::Field;
use spartan2::{
  neutronnova_zk::{
    NeutronNovaNIFS, NeutronNovaNifsStrategy, NeutronNovaZkSNARK, SmallValueNeutronNovaNIFS,
  },
  provider::T256HyraxEngine,
  traits::{Engine, circuit::SpartanCircuit},
};
use std::{marker::PhantomData, time::Duration};

type E = T256HyraxEngine;

/// Sizes in bytes to benchmark: 1 KB (16 steps) and 2 KB (32 steps).
const SIZES: &[usize] = &[1024, 2048];

/// SHA-256 block size in bytes.
const BLOCK_BYTES: usize = 64;

type RegularNifs = NeutronNovaNIFS<E>;
type SmallI64L0_3 = SmallValueNeutronNovaNIFS<E, i64, 3>;
type SmallI32L0_3 = SmallValueNeutronNovaNIFS<E, i32, 3>;

fn num_steps_for_size(size: usize) -> usize {
  size / BLOCK_BYTES
}

/// Step circuit: proves one SHA-256 compression on a 64-byte block.
///
/// The 512 input bits are allocated as precommitted witnesses; the previous
/// hash state uses the SHA-256 IV (constants).
#[derive(Clone, Debug)]
struct Sha256StepCircuit<Eng: Engine> {
  block: [u8; BLOCK_BYTES],
  _p: PhantomData<Eng>,
}

impl<Eng: Engine> Sha256StepCircuit<Eng> {
  fn new(block: [u8; BLOCK_BYTES]) -> Self {
    Self {
      block,
      _p: PhantomData,
    }
  }
}

impl<Eng: Engine> SpartanCircuit<Eng> for Sha256StepCircuit<Eng> {
  fn public_values(&self) -> Result<Vec<Eng::Scalar>, SynthesisError> {
    Ok(vec![Eng::Scalar::ZERO])
  }

  fn shared<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<Eng::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<Eng::Scalar>],
  ) -> Result<Vec<AllocatedNum<Eng::Scalar>>, SynthesisError> {
    // Allocate 512 bits of block input as witness (big-endian per byte).
    let input_bits: Vec<Boolean> = self
      .block
      .iter()
      .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1u8 == 1u8))
      .enumerate()
      .map(|(i, b)| {
        AllocatedBit::alloc(cs.namespace(|| format!("block bit {i}")), Some(b)).map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(input_bits.len(), 512);

    // Use SHA-256 IV for the current hash value (constants).
    const IV: [u32; 8] = [
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
      0x5be0cd19,
    ];
    let current_hash: Vec<UInt32> = IV.iter().map(|&v| UInt32::constant(v)).collect();

    // One SHA-256 compression per step.
    let _next = sha256_compression_function(
      cs.namespace(|| "sha256 compression"),
      &input_bits,
      &current_hash,
    )?;

    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(Eng::Scalar::ZERO))?;
    x.inputize(cs.namespace(|| "inputize x"))?;

    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<Eng::Scalar>],
    _: &[AllocatedNum<Eng::Scalar>],
    _: Option<&[Eng::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

/// Trivial core circuit: just exposes a single public input. Preserves the
/// "N step circuits + 1 core circuit" structure that NeutronNova requires.
#[derive(Clone, Debug)]
struct CoreCircuit<Eng: Engine>(PhantomData<Eng>);

impl<Eng: Engine> CoreCircuit<Eng> {
  fn new() -> Self {
    Self(PhantomData)
  }
}

impl<Eng: Engine> SpartanCircuit<Eng> for CoreCircuit<Eng> {
  fn public_values(&self) -> Result<Vec<Eng::Scalar>, SynthesisError> {
    Ok(vec![Eng::Scalar::ZERO])
  }

  fn shared<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    _: &mut CS,
  ) -> Result<Vec<AllocatedNum<Eng::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    cs: &mut CS,
    _: &[AllocatedNum<Eng::Scalar>],
  ) -> Result<Vec<AllocatedNum<Eng::Scalar>>, SynthesisError> {
    // One SHA-256 compression, matching the step circuit's shape.
    // Keeps the core circuit under 32,768 constraints (~26,352 for one compression)
    // while still representing a meaningful per-fold "core" workload.
    let input_bits: Vec<Boolean> = (0..512)
      .map(|i| {
        AllocatedBit::alloc(cs.namespace(|| format!("core bit {i}")), Some(false))
          .map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;

    const IV: [u32; 8] = [
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
      0x5be0cd19,
    ];
    let current_hash: Vec<UInt32> = IV.iter().map(|&v| UInt32::constant(v)).collect();

    let _next = sha256_compression_function(
      cs.namespace(|| "core sha256 compression"),
      &input_bits,
      &current_hash,
    )?;

    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(Eng::Scalar::ZERO))?;
    x.inputize(cs.namespace(|| "inputize x"))?;
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<Eng::Scalar>>(
    &self,
    _: &mut CS,
    _: &[AllocatedNum<Eng::Scalar>],
    _: &[AllocatedNum<Eng::Scalar>],
    _: Option<&[Eng::Scalar]>,
  ) -> Result<(), SynthesisError> {
    Ok(())
  }
}

/// Thread counts to benchmark. Override with BENCH_THREADS env var (comma-separated).
/// Defaults are capped at the host's available parallelism to avoid oversubscription.
fn thread_counts() -> Vec<usize> {
  if let Ok(val) = std::env::var("BENCH_THREADS") {
    val
      .split(',')
      .filter_map(|s| s.trim().parse().ok())
      .collect()
  } else {
    let max = std::thread::available_parallelism()
      .map(|n| n.get())
      .unwrap_or(4);
    vec![1, 2, 4, 8, 16]
      .into_iter()
      .filter(|&t| t <= max)
      .collect()
  }
}

fn make_step_circuits(num_steps: usize) -> Vec<Sha256StepCircuit<E>> {
  (0..num_steps)
    .map(|i| Sha256StepCircuit::<E>::new([i as u8; BLOCK_BYTES]))
    .collect()
}

fn report_proof_size<Nifs>(label: &str, size: usize, num_steps: usize, nifs_input: Nifs::Input)
where
  Nifs: NeutronNovaNifsStrategy<E>,
  NeutronNovaZkSNARK<E, Nifs>: serde::Serialize,
{
  let step_proto = Sha256StepCircuit::<E>::new([0u8; BLOCK_BYTES]);
  let core_proto = CoreCircuit::<E>::new();
  let (pk, _) = NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps).unwrap();
  let step_circuits = make_step_circuits(num_steps);
  let core_circuit = CoreCircuit::<E>::new();
  let prep = NeutronNovaZkSNARK::<E, Nifs>::prep_prove::<_, _>(
    &pk,
    &step_circuits,
    &core_circuit,
    &nifs_input,
  )
  .unwrap();
  let (proof, _) = NeutronNovaZkSNARK::<E, Nifs>::prove::<_, _>(
    &pk,
    &step_circuits,
    &core_circuit,
    prep,
    &nifs_input,
  )
  .unwrap();
  let proof_bytes = bincode::serialize(&proof).unwrap().len();
  println!(
    "NeutronNova SHA-256 backend={} size={}B num_steps={} (= {} compressions): proof_size={} bytes",
    label, size, num_steps, num_steps, proof_bytes
  );
}

fn bench_backend<Nifs>(
  g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
  pool: &rayon::ThreadPool,
  size: usize,
  num_steps: usize,
  nthreads: usize,
  label: &str,
  nifs_input: Nifs::Input,
) where
  Nifs: NeutronNovaNifsStrategy<E>,
{
  let prep_input = nifs_input.clone();
  g.bench_function(format!("prep_prove/{label}/{size}/t{nthreads}"), |b| {
    b.iter_batched(
      || {
        pool.install(|| {
          let step_proto = Sha256StepCircuit::<E>::new([0u8; BLOCK_BYTES]);
          let core_proto = CoreCircuit::<E>::new();
          NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps)
            .unwrap()
            .0
        })
      },
      |pk| {
        let nifs_input = prep_input.clone();
        pool.install(|| {
          let step_circuits = make_step_circuits(num_steps);
          let core_circuit = CoreCircuit::<E>::new();
          let _ = NeutronNovaZkSNARK::<E, Nifs>::prep_prove::<_, _>(
            &pk,
            &step_circuits,
            &core_circuit,
            &nifs_input,
          )
          .unwrap();
        });
      },
      BatchSize::LargeInput,
    );
  });

  let prove_input = nifs_input.clone();
  g.bench_function(format!("prove/{label}/{size}/t{nthreads}"), |b| {
    b.iter_batched(
      || {
        let nifs_input = prove_input.clone();
        pool.install(|| {
          let step_proto = Sha256StepCircuit::<E>::new([0u8; BLOCK_BYTES]);
          let core_proto = CoreCircuit::<E>::new();
          let (pk, _vk) =
            NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps).unwrap();
          let step_circuits = make_step_circuits(num_steps);
          let core_circuit = CoreCircuit::<E>::new();
          let prep = NeutronNovaZkSNARK::<E, Nifs>::prep_prove::<_, _>(
            &pk,
            &step_circuits,
            &core_circuit,
            &nifs_input,
          )
          .unwrap();
          (pk, step_circuits, core_circuit, prep, nifs_input)
        })
      },
      |(pk, step_circuits, core_circuit, prep, nifs_input)| {
        pool.install(|| {
          let _ = NeutronNovaZkSNARK::<E, Nifs>::prove::<_, _>(
            &pk,
            &step_circuits,
            &core_circuit,
            prep,
            &nifs_input,
          )
          .unwrap();
        });
      },
      BatchSize::LargeInput,
    );
  });

  let total_input = nifs_input.clone();
  g.bench_function(format!("total/{label}/{size}/t{nthreads}"), |b| {
    b.iter_batched(
      || {
        pool.install(|| {
          let step_proto = Sha256StepCircuit::<E>::new([0u8; BLOCK_BYTES]);
          let core_proto = CoreCircuit::<E>::new();
          let (pk, _) =
            NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps).unwrap();
          let step_circuits = make_step_circuits(num_steps);
          let core_circuit = CoreCircuit::<E>::new();
          (pk, step_circuits, core_circuit)
        })
      },
      |(pk, step_circuits, core_circuit)| {
        let nifs_input = total_input.clone();
        pool.install(|| {
          let prep = NeutronNovaZkSNARK::<E, Nifs>::prep_prove::<_, _>(
            &pk,
            &step_circuits,
            &core_circuit,
            &nifs_input,
          )
          .unwrap();
          let _ = NeutronNovaZkSNARK::<E, Nifs>::prove::<_, _>(
            &pk,
            &step_circuits,
            &core_circuit,
            prep,
            &nifs_input,
          )
          .unwrap();
        });
      },
      BatchSize::LargeInput,
    );
  });

  let verify_input = nifs_input;
  g.bench_function(format!("verify/{label}/{size}/t{nthreads}"), |b| {
    b.iter_batched(
      || {
        let nifs_input = verify_input.clone();
        pool.install(|| {
          let step_proto = Sha256StepCircuit::<E>::new([0u8; BLOCK_BYTES]);
          let core_proto = CoreCircuit::<E>::new();
          let (pk, vk) =
            NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps).unwrap();
          let step_circuits = make_step_circuits(num_steps);
          let core_circuit = CoreCircuit::<E>::new();
          let prep = NeutronNovaZkSNARK::<E, Nifs>::prep_prove::<_, _>(
            &pk,
            &step_circuits,
            &core_circuit,
            &nifs_input,
          )
          .unwrap();
          let (proof, _) = NeutronNovaZkSNARK::<E, Nifs>::prove::<_, _>(
            &pk,
            &step_circuits,
            &core_circuit,
            prep,
            &nifs_input,
          )
          .unwrap();
          (vk, proof)
        })
      },
      |(vk, proof)| {
        pool.install(|| {
          proof.verify(&vk, num_steps).unwrap();
        });
      },
      BatchSize::LargeInput,
    );
  });
}

fn neutronnova_benches(c: &mut Criterion) {
  let thread_counts = thread_counts();

  // Report proof sizes once per size (outside measurements).
  for &size in SIZES {
    let num_steps = num_steps_for_size(size);
    report_proof_size::<RegularNifs>("regular-field", size, num_steps, false);
    report_proof_size::<RegularNifs>("regular-small-round0", size, num_steps, true);
    report_proof_size::<SmallI64L0_3>("small-value-i64-l0-3", size, num_steps, ());
    report_proof_size::<SmallI32L0_3>("small-value-i32-l0-3", size, num_steps, ());
  }

  let mut g = c.benchmark_group("neutronnova_sha256");
  g.sample_size(10);
  g.warm_up_time(Duration::from_millis(100));
  g.measurement_time(Duration::from_secs(10));

  for &size in SIZES {
    let num_steps = num_steps_for_size(size);
    g.throughput(Throughput::Bytes(size as u64));

    for &nthreads in &thread_counts {
      let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .expect("failed to build rayon pool");

      g.bench_function(format!("setup/{size}/t{nthreads}"), |b| {
        b.iter(|| {
          pool.install(|| {
            let step_proto = Sha256StepCircuit::<E>::new([0u8; BLOCK_BYTES]);
            let core_proto = CoreCircuit::<E>::new();
            let _ = NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps).unwrap();
          });
        });
      });

      bench_backend::<RegularNifs>(
        &mut g,
        &pool,
        size,
        num_steps,
        nthreads,
        "regular-field",
        false,
      );
      bench_backend::<RegularNifs>(
        &mut g,
        &pool,
        size,
        num_steps,
        nthreads,
        "regular-small-round0",
        true,
      );
      bench_backend::<SmallI64L0_3>(
        &mut g,
        &pool,
        size,
        num_steps,
        nthreads,
        "small-value-i64-l0-3",
        (),
      );
      bench_backend::<SmallI32L0_3>(
        &mut g,
        &pool,
        size,
        num_steps,
        nthreads,
        "small-value-i32-l0-3",
        (),
      );
    }
  }
  g.finish();
}

criterion_group!(benches, neutronnova_benches);
criterion_main!(benches);
