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
//! Override backends with `BENCH_L0=field,0,1,2`; `0` is the regular small round-0 baseline.

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
  errors::SpartanError,
  neutronnova_zk::{
    NeutronNovaNIFS, NeutronNovaNifsStrategy, NeutronNovaPrepZkSNARK, NeutronNovaProverKey,
    NeutronNovaStepMatvecs, NeutronNovaVerifierKey, NeutronNovaZkSNARK, SmallValueNeutronNovaNIFS,
    SmallValueNeutronNovaStepMatvecs,
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

#[derive(Clone, Copy, Debug)]
enum BenchBackend {
  /// Full-field regular NeutronNova, with no small-value optimization.
  RegularField,
  /// Old regular NeutronNova optimized path: small witness synthesis and round-0 i64 path.
  RegularSmallRound0,
  /// Small-value accumulator backend with runtime l0.
  SmallValue { l0: usize },
}

impl BenchBackend {
  fn label(self) -> String {
    match self {
      BenchBackend::RegularField => "regular-field".to_string(),
      BenchBackend::RegularSmallRound0 => "regular-small-round0".to_string(),
      BenchBackend::SmallValue { l0 } => format!("small-value-l0-{l0}"),
    }
  }
}

/// NIFS backends to benchmark. `0` preserves the old regular optimized baseline;
/// `field` selects the regular full-field backend; positive values select the
/// small-value accumulator with runtime `l0 = value`.
fn bench_backends() -> Vec<BenchBackend> {
  std::env::var("BENCH_L0")
    .ok()
    .map(|val| {
      let values = val
        .split(',')
        .map(|raw| {
          let s = raw.trim();
          if s.is_empty() {
            panic!("invalid BENCH_L0 value: empty entry in {val:?}");
          }
          if matches!(s, "field" | "regular-field" | "full-field") {
            return BenchBackend::RegularField;
          }
          let l0 = s.parse::<usize>().unwrap_or_else(|_| {
            panic!("invalid BENCH_L0 value: {s:?}; supported values are field,0,1,2,3,4,5")
          });
          if l0 > 5 {
            panic!("unsupported BENCH_L0={l0}; supported values are field,0,1,2,3,4,5");
          }
          if l0 == 0 {
            BenchBackend::RegularSmallRound0
          } else {
            BenchBackend::SmallValue { l0 }
          }
        })
        .collect::<Vec<_>>();
      if values.is_empty() {
        vec![BenchBackend::RegularSmallRound0]
      } else {
        values
      }
    })
    .unwrap_or_else(|| vec![BenchBackend::RegularSmallRound0])
}

fn validate_backend_for_case(backend: BenchBackend, size: usize, num_steps: usize) {
  if let BenchBackend::SmallValue { l0 } = backend {
    let ell_b = num_steps.next_power_of_two().ilog2() as usize;
    if l0 > ell_b {
      panic!("BENCH_L0={l0} exceeds ell_b={ell_b} for size={size}B num_steps={num_steps}");
    }
  }
}

fn prove_with_backend<Nifs>(
  pk: &NeutronNovaProverKey<E>,
  step_circuits: &[Sha256StepCircuit<E>],
  core_circuit: &CoreCircuit<E>,
  prep: NeutronNovaPrepZkSNARK<E, Nifs::StepMatvecs, Nifs::Input>,
  nifs_input: &Nifs::Input,
) -> Result<
  (
    NeutronNovaZkSNARK<E>,
    NeutronNovaPrepZkSNARK<E, Nifs::StepMatvecs, Nifs::Input>,
  ),
  SpartanError,
>
where
  Nifs: NeutronNovaNifsStrategy<E>,
{
  NeutronNovaZkSNARK::<E>::prove::<Sha256StepCircuit<E>, CoreCircuit<E>, Nifs>(
    pk,
    step_circuits,
    core_circuit,
    prep,
    nifs_input,
  )
}

fn prep_and_prove_with_backend<Nifs>(
  pk: &NeutronNovaProverKey<E>,
  step_circuits: &[Sha256StepCircuit<E>],
  core_circuit: &CoreCircuit<E>,
  nifs_input: &Nifs::Input,
) -> Result<NeutronNovaZkSNARK<E>, SpartanError>
where
  Nifs: NeutronNovaNifsStrategy<E>,
{
  let prep = NeutronNovaZkSNARK::<E>::prep_prove::<Sha256StepCircuit<E>, CoreCircuit<E>, Nifs>(
    pk,
    step_circuits,
    core_circuit,
    nifs_input,
  )?;
  let (proof, _) = prove_with_backend::<Nifs>(pk, step_circuits, core_circuit, prep, nifs_input)?;
  Ok(proof)
}

type RegularPrep = NeutronNovaPrepZkSNARK<E, NeutronNovaStepMatvecs<E>, bool>;
type SmallValuePrep = NeutronNovaPrepZkSNARK<E, SmallValueNeutronNovaStepMatvecs<E>, usize>;

enum PreparedBenchCase {
  Regular {
    pk: NeutronNovaProverKey<E>,
    vk: NeutronNovaVerifierKey<E>,
    step_circuits: Vec<Sha256StepCircuit<E>>,
    core_circuit: CoreCircuit<E>,
    prep: RegularPrep,
    nifs_input: bool,
  },
  SmallValue {
    pk: NeutronNovaProverKey<E>,
    vk: NeutronNovaVerifierKey<E>,
    step_circuits: Vec<Sha256StepCircuit<E>>,
    core_circuit: CoreCircuit<E>,
    prep: SmallValuePrep,
    l0: usize,
  },
}

fn prove_prepared(
  case: PreparedBenchCase,
) -> Result<(NeutronNovaZkSNARK<E>, NeutronNovaVerifierKey<E>), SpartanError> {
  match case {
    PreparedBenchCase::Regular {
      pk,
      vk,
      step_circuits,
      core_circuit,
      prep,
      nifs_input,
    } => {
      let (proof, _) = prove_with_backend::<NeutronNovaNIFS<E>>(
        &pk,
        &step_circuits,
        &core_circuit,
        prep,
        &nifs_input,
      )?;
      Ok((proof, vk))
    }
    PreparedBenchCase::SmallValue {
      pk,
      vk,
      step_circuits,
      core_circuit,
      prep,
      l0,
    } => {
      let (proof, _) = prove_with_backend::<SmallValueNeutronNovaNIFS<E>>(
        &pk,
        &step_circuits,
        &core_circuit,
        prep,
        &l0,
      )?;
      Ok((proof, vk))
    }
  }
}

fn make_step_circuits(num_steps: usize) -> Vec<Sha256StepCircuit<E>> {
  (0..num_steps)
    .map(|i| Sha256StepCircuit::<E>::new([i as u8; BLOCK_BYTES]))
    .collect()
}

fn prepare_case(num_steps: usize, backend: BenchBackend) -> PreparedBenchCase {
  let step_proto = Sha256StepCircuit::<E>::new([0u8; BLOCK_BYTES]);
  let core_proto = CoreCircuit::<E>::new();
  let (pk, vk) = NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps).unwrap();
  let step_circuits = make_step_circuits(num_steps);
  let core_circuit = CoreCircuit::<E>::new();
  match backend {
    BenchBackend::RegularField | BenchBackend::RegularSmallRound0 => {
      let nifs_input = matches!(backend, BenchBackend::RegularSmallRound0);
      let prep = NeutronNovaZkSNARK::<E>::prep_prove::<_, _, NeutronNovaNIFS<E>>(
        &pk,
        &step_circuits,
        &core_circuit,
        &nifs_input,
      )
      .unwrap();
      PreparedBenchCase::Regular {
        pk,
        vk,
        step_circuits,
        core_circuit,
        prep,
        nifs_input,
      }
    }
    BenchBackend::SmallValue { l0 } => {
      let prep = NeutronNovaZkSNARK::<E>::prep_prove::<_, _, SmallValueNeutronNovaNIFS<E>>(
        &pk,
        &step_circuits,
        &core_circuit,
        &l0,
      )
      .unwrap();
      PreparedBenchCase::SmallValue {
        pk,
        vk,
        step_circuits,
        core_circuit,
        prep,
        l0,
      }
    }
  }
}

fn neutronnova_benches(c: &mut Criterion) {
  let thread_counts = thread_counts();
  let backends = bench_backends();
  for &backend in &backends {
    for &size in SIZES {
      validate_backend_for_case(backend, size, num_steps_for_size(size));
    }
  }

  // Report proof sizes once per size (outside measurements).
  for &backend in &backends {
    let backend_label = backend.label();
    for &size in SIZES {
      let num_steps = num_steps_for_size(size);
      let (proof, _) = prove_prepared(prepare_case(num_steps, backend)).unwrap();
      let proof_bytes = bincode::serialize(&proof).unwrap();
      println!(
        "NeutronNova SHA-256 backend={} size={}B num_steps={} (= {} compressions): proof_size={} bytes",
        backend_label,
        size,
        num_steps,
        num_steps,
        proof_bytes.len()
      );
    }
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

      for &backend in &backends {
        let backend_label = backend.label();

        g.bench_function(
          format!("prep_prove/{backend_label}/{size}/t{nthreads}"),
          |b| {
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
                pool.install(|| {
                  let step_circuits = make_step_circuits(num_steps);
                  let core_circuit = CoreCircuit::<E>::new();
                  match backend {
                    BenchBackend::RegularField | BenchBackend::RegularSmallRound0 => {
                      let nifs_input = matches!(backend, BenchBackend::RegularSmallRound0);
                      let _ = NeutronNovaZkSNARK::<E>::prep_prove::<_, _, NeutronNovaNIFS<E>>(
                        &pk,
                        &step_circuits,
                        &core_circuit,
                        &nifs_input,
                      )
                      .unwrap();
                    }
                    BenchBackend::SmallValue { l0 } => {
                      let _ = NeutronNovaZkSNARK::<E>::prep_prove::<
                        _,
                        _,
                        SmallValueNeutronNovaNIFS<E>,
                      >(&pk, &step_circuits, &core_circuit, &l0)
                      .unwrap();
                    }
                  }
                });
              },
              BatchSize::LargeInput,
            );
          },
        );

        g.bench_function(format!("prove/{backend_label}/{size}/t{nthreads}"), |b| {
          b.iter_batched(
            || pool.install(|| prepare_case(num_steps, backend)),
            |case| {
              pool.install(|| {
                let _ = prove_prepared(case).unwrap();
              });
            },
            BatchSize::LargeInput,
          );
        });

        g.bench_function(format!("total/{backend_label}/{size}/t{nthreads}"), |b| {
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
              pool.install(|| match backend {
                BenchBackend::RegularField | BenchBackend::RegularSmallRound0 => {
                  let nifs_input = matches!(backend, BenchBackend::RegularSmallRound0);
                  let _ = prep_and_prove_with_backend::<NeutronNovaNIFS<E>>(
                    &pk,
                    &step_circuits,
                    &core_circuit,
                    &nifs_input,
                  )
                  .unwrap();
                }
                BenchBackend::SmallValue { l0 } => {
                  let _ = prep_and_prove_with_backend::<SmallValueNeutronNovaNIFS<E>>(
                    &pk,
                    &step_circuits,
                    &core_circuit,
                    &l0,
                  )
                  .unwrap();
                }
              });
            },
            BatchSize::LargeInput,
          );
        });

        g.bench_function(format!("verify/{backend_label}/{size}/t{nthreads}"), |b| {
          b.iter_batched(
            || {
              pool.install(|| {
                let (proof, vk) = prove_prepared(prepare_case(num_steps, backend)).unwrap();
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
    }
  }
  g.finish();
}

criterion_group!(benches, neutronnova_benches);
criterion_main!(benches);
