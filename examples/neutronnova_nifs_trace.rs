// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

use std::{
  env,
  fmt::{self, Debug},
  marker::PhantomData,
  sync::{Mutex, OnceLock},
  time::Instant,
};

use bellpepper::gadgets::{sha256::sha256_compression_function, uint32::UInt32};
use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
};
use ff::Field;
use spartan2::{
  errors::SpartanError,
  neutronnova_zk::{NeutronNovaProverKey, NeutronNovaZkSNARK, SmallValueNeutronNovaNIFS},
  provider::T256HyraxEngine,
  traits::{Engine, circuit::SpartanCircuit},
};
use tracing::{Event, Subscriber, field::Visit};
use tracing_subscriber::{Layer, layer::Context, prelude::*};

type E = T256HyraxEngine;
const BLOCK_BYTES: usize = 64;
static CURRENT_CASE: OnceLock<Mutex<String>> = OnceLock::new();
static LAST_NIFS_MS: OnceLock<Mutex<Option<u128>>> = OnceLock::new();

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
    let input_bits: Vec<Boolean> = self
      .block
      .iter()
      .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1u8 == 1u8))
      .enumerate()
      .map(|(i, b)| {
        AllocatedBit::alloc(cs.namespace(|| format!("block bit {i}")), Some(b)).map(Boolean::from)
      })
      .collect::<Result<Vec<_>, _>>()?;

    const IV: [u32; 8] = [
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
      0x5be0cd19,
    ];
    let current_hash: Vec<UInt32> = IV.iter().map(|&v| UInt32::constant(v)).collect();

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

#[derive(Clone, Debug)]
struct CoreCircuit<Eng: Engine> {
  _p: PhantomData<Eng>,
}

impl<Eng: Engine> CoreCircuit<Eng> {
  fn new() -> Self {
    Self { _p: PhantomData }
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
    let input_bits: Vec<Boolean> = [0u8; BLOCK_BYTES]
      .iter()
      .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1u8 == 1u8))
      .enumerate()
      .map(|(i, b)| {
        AllocatedBit::alloc(cs.namespace(|| format!("core block bit {i}")), Some(b))
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

#[derive(Clone, Copy)]
enum Backend {
  RegularField,
  RegularSmall,
  SmallValueL0_3,
  SmallValueI32L0_3,
}

impl Backend {
  fn label(self) -> &'static str {
    match self {
      Backend::RegularField => "regular-field",
      Backend::RegularSmall => "regular-small-round0",
      Backend::SmallValueL0_3 => "small-value-l0-3",
      Backend::SmallValueI32L0_3 => "small-value-i32-l0-3",
    }
  }
}

struct NifsOnlyLayer;

struct NifsVisitor {
  message: Option<String>,
  elapsed_ms: Option<String>,
}

impl Visit for NifsVisitor {
  fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn Debug) {
    let value = format!("{value:?}");
    match field.name() {
      "message" => self.message = Some(value.trim_matches('"').to_string()),
      "elapsed_ms" => self.elapsed_ms = Some(value.trim_matches('"').to_string()),
      _ => {}
    }
  }

  fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
    if field.name() == "message" {
      self.message = Some(value.to_string());
    }
  }

  fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
    if field.name() == "elapsed_ms" {
      self.elapsed_ms = Some(value.to_string());
    }
  }

  fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
    if field.name() == "elapsed_ms" {
      self.elapsed_ms = Some(value.to_string());
    }
  }

  fn record_u128(&mut self, field: &tracing::field::Field, value: u128) {
    if field.name() == "elapsed_ms" {
      self.elapsed_ms = Some(value.to_string());
    }
  }

  fn record_i128(&mut self, field: &tracing::field::Field, value: i128) {
    if field.name() == "elapsed_ms" {
      self.elapsed_ms = Some(value.to_string());
    }
  }
}

impl<S> Layer<S> for NifsOnlyLayer
where
  S: Subscriber,
{
  fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
    let mut visitor = NifsVisitor {
      message: None,
      elapsed_ms: None,
    };
    event.record(&mut visitor);
    if visitor.message.as_deref() == Some("NIFS") {
      let label = CURRENT_CASE
        .get()
        .and_then(|case| case.lock().ok().map(|case| case.clone()))
        .unwrap_or_else(|| "unknown".to_string());
      let elapsed_ms = visitor.elapsed_ms.unwrap_or_else(|| "unknown".to_string());
      if let Ok(parsed) = elapsed_ms.parse::<u128>() {
        *LAST_NIFS_MS
          .get_or_init(|| Mutex::new(None))
          .lock()
          .expect("nifs mutex poisoned") = Some(parsed);
      }
      eprintln!("nifs_trace_nifs {label} nifs_ms={elapsed_ms}",);
    }
  }
}

fn parse_csv_usize(var: &str, default: &str) -> Vec<usize> {
  env::var(var)
    .unwrap_or_else(|_| default.to_string())
    .split(',')
    .filter_map(|value| value.trim().parse().ok())
    .collect()
}

fn make_step_circuits(num_steps: usize) -> Vec<Sha256StepCircuit<E>> {
  (0..num_steps)
    .map(|i| Sha256StepCircuit::<E>::new([i as u8; BLOCK_BYTES]))
    .collect()
}

fn run_regular(
  pk: &NeutronNovaProverKey<E>,
  step_circuits: &[Sha256StepCircuit<E>],
  core_circuit: &CoreCircuit<E>,
  nifs_input: bool,
) -> Result<(), SpartanError> {
  let prep =
    NeutronNovaZkSNARK::<E>::prep_prove::<_, _>(pk, step_circuits, core_circuit, &nifs_input)?;
  let (_proof, _prep) =
    NeutronNovaZkSNARK::<E>::prove::<_, _>(pk, step_circuits, core_circuit, prep, &nifs_input)?;
  Ok(())
}

fn run_small_value_l0_3(
  pk: &NeutronNovaProverKey<E>,
  step_circuits: &[Sha256StepCircuit<E>],
  core_circuit: &CoreCircuit<E>,
) -> Result<(), SpartanError> {
  let nifs_input = ();
  let prep = NeutronNovaZkSNARK::<E, SmallValueNeutronNovaNIFS<E, i64, 3>>::prep_prove::<_, _>(
    pk,
    step_circuits,
    core_circuit,
    &nifs_input,
  )?;
  let (_proof, _prep) = NeutronNovaZkSNARK::<E, SmallValueNeutronNovaNIFS<E, i64, 3>>::prove::<_, _>(
    pk,
    step_circuits,
    core_circuit,
    prep,
    &nifs_input,
  )?;
  Ok(())
}

fn run_small_value_i32_l0_3(
  pk: &NeutronNovaProverKey<E>,
  step_circuits: &[Sha256StepCircuit<E>],
  core_circuit: &CoreCircuit<E>,
) -> Result<(), SpartanError> {
  let nifs_input = ();
  let prep = NeutronNovaZkSNARK::<E, SmallValueNeutronNovaNIFS<E, i32, 3>>::prep_prove::<_, _>(
    pk,
    step_circuits,
    core_circuit,
    &nifs_input,
  )?;
  let (_proof, _prep) = NeutronNovaZkSNARK::<E, SmallValueNeutronNovaNIFS<E, i32, 3>>::prove::<_, _>(
    pk,
    step_circuits,
    core_circuit,
    prep,
    &nifs_input,
  )?;
  Ok(())
}

fn run_case(
  backend: Backend,
  bytes: usize,
  threads: usize,
  iter: usize,
) -> Result<(), SpartanError> {
  let num_steps = bytes / BLOCK_BYTES;
  let step_proto = Sha256StepCircuit::<E>::new([0u8; BLOCK_BYTES]);
  let core_proto = CoreCircuit::<E>::new();
  let (pk, _) = NeutronNovaZkSNARK::<E>::setup(&step_proto, &core_proto, num_steps)?;
  let step_circuits = make_step_circuits(num_steps);
  let core_circuit = CoreCircuit::<E>::new();
  let label = format!(
    "backend={} bytes={bytes} steps={num_steps} threads={threads} iter={iter}",
    backend.label()
  );
  *CURRENT_CASE
    .get_or_init(|| Mutex::new(String::new()))
    .lock()
    .expect("case mutex poisoned") = label.clone();
  *LAST_NIFS_MS
    .get_or_init(|| Mutex::new(None))
    .lock()
    .expect("nifs mutex poisoned") = None;

  let start = Instant::now();
  match backend {
    Backend::RegularField => run_regular(&pk, &step_circuits, &core_circuit, false)?,
    Backend::RegularSmall => run_regular(&pk, &step_circuits, &core_circuit, true)?,
    Backend::SmallValueL0_3 => run_small_value_l0_3(&pk, &step_circuits, &core_circuit)?,
    Backend::SmallValueI32L0_3 => run_small_value_i32_l0_3(&pk, &step_circuits, &core_circuit)?,
  }
  let nifs_ms = LAST_NIFS_MS
    .get_or_init(|| Mutex::new(None))
    .lock()
    .expect("nifs mutex poisoned")
    .map(|value| value.to_string())
    .unwrap_or_else(|| "unknown".to_string());
  eprintln!(
    "nifs_trace_result {label} nifs_ms={nifs_ms} prep_and_prove_ms={}",
    start.elapsed().as_millis()
  );
  Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  tracing_subscriber::registry()
    .with(NifsOnlyLayer)
    .try_init()?;

  let bytes_values = parse_csv_usize("NIFS_TRACE_BYTES", "1024,2048");
  let thread_values = parse_csv_usize("NIFS_TRACE_THREADS", "1,2,8");
  let iters = env::var("NIFS_TRACE_ITERS")
    .ok()
    .and_then(|value| value.parse().ok())
    .unwrap_or(3);
  let backends = [
    Backend::RegularField,
    Backend::RegularSmall,
    Backend::SmallValueL0_3,
    Backend::SmallValueI32L0_3,
  ];

  for bytes in bytes_values {
    for threads in &thread_values {
      let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(*threads)
        .build()?;
      for backend in backends {
        for iter in 0..iters {
          pool.install(|| run_case(backend, bytes, *threads, iter))?;
        }
      }
    }
  }

  Ok(())
}

impl fmt::Debug for Backend {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(self.label())
  }
}
