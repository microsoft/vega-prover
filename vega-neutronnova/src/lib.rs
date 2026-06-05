// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Vega-branded NeutronNova proof system.
//!
//! This crate exposes the NeutronNova folding-based optimization used by Vega
//! for repeated executions over split R1CS instances.

#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![deny(unsafe_code)]

/// NeutronNova folding proof system.
pub mod neutronnova_zk;

mod zk;
mod zk_sumcheck;

pub use neutronnova_zk::{
  NeutronNovaNIFS, NeutronNovaPrepZkSNARK, NeutronNovaProverKey, NeutronNovaVerifierKey,
  NeutronNovaZkSNARK,
};
pub use vega_core::{bellpepper, errors, provider, traits};
