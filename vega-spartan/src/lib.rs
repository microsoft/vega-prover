// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Vega-branded Spartan proof systems.
//!
//! This crate exposes the Spartan variants used by Vega for split R1CS
//! circuits with a precomputable/online witness split.

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

/// Non-zero-knowledge Spartan over split R1CS.
pub mod spartan;
/// Zero-knowledge Spartan over split R1CS.
pub mod spartan_zk;

mod zk;
mod zk_sumcheck;

pub use spartan::{SpartanPrepSNARK, SpartanSNARK};
pub use spartan_zk::{SpartanPrepZkSNARK, SpartanZkSNARK};
pub use vega_core::{bellpepper, errors, provider, traits};
