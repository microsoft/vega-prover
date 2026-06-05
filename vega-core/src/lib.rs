// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Shared proving infrastructure for Vega's split-R1CS proof systems.
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

// shared internal modules used by the branded proof-system crates
#[doc(hidden)]
#[allow(missing_docs)]
pub mod digest;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod math;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod nifs;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod r1cs;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod spartan_relaxed;

#[macro_use]
mod macros;

// public modules
pub mod bellpepper;
pub mod errors;
pub mod provider;
pub mod traits;

// shared internal modules used by the branded proof-system crates
#[doc(hidden)]
#[allow(missing_docs)]
pub mod big_num;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod polys;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod sumcheck;

/// Start a span + timer, return `(Span, Instant)`.
#[macro_export]
macro_rules! start_span {
    ($name:expr $(, $($fmt:tt)+)?) => {{
        let span       = tracing::info_span!($name $(, $($fmt)+)?);
        let span_clone = span.clone();    // lives as long as the guard
        let _guard      = span_clone.enter();
        (span, std::time::Instant::now())
    }};
}

/// The default width used for monolithic commitments.
pub const DEFAULT_COMMITMENT_WIDTH: usize = 2048;

use traits::{Engine, pcs::PCSEngineTrait};
/// Commitment key for an engine's polynomial commitment scheme.
pub type CommitmentKey<E> = <<E as traits::Engine>::PCS as PCSEngineTrait<E>>::CommitmentKey;
/// Verifier key for an engine's polynomial commitment scheme.
pub type VerifierKey<E> = <<E as traits::Engine>::PCS as PCSEngineTrait<E>>::VerifierKey;
/// Commitment for an engine's polynomial commitment scheme.
pub type Commitment<E> = <<E as Engine>::PCS as PCSEngineTrait<E>>::Commitment;
/// Polynomial commitment scheme selected by an engine.
pub type PCS<E> = <E as Engine>::PCS;
/// Blinding value for an engine's polynomial commitment scheme.
pub type Blind<E> = <<E as Engine>::PCS as PCSEngineTrait<E>>::Blind;
