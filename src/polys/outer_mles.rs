use crate::traits::Engine;
use serde::{Deserialize, Serialize};

/// Full field-valued step-circuit Az/Bz/Cz tables.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FieldStepMLEs<E: Engine> {
  /// Field-valued Az tables in original instance order.
  pub az: Vec<Vec<E::Scalar>>,
  /// Field-valued Bz tables in original instance order.
  pub bz: Vec<Vec<E::Scalar>>,
  /// Field-valued Cz tables in original instance order.
  pub cz: Vec<Vec<E::Scalar>>,
}

impl<E: Engine> FieldStepMLEs<E> {
  pub(crate) fn from_triples(
    field_mles: Vec<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>)>,
  ) -> Self {
    let mut az = Vec::with_capacity(field_mles.len());
    let mut bz = Vec::with_capacity(field_mles.len());
    let mut cz = Vec::with_capacity(field_mles.len());
    for (az_layer, bz_layer, cz_layer) in field_mles {
      az.push(az_layer);
      bz.push(bz_layer);
      cz.push(cz_layer);
    }

    Self { az, bz, cz }
  }

  pub(crate) fn len(&self) -> usize {
    self.az.len()
  }
}

/// Small-value Az/Bz tables and the global positions that must use field corrections.
///
/// Invariant: for every index in `field_positions`, every layer's small Az/Bz
/// table stores zero at that index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmallAbStepMLEs<AbLayer> {
  /// Small Az tables; global large positions are zeroed.
  pub az_small: Vec<AbLayer>,
  /// Small Bz tables; global large positions are zeroed.
  pub bz_small: Vec<AbLayer>,
  /// Sorted positions where any cached value did not fit in the small representation.
  pub field_positions: Vec<usize>,
}

/// Small-value Az/Bz/Cz tables and the global positions that must use field corrections.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmallAbcStepMLEs<AbLayer, CLayer> {
  /// Small Az/Bz tables and their large-position metadata.
  pub ab: SmallAbStepMLEs<AbLayer>,
  /// Small Cz tables; global large positions are zeroed.
  pub cz_small: Vec<CLayer>,
  /// Sorted positions where any cached C value did not fit in the small representation.
  pub c_field_positions: Vec<usize>,
}
