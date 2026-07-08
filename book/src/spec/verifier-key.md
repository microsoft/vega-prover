# Verifier key and digest

The verifier key is the public parameter set that `verify` consumes. It is produced once by setup from the step and core circuits and the canonical curve, and it is shared unchanged by the prover and the verifier. The verifier key is **not** serialized into the proof. Its role in the byte-exact contract is indirect but decisive: a 32-byte **digest** of the verifier key is the first value absorbed into the Fiat--Shamir transcript, so every later challenge depends on it. A byte-equivalent prover must therefore compute the identical digest, which requires both the identical verifier-key contents and the identical digest algorithm specified here.

## Verifier key contents

The verifier key is a structure with nine serialized fields, in this declaration order:

| Field | Type | Role |
| --- | --- | --- |
| `ck` | commitment key | Hyrax generators used to commit witness columns. |
| `vk_ee` | PCS verifier key | Hyrax evaluation-argument verifier parameters. |
| `S_step` | split R1CS shape | Constraint-system shape of the step circuit. |
| `S_core` | split R1CS shape | Constraint-system shape of the core circuit. |
| `vc_shape` | split multi-round R1CS shape | Shape of the in-circuit verifier circuit. |
| `vc_shape_regular` | R1CS shape | The verifier circuit's shape as a single-round R1CS. |
| `vc_ck` | commitment key | Generators for the verifier circuit's witness. |
| `vc_vk` | inner verifier key | Verifier key of the single-circuit relaxed proof. |
| `num_steps` | `usize` | Number of step instances (at least two). |

A tenth field caches the digest once computed; it is marked non-serialized and never contributes bytes.

The contents of these fields are fixed by setup: the same step circuit, core circuit, `num_steps`, and canonical curve produce the same verifier key. This chapter does not respecify setup; it specifies how the digest is computed from an existing verifier key.

## The digest

The digest is the SHA-256 hash of a single byte string \\(D\\) built from the verifier-key fields. The hash input is streamed field by field; there is no domain-separation tag, length prefix, or trailing marker around \\(D\\).

```text
digest = SHA-256(D)          // 32-byte output

D = bincode(ck)
 || bincode(vk_ee)
 || shape_raw(S_step)
 || shape_raw(S_core)
 || bincode(vc_shape)
 || bincode(vc_shape_regular)
 || bincode(vc_ck)
 || bincode(vc_vk)
 || bincode(num_steps)
```

This order is fixed by the verifier key's digest routine, which writes the fields in the declaration order above; a byte-equivalent prover must reproduce this exact sequence rather than rely on any automatic struct serialization. Two encodings appear:

- `bincode(v)` is the little-endian fixint encoding of [Serialization](serialization.md), applied recursively to the value's type. `bincode(num_steps)` is therefore an 8-byte little-endian `u64`. This is the raw number of step instances passed to setup; it is **not** rounded up to a power of two. The power-of-two padding used elsewhere in the protocol affects only the number of folding rounds, never this stored count.
- `shape_raw` is a compact hand-written encoding used only for the two circuit shapes. It is **not** bincode: it exists so the large constraint-system matrices hash quickly, and it is the reason the digest is a mix of two encodings rather than one uniform bincode call.

The nested layouts of the bincode-encoded fields (`ck`, `vk_ee`, `vc_shape`, `vc_shape_regular`, `vc_ck`, `vc_vk`) follow the [serialization](serialization.md) rules applied recursively to each type's fields, using the same primitive, sequence, field, and point encodings given there.

## Shape and matrix raw encoding

A split R1CS shape hashes as ten dimension counts followed by its three constraint matrices:

```text
shape_raw(S) =
     u64_le(S.num_cons)
  || u64_le(S.num_cons_unpadded)
  || u64_le(S.num_shared_unpadded)
  || u64_le(S.num_precommitted_unpadded)
  || u64_le(S.num_rest_unpadded)
  || u64_le(S.num_shared)
  || u64_le(S.num_precommitted)
  || u64_le(S.num_rest)
  || u64_le(S.num_public)
  || u64_le(S.num_challenges)
  || matrix_raw(S.A)
  || matrix_raw(S.B)
  || matrix_raw(S.C)
```

Each matrix is stored in compressed sparse row form: a `data` array of nonzero field values, a parallel `indices` array of column indices, an `indptr` array of row-start offsets, and a `cols` column count. It hashes as three lengths and the column count, followed by the three arrays:

```text
matrix_raw(M) =
     u64_le(len(M.data))
  || u64_le(len(M.indices))
  || u64_le(len(M.indptr))
  || u64_le(M.cols)
  || for d in M.data:    le32(d)      // scalar-field value, 32 little-endian bytes
  || for i in M.indices: u64_le(i)
  || for p in M.indptr:  u64_le(p)
```

The `data` values are scalar-field elements written with the same little-endian 32-byte encoding as a scalar on the wire ([Serialization](serialization.md#scalar-field-elements)); `to_repr` for the scalar field is little-endian. The `indices` and `indptr` entries are machine-word indices written as 8-byte little-endian `u64` values. All four length and count headers are 8-byte little-endian `u64` values, even though the underlying values are register-sized.

## The digest in the transcript

The 32-byte SHA-256 output is the digest. It enters the transcript as raw bytes:

```text
transcript.absorb("vk", digest)
```

The digest's transcript encoding is its 32 bytes in order, with no reversal and no field reduction: it is hashed as an opaque byte string, not as a field element. The exact position of this absorb in the schedule is specified in [The transcript schedule](transcript-schedule.md).
