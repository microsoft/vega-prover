# Byte encodings and serialization

This chapter specifies the primitive byte encodings used when algebraic objects are absorbed into the Fiat--Shamir transcript, and the inverse map used when transcript output is interpreted as a scalar challenge. The algebraic objects themselves are defined in [Fields, groups, and the engine](./fields-and-groups.md). The transcript operations that consume these bytes are described in [The Fiat--Shamir transcript](./transcript.md).

These encodings are not the proof-object wire format. The proof envelope and its serialized layout are specified separately in [Serialization and encodings](../spec/serialization.md).

## Transcript representation

Every value that can be absorbed into the transcript has a canonical byte string, exposed by `to_transcript_bytes()`. Absorbing a value appends the caller-supplied label followed immediately by this byte string to the running hash state.

There is no length prefix, domain separator, or delimiter between the label and the value bytes. There is also no delimiter between successive absorbed values. Unambiguous transcript construction therefore comes from the fixed protocol schedule and fixed-width primitive encodings, not from self-describing byte strings.

This chapter specifies the primitive encodings: scalar-field elements, base-field elements, group elements, and slices of these. Aggregate objects — such as polynomials, commitments, and R1CS instances — implement their own `to_transcript_bytes` in terms of these primitives, and some add internal structure. For example, a Hyrax commitment brackets its point bytes with the literal marker strings `poly_commitment_begin` and `poly_commitment_end`, and a univariate polynomial is absorbed in a compressed form that omits its linear coefficient and uses the field backend's native byte order. Those composite encodings are specified alongside the objects that define them and in [Serialization and encodings](../spec/serialization.md).

## Fixed-width primitive encodings

In the canonical engine, scalar-field elements and base-field elements occupy 32 bytes. An uncompressed group element occupies 64 bytes.

```text
scalar in F        32 bytes    canonical integer, big-endian
base-field elem    32 bytes    canonical integer, little-endian
point in G         64 bytes    x-coordinate || y-coordinate (each a 32-byte little-endian base-field element)
slice &[T]          variable    enc(T[0]) || enc(T[1]) || ... || enc(T[n-1])
challenge scalar   64 bytes -> scalar by little-endian reduction mod p_scal
```

The scalar, base-field, point, and slice encodings in this table are transcript input encodings. Scalars and base-field elements use opposite byte orders, for the backend reason explained below. The challenge-scalar line is the transcript output decoding rule and uses little-endian byte order, as described at the end of the chapter.

## Scalar-field elements

A scalar \\(s \in \mathbb{F}\\) is encoded as its canonical integer representative in \\([0,p\_{\mathrm{scal}})\\), written as exactly 32 bytes in most-significant-byte-first order.

Equivalently:

```text
enc_scalar(s) = be32(integer representative of s)
```

The `to_transcript_bytes` implementation takes the field backend's native canonical byte representation and reverses it. The scalar field's native representation is 32 little-endian bytes, so the scalar transcript encoding is big-endian.

## Base-field elements

A base-field element is encoded by the same `to_transcript_bytes` reversal, but the base field's native byte order is the opposite of the scalar field's. The base field's native canonical representation is 32 big-endian bytes, so reversing it yields a little-endian transcript encoding. A base-field element is therefore written as exactly 32 bytes in least-significant-byte-first (little-endian) order — the opposite of the scalar encoding.

```text
enc_base(x) = le32(integer representative of x)
```

This is the coordinate encoding used inside group-element encodings.

## Group elements

A group element is encoded in uncompressed affine form. If the input point is projective, it is first converted to affine form. Its affine coordinates are then encoded as base-field elements and concatenated:

```text
enc_point(P) = enc_base(P.x) || enc_base(P.y)
```

The result is exactly 64 bytes: 32 bytes for the x-coordinate followed by 32 bytes for the y-coordinate. The affine and projective transcript encodings of the same group element are identical.

There is no compressed-point encoding. The encoding reads the affine coordinates of the point, so the identity (point at infinity), which has no affine coordinates, has no byte encoding — the implementation fails rather than emitting bytes for it. In the protocol, points absorbed into the transcript are non-identity commitments.

## Slices and vectors

A slice `&[T]` of transcript-encodable values is encoded as the in-order concatenation of each element's transcript bytes:

```text
enc_slice([v0, v1, ..., v(n-1)]) =
    enc(v0) || enc(v1) || ... || enc(v(n-1))
```

There is no length prefix and no separator between elements. A vector of \\(n\\) scalars therefore encodes to exactly `32 * n` bytes. A vector of \\(n\\) group elements encodes to exactly `64 * n` bytes.

As with individual absorbs, the protocol schedule supplies the expected lengths and types. These slice encodings are not self-delimiting.

## Challenge decoding from transcript output

Transcript challenges are produced from exactly 64 output bytes. These 64 bytes are interpreted as a little-endian integer and reduced modulo the scalar-field prime \\(p\_{\mathrm{scal}}\\), yielding a scalar in \\(\mathbb{F}\\).

```text
from_uniform(bytes[0..64]) =
    little_endian_integer(bytes[0..64]) mod p_scal
```

This decoding is a self-contained rule, not the inverse of the input encodings. Scalars are absorbed as 32 big-endian bytes and base-field coordinates as 32 little-endian bytes, whereas a challenge squeezed from the transcript is decoded from 64 little-endian bytes before reduction.

The 64-byte input is wider than the scalar field size. Reducing this wide integer modulo \\(p\_{\mathrm{scal}}\\) gives the challenge scalar with statistical uniformity appropriate for the canonical 256-bit scalar field.
