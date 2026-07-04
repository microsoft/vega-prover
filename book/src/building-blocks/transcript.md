# The Fiat–Shamir transcript

The Vega transcript is the byte-level mechanism that turns prover messages into Fiat--Shamir challenges. For the canonical engine described in [Fields, groups, and the engine](./fields-and-groups.md), the transcript is based on Keccak256 and produces challenges in the scalar field \\(\mathbb{F}\\).

The transcript mechanism has four operations: `new`, `absorb`, `squeeze`, and `dom_sep`. This chapter specifies those operations byte-for-byte. It does not specify the protocol-level order in which \\(\mathrm{Vega}\_{\mathrm{MC}}\\) absorbs proof objects and squeezes challenges; that ordered schedule is fixed in [The transcript schedule](../spec/transcript-schedule.md). It also does not define how scalars, points, or vectors become bytes; those encodings are specified in [Byte encodings and serialization](./encodings.md).

## State and constants

A transcript instance has three pieces of internal state:

- `state`: a 64-byte array. This carries entropy from one challenge to the next.
- `round`: a `u16` counter. It starts at `0` and increments by one on every `squeeze`.
- `transcript`: a running Keccak256 hasher. It accumulates absorbed bytes since the last `squeeze`, and is reset to an empty Keccak256 hasher immediately after each `squeeze`.

The fixed byte constants are:

- `PERSONA_TAG = "NoTR"`, four ASCII bytes, used once when a transcript is created;
- `DOM_SEP_TAG = "NoDS"`, four ASCII bytes, used by `squeeze` and `dom_sep`;
- two one-byte challenge-half suffixes, `0x00` and `0x01`;
- a transcript state size of 64 bytes.

Keccak256 here means the `sha3` crate's Keccak256 function, with a 32-byte digest.

## The state-update primitive

All 64-byte transcript states and challenge preimages are produced by the same primitive. Given a Keccak256 hasher `K` and an input byte string `input`, `compute_updated_state(K, input)` performs the following steps:

1. update `K` with `input`;
2. clone the resulting hasher into two copies;
3. update the first copy with `0x00` and finalize it to obtain a 32-byte `lo` digest;
4. update the second copy with `0x01` and finalize it to obtain a 32-byte `hi` digest;
5. return `lo ‖ hi`, a 64-byte string.

Equivalently, if `K` already contains some byte prefix, the returned halves are:

```text
lo = Keccak256(K_prefix ‖ input ‖ 0x00)
hi = Keccak256(K_prefix ‖ input ‖ 0x01)
output = lo ‖ hi
```

The notation `K_prefix` denotes the bytes already accumulated in the hasher before `input` is appended.

## `new(label)`

`new(label)` initializes a fresh transcript under a static domain label.

The initial 64-byte state is computed from an empty Keccak256 hasher and the byte string `"NoTR" ‖ label`:

```text
state = compute_updated_state(empty Keccak256, "NoTR" ‖ label)
round = 0
transcript = empty Keccak256
```

The `PERSONA_TAG` is used only in this initialization step. The running `transcript` hasher starts empty; the initial label affects `state`, not the running absorbed-byte buffer.

## `absorb(label, value)`

`absorb(label, value)` appends bytes to the running hash:

```text
transcript.update(label)
transcript.update(value.to_transcript_bytes())
```

There is no length framing, no delimiter, and no implicit separator between the label and the encoded value. Successive absorbs append to the same running Keccak256 hasher until the next `squeeze`. The byte representation of `value` is the transcript encoding defined in [Byte encodings and serialization](./encodings.md).

`absorb` does not change `state` or `round` directly. Its bytes affect the next `squeeze`, because that operation clones and extends the running `transcript` hasher.

## `squeeze(label)`

`squeeze(label)` derives the next challenge scalar and rolls the transcript state forward.

First it builds the squeeze input:

```text
squeeze input = "NoDS" ‖ round(2 B, little-endian) ‖ state(64 B) ‖ label
```

The 2-byte round field is the current `u16` round counter in little-endian order. The 64-byte state is the current transcript `state` before this squeeze.

Next it computes a 64-byte output by applying the state-update primitive to a clone of the running `transcript` hasher:

```text
output = compute_updated_state(transcript clone, squeeze input)
```

Because the cloned hasher already contains all bytes absorbed since the previous squeeze, the two output halves are exactly:

```text
challenge half_b = Keccak256(
    absorbed_bytes_since_last_squeeze
    ‖ "NoDS"
    ‖ round(2 B, little-endian)
    ‖ state(64 B)
    ‖ label
    ‖ b
)   for b in {0x00, 0x01}

challenge output = half_0x00 ‖ half_0x01
```

After computing `output`, the transcript updates its internal state:

```text
round = round + 1
state = output
transcript = empty Keccak256
```

The returned challenge is `from_uniform(output)`: the 64-byte `output` interpreted as uniform input for a scalar in \\(\mathbb{F}\\). The scalar conversion is specified in [Byte encodings and serialization](./encodings.md).

## `dom_sep(bytes)`

`dom_sep(bytes)` folds a static domain separator into the running hash:

```text
transcript.update("NoDS")
transcript.update(bytes)
```

It does not squeeze a challenge, does not change `state`, and does not increment `round`. Its effect is to namespace later absorbed bytes in the same running hasher. For example, a sub-protocol can open by calling `dom_sep` before its own absorbs, so those absorbs are separated from the surrounding transcript context.

## Determinism and verifier reconstruction

The transcript is the sole source of Fiat--Shamir challenges. Two runs that execute the same transcript operations with the same byte strings, in the same order, produce the same sequence of challenge scalars. Any difference in label bytes, encoded value bytes, domain separators, or squeeze positions changes the subsequent transcript state.

The verifier reconstructs the transcript by replaying the same public labels and proof bytes, then re-derives each challenge with `squeeze`. This binds every challenge to the bytes that preceded it in the transcript. The complete \\(\mathrm{Vega}\_{\mathrm{MC}}\\) operation order is therefore part of the protocol specification and is given in [The transcript schedule](../spec/transcript-schedule.md).

## A small mechanism example

Consider a transcript created with a generic label:

```text
T = new("example transcript")
```

Initialization sets:

```text
state = compute_updated_state(empty Keccak256, "NoTR" ‖ "example transcript")
round = 0
transcript = empty Keccak256
```

Suppose two scalar values `a` and `b` are then absorbed under generic labels:

```text
absorb("a", a)
absorb("b", b)
```

The running hasher now contains the plain concatenation:

```text
"a" ‖ a.to_transcript_bytes() ‖ "b" ‖ b.to_transcript_bytes()
```

No state update has occurred yet. A subsequent challenge request

```text
squeeze("challenge")
```

uses that absorbed byte string as the Keccak256 prefix, then appends:

```text
"NoDS" ‖ 0x0000 ‖ state(64 B) ‖ "challenge" ‖ 0x00
"NoDS" ‖ 0x0000 ‖ state(64 B) ‖ "challenge" ‖ 0x01
```

to form the low and high halves of the 64-byte output. The transcript then sets `state` to that output, increments `round` to `1`, resets the running hasher to empty, and returns `from_uniform(output)` as the challenge scalar.
