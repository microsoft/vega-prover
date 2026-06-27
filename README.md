# vega-prover: Low-latency client-side ZK proving over signed data

This repository implements the ZK provers of [Vega](https://eprint.iacr.org/2025/2094). They are used for client-side ZK proving of statements over signed data. We focus on optimizing low proving latency, and on settings where statements are proven repeatedly over the same signed data.

## Running benchmarks

> [!IMPORTANT]
> We optimize for low ZK proving latency on the signed messages typical in practice — not for raw throughput on artificial, high-volume workloads.

The `benches/` directory contains SHA-256 benchmarks using [Criterion](https://github.com/bheisler/criterion.rs). Each benchmark measures setup, prep_prove, prove, and verify times across multiple iterations and thread counts, and reports proof sizes.

```bash
# Single-circuit (SC) prover: SHA-256 over 1 KiB and 2 KiB messages
RUSTFLAGS="-C target-cpu=native" cargo bench --bench sha256_vega_sc

# Multi-circuit (MC) prover: 32 SHA-256 step circuits (2048 bytes total)
RUSTFLAGS="-C target-cpu=native" cargo bench --bench sha256_vega_mc_zkp
```

Override thread counts with `BENCH_THREADS` (comma-separated):

```bash
BENCH_THREADS=1,8 RUSTFLAGS="-C target-cpu=native" cargo bench --bench sha256_vega_sc
```

## References

[Vega: Low-latency zero-knowledge proofs over existing credentials](https://eprint.iacr.org/2025/2094) \
Darya Kaviani, Srinath Setty \
IEEE S&P 2026

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
