# Bibliography

The proof system documented here builds on a line of work on transparent zkSNARKs and folding schemes. These are its primary references; each entry notes where the construction appears in the book.

**Vega.** Darya Kaviani and Srinath Setty. *Vega: Low-Latency Zero-Knowledge Proofs over Existing Credentials.* Cryptology ePrint Archive, Paper 2025/2094. <https://eprint.iacr.org/2025/2094>

The system this book specifies; \\(\mathrm{Vega}\_{\mathrm{MC}}\\) is its multi-circuit prover.

**Spartan.** Srinath Setty. *Spartan: Efficient and general-purpose zkSNARKs without trusted setup.* CRYPTO 2020. Cryptology ePrint Archive, Paper 2019/550. <https://eprint.iacr.org/2019/550>

The R1CS argument that reduces constraint satisfaction to an outer and inner sum-check and a committed-witness opening. See [The Spartan argument](../building-blocks/spartan.md).

**Nova.** Abhiram Kothapalli, Srinath Setty, and Ioanna Tzialla. *Nova: Recursive Zero-Knowledge Arguments from Folding Schemes.* CRYPTO 2022. Cryptology ePrint Archive, Paper 2021/370. <https://eprint.iacr.org/2021/370>

The folding scheme whose blinding property underlies the zero-knowledge fold of the verifier-circuit instance. See [Nova folding for zero-knowledge](../building-blocks/nova-zk.md).

**HyperNova.** Abhiram Kothapalli and Srinath Setty. *HyperNova: Recursive arguments for customizable constraint systems.* CRYPTO 2024. Cryptology ePrint Archive, Paper 2023/573. <https://eprint.iacr.org/2023/573>

Recursive arguments for customizable constraint systems, and the sum-check-based folding lineage this system draws on.

**NeutronNova.** Abhiram Kothapalli and Srinath Setty. *NeutronNova: Folding everything that reduces to zero-check.* Cryptology ePrint Archive, Paper 2024/1606. <https://eprint.iacr.org/2024/1606>

The zero-check folding scheme that accumulates the many step instances into one. See [NeutronNova folding](../building-blocks/neutronnova.md).
