# DNSPIR

Reference implementation of the PIR scheme and DNS protocol from "DNSPIR: Private Information Retrieval Optimized for Privacy-Preserving DNS Lookups". Lea Nuernberger, Simon Pohmann, Mattia Veroni, Christian Weinert. 2026. To appear in PETS'27.

This repository has been submitted for the artifact evaluation of PETS. See [the artifact appendix](/ARTIFACT-APPENDIX.md).

## Repository structure

A Cargo workspace with two crates:

| path                  | crate          | contents                                                                                                                                                                 |
|-----------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `crates/pir`          | `dnspir-pir`   | The PIR protocol itself, and nothing else. BFV arithmetic, the AVX-512 inner-product kernel, the preprocessed single-shard database, the two-level composition, and the byte-level wire format (`pir_wrapper`). Knows nothing about DNS. |
| `crates/dns-over-pir` | `dns-over-pir` | Everything else: the mapping of DNS names onto PIR slots, the zone-file parser, the TCP protocol, the authoritative server, the iterative resolver, and the CLI.            |

The dependency direction is strictly `dns-over-pir` → `dnspir-pir`. The
measurements reported in the paper concern `crates/pir`; the DNS crate
exists to show that the scheme composes into a working resolver.

## Building

Requires a nightly Rust toolchain — the exact version is pinned in
`rust-toolchain.toml` and `rustup` will fetch it automatically — and an
x86-64 CPU with AVX-512 (`avx512f`, `avx512dq`, `avx512bw`, `avx512vl`).

```sh
cargo build --release
```

Without AVX-512, build with `--features emulate_avx512` to substitute a
portable AVX2 emulation of the kernel. This is meant for checking
correctness on other machines; the numbers it produces are not meaningful.

## Testing

```sh
cargo test --workspace                    # unit tests of both crates
cargo test --workspace -- --ignored       # plus the slow one (see below)
```

The default run includes an end-to-end check over the bundled `root.txt`.
One test is `#[ignore]`d because it is slow: an exhaustive check of the
SIMD ring arithmetic in `crates/pir`.

## Benchmarking

The `bench` sub-command runs the PIR engine end to end — preprocessing, a
fixed list of queries, and a decryption check of every answer — with no DNS
layer involved. It reports the upload split into Galois keys and query
ciphertexts, the response size, and the mean/standard deviation of the
server's response time, broken down into the primary and secondary phase.

```sh
cargo run --release -- bench [--db-entries N] [--preprocessed-db-count N] [--force-full-dbs]
```

* `--db-entries N` — the number of database entries (each entry is 2560 B
  of payload) to advertise. This alone determines the database shape:
  the shard count, the entries per shard, and whether the conjugated query
  ciphertexts have to go on the wire. Default: `N² = 4194304`, i.e. one
  fully loaded fleet.
* `--preprocessed-db-count N` — actually preprocess only the first `N`
  shards and reuse them for the remaining ones. Response times are
  unaffected, but the memory footprint is, so this is what makes large
  entry counts measurable on a machine that cannot hold the full fleet.
  A full-size shard needs ~768 MB, a half-size shard ~384 MB.
* `--force-full-dbs` — use full-size shards even at entry counts where the
  protocol would pick the half-size layout. For measuring the trade-off
  between the two; not a configuration the protocol ever uses.

A run that fits comfortably in a few GB of RAM:

```sh
cargo run --release -- bench --db-entries 32768 --preprocessed-db-count 2
```

Reserve huge pages (`vm.nr_hugepages`) before benchmarking. The engine
falls back to regular pages, but the inner-product loop touches every byte
of the database on every query, so the TLB pressure is measurable.

## Running the resolver

Serve a zone, then resolve a name against it:

```sh
cargo run --release -- server [--port PORT] [--always-dump] <zone_file>
cargo run --release -- client [--server HOST[:PORT]] <domain_name>
```

The server picks its mode from the zone size: zones whose gzipped form fits
in 100 KB are served as a plain compressed dump (there is nothing to hide
if the client can hold the entire zone), larger ones are bucketed into a
PIR database. The bundled `root.txt` is just over that threshold, so it
exercises the PIR path:

```sh
cargo run --release -- server root.txt &
cargo run --release -- client --server 127.0.0.1:9000 a.root-servers.net
```

`root.txt` is the published root zone. If you want to test multi-hop
resolution scenarios, the easiest way would be to manually add entries like
```text
org.			172800	IN	NS	ns.test.org.
ns.test.org.	172800	IN	A	127.0.0.1
```
before running it.

## AI policy

Parts of this repository were written by an AI agent.

The split is by functionality rather than by line. Every idea in the code
below is attributed; a line-by-line accounting is neither claimed nor
implied, because generated and handwritten code was revised against each
other over time.

**`crates/pir` — handwritten, with minor exceptions.** The PIR protocol is
the contribution the paper makes, and it was written by hand: the BFV layer
(`bfv.rs`), the 8-lane modular arithmetic and its AVX-512 kernels
(`simd_zn.rs`, `avx.rs`, `align.rs`, `permute.rs`), the NTT, the layout and
preprocessing of a database shard and the Galois sub-group machinery
(`base_pir.rs`), the composition of shards into a two-level retrieval and
its wire format (`double_pir.rs`, `pir_wrapper.rs`), and every
cryptographic parameter. On the other hand, the database-shape rules
(`get_database_shape()`), potentially skipping the conjugated query
ciphertext, and the thread-local reuse of the secondary database buffer are
generated. Furthermore, the benchmark harness (`bench_wrapped_pir`), several
of the crate's tests, and most of the documentation comments are also generated.

**`crates/dns-over-pir` — generated.** The entire DNS layer is the agent's
work: the mapping of names onto PIR slots, the bucket format and the
discriminator scheme with its per-bucket salt search, the balls-into-bins
sizing bound, the zone parser and its memory optimisations, the wire
protocol, the server, the iterative resolver, the CLI, and the tests. The
design brief for it was a prose description; the realisation is generated.
`CLAUDE.md` and this README are generated too.

Everything generated was read and reviewed before being accepted, but it
was not written line by line by a human.

**The setup.** The agent is Anthropic's *Claude Code* (using the Opus 4.8
and Opus 5 models), run in its hosted mode via `claude.ai/code`, which starts
each task in a sandboxed container with a fresh clone of the repository. Tasks
were given as prose descriptions and reviewed before being accepted.

# Security

The PIR protocol implements advanced cryptography which has not undergone a security
review or any form of hardening. It is therefore **NOT SECURE FOR USE IN PRODUCTION**.