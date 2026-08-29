# Artifact Appendix (Required for all badges)

Paper title: "DNSPIR: Private Information Retrieval Optimized for Privacy-Preserving DNS Lookups"

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [x] **Reproduced**

## Description

This artifact contains the implementation of the PIR protocol (in `./crates/pir`) and the private DNS protocol (in `./crates/dns-over-pir`) proposed in the paper "DNSPIR: Private Information Retrieval Optimized for Privacy-Preserving DNS Lookups". Lea Nuernberger, Simon Pohmann, Mattia Veroni, Christian Weinert. 2026. To appear in PETS'27.

### Security/Privacy Issues and Ethical Concerns

The full-stack [Experiment 6](#experiment-6-functionality-of-dns-protocol) will start a server on the current machine, which (depending on security configurations) may be exposed on the local network.
There are no other security or ethical concerns with running this code.

Note also that the PIR protocol implemented in this artifact has not undergone a security review or any form of hardening. It is therefore **NOT SECURE FOR USE IN PRODUCTION**.

## Basic Requirements

### Hardware Requirements

Minimal hardware requirements:
 - a CPU with AVX-512 (preferred) or AVX2 (requires compilation with `--features=emulate_avx512`)
 - for benchmarking databases up to 11GB (as reported in the paper), 256GB of RAM are required; for experiments of reduced size, about 32GB are sufficient

The hardware used for the obtain the reported timings is
 - CPU: Intel(R) Xeon(R) Gold 5318S CPU @ 2.10GHz
 - RAM: 4 x 64GB of DDR4 3200 MHz

### Software Requirements 

We believe that any recent Linux OS and Rust compiler (nightly toolchain) will be sufficient to compile and run the artifact (subject to the above hardware requirements).

The reported benchmarks were performed on a system with the following software installed:
 - Linux (Ubuntu, update `5.15.0-176-generic`)
 - Rustup version 1.28.2
 - Rust nightly-2026-05-14 (automatically downloaded and used by rustup)
 - Rust packages bytemuck version 1, feanor-math version 3.5.18, flate2 version 1.0, memmap2 version 0.9.10, rand version 0.8.5, rand_distr version 0.4.3, rayon version 1.10.0, take_mut version 0.2.2, tracing version 0.1.41, tracing-chrome version 0.7.2, tracing-subscriber version 0.3.20 (all automatically downloaded and compiled by cargo)

### Estimated Time and Storage Consumption

Benchmarks for database sizes up to 11GB (as reported in the paper) take up to half an hour (less with multithreading) and up to 256GB of RAM. No disk space is required.

Smaller-scale experiments (cf. [Experiment 5](#experiment-5-pir-protocol-run-database-222-x-2560b-single-threaded-downscaled-version)) run within a few minutes and 8GB of RAM.

## Environment

### Accessibility

The complete artifact is hosted at [https://github.com/FeanorTheElf/dnspir.git](https://github.com/FeanorTheElf/dnspir.git).

### Set Up the Environment

To run below experiments, it is sufficient to have installed a recent version of rustup. 
Downloading and compiling dependencies and compiling and running the code will automatically be done by cargo as required.

To reproduce the timings for the large-database benchmarks, one must additionally reserve huge pages, using
```bash
sudo sysctl vm.nr_hugepages=102400
```
Note that this step is optional, and the implementation will fall back to normal memory if huge pages are not available.
The performance impact of using huge pages seems to be limited (less than 5%).

### Testing the Environment

In the directory root, run
```bash
cargo test
```
On a successful setup, all tests should pass.

Note that if your system only supports AVX2, instead run
```bash
cargo test --features=emulate_avx512
```

In either case, the expected output should look as follows.
```
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.01s
     Running unittests src/main.rs (target/debug/deps/dns_over_pir-5a8c2b1c8e34c383)

running 57 tests
test dns::bucket::tests::bucket_salt_changes_discriminator ... ok
test dns::bucket::tests::bucket_salt_roundtrip ... ok
[...]

test result: ok. 57 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.78s

     Running unittests src/lib.rs (target/debug/deps/pir-0fd3eeacb8041a84)

running 10 tests
test simd_zn::test_reduce ... ignored
test pir_wrapper::test_condense_uncondense ... ok
test base_pir::test_set_transformed_entries_small_subgroup ... ok
[...]

test result: ok. 9 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 10.82s

   Doc-tests pir

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: Performance of our PIR protocol

Our PIR protocol, as described in the paper mentioned above and implemented in `./crates/pir`, correctly realizes a PIR protocol. The protocol is reasonably efficient, and achieves communication costs and runtimes as shown in the below table. This table consists of the corresponding rows from Table 4 and Table 5 from the paper.

|     Database | Threads | Request | Response |  Time | Preproc |
|--------------|---------|---------|----------|-------|---------|
| 2^20 x 2560B |       1 |   52 KB |    22 KB | 4.5 s |   346 s |
| 2^20 x 2560B |       8 |   52 KB |    22 KB | 1.3 s |    42 s |
| 2^22 x 2560B |       1 |   60 KB |    22 KB |  17 s |  1367 s |
| 2^22 x 2560B |       8 |   60 KB |    22 KB | 4.3 s |   171 s |

### Experiments

All experiments take about 5 min of human time and up to 30 min of compute time. Disk storage is negligible in every case. 

#### Experiment 1: PIR protocol run, Database 2^20 x 2560B, single-threaded

The experiment runs our PIR protocol on a database consisting of 2^20 entries of size 2560B each, and measures preprocessing time, online time and communication cost.

To run, use
```bash
RAYON_NUM_THREADS=1 cargo run --release [--features=emulate_avx512] bench --db-entries 1048576
```

This example reproduces table row 1 from the table in [Main Result 1](#main-result-1-performance-of-our-pir-protocol).
Concretely, the output should look similar to the following.
```text
Primary databases: 64 x 16384 entries, conjugated primary query sent, conjugated secondary query omitted
Allocating 51539607552B for preprocessed database
Preparing database of 64 x 16384 x 2048 = 2147483648 elements of Z/1025Z
Preprocessed database 1/64
Preprocessed database 2/64
Preprocessed database 3/64
[...]
Preprocessed database 63/64
Preprocessed database 64/64
Preprocessing done in 385 s
Performing PIR queries
RAM bandwidth (MB/s): 15341.228108891997
PIR time: 4510 ms
Performed query 1/16
RAM bandwidth (MB/s): 15289.396468282715
PIR time: 4482 ms
Performed query 2/16
[...]
RAM bandwidth (MB/s): 15241.915513484053
PIR time: 4487 ms
Performed query 16/16
done
Communication:
  Galois keys:          25600 B
  Query (excl. keys):   26112 B
  Response:             22272 B
Response time (avg):    4493105 us
  Primary DBs (avg):    3379290 us
  Secondary DB (avg):   1110116 us
Response time stddev:   9561.81539849226
```

#### Experiment 2: PIR protocol run, Database 2^20 x 2560B, multi-threaded

The experiment runs our PIR protocol on a database consisting of 2^20 entries of size 2560B each, and measures preprocessing time, online time and communication cost.

To run, use
```bash
RAYON_NUM_THREADS=8 cargo run --release [--features=emulate_avx512] bench --db-entries 1048576
```

This example reproduces table row 2 from the table in [Main Result 1](#main-result-1-performance-of-our-pir-protocol).
The output should look similar to, and is interpreted analogously, as in [Experiment 1](#experiment-1-pir-protocol-run-database-220-x-2560b-single-threaded).

#### Experiment 3: PIR protocol run, Database 2^22 x 2560B, single-threaded

The experiment runs our PIR protocol on a database consisting of 2^22 entries of size 2560B each, and measures preprocessing time, online time and communication cost.

To run, use
```bash
RAYON_NUM_THREADS=1 cargo run --release [--features=emulate_avx512] bench --db-entries 4194304
```

This example reproduces table row 3 from the table in [Main Result 1](#main-result-1-performance-of-our-pir-protocol).
The output should look similar to, and is interpreted analogously, as in [Experiment 1](#experiment-1-pir-protocol-run-database-220-x-2560b-single-threaded).

#### Experiment 4: PIR protocol run, Database 2^22 x 2560B, multi-threaded

The experiment runs our PIR protocol on a database consisting of 2^22 entries of size 2560B each, and measures preprocessing time, online time and communication cost.

To run, use
```bash
RAYON_NUM_THREADS=8 cargo run --release [--features=emulate_avx512] bench --db-entries 4194304
```

This example reproduces table row 4 from the table in [Main Result 1](#main-result-1-performance-of-our-pir-protocol).
The output should look similar to, and is interpreted analogously, as in [Experiment 1](#experiment-1-pir-protocol-run-database-220-x-2560b-single-threaded).

#### Experiment 5: PIR protocol run, Database 2^22 x 2560B, multi-threaded, downscaled version

The experiment runs our PIR protocol on a database consisting of 2^22 entries of size 2560B each, but reduces RAM usage and preprocessing time by using a database consisting of a single 2^14 x 2560B block, replicated 256 times.
As a result, this still verifies correctness of the protocol, and reproduces the figures for communication size. However, due to multi-threading, and since the reduced storage amount greatly improves cache hit rates, this will not reproduce runtime numbers.

To run, use
```bash
cargo run --release [--features=emulate_avx512] bench --db-entries 4194304 --preprocessed-db-count 1
```

This example reproduces the communication values given in table rows 3 and 4 from the table in [Main Result 1](#main-result-1-performance-of-our-pir-protocol).
The output should look similar to, and is interpreted analogously, as in [Experiment 1](#experiment-1-pir-protocol-run-database-220-x-2560b-single-threaded).

#### Experiment 6: Functionality of DNS protocol

This example starts a server hosting the DNS root zone via PIR and runs a standard DNS query against it.

To run, use
```bash
cargo run --release [--features=emulate_avx512] server root.txt
```
Then, in a second terminal, run
```bash
cargo run --release [--features=emulate_avx512] client ns1.dns.nic.aaa --server 127.0.0.1
```

The output should look similar to the following.
```text
Resolving ns1.dns.nic.aaa (iterative PIR-DNS lookup)
  hop 1: 127.0.0.1:9000 hosts zone .
ns1.dns.nic.aaa A 156.154.144.2
ns1.dns.nic.aaa AAAA 2610:a1:1071::2
Resolution took 0.164 s; sent 43018 B, received 22301 B over the wire
```

## Limitations

The end-to-end DNS benchmarks in the paper (cf. Table 6) are not reproducible with the artifact as given, since they additionally require the DNS zone files for .org and .com, which we cannot include as part of this repository due to licensing issues.
Additionally, these benchmarks would require the client to interact with multiple servers on a local network, which cannot easily be realized using VMs, due to the hardware requirements of each PIR server.
Therefore, we believe the artifact evaluation for the DNS protocol should be restricted to basic functionality.
However, since our PIR implementation is the core of the whole protocol, we believe that there is still significant value in the full artifact review for that part. 
