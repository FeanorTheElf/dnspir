//! Proof-of-Concept private DNS resolver built on top of the PIR engine.
//!
//! The architecture lives in five files:
//!
//! * [`bucket`] — wire layout of a single PIR slot's contents (hashing of
//!   domain names to slot indices and discriminators, packing of records
//!   into a fixed-size byte buffer).
//! * [`protocol`] — length-prefixed TCP request/response framing and the
//!   commands a client may issue (`INFO` / `QUERY` / `DUMP`).
//! * [`zone`] — parser, serializer, and gzip-based dump format for the
//!   simplified zone-file text the PoC accepts. Interns NS hostnames to
//!   keep memory bounded on huge zones like `.com`.
//! * [`server`] — startup, dump-vs-PIR mode selection, bucket build, PIR
//!   preprocessing, TCP accept loop.
//! * [`client`] — iterative resolver, suffix-walk per hop, optional global
//!   zone cache, hybrid (PIR or local-dump) per-hop server connections.
//!
//! The PIR engine underneath is the separate `dnspir-pir` crate, imported
//! here as `pir`; nothing in that crate knows about DNS. See `CLAUDE.md` at
//! the repository root for the higher-level architecture description and
//! how the two layers fit together.

pub mod bucket;
pub mod client;
pub mod protocol;
pub mod server;
pub mod zone;

/// Default TCP port the PoC client uses when an NS-glue IP is followed
/// without an explicit port, and the default port the server binds when
/// `--port` is omitted.
pub const DEFAULT_PORT: u16 = 9000;
