//! Command-line entry point of the DNSPIR proof of concept.
//!
//! Three sub-commands share one binary:
//!
//! * `client <domain>` — resolve a name iteratively over PIR, starting from
//!   the cached global zone or from `--server`.
//! * `server <zone_file>` — load a zone, build the PIR buckets (or a
//!   compressed dump for small zones) and serve them over TCP.
//! * `bench` — run the raw PIR engine end-to-end, without any DNS layer,
//!   for a given database shape.
//!
//! The DNS layer lives in [`dns`]; the PIR engine it sits on is the
//! separate `dnspir-pir` crate (imported as `pir`).

#![allow(non_snake_case)]

use std::net::SocketAddr;
use std::process::ExitCode;

use crate::dns::DEFAULT_PORT;

mod dns;

fn print_usage(name: &str) {
    eprintln!("usage:");
    eprintln!("  {} client [--server HOST[:PORT]] <domain_name>", name);
    eprintln!("  {} server [--port PORT] [--always-dump] <zone_file>", name);
    eprintln!(
        "  {} bench [--db-entries N] [--preprocessed-db-count N] [--force-full-dbs]",
        name
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        return ExitCode::from(2);
    }
    match args[1].as_str() {
        "client" => run_client(&args),
        "server" => run_server(&args),
        "bench" => run_bench(&args),
        _ => {
            print_usage(&args[0]);
            ExitCode::from(2)
        }
    }
}

fn run_client(args: &[String]) -> ExitCode {
    let mut server: Option<SocketAddr> = None;
    let mut domain: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--server" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--server requires an argument");
                    return ExitCode::from(2);
                }
                match parse_server(&args[i]) {
                    Some(s) => server = Some(s),
                    None => {
                        eprintln!("could not parse server address: {}", args[i]);
                        return ExitCode::from(2);
                    }
                }
            }
            other => {
                if domain.is_some() {
                    eprintln!("unexpected argument: {}", other);
                    return ExitCode::from(2);
                }
                domain = Some(other.to_owned());
            }
        }
        i += 1;
    }
    let Some(domain) = domain else {
        eprintln!("missing <domain_name>");
        return ExitCode::from(2);
    };
    match dns::client::run(&domain, server) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("client error: {}", e);
            ExitCode::from(1)
        }
    }
}

fn run_server(args: &[String]) -> ExitCode {
    let mut port: u16 = DEFAULT_PORT;
    let mut zone_file: Option<String> = None;
    let mut opts = dns::server::ServerOptions::default();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--port requires an argument");
                    return ExitCode::from(2);
                }
                match args[i].parse::<u16>() {
                    Ok(p) => port = p,
                    Err(_) => {
                        eprintln!("invalid port: {}", args[i]);
                        return ExitCode::from(2);
                    }
                }
            }
            "--always-dump" => {
                opts.always_dump = true;
            }
            other => {
                if zone_file.is_some() {
                    eprintln!("unexpected argument: {}", other);
                    return ExitCode::from(2);
                }
                zone_file = Some(other.to_owned());
            }
        }
        i += 1;
    }
    let Some(zone_file) = zone_file else {
        eprintln!("missing <zone_file>");
        return ExitCode::from(2);
    };
    match dns::server::run(&zone_file, port, opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("server error: {}", e);
            ExitCode::from(1)
        }
    }
}

fn run_bench(args: &[String]) -> ExitCode {
    use pir::base_pir::LOG2_N;
    use pir::pir_wrapper::bench_wrapped_pir;
    // Default: one full fleet at maximum capacity (N² entries).
    let default_entries = (1usize << LOG2_N) * (1 << LOG2_N);
    let mut db_entries: usize = default_entries;
    let mut preprocessed_db_count: Option<usize> = None;
    // Bench-only escape hatch: measure the full-size-shard layout for entry
    // counts where the protocol rule would use half-size shards.
    let mut force_full_dbs = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--db-entries" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--db-entries requires an argument");
                    return ExitCode::from(2);
                }
                match args[i].parse::<usize>() {
                    Ok(n) if n > 0 => db_entries = n,
                    _ => {
                        eprintln!("invalid --db-entries value: {}", args[i]);
                        return ExitCode::from(2);
                    }
                }
            }
            "--preprocessed-db-count" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--preprocessed-db-count requires an argument");
                    return ExitCode::from(2);
                }
                match args[i].parse::<usize>() {
                    Ok(n) if n > 0 => preprocessed_db_count = Some(n),
                    _ => {
                        eprintln!(
                            "invalid --preprocessed-db-count value: {}",
                            args[i]
                        );
                        return ExitCode::from(2);
                    }
                }
            }
            "--force-full-dbs" => {
                force_full_dbs = true;
            }
            other => {
                eprintln!("unexpected argument: {}", other);
                return ExitCode::from(2);
            }
        }
        i += 1;
    }
    bench_wrapped_pir(preprocessed_db_count, db_entries, 1, force_full_dbs);
    ExitCode::SUCCESS
}

fn parse_server(s: &str) -> Option<SocketAddr> {
    if let Ok(addr) = s.parse::<SocketAddr>() {
        return Some(addr);
    }
    // Allow a bare host or host without port.
    if let Ok(ip) = s.parse::<std::net::IpAddr>() {
        return Some(SocketAddr::new(ip, DEFAULT_PORT));
    }
    None
}
