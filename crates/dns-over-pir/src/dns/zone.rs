//! Zone-file parser, serializer and compressor for the simplified format
//! used by the PoC.
//!
//! Each line of the input file has five tab-separated fields:
//! ```text
//! <name>\t<ttl>\t<class>\t<type>\t<rdata>
//! ```
//! Only types `a`, `aaaa` and `ns` are recognized. Anything else is ignored.
//!
//! Small zones (after filtering and gzip compression) can be sent in full to
//! the client instead of through PIR — see [`serialize`] and
//! [`compress_bytes`].

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

use crate::dns::bucket::canonical;

/// All A / AAAA / NS records the zone holds for a single canonical name.
/// Empty `Vec`s mean "no records of that kind for this name".
#[derive(Default, Debug)]
pub struct NameRecords {
    /// IPv4 records.
    pub a: Vec<Ipv4Addr>,
    /// IPv6 records.
    pub aaaa: Vec<Ipv6Addr>,
    /// NS records, stored as indices into [`Zone::ns_pool`]. Many domains
    /// share the same NS hostnames (cloudflare, godaddy, google, …), so the
    /// strings are interned at parse time to keep memory under control on
    /// huge zones like `.com`.
    pub ns: Vec<u32>,
}

#[derive(Debug)]
pub struct Zone {
    /// canonical name -> records
    pub records: HashMap<Box<str>, NameRecords>,
    /// The zone the file describes, e.g. "org". Inferred from the longest
    /// common suffix of all record names.
    pub zone_name: String,
    /// Pool of distinct NS hostnames referenced by `records`. Index space
    /// matches every `NameRecords::ns` entry; resolve via
    /// [`Zone::ns_hostname`].
    pub ns_pool: Vec<Box<str>>,
}

impl Zone {
    /// Opens and parses the zone file at `path`. Convenience wrapper over
    /// [`Zone::parse`] that takes care of `File::open` + `BufReader`.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let f = File::open(path)?;
        Self::parse(BufReader::new(f))
    }

    /// Reads a zone in the simplified BIND-style whitespace-separated format
    /// from `r`. Runs of spaces and/or tabs between fields are collapsed —
    /// this is what lets the parser handle real zone files (including the
    /// root zone) which use column-aligned padding.
    ///
    /// The parser is allocation-aware: it bails on every line whose type
    /// isn't A/AAAA/NS *before* touching the canonical-name string or the
    /// records HashMap, and interns NS hostnames against an intermediate
    /// dedup map that is dropped before this function returns. Memory
    /// footprint scales with `O(distinct names + distinct NS targets)`
    /// rather than `O(records in file)`.
    pub fn parse<R: BufRead>(reader: R) -> std::io::Result<Self> {
        let mut records: HashMap<Box<str>, NameRecords> = HashMap::new();
        let mut ns_pool: Vec<Box<str>> = Vec::new();
        let mut ns_intern: HashMap<Box<str>, u32> = HashMap::new();
        let mut zone_suffix: Option<String> = None;

        // Reuse one String across all reads so we don't allocate per-line.
        let mut reader = reader;
        let mut line_buf = String::new();
        let mut lineno = 0usize;
        loop {
            line_buf.clear();
            let n = reader.read_line(&mut line_buf)?;
            if n == 0 {
                break;
            }
            lineno += 1;
            let line = line_buf.trim();
            if line.is_empty() || line.starts_with(';') {
                continue;
            }
            // Tokenize on any ASCII whitespace, collapsing runs so that
            // column-aligned tab/space padding stays invisible.
            let mut fields = line.split_ascii_whitespace();
            let name_raw = match fields.next() {
                Some(n) => n,
                None => continue,
            };
            // TTL and class are both optional in BIND zone files (either or
            // both may be omitted). Skip leading tokens that look like a TTL
            // (all-digits) or a class (IN/CS/CH/HS) until we land on what
            // should be the record type.
            let rtype_raw = loop {
                let Some(tok) = fields.next() else { break ""; };
                let is_ttl = !tok.is_empty() && tok.bytes().all(|b| b.is_ascii_digit());
                let is_class = tok.eq_ignore_ascii_case("in")
                    || tok.eq_ignore_ascii_case("cs")
                    || tok.eq_ignore_ascii_case("ch")
                    || tok.eq_ignore_ascii_case("hs");
                if !is_ttl && !is_class {
                    break tok;
                }
            };
            if rtype_raw.is_empty() {
                continue;
            }

            // Cheap type check *before* allocating anything. The real .com
            // zone has many unwanted record types per name (DS, RRSIG,
            // NSEC3PARAM, DNSKEY…); skipping them here means we don't pay
            // for a String allocation and a HashMap slot for each.
            let kind: u8 = if rtype_raw.eq_ignore_ascii_case("a") {
                0
            } else if rtype_raw.eq_ignore_ascii_case("aaaa") {
                1
            } else if rtype_raw.eq_ignore_ascii_case("ns") {
                2
            } else {
                continue;
            };

            let rdata = match fields.next() {
                Some(r) => r,
                None => continue,
            };

            // From here on the line is a keeper; pay the allocations.
            let name = canonical(name_raw);

            // Get the per-name record without allocating a fresh `Box<str>`
            // key on every call: the entry API consumes the key
            // unconditionally, so on the common "name already exists" path
            // (e.g. the 2nd/3rd/4th NS record for the same domain) we
            // would otherwise allocate-and-drop a Box<str> per line.
            if !records.contains_key(name.as_str()) {
                records.insert(name.as_str().into(), NameRecords::default());
            }
            let entry = records.get_mut(name.as_str()).expect("just inserted");

            match kind {
                0 => {
                    let Ok(ip) = rdata.parse::<Ipv4Addr>() else {
                        eprintln!("warn: bad A rdata on line {}: {}", lineno, rdata);
                        continue;
                    };
                    entry.a.push(ip);
                }
                1 => {
                    let Ok(ip) = rdata.parse::<Ipv6Addr>() else {
                        eprintln!("warn: bad AAAA rdata on line {}: {}", lineno, rdata);
                        continue;
                    };
                    entry.aaaa.push(ip);
                }
                2 => {
                    let ns_canon = canonical(rdata);
                    let idx = if let Some(&i) = ns_intern.get(ns_canon.as_str()) {
                        i
                    } else {
                        let i = u32::try_from(ns_pool.len())
                            .expect("more than 2^32 distinct NS hostnames");
                        let key: Box<str> = ns_canon.as_str().into();
                        ns_pool.push(key.clone());
                        ns_intern.insert(key, i);
                        i
                    };
                    entry.ns.push(idx);
                }
                _ => unreachable!(),
            }

            // Track the longest dot-bounded suffix shared by every kept
            // record. The fast path — when the new name is already in the
            // current suffix — does no allocation at all, which matters on
            // zones where every name ends in the same TLD (i.e. all of them).
            match zone_suffix {
                None => zone_suffix = Some(name),
                Some(ref current) => {
                    if !name_in_suffix(&name, current) {
                        zone_suffix = Some(common_suffix_str(current, &name));
                    }
                }
            }
        }

        // The intern map only existed to dedup during parsing; drop it
        // before returning so its memory is freed.
        drop(ns_intern);

        let zone_name = zone_suffix.unwrap_or_default();
        Ok(Zone {
            records,
            zone_name,
            ns_pool,
        })
    }

    /// Returns the interned NS hostname at the given index. Companion to the
    /// `u32` values stored in [`NameRecords::ns`].
    pub fn ns_hostname(&self, idx: u32) -> &str {
        &self.ns_pool[idx as usize]
    }

    /// Convenience iterator over the NS hostnames of one record set.
    pub fn ns_hostnames<'a>(&'a self, rec: &'a NameRecords) -> impl Iterator<Item = &'a str> {
        rec.ns.iter().map(|&i| self.ns_hostname(i))
    }

    /// Look up an A or AAAA record for `name` in this zone. Used for glue.
    pub fn glue_for(&self, name: &str) -> Option<IpAddr> {
        let rec = self.records.get(name)?;
        if let Some(ip) = rec.a.first() {
            return Some(IpAddr::V4(*ip));
        }
        if let Some(ip) = rec.aaaa.first() {
            return Some(IpAddr::V6(*ip));
        }
        None
    }
}

/// True when `name` is identical to `zone` or a dot-bounded sub-name of it.
/// (Same semantics as the helper in `dns::client`, duplicated here so the
/// zone module stays self-contained.)
fn name_in_suffix(name: &str, zone: &str) -> bool {
    if zone.is_empty() {
        return true;
    }
    if name == zone {
        return true;
    }
    name.len() > zone.len()
        && name.ends_with(zone)
        && name.as_bytes()[name.len() - zone.len() - 1] == b'.'
}

/// Returns the longest dot-bounded suffix shared between two canonical
/// names. Walks the labels from the right; allocates only the result.
fn common_suffix_str(a: &str, b: &str) -> String {
    let mut shared: Vec<&str> = Vec::new();
    let mut a_iter = a.rsplit('.');
    let mut b_iter = b.rsplit('.');
    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(al), Some(bl)) if al == bl => shared.push(al),
            _ => break,
        }
    }
    shared.reverse();
    shared.join(".")
}

/// Deterministic placeholder IP for a nameserver hostname whose A/AAAA record
/// is not in this zone. The spec allows us to make these up, but we want them
/// stable across runs so client-side debugging stays sane.
pub fn dummy_ip(hostname: &str) -> IpAddr {
    // Drop the result into 192.0.2.0/24 (TEST-NET-1, RFC 5737) so we don't
    // collide with real public IPs.
    let mut h = 0u32;
    for &b in hostname.as_bytes() {
        h = h.wrapping_mul(31).wrapping_add(b as u32);
    }
    IpAddr::V4(Ipv4Addr::new(192, 0, 2, (h & 0xFF) as u8))
}

/// Emits this zone back to the simplified tab-separated text format,
/// keeping only A / AAAA / NS records and sorting by name for stable
/// output. The result round-trips through [`Zone::parse`].
pub fn serialize(zone: &Zone) -> String {
    serialize_bounded(zone, usize::MAX).expect("unbounded serialize cannot overflow")
}

/// Like [`serialize`], but bails out as soon as the produced text would
/// exceed `cap` bytes. Returns `Err(produced_bytes_so_far)` in that case
/// so the caller can react without paying for the rest of the output —
/// useful when the only question being asked of the serialization is
/// "would the gzipped form fit under some small threshold?", since
/// serializing a multi-GB `.com` zone just to discard it is wasteful.
pub fn serialize_bounded(zone: &Zone, cap: usize) -> Result<String, usize> {
    let mut names: Vec<&str> = zone.records.keys().map(|k| k.as_ref()).collect();
    names.sort();
    let mut out = String::new();
    for name in names {
        let rec = &zone.records[name];
        for ip in &rec.a {
            let _ = writeln!(out, "{}.\t3600\tin\ta\t{}", name, ip);
            if out.len() > cap {
                return Err(out.len());
            }
        }
        for ip in &rec.aaaa {
            let _ = writeln!(out, "{}.\t3600\tin\taaaa\t{}", name, ip);
            if out.len() > cap {
                return Err(out.len());
            }
        }
        for &ns_idx in &rec.ns {
            let nshost = zone.ns_hostname(ns_idx);
            let _ = writeln!(out, "{}.\t3600\tin\tns\t{}.", name, nshost);
            if out.len() > cap {
                return Err(out.len());
            }
        }
    }
    Ok(out)
}

/// Gzip-compresses an arbitrary byte buffer at the best ratio level 9.
pub fn compress_bytes(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::best());
    enc.write_all(data)?;
    enc.finish()
}

/// Inverse of [`compress_bytes`].
pub fn decompress_bytes(gz: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut dec = GzDecoder::new(gz);
    let mut out = Vec::new();
    dec.read_to_end(&mut out)?;
    Ok(out)
}

/// Convenience: serialize the zone, gzip the result.
pub fn dump(zone: &Zone) -> std::io::Result<Vec<u8>> {
    compress_bytes(serialize(zone).as_bytes())
}

/// Convenience: gunzip the bytes, parse as a zone.
pub fn load_dump(gz: &[u8]) -> std::io::Result<Zone> {
    let text = decompress_bytes(gz)?;
    Zone::parse(text.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_text(text: &str) -> Zone {
        Zone::parse(text.as_bytes()).unwrap()
    }

    #[test]
    fn serialize_roundtrip() {
        let zone = parse_text(
            "a.org.\t3600\tin\ta\t1.2.3.4\n\
             a.org.\t3600\tin\taaaa\t2001:db8::1\n\
             b.org.\t3600\tin\tns\tns1.b.org.\n\
             ns1.b.org.\t3600\tin\ta\t10.0.0.1\n",
        );
        let again = parse_text(&serialize(&zone));
        assert_eq!(again.zone_name, zone.zone_name);
        assert_eq!(again.records.len(), zone.records.len());
        for (k, v) in &zone.records {
            let v2 = again.records.get(k).expect("name preserved");
            assert_eq!(v.a, v2.a, "A for {}", k);
            assert_eq!(v.aaaa, v2.aaaa, "AAAA for {}", k);
            // NS indices may differ between two parses because they index
            // into separate interned pools — compare the resolved hostnames.
            let resolved: Vec<&str> = zone.ns_hostnames(v).collect();
            let resolved2: Vec<&str> = again.ns_hostnames(v2).collect();
            assert_eq!(resolved, resolved2, "NS for {}", k);
        }
    }

    #[test]
    fn compress_decompress_roundtrip() {
        let zone = parse_text(
            "host.org.\t3600\tin\ta\t1.2.3.4\n\
             host.org.\t3600\tin\taaaa\t::1\n\
             apex.org.\t3600\tin\tns\tns.elsewhere.com.\n",
        );
        let gz = dump(&zone).unwrap();
        let zone2 = load_dump(&gz).unwrap();
        assert_eq!(zone.records.len(), zone2.records.len());
        assert_eq!(zone.zone_name, zone2.zone_name);
        // Gzip should shrink — sanity check.
        let raw = serialize(&zone);
        assert!(
            gz.len() < raw.len(),
            "gz {} >= raw {}",
            gz.len(),
            raw.len()
        );
    }

    #[test]
    fn root_zone_has_empty_zone_name() {
        let zone = parse_text(
            "org.\t3600\tin\tns\ttld-org.example.\n\
             com.\t3600\tin\tns\ttld-com.example.\n\
             tld-org.example.\t3600\tin\ta\t1.2.3.4\n\
             tld-com.example.\t3600\tin\ta\t5.6.7.8\n",
        );
        assert_eq!(zone.zone_name, "", "root zone should have empty zone_name");
    }

    /// BIND-style zone files (including the real root zone) often align columns
    /// using runs of tabs:
    ///
    /// ```text
    /// org.\t\t\t172800\tIN\tNS\tns.test.org.
    /// ```
    ///
    /// The parser must collapse such runs rather than treating every
    /// consecutive tab as a field separator — otherwise the type field
    /// "drifts" onto the TTL column and the record is silently dropped.
    #[test]
    fn parser_handles_multi_tab_alignment() {
        let zone = parse_text(
            "org.\t\t\t172800\tIN\tNS\tns.test.org.\n\
             ns.test.org.\t\t172800\tIN\tA\t1.2.3.4\n",
        );
        let org = zone.records.get("org").expect("`org` record present");
        let ns_names: Vec<&str> = zone.ns_hostnames(org).collect();
        assert_eq!(ns_names, vec!["ns.test.org"]);
        let ns = zone
            .records
            .get("ns.test.org")
            .expect("`ns.test.org` record present");
        assert_eq!(ns.a, vec!["1.2.3.4".parse::<Ipv4Addr>().unwrap()]);
    }

    /// BIND zone files may omit the TTL column (it then falls back to the
    /// last `$TTL` directive or the SOA minimum). Hand-written zones
    /// frequently look like `name IN A 1.2.3.4` with no TTL at all. The
    /// parser must accept that shape too rather than silently dropping
    /// every record because the type column "drifts" onto the rdata field.
    #[test]
    fn parser_handles_missing_ttl() {
        let zone = parse_text(
            "private-dns.com.        IN  NS    ns.private-dns.com.\n\
             ns.private-dns.com.     IN  A     10.203.3.21\n\
             www.private-dns.com.    IN  A     42.42.42.42\n",
        );
        assert_eq!(zone.zone_name, "private-dns.com");
        let www = zone
            .records
            .get("www.private-dns.com")
            .expect("www record must be parsed when TTL is omitted");
        assert_eq!(www.a, vec!["42.42.42.42".parse::<Ipv4Addr>().unwrap()]);
        let apex = zone.records.get("private-dns.com").expect("apex parsed");
        let ns_names: Vec<&str> = zone.ns_hostnames(apex).collect();
        assert_eq!(ns_names, vec!["ns.private-dns.com"]);
    }

    /// The class column (IN/CH/HS/CS) is also optional, and can appear
    /// either before or after the TTL when present. Accept all four shapes
    /// of the `name [TTL] [CLASS] TYPE rdata` prefix.
    #[test]
    fn parser_handles_class_in_either_order() {
        let zone = parse_text(
            "a.example.\t3600\tIN\tA\t1.1.1.1\n\
             b.example.\tIN\t3600\tA\t2.2.2.2\n\
             c.example.\t3600\tA\t3.3.3.3\n\
             d.example.\tA\t4.4.4.4\n",
        );
        for (name, ip) in [
            ("a.example", "1.1.1.1"),
            ("b.example", "2.2.2.2"),
            ("c.example", "3.3.3.3"),
            ("d.example", "4.4.4.4"),
        ] {
            let r = zone.records.get(name).unwrap_or_else(|| panic!("{}", name));
            assert_eq!(r.a, vec![ip.parse::<Ipv4Addr>().unwrap()]);
        }
    }

    /// The same BIND file uses space-padded alignment alongside tabs. Treat
    /// any run of ASCII whitespace as a single field separator.
    #[test]
    fn parser_handles_space_padded_alignment() {
        let zone = parse_text(
            "host.org.    3600  IN  A    10.0.0.1\n\
             host.org.    3600  IN  AAAA ::1\n",
        );
        let r = zone.records.get("host.org").unwrap();
        assert_eq!(r.a, vec!["10.0.0.1".parse::<Ipv4Addr>().unwrap()]);
        assert_eq!(r.aaaa, vec!["::1".parse::<Ipv6Addr>().unwrap()]);
    }

    /// Loads the real bundled root zone and sanity-checks a few apex names.
    /// Cheap enough to run by default; bundled file is ~25k lines.
    #[test]
    fn parser_handles_real_root_zone() {
        // Bundled data files live at the workspace root, not in this
        // package's directory (which is what `cargo test` makes the CWD).
        let zone = Zone::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../../root.txt"))
            .expect("root.txt missing");
        // `org.` has several NS records and no direct A in the real root.
        let org = zone.records.get("org").expect("org must be parsed");
        assert!(
            !org.ns.is_empty(),
            "org must have NS records, got {:?}",
            org
        );
        // `a.root-servers.net.` is glue with a known A record.
        let a_root = zone
            .records
            .get("a.root-servers.net")
            .expect("a.root-servers.net must be parsed");
        assert!(
            !a_root.a.is_empty(),
            "a.root-servers.net must have A record, got {:?}",
            a_root
        );
        // Root zone has no common suffix across TLDs.
        assert_eq!(zone.zone_name, "");
    }

    /// The real root zone has many record types we don't care about (SOA,
    /// RRSIG, NSEC, DS, DNSKEY, ZONEMD…). They must be ignored without
    /// disturbing the A/AAAA/NS records on the same names.
    #[test]
    fn parser_ignores_unknown_types_alongside_supported_ones() {
        let zone = parse_text(
            ".\t86400\tIN\tSOA\ta.root-servers.net. nstld.verisign-grs.com. 1 2 3 4 5\n\
             .\t86400\tIN\tRRSIG\tSOA 8 0 86400 20260528170000 20260515160000 54393 . abc=\n\
             org.\t\t\t172800\tIN\tNS\ta0.org.afilias-nst.info.\n\
             org.\t\t\t86400\tIN\tDS\t26974 8 2 4FEDE294C53F438A158C41D39489CD78A86BEB0D8A0AEAFF14745C0D16E1DE32\n\
             org.\t\t\t86400\tIN\tNSEC\torganic. NS DS RRSIG NSEC\n",
        );
        let org = zone.records.get("org").expect("`org` record present");
        let ns_names: Vec<&str> = zone.ns_hostnames(org).collect();
        assert_eq!(ns_names, vec!["a0.org.afilias-nst.info"]);
        assert!(org.a.is_empty());
        assert!(org.aaaa.is_empty());
    }
}
