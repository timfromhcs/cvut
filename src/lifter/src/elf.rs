// SPDX-License-Identifier: Apache-2.0
// ELF64 cubin parser and instruction extractor
//
// Security posture: all offsets/sizes derived from untrusted file bytes are
// validated with bounds checks and checked arithmetic. Malformed input
// returns `Err` -- never panics.

#[allow(dead_code)]
pub struct CubinSection {
    pub name: String,
    pub data: Vec<u8>,
}

fn read_u16_le(bytes: &[u8], off: usize) -> Result<u16, String> {
    let end = off
        .checked_add(2)
        .ok_or_else(|| "u16 read overflow".to_string())?;
    if end > bytes.len() {
        return Err("unexpected end of file reading u16".into());
    }
    Ok(u16::from_le_bytes([bytes[off], bytes[off + 1]]))
}

fn read_u32_le(bytes: &[u8], off: usize) -> Result<u32, String> {
    let end = off
        .checked_add(4)
        .ok_or_else(|| "u32 read overflow".to_string())?;
    if end > bytes.len() {
        return Err("unexpected end of file reading u32".into());
    }
    Ok(u32::from_le_bytes([
        bytes[off],
        bytes[off + 1],
        bytes[off + 2],
        bytes[off + 3],
    ]))
}

fn read_u64_le(bytes: &[u8], off: usize) -> Result<u64, String> {
    let end = off
        .checked_add(8)
        .ok_or_else(|| "u64 read overflow".to_string())?;
    if end > bytes.len() {
        return Err("unexpected end of file reading u64".into());
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[off..end]);
    Ok(u64::from_le_bytes(arr))
}

fn checked_range(base: usize, len: usize, file_len: usize) -> Result<(usize, usize), String> {
    let end = base
        .checked_add(len)
        .ok_or_else(|| "file range arithmetic overflow".to_string())?;
    if end > file_len {
        return Err(format!(
            "file range [{}, {}) exceeds file size {}",
            base, end, file_len
        ));
    }
    Ok((base, end))
}

pub fn parse_cubin_instructions(bytes: &[u8]) -> Result<Vec<[u32; 4]>, String> {
    if bytes.len() >= 4 && &bytes[0..4] == b"\x7fELF" {
        // ELF64 format
        if bytes.len() < 64 {
            return Err("ELF file too small".into());
        }

        // Validate ELF identification: class=2 (64-bit), data=1 (LE), version=1.
        if bytes[4] != 2 {
            return Err(format!(
                "unsupported ELF class {} (only 64-bit ELF supported)",
                bytes[4]
            ));
        }
        if bytes[5] != 1 {
            return Err(format!(
                "unsupported ELF endianness {} (only little-endian supported)",
                bytes[5]
            ));
        }
        if bytes[6] != 1 {
            return Err(format!("unsupported ELF version {}", bytes[6]));
        }

        let e_shoff = read_u64_le(bytes, 40)? as usize;
        let e_shentsize = read_u16_le(bytes, 58)? as usize;
        let e_shnum = read_u16_le(bytes, 60)? as usize;
        let e_shstrndx = read_u16_le(bytes, 62)? as usize;

        if e_shentsize < 64 {
            return Err(format!(
                "invalid section header entry size {} (expected >= 64)",
                e_shentsize
            ));
        }
        const MAX_SECTIONS: usize = 4096;
        if e_shnum == 0 || e_shnum > MAX_SECTIONS {
            return Err(format!("invalid section count {}", e_shnum));
        }
        if e_shstrndx >= e_shnum {
            return Err("Invalid shstrndx".into());
        }

        // Section-header table must fit inside the file (checked arithmetic).
        let table_len = e_shnum
            .checked_mul(e_shentsize)
            .ok_or_else(|| "section table size overflow".to_string())?;
        checked_range(e_shoff, table_len, bytes.len())
            .map_err(|_| "section header table exceeds file size".to_string())?;

        let strtab_hdr = e_shoff
            .checked_add(
                e_shstrndx
                    .checked_mul(e_shentsize)
                    .ok_or_else(|| "strtab header offset overflow".to_string())?,
            )
            .ok_or_else(|| "strtab header offset overflow".to_string())?;
        // sh_offset(24) and sh_size(32) within the header entry.
        let strtab_offset = read_u64_le(
            bytes,
            strtab_hdr
                .checked_add(24)
                .ok_or_else(|| "strtab offset field overflow".to_string())?,
        )? as usize;
        let strtab_size = read_u64_le(
            bytes,
            strtab_hdr
                .checked_add(32)
                .ok_or_else(|| "strtab size field overflow".to_string())?,
        )? as usize;
        let (st_start, st_end) = checked_range(strtab_offset, strtab_size, bytes.len())
            .map_err(|_| "string table exceeds file size".to_string())?;
        let strtab = &bytes[st_start..st_end];
        if strtab.is_empty() || strtab[0] != 0 {
            return Err("invalid string table (must be NUL-prefixed)".into());
        }

        let mut text_data: Option<&[u8]> = None;

        for i in 0..e_shnum {
            let sh_offset = e_shoff
                .checked_add(
                    i.checked_mul(e_shentsize)
                        .ok_or_else(|| "section header offset overflow".to_string())?,
                )
                .ok_or_else(|| "section header offset overflow".to_string())?;
            // Entry itself already proven in-range by the table check above.
            let sh_name = read_u32_le(bytes, sh_offset)? as usize;
            let offset = read_u64_le(
                bytes,
                sh_offset
                    .checked_add(24)
                    .ok_or_else(|| "sh_offset field overflow".to_string())?,
            )? as usize;
            let size = read_u64_le(
                bytes,
                sh_offset
                    .checked_add(32)
                    .ok_or_else(|| "sh_size field overflow".to_string())?,
            )? as usize;

            if sh_name >= strtab.len() {
                return Err(format!(
                    "section {} has out-of-bounds name index {}",
                    i, sh_name
                ));
            }
            let name_end_rel = strtab[sh_name..]
                .iter()
                .position(|&c| c == 0)
                .ok_or_else(|| format!("section {} name is unterminated", i))?;
            let name_end = sh_name
                .checked_add(name_end_rel)
                .ok_or_else(|| "section name range overflow".to_string())?;
            let name = std::str::from_utf8(&strtab[sh_name..name_end])
                .map_err(|_| format!("section {} name is not valid UTF-8", i))?;

            if name.starts_with(".text") && size > 0 {
                let (t_start, t_end) = checked_range(offset, size, bytes.len())
                    .map_err(|_| format!("section '{}' exceeds file size", name))?;
                let section_bytes = &bytes[t_start..t_end];
                if section_bytes.len() % 16 != 0 {
                    return Err(format!(
                        "section '{}' size {} is not a multiple of 16 bytes",
                        name,
                        section_bytes.len()
                    ));
                }
                text_data = Some(section_bytes);
                break;
            }
        }

        let data = text_data.ok_or_else(|| "No .text section found in ELF cubin".to_string())?;
        parse_raw_instructions(data)
    } else {
        // Raw binary or hex
        parse_raw_instructions(bytes)
    }
}

pub fn parse_raw_instructions(bytes: &[u8]) -> Result<Vec<[u32; 4]>, String> {
    if bytes.len() % 16 != 0 {
        // Try parsing textual hex
        if let Ok(text) = std::str::from_utf8(bytes) {
            let words: Vec<u32> = text
                .split_whitespace()
                .filter_map(|w| {
                    let cleaned = w.trim_start_matches("0x").trim_end_matches([',', ';']);
                    u32::from_str_radix(cleaned, 16).ok()
                })
                .collect();
            if words.len() % 4 == 0 && !words.is_empty() {
                let mut instrs = Vec::new();
                let (chunks, _) = words.as_chunks::<4>();
                for chunk in chunks {
                    instrs.push(*chunk);
                }
                return Ok(instrs);
            }
        }
        return Err(format!(
            "Instruction stream size {} is not a multiple of 16 bytes",
            bytes.len()
        ));
    }

    let mut instrs = Vec::new();
    let (chunks, _) = bytes.as_chunks::<16>();
    for chunk in chunks {
        // `chunk` is exactly 16 bytes by construction; manual indexing keeps
        // malformed-input handling panic-free (no unwrap on external data).
        let w0 = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let w1 = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let w2 = u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
        let w3 = u32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);
        instrs.push([w0, w1, w2, w3]);
    }
    Ok(instrs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_raw_instructions_16byte() {
        let raw = [1u8; 32];
        let instrs = parse_raw_instructions(&raw).unwrap();
        assert_eq!(instrs.len(), 2);
    }

    #[test]
    fn test_parse_raw_instructions_invalid_size() {
        let raw = [1u8; 15];
        assert!(parse_raw_instructions(&raw).is_err());
    }

    #[test]
    fn test_parse_text_hex_instructions() {
        let text = b"0x00000001 0x00000002 0x00000003 0x00000004";
        let instrs = parse_raw_instructions(text).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0], [1, 2, 3, 4]);
    }

    #[test]
    fn test_parse_cubin_too_small() {
        let bytes = b"\x7fELF_short";
        assert!(parse_cubin_instructions(bytes).is_err());
    }

    #[test]
    fn test_reject_bad_class_and_endianness() {
        // Valid-size ELF header with wrong class/data bytes must be rejected, not panicked.
        let mut hdr = vec![0u8; 64];
        hdr[0..4].copy_from_slice(b"\x7fELF");
        hdr[4] = 1; // 32-bit -> unsupported
        hdr[5] = 1;
        hdr[6] = 1;
        assert!(parse_cubin_instructions(&hdr).is_err());

        let mut hdr2 = vec![0u8; 64];
        hdr2[0..4].copy_from_slice(b"\x7fELF");
        hdr2[4] = 2;
        hdr2[5] = 2; // big-endian -> unsupported
        hdr2[6] = 1;
        assert!(parse_cubin_instructions(&hdr2).is_err());
    }

    #[test]
    fn test_reject_truncated_section_table_no_panic() {
        // e_shoff points far past EOF; must return Err, not panic.
        let mut hdr = vec![0u8; 64];
        hdr[0..4].copy_from_slice(b"\x7fELF");
        hdr[4] = 2;
        hdr[5] = 1;
        hdr[6] = 1;
        // e_shoff = 0xFFFF_FFFF at bytes 40..48
        hdr[40..48].copy_from_slice(&0xFFFFFFFFu64.to_le_bytes());
        hdr[58..60].copy_from_slice(&64u16.to_le_bytes());
        hdr[60..62].copy_from_slice(&3u16.to_le_bytes());
        hdr[62..64].copy_from_slice(&2u16.to_le_bytes());
        assert!(parse_cubin_instructions(&hdr).is_err());
    }

    #[test]
    fn test_reject_overflow_section_size_no_panic() {
        // e_shnum * e_shentsize overflow / OOB must be Err.
        let mut hdr = vec![0u8; 64];
        hdr[0..4].copy_from_slice(b"\x7fELF");
        hdr[4] = 2;
        hdr[5] = 1;
        hdr[6] = 1;
        hdr[40..48].copy_from_slice(&64u64.to_le_bytes());
        hdr[58..60].copy_from_slice(&64u16.to_le_bytes());
        hdr[60..62].copy_from_slice(&4095u16.to_le_bytes());
        hdr[62..64].copy_from_slice(&4094u16.to_le_bytes());
        // Only 64 bytes present, table claims ~256KB -> must be Err.
        assert!(parse_cubin_instructions(&hdr).is_err());
    }

    #[test]
    fn test_fuzz_no_panic_on_arbitrary_bytes() {
        // Deterministic pseudo-fuzz: no input may panic.
        let mut x: u32 = 0x12345678;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x & 0xFF) as u8
        };
        for len in [0usize, 1, 3, 4, 15, 16, 63, 64, 128, 256] {
            for _ in 0..50 {
                let buf: Vec<u8> = (0..len).map(|_| next()).collect();
                let _ = parse_cubin_instructions(&buf);
            }
        }
        // All-0xFF and all-'A' edge inputs.
        let _ = parse_cubin_instructions(&vec![0xFFu8; 128]);
        let _ = parse_cubin_instructions(&vec![0x41u8; 64]);
    }
}
