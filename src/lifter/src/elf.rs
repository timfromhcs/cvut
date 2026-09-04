// SPDX-License-Identifier: Apache-2.0
// ELF64 cubin parser and instruction extractor

#[allow(dead_code)]
pub struct CubinSection {
    pub name: String,
    pub data: Vec<u8>,
}

pub fn parse_cubin_instructions(bytes: &[u8]) -> Result<Vec<[u32; 4]>, String> {
    if bytes.len() >= 4 && &bytes[0..4] == b"\x7fELF" {
        // ELF64 format
        if bytes.len() < 64 {
            return Err("ELF file too small".into());
        }

        let e_shoff = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
        let e_shentsize = u16::from_le_bytes(bytes[58..60].try_into().unwrap()) as usize;
        let e_shnum = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
        let e_shstrndx = u16::from_le_bytes(bytes[62..64].try_into().unwrap()) as usize;

        if e_shstrndx >= e_shnum {
            return Err("Invalid shstrndx".into());
        }

        let strtab_sh_offset = e_shoff + e_shstrndx * e_shentsize;
        let strtab_offset = u64::from_le_bytes(
            bytes[strtab_sh_offset + 24..strtab_sh_offset + 32]
                .try_into()
                .unwrap(),
        ) as usize;
        let strtab_size = u64::from_le_bytes(
            bytes[strtab_sh_offset + 32..strtab_sh_offset + 40]
                .try_into()
                .unwrap(),
        ) as usize;
        let strtab = &bytes[strtab_offset..strtab_offset + strtab_size];

        let mut text_data = None;

        for i in 0..e_shnum {
            let sh_offset = e_shoff + i * e_shentsize;
            let sh_name =
                u32::from_le_bytes(bytes[sh_offset..sh_offset + 4].try_into().unwrap()) as usize;
            let offset =
                u64::from_le_bytes(bytes[sh_offset + 24..sh_offset + 32].try_into().unwrap())
                    as usize;
            let size = u64::from_le_bytes(bytes[sh_offset + 32..sh_offset + 40].try_into().unwrap())
                as usize;

            let name_end = strtab[sh_name..].iter().position(|&c| c == 0).unwrap_or(0);
            let name = std::str::from_utf8(&strtab[sh_name..sh_name + name_end]).unwrap_or("");

            if name.starts_with(".text") && size > 0 {
                text_data = Some(&bytes[offset..offset + size]);
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
                for chunk in words.chunks_exact(4) {
                    instrs.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
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
    for chunk in bytes.chunks_exact(16) {
        let w0 = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let w1 = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
        let w2 = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
        let w3 = u32::from_le_bytes(chunk[12..16].try_into().unwrap());
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
}
