// SPDX-License-Identifier: Apache-2.0
// SASS 128-bit instruction decoder derived directly from Mesa NAK (sm70_encode.rs)

#[allow(dead_code, non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Nop,
    S2R,
    LDC,
    MOV,
    IADD3,
    IMAD,
    IMAD64,
    ISETP,
    LDG,
    STG,
    LDS,
    STS,
    FADD,
    FMUL,
    FFMA,
    HMMA,
    LDSM,
    SHFL,
    BarSync,
    BRA,
    BSSY,
    BSYNC,
    EXIT,
    Unknown(u16),
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialReg {
    LaneId = 0x00,
    TidX = 0x21,
    TidY = 0x22,
    TidZ = 0x23,
    CtaIdX = 0x25,
    CtaIdY = 0x26,
    CtaIdZ = 0x27,
    LaneMaskEq = 0x38,
    LaneMaskLt = 0x39,
    LaneMaskLe = 0x3a,
    LaneMaskGt = 0x3b,
    LaneMaskGe = 0x3c,
    ClockLo = 0x50,
    ClockHi = 0x51,
    Unknown = 0xff,
}

impl From<u8> for SpecialReg {
    fn from(val: u8) -> Self {
        match val {
            0x00 => SpecialReg::LaneId,
            0x21 => SpecialReg::TidX,
            0x22 => SpecialReg::TidY,
            0x23 => SpecialReg::TidZ,
            0x25 => SpecialReg::CtaIdX,
            0x26 => SpecialReg::CtaIdY,
            0x27 => SpecialReg::CtaIdZ,
            0x38 => SpecialReg::LaneMaskEq,
            0x39 => SpecialReg::LaneMaskLt,
            0x3a => SpecialReg::LaneMaskLe,
            0x3b => SpecialReg::LaneMaskGt,
            0x3c => SpecialReg::LaneMaskGe,
            0x50 => SpecialReg::ClockLo,
            0x51 => SpecialReg::ClockHi,
            _ => SpecialReg::Unknown,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DecodedInstruction {
    pub opcode: Opcode,
    pub pred_reg: u8,
    pub pred_inv: bool,
    pub dst: u8,
    pub src0: u8,
    pub src1: u8,
    pub src2: u8,
    pub imm32: u32,
    pub offset: u32,
    pub target_offset: i32,
    pub special_reg: SpecialReg,
    pub raw: [u32; 4],
}

pub fn get_bits(inst: &[u32; 4], start: usize, end: usize) -> u64 {
    let mut res = 0u64;
    for bit in start..end {
        let word_idx = bit / 32;
        let bit_idx = bit % 32;
        let bit_val = ((inst[word_idx] >> bit_idx) & 1) as u64;
        res |= bit_val << (bit - start);
    }
    res
}

pub fn decode_instruction(inst: &[u32; 4]) -> DecodedInstruction {
    let raw_op12 = get_bits(inst, 0, 12) as u16;
    let alu_op9 = get_bits(inst, 0, 9) as u16;
    let _alu_form = get_bits(inst, 9, 12) as u8;

    let pred_reg = get_bits(inst, 12, 15) as u8;
    let pred_inv = get_bits(inst, 15, 16) != 0;
    let dst = get_bits(inst, 16, 24) as u8;
    let src0 = get_bits(inst, 24, 32) as u8;
    let src1 = get_bits(inst, 32, 40) as u8;
    let src2 = get_bits(inst, 64, 72) as u8;
    let imm32 = get_bits(inst, 32, 64) as u32;
    let offset = get_bits(inst, 40, 64) as u32;
    let sr_idx = get_bits(inst, 72, 80) as u8;

    // Relative target offset for branches (bits 34..64 or 34..82)
    let raw_target = get_bits(inst, 34, 64) as u32;
    let target_offset = raw_target as i32;

    let opcode = match raw_op12 {
        0x919 | 0x9c3 => Opcode::S2R,
        0x981 | 0x381 => Opcode::LDG,
        0x986 | 0x386 => Opcode::STG,
        0x984 => Opcode::LDS,
        0x988 | 0x388 => Opcode::STS,
        0x9c0 | 0x3c0 => Opcode::LDC,
        0x947 | 0x547 => Opcode::BRA,
        0x945 => Opcode::BSSY,
        0x941 => Opcode::BSYNC,
        0x94d => Opcode::EXIT,
        0xb1d => Opcode::BarSync,
        0x23c => Opcode::HMMA,
        0x83b => Opcode::LDSM,
        0x389 | 0x589 | 0x989 => Opcode::SHFL,
        _ => match alu_op9 {
            0x002 => Opcode::MOV,
            0x010 => Opcode::IADD3,
            0x024 | 0x0a4 => Opcode::IMAD,
            0x025 | 0x0a5 => Opcode::IMAD64,
            0x00c | 0x08c => Opcode::ISETP,
            0x021 | 0x054 => Opcode::FADD,
            0x020 | 0x056 => Opcode::FMUL,
            0x023 | 0x055 => Opcode::FFMA,
            0x118 => Opcode::Nop,
            _ => Opcode::Unknown(raw_op12),
        },
    };

    DecodedInstruction {
        opcode,
        pred_reg,
        pred_inv,
        dst,
        src0,
        src1,
        src2,
        imm32,
        offset,
        target_offset,
        special_reg: SpecialReg::from(sr_idx),
        raw: *inst,
    }
}
