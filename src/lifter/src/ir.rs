// SPDX-License-Identifier: Apache-2.0
//! SASS intermediate representation (IR).
//!
//! This module separates *decoding* (bit patterns, see `decoder.rs`) from
//! *validation* and *lowering* (semantics, see `lifter.rs`).
//!
//! Honesty contract: every [`DecodedInstruction`] maps to exactly one
//! [`IrInstruction`]. Instructions whose semantics CVUT does not model map to
//! [`SassOp::Unsupported`] with a human-readable reason. The SPIR-V emitter
//! must fail safely on `Unsupported` rather than silently emitting a valid
//! looking but semantically wrong instruction.

use crate::decoder::{DecodedInstruction, Opcode, SpecialReg};

/// Single SASS operation in IR form. Operands are explicit so the emitter
/// cannot silently drop data dependencies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SassOp {
    Mov {
        dst: u8,
        src: u8,
        imm: Option<u32>,
    },
    IAdd3 {
        dst: u8,
        src0: u8,
        src1: u8,
        src2: u8,
    },
    IMad {
        dst: u8,
        src0: u8,
        src1: u8,
        src2: u8,
    },
    ISetP {
        pred: u8,
        src0: u8,
        src1: u8,
    },
    FAdd {
        dst: u8,
        src0: u8,
        src1: u8,
    },
    FMul {
        dst: u8,
        src0: u8,
        src1: u8,
    },
    FFma {
        dst: u8,
        src0: u8,
        src1: u8,
        src2: u8,
    },
    S2R {
        dst: u8,
        sr: SpecialReg,
    },
    Ldc {
        dst: u8,
        offset: u32,
    },
    Ldg {
        dst: u8,
        addr_reg: u8,
        offset: u32,
    },
    Stg {
        addr_reg: u8,
        src: u8,
        offset: u32,
    },
    Lds {
        dst: u8,
        addr_reg: u8,
    },
    Sts {
        addr_reg: u8,
        src: u8,
    },
    Bra {
        target: Option<u32>,
        pred: u8,
        pred_inv: bool,
    },
    Exit,
    Barrier,
    Nop,
    /// Explicitly unsupported semantics. `reason` must name the missing feature.
    Unsupported {
        raw_opcode: u16,
        reason: String,
    },
}

/// One IR instruction: operation plus predicate guard and source address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrInstruction {
    pub op: SassOp,
    /// Predicate register guarding this instruction (7 == PT, always true).
    pub pred_guard: u8,
    pub pred_inv: bool,
    /// Byte address of the source SASS instruction (index * 16).
    pub address: u32,
}

impl IrInstruction {
    #[allow(dead_code)]
    pub fn is_unsupported(&self) -> bool {
        matches!(self.op, SassOp::Unsupported { .. })
    }
}

/// Basic block: a straight-line run of IR instructions with one entry point.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BasicBlock {
    pub id: usize,
    pub start_address: u32,
    pub instrs: Vec<IrInstruction>,
    /// Resolved successor block ids (empty == function exit).
    pub successors: Vec<usize>,
}

/// Lower a single decoded instruction to IR. Never panics; unknown opcodes
/// become `Unsupported`.
pub fn lower_to_ir(decoded: &DecodedInstruction, address: u32) -> IrInstruction {
    let pred_guard = decoded.pred_reg;
    let pred_inv = decoded.pred_inv;
    let op = match decoded.opcode {
        Opcode::MOV => SassOp::Mov {
            dst: decoded.dst,
            src: decoded.src0,
            imm: if decoded.imm32 != 0 {
                Some(decoded.imm32)
            } else {
                None
            },
        },
        Opcode::IADD3 => SassOp::IAdd3 {
            dst: decoded.dst,
            src0: decoded.src0,
            src1: decoded.src1,
            src2: decoded.src2,
        },
        Opcode::IMAD | Opcode::IMAD64 => SassOp::IMad {
            dst: decoded.dst,
            src0: decoded.src0,
            src1: decoded.src1,
            src2: decoded.src2,
        },
        Opcode::ISETP => SassOp::ISetP {
            pred: decoded.dst.min(7),
            src0: decoded.src0,
            src1: decoded.src1,
        },
        Opcode::FADD => SassOp::FAdd {
            dst: decoded.dst,
            src0: decoded.src0,
            src1: decoded.src1,
        },
        Opcode::FMUL => SassOp::FMul {
            dst: decoded.dst,
            src0: decoded.src0,
            src1: decoded.src1,
        },
        Opcode::FFMA => SassOp::FFma {
            dst: decoded.dst,
            src0: decoded.src0,
            src1: decoded.src1,
            src2: decoded.src2,
        },
        Opcode::S2R => {
            if decoded.special_reg == SpecialReg::Unknown {
                SassOp::Unsupported {
                    raw_opcode: 0x919,
                    reason: "S2R with unknown special register".into(),
                }
            } else {
                SassOp::S2R {
                    dst: decoded.dst,
                    sr: decoded.special_reg,
                }
            }
        }
        Opcode::LDC => SassOp::Ldc {
            dst: decoded.dst,
            offset: decoded.offset,
        },
        Opcode::LDG => SassOp::Ldg {
            dst: decoded.dst,
            addr_reg: decoded.src0,
            offset: decoded.offset,
        },
        Opcode::STG => SassOp::Stg {
            addr_reg: decoded.src0,
            src: decoded.src1,
            offset: decoded.offset,
        },
        Opcode::LDS => SassOp::Lds {
            dst: decoded.dst,
            addr_reg: decoded.src0,
        },
        Opcode::STS => SassOp::Sts {
            addr_reg: decoded.src0,
            src: decoded.src1,
        },
        Opcode::BRA => SassOp::Bra {
            target: resolve_branch_target(address, decoded.target_offset),
            pred: decoded.pred_reg,
            pred_inv: decoded.pred_inv,
        },
        Opcode::BSSY | Opcode::BSYNC | Opcode::BarSync => SassOp::Barrier,
        Opcode::EXIT => SassOp::Exit,
        Opcode::Nop => SassOp::Nop,
        Opcode::LDSM => SassOp::Unsupported {
            raw_opcode: 0x83b,
            reason: "LDSM matrix load not modeled; needs shared-memory matrix layout".into(),
        },
        Opcode::SHFL => SassOp::Unsupported {
            raw_opcode: 0x389,
            reason: "SHFL warp shuffle not modeled; needs subgroup lowering".into(),
        },
        Opcode::HMMA => SassOp::Unsupported {
            raw_opcode: 0x23c,
            reason: "HMMA tensor-core op not modeled; needs VK_KHR_cooperative_matrix".into(),
        },
        Opcode::Unknown(raw) => SassOp::Unsupported {
            raw_opcode: raw,
            reason: format!("reserved/unknown opcode {:#06x}", raw),
        },
    };
    IrInstruction {
        op,
        pred_guard,
        pred_inv,
        address,
    }
}

/// Lower a whole decoded stream, preserving order and addresses.
pub fn lower_stream(decoded: &[DecodedInstruction]) -> Vec<IrInstruction> {
    decoded
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let addr = (i as u32).saturating_mul(16);
            lower_to_ir(d, addr)
        })
        .collect()
}

/// Resolve a branch target byte address safely.
/// `target_offset` is a signed offset in bytes relative to the branch
/// instruction address (matches the lifter's historic interpretation).
/// Returns `None` when the target is misaligned or outside the function.
pub fn resolve_branch_target(pc: u32, target_offset: i32) -> Option<u32> {
    let target = (pc as i64).checked_add(target_offset as i64)?;
    if target < 0 || target % 16 != 0 {
        return None;
    }
    let t = target as u64;
    if t > u32::MAX as u64 {
        return None;
    }
    Some(t as u32)
}

/// Build basic blocks from an IR stream. Leaders are instruction 0, branch
/// targets, and fall-throughs after branches/exits. Unresolvable branch
/// targets produce a block with no successor (fail-safe: emitter treats this
/// as an error when the branch is reached).
pub fn build_cfg(ir: &[IrInstruction]) -> Vec<BasicBlock> {
    use std::collections::BTreeSet;
    if ir.is_empty() {
        return Vec::new();
    }
    let addr_to_idx: std::collections::HashMap<u32, usize> = ir
        .iter()
        .enumerate()
        .map(|(i, ins)| (ins.address, i))
        .collect();

    let mut leaders = BTreeSet::new();
    leaders.insert(0usize);
    for (i, ins) in ir.iter().enumerate() {
        match &ins.op {
            SassOp::Bra { target, .. } => {
                if let Some(t) = target {
                    if let Some(&idx) = addr_to_idx.get(t) {
                        leaders.insert(idx);
                    }
                }
                if i + 1 < ir.len() {
                    leaders.insert(i + 1);
                }
            }
            SassOp::Exit if i + 1 < ir.len() => {
                leaders.insert(i + 1);
            }
            SassOp::Exit => {}
            _ => {}
        }
    }

    let mut blocks = Vec::new();
    let mut sorted: Vec<usize> = leaders.into_iter().collect();
    sorted.sort_unstable();
    for (bid, &start) in sorted.iter().enumerate() {
        let end = if bid + 1 < sorted.len() {
            sorted[bid + 1]
        } else {
            ir.len()
        };
        let instrs = ir[start..end].to_vec();
        let start_address = instrs.first().map(|x| x.address).unwrap_or(0);
        let mut successors = Vec::new();
        if let Some(last) = instrs.last() {
            match &last.op {
                SassOp::Exit => {}
                SassOp::Bra { target, .. } => {
                    if let Some(t) = target {
                        if let Some(&idx) = addr_to_idx.get(t) {
                            // find block containing idx
                            for (obid, &ostart) in sorted.iter().enumerate() {
                                let oend = if obid + 1 < sorted.len() {
                                    sorted[obid + 1]
                                } else {
                                    ir.len()
                                };
                                if idx >= ostart && idx < oend {
                                    successors.push(obid);
                                    break;
                                }
                            }
                        }
                    }
                    // Conditional branches also fall through.
                    if last.pred_guard != 7 || last.pred_inv {
                        if bid + 1 < sorted.len() {
                            successors.push(bid + 1);
                        }
                    } else if successors.is_empty() {
                        // Unconditional branch with unresolvable target: no
                        // successor -> emitter must report an error.
                    }
                }
                _ => {
                    if bid + 1 < sorted.len() {
                        successors.push(bid + 1);
                    }
                }
            }
        }
        blocks.push(BasicBlock {
            id: bid,
            start_address,
            instrs,
            successors,
        });
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{DecodedInstruction, Opcode, SpecialReg};

    fn mk(op: Opcode) -> DecodedInstruction {
        DecodedInstruction {
            opcode: op,
            pred_reg: 7,
            pred_inv: false,
            dst: 0,
            src0: 1,
            src1: 2,
            src2: 3,
            imm32: 0,
            offset: 0,
            target_offset: 0,
            special_reg: SpecialReg::Unknown,
            raw: [0; 4],
        }
    }

    #[test]
    fn test_lower_known_ops_preserve_operands() {
        let mut d = mk(Opcode::FADD);
        d.dst = 14;
        d.src0 = 12;
        d.src1 = 13;
        let ir = lower_to_ir(&d, 0);
        assert_eq!(
            ir.op,
            SassOp::FAdd {
                dst: 14,
                src0: 12,
                src1: 13
            }
        );
        assert!(!ir.is_unsupported());
    }

    #[test]
    fn test_unknown_becomes_explicit_unsupported() {
        let d = mk(Opcode::Unknown(0xfff));
        let ir = lower_to_ir(&d, 16);
        assert!(ir.is_unsupported());
    }

    #[test]
    fn test_hmma_shfl_are_unsupported_not_silent() {
        assert!(lower_to_ir(&mk(Opcode::HMMA), 0).is_unsupported());
        assert!(lower_to_ir(&mk(Opcode::SHFL), 0).is_unsupported());
        assert!(lower_to_ir(&mk(Opcode::LDSM), 0).is_unsupported());
    }

    #[test]
    fn test_branch_target_resolution_is_safe() {
        assert_eq!(resolve_branch_target(0, 64), Some(64));
        // Misaligned target is unresolved, not wrapped.
        assert_eq!(resolve_branch_target(0, 60 + 2), None);
        // Negative overflow is unresolved.
        assert_eq!(resolve_branch_target(0, i32::MIN), None);
    }

    #[test]
    fn test_cfg_splits_at_branch_and_exit() {
        let mut bra = mk(Opcode::BRA);
        bra.pred_reg = 7;
        bra.target_offset = 32;
        let exit = mk(Opcode::EXIT);
        let decoded = vec![mk(Opcode::Nop), bra, mk(Opcode::Nop), exit];
        let ir = lower_stream(&decoded);
        let cfg = build_cfg(&ir);
        // Leaders: 0, fall-through after BRA (idx 2), target idx 2, after EXIT none.
        assert!(cfg.len() >= 2);
        assert_eq!(cfg[0].start_address, 0);
    }

    #[test]
    fn test_fuzz_lower_never_panics() {
        // Every u12 opcode value must lower without panicking.
        for raw in 0..4096u16 {
            let inst = [raw as u32, 0, 0, 0];
            let d = crate::decoder::decode_instruction(&inst);
            let ir = lower_to_ir(&d, 0);
            let _ = ir.is_unsupported();
        }
    }
}
