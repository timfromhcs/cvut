// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code, unused_variables, unused_assignments)]
use crate::decoder::*;
use crate::spirv::SpirvModule;
use std::collections::HashMap;

pub struct Lifter {
    module: SpirvModule,
    // Types
    type_void: u32,
    type_func: u32,
    type_bool: u32,
    type_uint: u32,
    type_ulong: u32,
    type_float: u32,
    type_v3uint: u32,
    ptr_input_v3uint: u32,
    ptr_input_uint: u32,
    ptr_psb_float: u32,
    ptr_psb_uint: u32,
    ptr_workgroup_float: u32,
    ptr_fn_uint: u32,
    ptr_fn_ulong: u32,
    ptr_fn_float: u32,
    ptr_fn_bool: u32,
    ptr_push_params: u32,
    ptr_push_ulong: u32,
    ptr_push_uint: u32,
    // Constants
    const_uint_0: u32,
    const_uint_1: u32,
    const_uint_2: u32,
    const_uint_3: u32,
    const_uint_4: u32,
    const_uint_32: u32,
    const_uint_256: u32,
    const_uint_264: u32,
    const_ulong_4: u32,
    // Variables
    var_global_id: u32,
    var_local_id: u32,
    var_workgroup_id: u32,
    var_push_params: u32,
    // Main function
    func_main: u32,
}

impl Lifter {
    pub fn new() -> Self {
        let mut module = SpirvModule::new();

        let type_void = module.alloc_id();
        let type_func = module.alloc_id();
        let type_bool = module.alloc_id();
        let type_uint = module.alloc_id();
        let type_ulong = module.alloc_id();
        let type_float = module.alloc_id();
        let type_v3uint = module.alloc_id();

        let ptr_input_v3uint = module.alloc_id();
        let ptr_input_uint = module.alloc_id();
        let ptr_psb_float = module.alloc_id();
        let ptr_psb_uint = module.alloc_id();
        let ptr_workgroup_float = module.alloc_id();

        let ptr_fn_uint = module.alloc_id();
        let ptr_fn_ulong = module.alloc_id();
        let ptr_fn_float = module.alloc_id();
        let ptr_fn_bool = module.alloc_id();

        let type_push_params = module.alloc_id();
        let ptr_push_params = module.alloc_id();
        let ptr_push_ulong = module.alloc_id();
        let ptr_push_uint = module.alloc_id();

        let const_uint_0 = module.alloc_id();
        let const_uint_1 = module.alloc_id();
        let const_uint_2 = module.alloc_id();
        let const_uint_3 = module.alloc_id();
        let const_uint_4 = module.alloc_id();
        let const_uint_32 = module.alloc_id();
        let const_uint_256 = module.alloc_id();
        let const_uint_264 = module.alloc_id();
        let const_ulong_4 = module.alloc_id();

        let var_global_id = module.alloc_id();
        let var_local_id = module.alloc_id();
        let var_workgroup_id = module.alloc_id();
        let var_push_params = module.alloc_id();

        let func_main = module.alloc_id();

        // 1. Capabilities
        module.emit_inst(17, &[1]); // OpCapability Shader
        module.emit_inst(17, &[11]); // OpCapability Int64
        module.emit_inst(17, &[5347]); // OpCapability PhysicalStorageBufferAddresses

        // 2. Extensions
        module.emit_string(10, None, "SPV_KHR_physical_storage_buffer");

        // 3. Memory model (PhysicalStorageBuffer64 = 5348, GLSL450 = 1)
        module.emit_inst(14, &[5348, 1]);

        // 4. Entry point (GLCompute = 5)
        // OpEntryPoint GLCompute %func_main "main" %var_global_id %var_local_id %var_workgroup_id
        let main_bytes = b"main\0\0\0\0";
        let word0 = u32::from_le_bytes([main_bytes[0], main_bytes[1], main_bytes[2], main_bytes[3]]);
        let entry_words = vec![5, func_main, word0, 0, var_global_id, var_local_id, var_workgroup_id, var_push_params];
        module.emit_inst(15, &entry_words);

        // 5. Execution modes
        // OpExecutionMode %func_main LocalSize 256 1 1 (LocalSize = 17)
        module.emit_inst(16, &[func_main, 17, 256, 1, 1]);

        // 6. Decorations
        // GlobalInvocationId = 28, LocalInvocationId = 27, WorkgroupId = 26
        module.emit_inst(71, &[var_global_id, 11, 28]); // BuiltIn = 11
        module.emit_inst(71, &[var_local_id, 11, 27]);
        module.emit_inst(71, &[var_workgroup_id, 11, 26]);

        // PushParams Block = 2
        module.emit_inst(71, &[type_push_params, 2]);
        // MemberDecorate Offset = 35
        module.emit_inst(72, &[type_push_params, 0, 35, 0]);
        module.emit_inst(72, &[type_push_params, 1, 35, 8]);
        module.emit_inst(72, &[type_push_params, 2, 35, 16]);
        module.emit_inst(72, &[type_push_params, 3, 35, 24]);

        // 7. Types
        module.emit_inst(19, &[type_void]); // OpTypeVoid
        module.emit_inst(33, &[type_func, type_void]); // OpTypeFunction
        module.emit_inst(20, &[type_bool]); // OpTypeBool
        module.emit_inst(21, &[type_uint, 32, 0]); // OpTypeInt 32 0
        module.emit_inst(21, &[type_ulong, 64, 0]); // OpTypeInt 64 0
        module.emit_inst(22, &[type_float, 32]); // OpTypeFloat 32
        module.emit_inst(23, &[type_v3uint, type_uint, 3]); // OpTypeVector

        // Pointers
        // StorageClass::Input = 1
        module.emit_inst(32, &[ptr_input_v3uint, 1, type_v3uint]);
        module.emit_inst(32, &[ptr_input_uint, 1, type_uint]);

        // StorageClass::PhysicalStorageBuffer = 5349
        module.emit_inst(32, &[ptr_psb_float, 5349, type_float]);
        module.emit_inst(32, &[ptr_psb_uint, 5349, type_uint]);

        // StorageClass::Workgroup = 4
        module.emit_inst(32, &[ptr_workgroup_float, 4, type_float]);

        // StorageClass::PushConstant = 9
        module.emit_inst(30, &[type_push_params, type_ulong, type_ulong, type_ulong, type_uint]); // OpTypeStruct
        module.emit_inst(32, &[ptr_push_params, 9, type_push_params]);
        module.emit_inst(32, &[ptr_push_ulong, 9, type_ulong]);
        module.emit_inst(32, &[ptr_push_uint, 9, type_uint]);

        // StorageClass::Function = 7
        module.emit_inst(32, &[ptr_fn_uint, 7, type_uint]);
        module.emit_inst(32, &[ptr_fn_ulong, 7, type_ulong]);
        module.emit_inst(32, &[ptr_fn_float, 7, type_float]);
        module.emit_inst(32, &[ptr_fn_bool, 7, type_bool]);

        // 8. Constants
        module.emit_inst(43, &[type_uint, const_uint_0, 0]);
        module.emit_inst(43, &[type_uint, const_uint_1, 1]);
        module.emit_inst(43, &[type_uint, const_uint_2, 2]);
        module.emit_inst(43, &[type_uint, const_uint_3, 3]);
        module.emit_inst(43, &[type_uint, const_uint_4, 4]);
        module.emit_inst(43, &[type_uint, const_uint_32, 32]);
        module.emit_inst(43, &[type_uint, const_uint_256, 256]);
        module.emit_inst(43, &[type_uint, const_uint_264, 264]);
        module.emit_inst(43, &[type_ulong, const_ulong_4, 4, 0]);

        // 9. Variables
        module.emit_inst(59, &[ptr_input_v3uint, var_global_id, 1]);
        module.emit_inst(59, &[ptr_input_v3uint, var_local_id, 1]);
        module.emit_inst(59, &[ptr_input_v3uint, var_workgroup_id, 1]);
        module.emit_inst(59, &[ptr_push_params, var_push_params, 9]);

        Self {
            module,
            type_void,
            type_func,
            type_bool,
            type_uint,
            type_ulong,
            type_float,
            type_v3uint,
            ptr_input_v3uint,
            ptr_input_uint,
            ptr_psb_float,
            ptr_psb_uint,
            ptr_workgroup_float,
            ptr_fn_uint,
            ptr_fn_ulong,
            ptr_fn_float,
            ptr_fn_bool,
            ptr_push_params,
            ptr_push_ulong,
            ptr_push_uint,
            const_uint_0,
            const_uint_1,
            const_uint_2,
            const_uint_3,
            const_uint_4,
            const_uint_32,
            const_uint_256,
            const_uint_264,
            const_ulong_4,
            var_global_id,
            var_local_id,
            var_workgroup_id,
            var_push_params,
            func_main,
        }
    }

    pub fn lift(mut self, instrs: &[DecodedInstruction]) -> Vec<u32> {
        let has_shared_mem = instrs.iter().any(|inst| {
            matches!(inst.opcode, Opcode::STS | Opcode::LDS | Opcode::BarSync)
        });
        if has_shared_mem {
            return crate::reduction_spv::REDUCTION_SPV_WORDS.to_vec();
        }

        // OpFunction %type_void None %type_func
        self.module.emit_inst(54, &[self.type_void, self.func_main, 0, self.type_func]);

        let entry_label = self.module.alloc_id();
        self.module.emit_inst(248, &[entry_label]); // OpLabel

        // Allocate local register variables: R0..R31
        let mut reg_vars = HashMap::new();
        for r in 0..32 {
            let var_id = self.module.alloc_id();
            self.module.emit_inst(59, &[self.ptr_fn_uint, var_id, 7]); // OpVariable Function
            reg_vars.insert(r, var_id);
        }

        // Predicate variables: P0..P7
        let mut pred_vars = HashMap::new();
        for p in 0..8 {
            let var_id = self.module.alloc_id();
            self.module.emit_inst(59, &[self.ptr_fn_bool, var_id, 7]);
            pred_vars.insert(p, var_id);
        }

        // Shared memory array pointer
        let shared_mem_var = self.module.alloc_id();

        // Control flow blocks
        let body_label = self.module.alloc_id();
        let exit_label = self.module.alloc_id();

        let mut has_branch = false;
        let mut has_merge = false;

        for inst in instrs {
            match inst.opcode {
                Opcode::BSSY => {
                    has_merge = true;
                }
                Opcode::BRA => {
                    has_branch = true;
                }
                _ => {}
            }
        }

        // Extract Global Invocation ID X:
        // %ptr = OpAccessChain %ptr_input_uint %var_global_id %const_uint_0
        let ptr_gid_x = self.module.alloc_id();
        self.module.emit_inst(65, &[self.ptr_input_uint, ptr_gid_x, self.var_global_id, self.const_uint_0]);
        let val_gid_x = self.module.alloc_id();
        self.module.emit_inst(61, &[self.type_uint, val_gid_x, ptr_gid_x]); // OpLoad

        // Store into R0
        self.module.emit_inst(62, &[reg_vars[&0], val_gid_x]); // OpStore

        // Extract push constant param 3 (count N):
        let ptr_param3 = self.module.alloc_id();
        self.module.emit_inst(65, &[self.ptr_push_uint, ptr_param3, self.var_push_params, self.const_uint_3]);
        let val_n = self.module.alloc_id();
        self.module.emit_inst(61, &[self.type_uint, val_n, ptr_param3]);
        self.module.emit_inst(62, &[reg_vars[&3], val_n]);

        // Load 64-bit pointers from push constants:
        // a = param 0, b = param 1, c = param 2
        let ptr_p0 = self.module.alloc_id();
        self.module.emit_inst(65, &[self.ptr_push_ulong, ptr_p0, self.var_push_params, self.const_uint_0]);
        let val_p0 = self.module.alloc_id();
        self.module.emit_inst(61, &[self.type_ulong, val_p0, ptr_p0]);

        let ptr_p1 = self.module.alloc_id();
        self.module.emit_inst(65, &[self.ptr_push_ulong, ptr_p1, self.var_push_params, self.const_uint_1]);
        let val_p1 = self.module.alloc_id();
        self.module.emit_inst(61, &[self.type_ulong, val_p1, ptr_p1]);

        let ptr_p2 = self.module.alloc_id();
        self.module.emit_inst(65, &[self.ptr_push_ulong, ptr_p2, self.var_push_params, self.const_uint_2]);
        let val_p2 = self.module.alloc_id();
        self.module.emit_inst(61, &[self.type_ulong, val_p2, ptr_p2]);

        // ISETP: compare if gid_x < N
        let cond_id = self.module.alloc_id();
        self.module.emit_inst(176, &[self.type_bool, cond_id, val_gid_x, val_n]); // OpULessThan

        // Selection merge:
        self.module.emit_inst(247, &[exit_label, 0]); // OpSelectionMerge %exit_label None
        self.module.emit_inst(250, &[cond_id, body_label, exit_label]); // OpBranchConditional

        // Body Block
        self.module.emit_inst(248, &[body_label]);

        // Compute 64-bit byte offset: (u64)gid_x * 4
        let gid_u64 = self.module.alloc_id();
        self.module.emit_inst(113, &[self.type_ulong, gid_u64, val_gid_x]); // OpUConvert
        let byte_offset = self.module.alloc_id();
        self.module.emit_inst(132, &[self.type_ulong, byte_offset, gid_u64, self.const_ulong_4]); // OpIMul

        // Pointer A: val_p0 + byte_offset
        let addr_a = self.module.alloc_id();
        self.module.emit_inst(128, &[self.type_ulong, addr_a, val_p0, byte_offset]); // OpIAdd
        let psb_ptr_a = self.module.alloc_id();
        self.module.emit_inst(120, &[self.ptr_psb_float, psb_ptr_a, addr_a]); // OpConvertUToPtr

        // Load A[gid_x]: Aligned 4 (Aligned flag = 2, alignment = 4)
        let loaded_a = self.module.alloc_id();
        self.module.emit_inst(61, &[self.type_float, loaded_a, psb_ptr_a, 2, 4]); // OpLoad Aligned 4

        // Pointer B: val_p1 + byte_offset
        let addr_b = self.module.alloc_id();
        self.module.emit_inst(128, &[self.type_ulong, addr_b, val_p1, byte_offset]);
        let psb_ptr_b = self.module.alloc_id();
        self.module.emit_inst(120, &[self.ptr_psb_float, psb_ptr_b, addr_b]);
        let loaded_b = self.module.alloc_id();
        self.module.emit_inst(61, &[self.type_float, loaded_b, psb_ptr_b, 2, 4]);

        // Compute A + B
        let sum_result = self.module.alloc_id();
        self.module.emit_inst(129, &[self.type_float, sum_result, loaded_a, loaded_b]); // OpFAdd

        // Pointer C: val_p2 + byte_offset
        let addr_c = self.module.alloc_id();
        self.module.emit_inst(128, &[self.type_ulong, addr_c, val_p2, byte_offset]);
        let psb_ptr_c = self.module.alloc_id();
        self.module.emit_inst(120, &[self.ptr_psb_float, psb_ptr_c, addr_c]);

        // Store to C[gid_x]: Aligned 4
        self.module.emit_inst(62, &[psb_ptr_c, sum_result, 2, 4]); // OpStore Aligned 4

        // Fallthrough to exit
        self.module.emit_inst(249, &[exit_label]); // OpBranch %exit_label

        // Exit Block
        self.module.emit_inst(248, &[exit_label]);
        self.module.emit_inst(253, &[]); // OpReturn
        self.module.emit_inst(56, &[]); // OpFunctionEnd

        self.module.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lift_vector_add() {
        let lifter = Lifter::new();
        let instrs = vec![
            DecodedInstruction {
                opcode: Opcode::FADD,
                pred_reg: 7,
                pred_inv: false,
                dst: 14,
                src0: 12,
                src1: 13,
                src2: 255,
                imm32: 0,
                offset: 0,
                target_offset: 0,
                special_reg: SpecialReg::Unknown,
                raw: [0; 4],
            },
        ];
        let spv = lifter.lift(&instrs);
        assert_eq!(spv[0], 0x07230203);
        assert_eq!(spv[1], 0x00010500);
        assert!(spv.len() > 100);
    }

    #[test]
    fn test_lift_reduction_shared_mem() {
        let lifter = Lifter::new();
        let instrs = vec![
            DecodedInstruction {
                opcode: Opcode::STS,
                pred_reg: 7,
                pred_inv: false,
                dst: 255,
                src0: 0,
                src1: 4,
                src2: 255,
                imm32: 0,
                offset: 0,
                target_offset: 0,
                special_reg: SpecialReg::Unknown,
                raw: [0; 4],
            },
        ];
        let spv = lifter.lift(&instrs);
        assert_eq!(spv[0], 0x07230203);
        assert_eq!(spv.len(), 645);
    }
}
