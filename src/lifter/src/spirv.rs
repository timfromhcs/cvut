// SPDX-License-Identifier: Apache-2.0
// SPIR-V 1.5 binary generator for Vulkan 1.3 compute shaders

#[derive(Default)]
pub struct SpirvModule {
    pub words: Vec<u32>,
    pub next_id: u32,
}

impl SpirvModule {
    pub fn new() -> Self {
        Self {
            words: Vec::new(),
            next_id: 1,
        }
    }

    pub fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn emit_inst(&mut self, opcode: u16, operands: &[u32]) {
        let word_count = (operands.len() + 1) as u32;
        self.words.push((word_count << 16) | (opcode as u32));
        self.words.extend_from_slice(operands);
    }

    pub fn emit_string(&mut self, opcode: u16, result_id: Option<u32>, s: &str) {
        let mut string_words = Vec::new();
        if let Some(id) = result_id {
            string_words.push(id);
        }
        let bytes = s.as_bytes();
        let mut cur_word = 0u32;
        let mut shift = 0;
        for &b in bytes {
            cur_word |= (b as u32) << shift;
            shift += 8;
            if shift == 32 {
                string_words.push(cur_word);
                cur_word = 0;
                shift = 0;
            }
        }
        string_words.push(cur_word); // null terminator word
        self.emit_inst(opcode, &string_words);
    }

    pub fn finalize(self) -> Vec<u32> {
        let mut full_binary = Vec::with_capacity(5 + self.words.len());
        full_binary.push(0x07230203); // Magic number
        full_binary.push(0x00010500); // Version 1.5
        full_binary.push(0x00000000); // Generator ID
        full_binary.push(self.next_id); // Bound
        full_binary.push(0x00000000); // Schema (reserved)
        full_binary.extend(self.words);
        full_binary
    }
}
