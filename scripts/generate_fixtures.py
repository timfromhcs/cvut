import struct
import os

def set_bits(inst, start, end, val):
    # inst is list of 4 32-bit ints
    for b in range(start, end):
        word_idx = b // 32
        bit_idx = b % 32
        bit_val = (val >> (b - start)) & 1
        inst[word_idx] |= (bit_val << bit_idx)

def encode_sass_instr(opcode, pred=7, pred_inv=0, dst=255, src0=255, src1=255, src2=255, imm32=0, offset=0, sr_idx=0, target_offset=0):
    inst = [0, 0, 0, 0]
    set_bits(inst, 0, 12, opcode)
    set_bits(inst, 12, 15, pred)
    set_bits(inst, 15, 16, pred_inv)
    set_bits(inst, 16, 24, dst)
    set_bits(inst, 24, 32, src0)
    set_bits(inst, 32, 40, src1)
    set_bits(inst, 64, 72, src2)
    if imm32 != 0:
        set_bits(inst, 32, 64, imm32)
    if offset != 0:
        set_bits(inst, 40, 64, offset)
    if sr_idx != 0:
        set_bits(inst, 72, 80, sr_idx)
    if target_offset != 0:
        set_bits(inst, 34, 64, target_offset & 0x3fffffff)
    return struct.pack("<4I", *inst)

def build_cubin(section_name, instructions):
    code_bytes = b"".join(instructions)
    
    # Pad code_bytes to 16-byte alignment
    if len(code_bytes) % 16 != 0:
        code_bytes += b"\x00" * (16 - (len(code_bytes) % 16))

    shstrtab = b"\x00" + section_name.encode("ascii") + b"\x00.shstrtab\x00"
    sec_name_off = 1
    shstrtab_name_off = 1 + len(section_name) + 1

    ehdr_size = 64
    code_off = ehdr_size
    shstrtab_off = code_off + len(code_bytes)
    shdr_off = shstrtab_off + len(shstrtab)
    if shdr_off % 8 != 0:
        pad = 8 - (shdr_off % 8)
        shstrtab += b"\x00" * pad
        shdr_off += pad

    # 3 sections: null, text, shstrtab
    shnum = 3
    shentsize = 64
    shstrndx = 2

    # ELF64 Header
    e_ident = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8
    e_type = 2      # ET_EXEC
    e_machine = 190 # EM_CUDA
    e_version = 1
    e_entry = 0
    e_phoff = 0
    e_shoff = shdr_off
    e_flags = 0x0050 # SM 80 (Ampere)
    e_ehsize = ehdr_size
    e_phentsize = 0
    e_phnum = 0
    e_shentsize = shentsize
    e_shnum = shnum
    e_shstrndx = shstrndx

    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        e_ident, e_type, e_machine, e_version, e_entry,
        e_phoff, e_shoff, e_flags, e_ehsize, e_phentsize,
        e_phnum, e_shentsize, e_shnum, e_shstrndx
    )

    # Section Headers
    # 0: NULL
    sh0 = b"\x00" * 64
    # 1: PROGBITS (code)
    sh1 = struct.pack(
        "<IIQQQQIIQQ",
        sec_name_off, 1, 6, 0, code_off, len(code_bytes), 0, 0, 16, 16
    )
    # 2: STRTAB (shstrtab)
    sh2 = struct.pack(
        "<IIQQQQIIQQ",
        shstrtab_name_off, 3, 0, 0, shstrtab_off, len(shstrtab), 0, 0, 1, 0
    )

    return ehdr + code_bytes + shstrtab + sh0 + sh1 + sh2

def main():
    os.makedirs("tests/fixtures", exist_ok=True)

    # Fixture 1: SM80 Vector Add SASS
    # Instructions derived directly from Mesa NAK encodings
    vadd_instrs = [
        # S2R R0, SR_CTAID_X (0x919, SR 0x25)
        encode_sass_instr(0x919, dst=0, sr_idx=0x25),
        # S2R R1, SR_TID_X (0x919, SR 0x21)
        encode_sass_instr(0x919, dst=1, sr_idx=0x21),
        # IMAD R0, R0, 256, R1 (0x024)
        encode_sass_instr(0x024, dst=0, src0=0, imm32=256, src2=1),
        # LDC.U32 R3, c[0x0][0x18] (0x9c0)
        encode_sass_instr(0x9c0, dst=3, offset=24),
        # ISETP.GE.AND P0, PT, R0, R3, PT (0x00c)
        encode_sass_instr(0x00c, dst=0, src0=0, src1=3),
        # BSSY 0x40 (0x945)
        encode_sass_instr(0x945, target_offset=0x40),
        # @P0 BRA 0x40 (0x947)
        encode_sass_instr(0x947, pred=0, pred_inv=0, target_offset=0x40),
        # LDG.E R12, [R4] (0x981)
        encode_sass_instr(0x981, dst=12, src0=4),
        # LDG.E R13, [R6] (0x981)
        encode_sass_instr(0x981, dst=13, src0=6),
        # FADD R14, R12, R13 (0x021)
        encode_sass_instr(0x021, dst=14, src0=12, src1=13),
        # STG.E [R8], R14 (0x986)
        encode_sass_instr(0x986, src0=8, src1=14),
        # BSYNC (0x941)
        encode_sass_instr(0x941),
        # EXIT (0x94d)
        encode_sass_instr(0x94d),
    ]

    cubin_vadd = build_cubin(".text.vector_add", vadd_instrs)
    with open("tests/fixtures/sm80_vector_add.cubin", "wb") as f:
        f.write(cubin_vadd)
    print("Generated tests/fixtures/sm80_vector_add.cubin ({} bytes)".format(len(cubin_vadd)))

    # Fixture 2: SM80 Reduction Pure SASS
    reduc_instrs = [
        # S2R R0, SR_TID_X (0x919, SR 0x21)
        encode_sass_instr(0x919, dst=0, sr_idx=0x21),
        # S2R R1, SR_CTAID_X (0x919, SR 0x25)
        encode_sass_instr(0x919, dst=1, sr_idx=0x25),
        # LDG.E R4, [R2] (Load input element)
        encode_sass_instr(0x981, dst=4, src0=2),
        # STS [R0], R4 (Store to shared memory)
        encode_sass_instr(0x988, src0=0, src1=4),
        # BAR.SYNC (0xb1d)
        encode_sass_instr(0xb1d),
        # LDS R5, [R0+16] (Load partner from shared mem)
        encode_sass_instr(0x984, dst=5, src0=0, offset=16),
        # FADD R4, R4, R5
        encode_sass_instr(0x021, dst=4, src0=4, src1=5),
        # BAR.SYNC
        encode_sass_instr(0xb1d),
        # STG.E [R3], R4 (Store reduced result)
        encode_sass_instr(0x986, src0=3, src1=4),
        # EXIT
        encode_sass_instr(0x94d),
    ]

    cubin_reduc = build_cubin(".text.reduction", reduc_instrs)
    with open("tests/fixtures/sm80_reduction_pure_sass.cubin", "wb") as f:
        f.write(cubin_reduc)
    print("Generated tests/fixtures/sm80_reduction_pure_sass.cubin ({} bytes)".format(len(cubin_reduc)))

if __name__ == "__main__":
    main()
