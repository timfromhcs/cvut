// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::manual_is_multiple_of)]

mod decoder;
mod elf;
mod lifter;
mod reduction_spv;
mod spirv;

use std::env;
use std::fs;
use std::path::Path;
use std::process::exit;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut input_path = String::new();
    let mut output_path = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                if i + 1 < args.len() {
                    input_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = args[i + 1].clone();
                    i += 1;
                }
            }
            _ => {
                if args[i].starts_with("--input=") {
                    input_path = args[i]["--input=".len()..].to_string();
                } else if args[i].starts_with("--output=") {
                    output_path = args[i]["--output=".len()..].to_string();
                }
            }
        }
        i += 1;
    }

    if input_path.is_empty() || output_path.is_empty() {
        eprintln!("Usage: sass_lifter --input <cubin_file> --output <spv_file>");
        exit(1);
    }

    println!("[SASS_LIFTER] Reading input binary: {}", input_path);
    let bytes = match fs::read(&input_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read input file {}: {}", input_path, e);
            exit(1);
        }
    };

    let raw_instrs = match elf::parse_cubin_instructions(&bytes) {
        Ok(instrs) => instrs,
        Err(e) => {
            eprintln!("Failed to parse cubin instructions: {}", e);
            exit(1);
        }
    };

    println!(
        "[SASS_LIFTER] Parsed {} 128-bit SASS instructions.",
        raw_instrs.len()
    );

    let mut decoded = Vec::new();
    for inst in &raw_instrs {
        let d = decoder::decode_instruction(inst);
        println!("[SASS_LIFTER] Decoded opcode: {:?}", d.opcode);
        decoded.push(d);
    }

    let lifter = lifter::Lifter::new();
    let spv_words = lifter.lift(&decoded);

    let mut spv_bytes = Vec::with_capacity(spv_words.len() * 4);
    for word in spv_words {
        spv_bytes.extend_from_slice(&word.to_le_bytes());
    }

    if let Some(parent) = Path::new(&output_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    if let Err(e) = fs::write(&output_path, &spv_bytes) {
        eprintln!("Failed to write SPIR-V output to {}: {}", output_path, e);
        exit(1);
    }

    println!(
        "[SASS_LIFTER] Successfully lifted SASS to SPIR-V: {} ({} bytes)",
        output_path,
        spv_bytes.len()
    );
}
