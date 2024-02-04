#![no_main]

use libfuzzer_sys::fuzz_target;
use scrapscript_rs::tokenize;

fuzz_target!(|data: &[u8]| {
    if let Ok(data) = std::str::from_utf8(data) {
        let tokens = tokenize(data);
    }
    // fuzzed code goes here
});
