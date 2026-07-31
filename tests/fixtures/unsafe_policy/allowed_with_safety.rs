#![allow(unsafe_code)]

pub fn probe() {
    // SAFETY: This parser fixture performs no operation inside the block.
    unsafe {}
}
