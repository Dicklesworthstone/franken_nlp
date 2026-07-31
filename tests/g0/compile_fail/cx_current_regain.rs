//! Negative OQ-35 fixture: the pin does not provide a generic
//! `Cx::<cap::None>::current()` lookup. This proves API absence only; it is
//! not an ambient-authority or post-restriction-regain test.

use asupersync::cx::{Cx, cap};

fn no_generic_restricted_current_api() -> Cx<cap::All> {
    Cx::<cap::None>::current().expect("compile-only fixture has no runtime")
}

fn main() {}
