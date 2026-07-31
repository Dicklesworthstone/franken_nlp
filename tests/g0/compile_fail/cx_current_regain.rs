//! Negative OQ-35 fixture: a lookup through the statically restricted
//! `Cx<cap::None>` view cannot be assigned an all-capability context.

use asupersync::cx::{Cx, cap};

fn illegal_current_regain() -> Cx<cap::All> {
    Cx::<cap::None>::current().expect("compile-only fixture has no runtime")
}

fn main() {}
