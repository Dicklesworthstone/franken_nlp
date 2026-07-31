//! Negative OQ-35 fixture: a statically restricted `Cx<cap::None>` lookup
//! cannot be assigned an all-capability context. Ambient `Cx::current()`
//! runtime-mask behavior is covered by the ordinary census semantics probe,
//! not this type-only fixture.

use asupersync::cx::{Cx, cap};

fn illegal_static_current_retype() -> Cx<cap::All> {
    Cx::<cap::None>::current().expect("compile-only fixture has no runtime")
}

fn main() {}
