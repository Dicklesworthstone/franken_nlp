//! Negative OQ-35 fixture: `restrict` must be monotone and cannot restore
//! every capability after a leaf has received `cap::None`.

use asupersync::cx::{Cx, cap};

fn illegal_widening(cx: Cx<cap::None>) -> Cx<cap::All> {
    cx.restrict::<cap::All>()
}

fn main() {}
