//! Test-only i64 oracle and deterministic scalar/SIMD differential fixtures.

use sha2::{Digest, Sha256};

use franken_nlp::native_engine::int8::{
    MAX_S8_S8_K_10752, MAX_U8_S8_CORRECTION_K_10752, MAX_U8_S8_RAW_K_10752,
    MAX_U8_S8_RAW_PLUS_CORRECTION_K_10752, MODEL_KS, MODEL_NS, dot_s8s8, dot_u8s8_xor128,
    unpack_int4_to_i8,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixtureShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DomainProfile {
    ExhaustiveSmall,
    RandomFull { cases: usize },
    AllExtremes,
    AlternatingExtremes,
}

impl DomainProfile {
    pub const fn label(self) -> &'static str {
        match self {
            Self::ExhaustiveSmall => "exhaustive-small",
            Self::RandomFull { .. } => "random-full-domain",
            Self::AllExtremes => "all-extremes",
            Self::AlternatingExtremes => "alternating-extremes",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DotFixture {
    pub shape: FixtureShape,
    pub seed: u64,
    pub profile: DomainProfile,
    pub input: Vec<i8>,
    pub weights: Vec<i8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OffsetFixture {
    pub k: usize,
    pub seed: u64,
    pub input: Vec<u8>,
    pub weights: Vec<i8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Int4Fixture {
    pub group_size: usize,
    pub seed: u64,
    pub input: Vec<i8>,
    pub packed_weights: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OracleReport {
    pub expected: i32,
    pub max_abs_product: i64,
    pub max_abs_intermediate: i64,
    pub bound: i64,
}

/// Deterministic fixture generator for scalar and SIMD differential tests.
#[derive(Clone, Copy, Debug)]
pub struct FixtureGenerator {
    seed: u64,
}

impl FixtureGenerator {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate the specified full-domain fixture class for one shape.
    #[must_use]
    pub fn dot_fixtures(self, shape: FixtureShape, profile: DomainProfile) -> Vec<DotFixture> {
        match profile {
            DomainProfile::ExhaustiveSmall => exhaustive_small(shape, self.seed),
            DomainProfile::RandomFull { cases } => (0..cases)
                .map(|case| random_fixture(shape, self.seed.wrapping_add(case as u64), profile))
                .collect(),
            DomainProfile::AllExtremes => all_extreme_fixtures(shape, self.seed),
            DomainProfile::AlternatingExtremes => vec![alternating_fixture(shape, self.seed)],
        }
    }

    /// Generate all tail classes `0/1/W-1/W/W+1` for a candidate vector width.
    #[must_use]
    pub fn tail_fixtures(self, width: usize) -> Vec<DotFixture> {
        let mut lengths = vec![
            0,
            1,
            width.saturating_sub(1),
            width,
            width.saturating_add(1),
        ];
        lengths.sort_unstable();
        lengths.dedup();
        lengths
            .into_iter()
            .enumerate()
            .map(|(index, k)| {
                random_fixture(
                    FixtureShape { m: 1, n: 1, k },
                    self.seed.wrapping_add(index as u64),
                    DomainProfile::RandomFull { cases: 1 },
                )
            })
            .collect()
    }

    /// Offset-domain raw/correction extrema for the canonical XOR-0x80 path.
    #[must_use]
    pub fn offset_extremes(self, k: usize) -> Vec<OffsetFixture> {
        [
            (0_u8, -128_i8),
            (255_u8, -128_i8),
            (0_u8, 127_i8),
            (255_u8, 127_i8),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (activation, weight))| OffsetFixture {
            k,
            seed: self.seed.wrapping_add(index as u64),
            input: vec![activation; k],
            weights: vec![weight; k],
        })
        .collect()
    }

    /// Int4 group fixtures cover both nibbles, sign extension, and extremes.
    #[must_use]
    pub fn int4_group_fixtures(self, group_size: usize) -> Vec<Int4Fixture> {
        assert!(
            group_size > 0 && group_size % 2 == 0,
            "int4 group must be nonzero and even"
        );
        let patterns = [(0x88_u8, -128_i8), (0x77_u8, 127_i8), (0x78_u8, -128_i8)];
        patterns
            .into_iter()
            .enumerate()
            .map(|(index, (packed, activation))| Int4Fixture {
                group_size,
                seed: self.seed.wrapping_add(index as u64),
                input: vec![activation; group_size],
                packed_weights: vec![packed; group_size / 2],
            })
            .collect()
    }
}

/// Check the declared K=10752 bounds and their smaller-K instances numerically.
pub fn verify_declared_bounds() -> Result<(), String> {
    for k in MODEL_KS {
        let signed = s8_s8_bound(k);
        let raw = u8_s8_raw_bound(k);
        let correction = u8_s8_correction_bound(k);
        let conservative = raw + correction;
        for (name, value) in [
            ("s8*s8", signed),
            ("u8*s8 raw", raw),
            ("u8*s8 correction", correction),
            ("u8*s8 conservative", conservative),
        ] {
            if value > i64::from(i32::MAX) {
                return Err(format!("bound overflow k={k} class={name} value={value}"));
            }
        }
        if k == 10752
            && (signed != MAX_S8_S8_K_10752
                || raw != MAX_U8_S8_RAW_K_10752
                || correction != MAX_U8_S8_CORRECTION_K_10752
                || conservative != MAX_U8_S8_RAW_PLUS_CORRECTION_K_10752)
        {
            return Err(format!(
                "published K=10752 bounds drifted signed={signed} raw={raw} correction={correction} conservative={conservative}"
            ));
        }
    }
    Ok(())
}

/// Recompute a signed dot in i64 and compare every i32-stage result.
pub fn verify_s8_fixture(fixture: &DotFixture) -> Result<OracleReport, String> {
    if fixture.input.len() != fixture.weights.len() || fixture.input.len() != fixture.shape.k {
        return Err(format!(
            "shape/input mismatch shape={:?} seed={} input={} weights={}",
            fixture.shape,
            fixture.seed,
            fixture.input.len(),
            fixture.weights.len()
        ));
    }
    let mut sum = 0_i64;
    let mut max_abs_product = 0_i64;
    let mut max_abs_intermediate = 0_i64;
    for (index, (&input, &weight)) in fixture.input.iter().zip(&fixture.weights).enumerate() {
        let product = i64::from(input) * i64::from(weight);
        sum += product;
        max_abs_product = max_abs_product.max(product.abs());
        max_abs_intermediate = max_abs_intermediate.max(sum.abs());
        if sum < i64::from(i32::MIN) || sum > i64::from(i32::MAX) {
            return Err(format!(
                "i32-range violation shape={:?} seed={} element={} i64={} bound={}",
                fixture.shape,
                fixture.seed,
                index,
                sum,
                s8_s8_bound(fixture.shape.k)
            ));
        }
    }
    let expected = i32::try_from(sum).map_err(|_| "checked range must narrow".to_owned())?;
    let actual = dot_s8s8(&fixture.input, &fixture.weights).map_err(|error| error.to_string())?;
    if actual != expected {
        return Err(format!(
            "scalar/i64 mismatch shape={:?} seed={} i32={} i64={} max_intermediate={} bound={}",
            fixture.shape,
            fixture.seed,
            actual,
            sum,
            max_abs_intermediate,
            s8_s8_bound(fixture.shape.k)
        ));
    }
    let bound = s8_s8_bound(fixture.shape.k);
    if max_abs_intermediate > bound {
        return Err(format!(
            "signed bound violation shape={:?} seed={} max_intermediate={} bound={bound}",
            fixture.shape, fixture.seed, max_abs_intermediate
        ));
    }
    Ok(OracleReport {
        expected,
        max_abs_product,
        max_abs_intermediate,
        bound,
    })
}

/// Recompute raw u8 MAC, row sum, and correction in i64 before narrowing.
pub fn verify_offset_fixture(fixture: &OffsetFixture) -> Result<OracleReport, String> {
    if fixture.input.len() != fixture.weights.len() || fixture.input.len() != fixture.k {
        return Err(format!(
            "offset shape/input mismatch k={} seed={} input={} weights={}",
            fixture.k,
            fixture.seed,
            fixture.input.len(),
            fixture.weights.len()
        ));
    }
    let mut raw = 0_i64;
    let mut row_sum = 0_i64;
    let mut max_abs_product = 0_i64;
    let mut max_abs_intermediate = 0_i64;
    for (index, (&input, &weight)) in fixture.input.iter().zip(&fixture.weights).enumerate() {
        let product = i64::from(input) * i64::from(weight);
        raw += product;
        row_sum += i64::from(weight);
        let correction = 128_i64 * row_sum;
        let corrected = raw - correction;
        max_abs_product = max_abs_product.max(product.abs());
        max_abs_intermediate = max_abs_intermediate
            .max(raw.abs())
            .max(correction.abs())
            .max(corrected.abs());
        for (stage, value, bound) in [
            ("raw", raw, u8_s8_raw_bound(fixture.k)),
            ("correction", correction, u8_s8_correction_bound(fixture.k)),
            (
                "raw+correction",
                raw.abs() + correction.abs(),
                u8_s8_raw_bound(fixture.k) + u8_s8_correction_bound(fixture.k),
            ),
        ] {
            if value.abs() > bound || value < i64::from(i32::MIN) || value > i64::from(i32::MAX) {
                return Err(format!(
                    "offset stage violation k={} seed={} element={} stage={} i64={} bound={}",
                    fixture.k, fixture.seed, index, stage, value, bound
                ));
            }
        }
    }
    let correction = 128_i64 * row_sum;
    let expected_i64 = raw - correction;
    let expected =
        i32::try_from(expected_i64).map_err(|_| "offset result cannot narrow".to_owned())?;
    let actual =
        dot_u8s8_xor128(&fixture.input, &fixture.weights).map_err(|error| error.to_string())?;
    if actual != expected {
        return Err(format!(
            "offset scalar/i64 mismatch k={} seed={} i32={} i64={} raw={} correction={} max_intermediate={}",
            fixture.k, fixture.seed, actual, expected_i64, raw, correction, max_abs_intermediate
        ));
    }
    Ok(OracleReport {
        expected,
        max_abs_product,
        max_abs_intermediate,
        bound: u8_s8_raw_bound(fixture.k) + u8_s8_correction_bound(fixture.k),
    })
}

/// Verify signed-int4 unpacking remains in the i8 MAC domain and matches i64.
pub fn verify_int4_fixture(fixture: &Int4Fixture) -> Result<OracleReport, String> {
    let weights = unpack_int4_to_i8(&fixture.packed_weights);
    if weights.len() != fixture.group_size || fixture.input.len() != fixture.group_size {
        return Err(format!(
            "int4 group mismatch group={} seed={} input={} unpacked={}",
            fixture.group_size,
            fixture.seed,
            fixture.input.len(),
            weights.len()
        ));
    }
    if weights.iter().any(|&weight| !(-8..=7).contains(&weight)) {
        return Err(format!(
            "int4 sign-extension violation group={} seed={}",
            fixture.group_size, fixture.seed
        ));
    }
    verify_s8_fixture(&DotFixture {
        shape: FixtureShape {
            m: 1,
            n: 1,
            k: fixture.group_size,
        },
        seed: fixture.seed,
        profile: DomainProfile::AllExtremes,
        input: fixture.input.clone(),
        weights,
    })
}

/// Hash a fixture's exact generated bytes for deterministic consumer checks.
#[must_use]
pub fn fixture_digest(fixture: &DotFixture) -> String {
    let mut digest = Sha256::new();
    digest.update(fixture.seed.to_le_bytes());
    digest.update((fixture.shape.m as u64).to_le_bytes());
    digest.update((fixture.shape.n as u64).to_le_bytes());
    digest.update((fixture.shape.k as u64).to_le_bytes());
    digest.update(fixture.profile.label().as_bytes());
    digest.update([0]);
    for &value in &fixture.input {
        digest.update(value.to_le_bytes());
    }
    digest.update([0xff]);
    for &value in &fixture.weights {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// Shape descriptors cover every fixed K/N pair without allocating model-sized matrices.
#[must_use]
pub fn model_shape_catalog() -> Vec<FixtureShape> {
    MODEL_KS
        .into_iter()
        .flat_map(|k| {
            MODEL_NS
                .into_iter()
                .map(move |n| FixtureShape { m: 1, n, k })
        })
        .collect()
}

pub const fn s8_s8_bound(k: usize) -> i64 {
    (k as i64) * 16_384
}

pub const fn u8_s8_raw_bound(k: usize) -> i64 {
    (k as i64) * 32_640
}

pub const fn u8_s8_correction_bound(k: usize) -> i64 {
    (k as i64) * 16_384
}

fn exhaustive_small(shape: FixtureShape, seed: u64) -> Vec<DotFixture> {
    assert_eq!(
        shape.k, 1,
        "full-domain exhaustive fixtures are intentionally K=1"
    );
    let mut fixtures = Vec::with_capacity(256 * 256);
    for input in i8::MIN..=i8::MAX {
        for weight in i8::MIN..=i8::MAX {
            fixtures.push(DotFixture {
                shape,
                seed,
                profile: DomainProfile::ExhaustiveSmall,
                input: vec![input],
                weights: vec![weight],
            });
        }
    }
    fixtures
}

fn random_fixture(shape: FixtureShape, seed: u64, profile: DomainProfile) -> DotFixture {
    let mut state = seed;
    let input = (0..shape.k).map(|_| next_i8(&mut state)).collect();
    let weights = (0..shape.k).map(|_| next_i8(&mut state)).collect();
    DotFixture {
        shape,
        seed,
        profile,
        input,
        weights,
    }
}

fn all_extreme_fixtures(shape: FixtureShape, seed: u64) -> Vec<DotFixture> {
    [
        (-128_i8, -128_i8),
        (127_i8, 127_i8),
        (-128_i8, 127_i8),
        (127_i8, -128_i8),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (input, weight))| DotFixture {
        shape,
        seed: seed.wrapping_add(index as u64),
        profile: DomainProfile::AllExtremes,
        input: vec![input; shape.k],
        weights: vec![weight; shape.k],
    })
    .collect()
}

fn alternating_fixture(shape: FixtureShape, seed: u64) -> DotFixture {
    let input = (0..shape.k)
        .map(|index| if index % 2 == 0 { -128 } else { 127 })
        .collect();
    let weights = (0..shape.k)
        .map(|index| if index % 2 == 0 { 127 } else { -128 })
        .collect();
    DotFixture {
        shape,
        seed,
        profile: DomainProfile::AlternatingExtremes,
        input,
        weights,
    }
}

fn next_i8(state: &mut u64) -> i8 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (value ^ (value >> 31)) as u8 as i8
}
