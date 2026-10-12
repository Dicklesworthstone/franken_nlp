#!/usr/bin/env bash
# Finite dispatched/cohort integer smoke, not full fixed-N model qualification.
# Never builds a binary. The controller supplies its already-built FNLP_BIN.
# Unlike the old inventory expansion, every PASS below requires actual checks.
set -euo pipefail

fnlp_bin="${FNLP_BIN:-fnlp}"
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if ! command -v jq >/dev/null 2>&1; then
  printf 'E2E_SUMMARY timestamp=%s status=FAIL scope=finite_synthetic_int8 reason=missing_jq\n' "$timestamp" >&2
  exit 1
fi

# Validate the complete finite transcript before forwarding its records.
# pipefail preserves a failing fnlp exit even if it prints a terminal pass.
if "$fnlp_bin" robot selftest --details | jq -sec '
  def tier_id: if . == "scalar" then "s_scalar_i32" else . end;
  def case_key: [.operation, .pattern, .m, .k, .n];
  def expected_cases:
    [ [0,1,7,8,9,31,32,33,63,64,65,127,128,129,3072,6144,10751,10752][] as $k
      | [["gemv",1,3],["gemm",1,3],["gemm",4,3],["gemm",5,3],["cohort",4,1]][] as $shape
      | ["extremes","mixed"][] as $pattern
      | [$shape[0],$pattern,$shape[1],$k,$shape[2]] ] | sort;
  def detected($f; $tier):
    if $tier == "scalar" then true
    elif $f.architecture == "aarch64" then
      if $tier == "a1_smmla" then $f.aarch64_i8mm == true
      elif $tier == "a2_dotprod" then $f.aarch64_dotprod == true
      else $tier == "a3_autovec" end
    elif $f.architecture == "x86_64" then
      if $tier == "x3a_avx2_low7_high_bit" or $tier == "x3b_avx2_widened_i16" then $f.x86_avx2 == true
      elif $tier == "x2_avx_vnni" then $f.x86_avx2 == true and $f.x86_avxvnni == true
      elif $tier == "x1b_avx512_vnni_ymm" then
        $f.x86_avx2 == true and $f.x86_avx512f == true and $f.x86_avx512vnni == true and $f.x86_avx512vl == true
      elif $tier == "x1a_avx512_vnni_zmm" then $f.x86_avx512f == true and $f.x86_avx512vnni == true
      else false end
    else false end;
  (["scalar","a1_smmla","a2_dotprod","a3_autovec","x1a_avx512_vnni_zmm",
    "x1b_avx512_vnni_ymm","x2_avx_vnni","x3a_avx2_low7_high_bit","x3b_avx2_widened_i16"] | sort) as $known
  | . as $records
  | .[0].data as $start | .[-1].data as $complete
  | [.[] | select(.event == "selftest_case")] as $cases
  | [.[] | select(.event == "selftest_tier")] as $tiers
  | expected_cases as $expected
  | if
      (length >= 3 and length <= 1631)
      and all(.[]; .protocol == "fnlp.cpu-selftest" and .schema_version == 1
        and (.event == "selftest_start" or .event == "selftest_case" or .event == "selftest_tier" or .event == "selftest_complete"))
      and .[0].event == "selftest_start" and .[-1].event == "selftest_complete"
      and ([.[] | select(.event == "selftest_start")] | length) == 1
      and ([.[] | select(.event == "selftest_complete")] | length) == 1
      and $start.suite_version == "cpu-int8-execution-v1" and $complete.suite_version == $start.suite_version
      and $start.cases_per_tier == 180 and $complete.cases_per_tier == 180
      and ($start.detected_features | keys) == (["architecture","aarch64_dotprod","aarch64_i8mm","x86_avx2","x86_avx512f","x86_avx512vl","x86_avx512vnni","x86_avxvnni"] | sort)
      and ($start.detected_features.architecture == "aarch64" or $start.detected_features.architecture == "x86_64" or $start.detected_features.architecture == "other")
      and all($start.detected_features | to_entries[] | select(.key != "architecture"); (.value | type) == "boolean")
      and $start.requested_tier == null and $complete.requested_tier == null
      and $complete.scope == "finite_synthetic_integer_cases_only_not_model_or_performance_qualification"
      and $complete.verdict == "pass" and $complete.cases_run > 0
      and ([$tiers[].data.tier] | sort) == $known
      and $complete.tiers_considered == 9
      and $complete.tiers_skipped == ([$tiers[] | select(.data.status == "skipped_unsupported_isa")] | length)
      and $complete.cases_run == ($cases | length) and $complete.cases_passed == $complete.cases_run
      and $complete.cases_run == ([$tiers[].data.cases_run] | add)
      and all($tiers[];
        .data as $tier
        | ([$cases[] | select(.data.tier == $tier.tier) | .data | case_key] | sort) as $seen
        | if $tier.status == "pass" then
            detected($start.detected_features; $tier.tier)
            and $tier.cases_run == 180 and $tier.cases_passed == 180 and $seen == $expected
          else $tier.status == "skipped_unsupported_isa"
            and (detected($start.detected_features; $tier.tier) | not)
            and $tier.cases_run == 0 and $tier.cases_passed == 0 and ($seen | length) == 0 end)
      and all($cases[];
        .data as $case
        | ($case.tier | tier_id) as $id
        | $case.failure == null and $case.first_mismatch == null
          and $case.observed_kernel == $case.requested_kernel
          and $case.requested_kernel == (if $case.operation == "cohort" then $id else $id + "_int8_" + $case.operation end)
          and ($case.expected_sha256 | test("^[0-9a-f]{64}$"))
          and $case.actual_sha256 == $case.expected_sha256)
    then $records[] else error("incomplete or failed finite CPU selftest transcript") end
'; then
  printf 'E2E_SUMMARY timestamp=%s status=PASS scope=finite_synthetic_int8 source=robot_selftest\n' "$timestamp" >&2
else
  printf 'E2E_SUMMARY timestamp=%s status=FAIL scope=finite_synthetic_int8 reason=selftest_or_transcript_failure\n' "$timestamp" >&2
  exit 1
fi
