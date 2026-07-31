#!/bin/sh
# Local-fixture tests for fetch_model.sh. The retained fixture directory is
# printed on completion so an operator can inspect every quarantine/journal.

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
FETCH="$ROOT/scripts/fetch_model.sh"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/fnlp-fetch-model-test.XXXXXX")
REV=f56ec5a9650268aa098496734743c25ea778bd2d
CASES=0
FAILED=
SERVER_PID=

log() { printf '%s FETCH_MODEL_TEST %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
fail_case() { FAILED="${FAILED}${FAILED:+,}$1"; log "CASE=$1 RESULT=FAIL detail=$2"; }
pass_case() { log "CASE=$1 RESULT=PASS detail=$2"; }
finish() {
    if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi
    if [ -n "$FAILED" ]; then
        log "FETCH_MODEL_TESTS RESULT=FAIL cases=$CASES failed=$FAILED retained_work=$WORK"
        exit 1
    fi
    log "FETCH_MODEL_TESTS RESULT=PASS cases=$CASES failed=none retained_work=$WORK"
}
trap finish EXIT

command -v python3 >/dev/null 2>&1 || { log "FETCH_MODEL_TESTS RESULT=FAIL cases=0 failed=missing-python3 retained_work=$WORK"; exit 1; }

FIXTURE="$WORK/fixture/resolve/$REV"
mkdir -p "$FIXTURE"
printf 'alpha fixture bytes\n' > "$FIXTURE/alpha.bin"
printf 'beta fixture bytes are slightly longer\n' > "$FIXTURE/beta.bin"
ALPHA_BYTES=$(wc -c < "$FIXTURE/alpha.bin" | tr -d '[:space:]')
BETA_BYTES=$(wc -c < "$FIXTURE/beta.bin" | tr -d '[:space:]')
ALPHA_SHA=$(shasum -a 256 "$FIXTURE/alpha.bin" | awk '{print $1}')
BETA_SHA=$(shasum -a 256 "$FIXTURE/beta.bin" | awk '{print $1}')
CATALOG="$WORK/catalog.txt"
printf 'alpha.bin|%s|%s\nbeta.bin|%s|%s\n' "$ALPHA_BYTES" "$ALPHA_SHA" "$BETA_BYTES" "$BETA_SHA" > "$CATALOG"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$WORK/fixture" > "$WORK/server.log" 2>&1 &
SERVER_PID=$!
BASE="http://127.0.0.1:$PORT/resolve/$REV"

run_fetch() {
    dest=$1
    shift
    FNLP_FETCH_ALLOW_TEST_BASE_URL=1 "$FETCH" --dest "$dest" --catalog "$CATALOG" --test-base-url "$BASE" "$@"
}

secure_mkdir() {
    old_umask=$(umask)
    umask 077
    mkdir -p "$1"
    umask "$old_umask"
}

# 1. Fresh download verifies length/digest and atomically activates both files.
CASES=$((CASES + 1)); DEST1="$WORK/case1"
if run_fetch "$DEST1" > "$WORK/case1.out" 2> "$WORK/case1.err" && cmp "$FIXTURE/alpha.bin" "$DEST1/alpha.bin" >/dev/null && cmp "$FIXTURE/beta.bin" "$DEST1/beta.bin" >/dev/null; then pass_case 1 fresh-download; else fail_case 1 fresh-download; fi

# 2. A secure matching journal makes a truncated partial resume with Range.
CASES=$((CASES + 1)); DEST2="$WORK/case2"; secure_mkdir "$DEST2/.fnlp-fetch-journals"
head -c 5 "$FIXTURE/alpha.bin" > "$DEST2/alpha.bin.partial"
printf 'url=%s/alpha.bin\nrevision=%s\nname=alpha.bin\nbytes=%s\nsha256=%s\n' "$BASE" "$REV" "$ALPHA_BYTES" "$ALPHA_SHA" > "$DEST2/.fnlp-fetch-journals/alpha.bin.journal"
if run_fetch "$DEST2" > "$WORK/case2.out" 2> "$WORK/case2.err" && cmp "$FIXTURE/alpha.bin" "$DEST2/alpha.bin" >/dev/null && grep -q 'DECISION=RESUME file=alpha.bin' "$WORK/case2.err"; then pass_case 2 journal-resume; else fail_case 2 journal-resume; fi

# 3. A partial without its journal is quarantined and restarted from zero.
CASES=$((CASES + 1)); DEST3="$WORK/case3"; secure_mkdir "$DEST3"
printf 'unbound bytes' > "$DEST3/alpha.bin.partial"
if run_fetch "$DEST3" > "$WORK/case3.out" 2> "$WORK/case3.err" && cmp "$FIXTURE/alpha.bin" "$DEST3/alpha.bin" >/dev/null && grep -q 'unbound-or-mismatched-partial' "$DEST3/quarantine/quarantine.log"; then pass_case 3 unbound-partial-quarantine; else fail_case 3 unbound-partial-quarantine; fi

# 4. A symlinked partial is refused rather than followed or quarantined as data.
CASES=$((CASES + 1)); DEST4="$WORK/case4"; secure_mkdir "$DEST4"
ln -s "$FIXTURE/alpha.bin" "$DEST4/alpha.bin.partial"
if run_fetch "$DEST4" > "$WORK/case4.out" 2> "$WORK/case4.err"; then fail_case 4 symlink-refusal; elif grep -q 'refused symlink partial' "$WORK/case4.err"; then pass_case 4 symlink-refusal; else fail_case 4 symlink-refusal; fi

# 5. A corrupt final file is retained in quarantine with its observed digest.
CASES=$((CASES + 1)); DEST5="$WORK/case5"; secure_mkdir "$DEST5"
printf 'corrupt old final' > "$DEST5/alpha.bin"
if run_fetch "$DEST5" > "$WORK/case5.out" 2> "$WORK/case5.err" && cmp "$FIXTURE/alpha.bin" "$DEST5/alpha.bin" >/dev/null && grep -q 'existing-file-verification-failed' "$DEST5/quarantine/quarantine.log"; then pass_case 5 corrupt-final-quarantine; else fail_case 5 corrupt-final-quarantine; fi

# 6. Check-only fully hashes: complete passes, one tampered byte fails with both values.
CASES=$((CASES + 1)); DEST6="$WORK/case6"
if run_fetch "$DEST6" > "$WORK/case6-download.out" 2> "$WORK/case6-download.err" && FNLP_FETCH_ALLOW_TEST_BASE_URL=1 "$FETCH" --dest "$DEST6" --catalog "$CATALOG" --test-base-url "$BASE" --check-only > "$WORK/case6-pass.out" 2> "$WORK/case6-pass.err"; then
    printf 'X' >> "$DEST6/alpha.bin"
    if FNLP_FETCH_ALLOW_TEST_BASE_URL=1 "$FETCH" --dest "$DEST6" --catalog "$CATALOG" --test-base-url "$BASE" --check-only > "$WORK/case6-fail.out" 2> "$WORK/case6-fail.err"; then fail_case 6 check-only-tamper; elif grep -q 'expected_sha256=' "$WORK/case6-fail.err" && grep -q 'observed_sha256=' "$WORK/case6-fail.err"; then pass_case 6 check-only-tamper; else fail_case 6 check-only-tamper; fi
else fail_case 6 check-only-setup; fi

# 7. A non-default revision cannot inherit the default catalog identity.
CASES=$((CASES + 1)); DEST7="$WORK/case7"
if "$FETCH" --dest "$DEST7" --revision deadbeef > "$WORK/case7.out" 2> "$WORK/case7.err"; then fail_case 7 untrusted-revision-refusal; elif [ "$?" -eq 2 ] && grep -q 'UNTRUSTED_REVISION_REFUSED' "$WORK/case7.err"; then pass_case 7 untrusted-revision-refusal; else fail_case 7 untrusted-revision-refusal; fi

# 8. A failed resume prints the exact same --dest invocation and journal path.
CASES=$((CASES + 1)); DEST8="$WORK/case8"; secure_mkdir "$DEST8/.fnlp-fetch-journals"
head -c 5 "$FIXTURE/alpha.bin" > "$DEST8/alpha.bin.partial"
printf 'url=http://127.0.0.1:1/alpha.bin\nrevision=%s\nname=alpha.bin\nbytes=%s\nsha256=%s\n' "$REV" "$ALPHA_BYTES" "$ALPHA_SHA" > "$DEST8/.fnlp-fetch-journals/alpha.bin.journal"
BAD_BASE=http://127.0.0.1:1
if FNLP_FETCH_ALLOW_TEST_BASE_URL=1 "$FETCH" --dest "$DEST8" --catalog "$CATALOG" --test-base-url "$BAD_BASE" > "$WORK/case8.out" 2> "$WORK/case8.err"; then fail_case 8 interrupted-resume-guidance; elif grep -Fq "scripts/fetch_model.sh --dest '$DEST8'" "$WORK/case8.err" && grep -q 'journal_dir=' "$WORK/case8.err"; then pass_case 8 interrupted-resume-guidance; else fail_case 8 interrupted-resume-guidance; fi

# 9. The model-gated check-only path is an honest no-model skip, not a cache hit.
CASES=$((CASES + 1)); DEST9="$WORK/case9"
if FNLP_FETCH_ALLOW_TEST_BASE_URL=1 "$FETCH" --dest "$DEST9" --catalog "$CATALOG" --test-base-url "$BASE" --check-only > "$WORK/case9.out" 2> "$WORK/case9.err" && grep -q 'CHECK_ONLY RESULT=SKIPPED_NO_MODEL' "$WORK/case9.err"; then pass_case 9 check-only-no-model-skip; else fail_case 9 check-only-no-model-skip; fi
