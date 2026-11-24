#!/bin/sh
# Local-fixture tests for fetch_model.sh. The retained fixture directory is
# printed on completion so an operator can inspect every quarantine/journal.

set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
FETCH="$ROOT/scripts/fetch_model.sh"
REAL_CURL=$(command -v curl)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/fnlp-fetch-model-test.XXXXXX")
REV=f56ec5a9650268aa098496734743c25ea778bd2d
CASES=0
FAILED=
SERVER_PID=
TLS_SERVER_PID=

log() { printf '%s FETCH_MODEL_TEST %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
fail_case() { FAILED="${FAILED}${FAILED:+,}$1"; log "CASE=$1 RESULT=FAIL detail=$2"; }
pass_case() { log "CASE=$1 RESULT=PASS detail=$2"; }
finish() {
    if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi
    if [ -n "$TLS_SERVER_PID" ]; then kill "$TLS_SERVER_PID" 2>/dev/null || true; fi
    if [ -n "$FAILED" ]; then
        log "FETCH_MODEL_TESTS RESULT=FAIL cases=$CASES failed=$FAILED retained_work=$WORK"
        exit 1
    fi
    log "FETCH_MODEL_TESTS RESULT=PASS cases=$CASES failed=none retained_work=$WORK"
}
trap finish EXIT

run_disk_cases() {
    # Exercise the production functions with a deterministic disk/transport
    # model. No sockets, source-model assets or production test hooks are used.
    sed '/^while \[ "\$#" -gt 0 \]; do/,$d' "$FETCH" > "$WORK/disk-functions.sh"
    cat > "$WORK/disk-tests.sh" <<'EOF'
#!/bin/sh
set -u
. "$1/disk-functions.sh"
DISK_WORK=$1
DISK_FIXTURE="$DISK_WORK/disk-fixture"
mkdir -p "$DISK_FIXTURE"
printf 'alpha fixture bytes\n' > "$DISK_FIXTURE/alpha.bin"
printf 'beta fixture bytes are slightly longer\n' > "$DISK_FIXTURE/beta.bin"
DISK_ALPHA=$(bytes "$DISK_FIXTURE/alpha.bin")
DISK_BETA=$(bytes "$DISK_FIXTURE/beta.bin")
DISK_ALPHA_SHA=$(sha256 "$DISK_FIXTURE/alpha.bin")
DISK_BETA_SHA=$(sha256 "$DISK_FIXTURE/beta.bin")
CATALOG="$DISK_WORK/disk-catalog.txt"
printf 'alpha.bin|%s|%s\nbeta.bin|%s|%s\n' "$DISK_ALPHA" "$DISK_ALPHA_SHA" "$DISK_BETA" "$DISK_BETA_SHA" > "$CATALOG"
TEST_BASE=https://fixture.invalid/source
catalog_stats

available_bytes() {
    printf 'probe\n' >> "$DEST/space-probes"
    printf '%s\n' "$DISK_FREE"
}
download_with_progress() {
    printf 'transfer\n' >> "$DEST/transfers"
    [ "$DISK_REFUSE_RANGE" -eq 0 ] || return 33
    disk_transfer_start=$(bytes "$2")
    disk_source=${2##*/}
    disk_source=${disk_source%.partial}
    tail -c "+$((disk_transfer_start + 1))" "$DISK_FIXTURE/$disk_source" >> "$2" || return 92
    DISK_FREE=$((DISK_FREE - $3 + disk_transfer_start))
}
disk_setup() {
    DEST="$DISK_WORK/disk-$1"
    VERIFIED_NAMES=' '
    LOCK_DIR=
    LOCK_OWNED=0
    DISK_REFUSE_RANGE=0
    ensure_destination
}
disk_journal() {
    journal_text "$TEST_BASE/$1" "$1" "$2" "$3" > "$JOURNAL_DIR/$1.journal"
}
disk_check() {
    "$@" || { printf 'DISK_TEST RESULT=FAIL assertion=%s\n' "$*" >&2; exit 1; }
}
disk_pass() { printf 'DISK_TEST case=%s RESULT=PASS\n' "$1"; }
disk_retained_one() {
    disk_retained_count=0
    for disk_retained_path in "$DEST/quarantine/$1".*; do
        [ -f "$disk_retained_path" ] || continue
        disk_check cmp "$2" "$disk_retained_path"
        disk_retained_count=$((disk_retained_count + 1))
    done
    disk_check test "$disk_retained_count" -eq 1
}

disk_setup cached
cp "$DISK_FIXTURE/alpha.bin" "$DEST/alpha.bin"
cp "$DISK_FIXTURE/beta.bin" "$DEST/beta.bin"
DISK_FREE=0
verify_existing_closure
download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
download_one beta.bin "$DISK_BETA" "$DISK_BETA_SHA"
disk_check test ! -e "$DEST/space-probes"
disk_check test ! -e "$DEST/transfers"
disk_pass cached-closure-without-free-space-or-network

disk_setup fresh
DISK_FREE=$((CATALOG_TOTAL + MARGIN_BYTES))
verify_existing_closure
disk_check test "$(remaining_download_bytes)" -eq "$CATALOG_TOTAL"
download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
disk_check test "$(remaining_download_bytes)" -eq "$DISK_BETA"
download_one beta.bin "$DISK_BETA" "$DISK_BETA_SHA"
disk_check test "$DISK_FREE" -eq "$MARGIN_BYTES"
disk_check test "$(remaining_download_bytes)" -eq 0
disk_check cmp "$DISK_FIXTURE/alpha.bin" "$DEST/alpha.bin"
disk_check cmp "$DISK_FIXTURE/beta.bin" "$DEST/beta.bin"
disk_pass exact-capacity-survives-sequential-progress

disk_setup later-hit
cp "$DISK_FIXTURE/beta.bin" "$DEST/beta.bin"
DISK_FREE=$((DISK_ALPHA + MARGIN_BYTES))
verify_existing_closure
disk_check test "$(remaining_download_bytes)" -eq "$DISK_ALPHA"
download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
download_one beta.bin "$DISK_BETA" "$DISK_BETA_SHA"
disk_check test "$DISK_FREE" -eq "$MARGIN_BYTES"
disk_pass later-verified-source-is-not-reserved-again

disk_setup bound-resume
head -c 5 "$DISK_FIXTURE/alpha.bin" > "$DEST/alpha.bin.partial"
head -c 7 "$DISK_FIXTURE/beta.bin" > "$DEST/beta.bin.partial"
disk_journal alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
disk_journal beta.bin "$DISK_BETA" "$DISK_BETA_SHA"
DISK_FREE=$((CATALOG_TOTAL - 12 + MARGIN_BYTES))
verify_existing_closure
disk_check test "$(remaining_download_bytes)" -eq "$((CATALOG_TOTAL - 12))"
download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
download_one beta.bin "$DISK_BETA" "$DISK_BETA_SHA"
disk_check test "$DISK_FREE" -eq "$MARGIN_BYTES"
disk_check cmp "$DISK_FIXTURE/alpha.bin" "$DEST/alpha.bin"
disk_check cmp "$DISK_FIXTURE/beta.bin" "$DEST/beta.bin"
disk_pass all-journal-bound-prefixes-earn-storage-credit

disk_setup unbound-resume
head -c 5 "$DISK_FIXTURE/alpha.bin" > "$DEST/alpha.bin.partial"
head -c 7 "$DISK_FIXTURE/beta.bin" > "$DEST/beta.bin.partial"
disk_journal beta.bin "$DISK_BETA" "$(printf '%064d' 0)"
disk_check test "$(remaining_download_bytes)" -eq "$CATALOG_TOTAL"
DISK_FREE=$((CATALOG_TOTAL + MARGIN_BYTES))
download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
download_one beta.bin "$DISK_BETA" "$DISK_BETA_SHA"
disk_check test -s "$DEST/quarantine/quarantine.log"
disk_check test "$DISK_FREE" -eq "$MARGIN_BYTES"
disk_pass unbound-prefixes-stay-charged-and-quarantined

disk_setup corrupt-final
cp "$DISK_FIXTURE/alpha.bin" "$DEST/alpha.bin"
printf X | dd of="$DEST/alpha.bin" bs=1 count=1 conv=notrunc 2>/dev/null
cp "$DISK_FIXTURE/beta.bin" "$DEST/beta.bin"
verify_existing_closure
disk_check test "$(remaining_download_bytes)" -eq "$DISK_ALPHA"
DISK_FREE=$((DISK_ALPHA + MARGIN_BYTES))
download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
disk_check test -s "$DEST/quarantine/quarantine.log"
disk_check cmp "$DISK_FIXTURE/alpha.bin" "$DEST/alpha.bin"
disk_pass same-length-corruption-never-earns-cache-credit

disk_setup complete-verified
cp "$DISK_FIXTURE/alpha.bin" "$DEST/alpha.bin.partial"
disk_journal alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
cp "$DISK_FIXTURE/beta.bin" "$DEST/beta.bin"
DISK_FREE=0
verify_existing_closure
download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
disk_check cmp "$DISK_FIXTURE/alpha.bin" "$DEST/alpha.bin"
disk_check test ! -e "$DEST/alpha.bin.partial"
disk_check test ! -e "$DEST/transfers"
disk_check test ! -e "$DEST/space-probes"
disk_check test ! -e "$DEST/quarantine"
disk_pass verified-complete-partial-publishes-without-transfer-or-free-space

DISK_REJECTED="$DISK_WORK/rejected-alpha.bin"
cp "$DISK_FIXTURE/alpha.bin" "$DISK_REJECTED"
printf X | dd of="$DISK_REJECTED" bs=1 count=1 conv=notrunc 2>/dev/null
disk_setup complete-rejected
cp "$DISK_REJECTED" "$DEST/alpha.bin.partial"
disk_journal alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
cp "$JOURNAL_DIR/alpha.bin.journal" "$DISK_WORK/rejected-alpha.journal"
cp "$DISK_FIXTURE/beta.bin" "$DEST/beta.bin"
DISK_FREE=$((DISK_ALPHA + MARGIN_BYTES))
verify_existing_closure
download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
disk_retained_one alpha.bin.partial "$DISK_REJECTED"
disk_retained_one alpha.bin.journal "$DISK_WORK/rejected-alpha.journal"
disk_check cmp "$DISK_FIXTURE/alpha.bin" "$DEST/alpha.bin"
disk_check test "$(wc -l < "$DEST/transfers" | tr -d '[:space:]')" -eq 1
disk_check test "$DISK_FREE" -eq "$MARGIN_BYTES"
disk_pass corrupt-complete-partial-is-preserved-and-replaced-in-one-invocation

disk_setup complete-rejected-no-space
cp "$DISK_REJECTED" "$DEST/alpha.bin.partial"
disk_journal alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
cp "$DISK_FIXTURE/beta.bin" "$DEST/beta.bin"
DISK_FREE=$MARGIN_BYTES
verify_existing_closure
disk_check test "$(remaining_download_bytes)" -eq 0
(download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA")
disk_check test "$?" -eq 4
disk_retained_one alpha.bin.partial "$DISK_REJECTED"
disk_retained_one alpha.bin.journal "$DISK_WORK/rejected-alpha.journal"
disk_check test ! -e "$DEST/transfers"
disk_check test ! -e "$DEST/alpha.bin"
disk_check test ! -s "$DEST/alpha.bin.partial"
disk_check test "$(remaining_download_bytes)" -eq "$DISK_ALPHA"
disk_pass quarantined-partial-loses-disk-credit-before-fresh-transfer

disk_setup incomplete-rejected
head -c 5 "$DISK_REJECTED" > "$DEST/alpha.bin.partial"
disk_journal alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
cp "$DISK_FIXTURE/beta.bin" "$DEST/beta.bin"
DISK_FREE=$((DISK_ALPHA - 5 + MARGIN_BYTES))
verify_existing_closure
(download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA")
disk_check test "$?" -eq 1
disk_retained_one alpha.bin.partial "$DISK_REJECTED"
disk_retained_one alpha.bin.journal "$DISK_WORK/rejected-alpha.journal"
disk_check test ! -e "$DEST/alpha.bin"
disk_check test ! -e "$DEST/alpha.bin.partial"
disk_check test ! -e "$JOURNAL_DIR/alpha.bin.journal"
DISK_FREE=$((DISK_ALPHA + MARGIN_BYTES))
download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
disk_check cmp "$DISK_FIXTURE/alpha.bin" "$DEST/alpha.bin"
disk_check test "$(wc -l < "$DEST/transfers" | tr -d '[:space:]')" -eq 2
disk_pass rejected-resume-never-traps-the-next-invocation-in-a-bad-prefix

disk_setup insufficient
DISK_FREE=$((CATALOG_TOTAL + MARGIN_BYTES - 1))
(download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA")
disk_check test "$?" -eq 4
disk_check test ! -e "$DEST/transfers"
disk_check test ! -s "$DEST/alpha.bin.partial"
disk_pass one-byte-short-refuses-before-transport

disk_setup range-restart
cp "$DISK_FIXTURE/beta.bin" "$DEST/beta.bin"
head -c 5 "$DISK_FIXTURE/alpha.bin" > "$DEST/alpha.bin.partial"
disk_journal alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
verify_existing_closure
DISK_FREE=$((DISK_ALPHA - 5 + MARGIN_BYTES))
DISK_REFUSE_RANGE=1
(download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA")
disk_check test "$?" -eq 4
disk_check test "$(wc -l < "$DEST/transfers" | tr -d '[:space:]')" -eq 1
disk_check test ! -s "$DEST/alpha.bin.partial"
disk_check test -s "$DEST/quarantine/quarantine.log"
disk_pass range-restart-requires-new-space-before-second-transfer

disk_setup nonregular-prefix
ln -s "$DISK_FIXTURE/alpha.bin" "$DEST/alpha.bin.partial"
disk_journal alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
disk_check test "$(remaining_download_bytes)" -eq "$CATALOG_TOTAL"
disk_pass symlink-prefix-earns-no-storage-credit

disk_setup oversized-prefix
cat "$DISK_FIXTURE/alpha.bin" "$DISK_FIXTURE/alpha.bin" > "$DEST/alpha.bin.partial"
disk_journal alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
disk_check test "$(remaining_download_bytes)" -eq "$CATALOG_TOTAL"
disk_pass oversized-prefix-earns-no-storage-credit

disk_setup changed-catalog
cp "$DISK_FIXTURE/beta.bin" "$DEST/beta.bin"
# A publisher can atomically update the catalog while a multi-GB transfer
# holds this invocation's lock. Credit and admission still use both initially
# validated rows, with beta verified and alpha outstanding.
printf 'new.bin|1|%s\n' "$DISK_ALPHA_SHA" > "$CATALOG.next"
mv "$CATALOG.next" "$CATALOG"
DISK_FREE=$((DISK_ALPHA + MARGIN_BYTES))
verify_existing_closure
disk_check test "$(remaining_download_bytes)" -eq "$DISK_ALPHA"
download_one alpha.bin "$DISK_ALPHA" "$DISK_ALPHA_SHA"
disk_check test "$(remaining_download_bytes)" -eq 0
disk_check cmp "$DISK_FIXTURE/alpha.bin" "$DEST/alpha.bin"
disk_pass changed-catalog-cannot-redirect-inflight-admission

for invalid in 01 08 -1 9223372036854775808 999999999999999999999999999; do
    if valid_byte_count "$invalid"; then exit 1; fi
done
disk_check valid_byte_count 0
disk_check valid_byte_count 9223372036854775807
CATALOG="$DISK_WORK/overflow-catalog.txt"
printf 'alpha.bin|9223372036854775807|%s\n' "$DISK_ALPHA_SHA" > "$CATALOG"
(catalog_stats)
disk_check test "$?" -eq 2
disk_pass byte-count-and-margin-overflow-refuse
EOF
    CASES=$((CASES + 1))
    if sh "$WORK/disk-tests.sh" "$WORK" > "$WORK/disk-tests.out" 2> "$WORK/disk-tests.err"; then
        cat "$WORK/disk-tests.out"
        pass_case disk-admission cached-and-resumable-remaining-space
    else
        cat "$WORK/disk-tests.out"
        cat "$WORK/disk-tests.err" >&2
        fail_case disk-admission cached-and-resumable-remaining-space
    fi
}

run_interrupt_cases() {
    # Exercise the actual fetch script and an owned long-lived transfer child.
    # The fake transport has an explicit drain latch, so no server, network,
    # large model or wall-clock race is needed to inspect lock ownership.
    CASES=$((CASES + 1))
    if python3 - "$FETCH" "$WORK" "$REV" > "$WORK/interrupt-tests.out" 2> "$WORK/interrupt-tests.err" <<'PY'
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

fetch = Path(sys.argv[1])
root = Path(sys.argv[2]) / "interrupt-cases"
revision = sys.argv[3]
root.mkdir()
shim = root / "bin"
shim.mkdir()
curl = shim / "curl"
curl.write_text(r'''#!/usr/bin/env python3
import os
from pathlib import Path
import signal
import sys
import time

case = Path(os.environ["FNLP_TEST_TRANSFER_CASE"])
args = sys.argv[1:]
output = Path(args[args.index("--output") + 1])
payload = b"pinned conversion source bytes\n"
if os.environ["FNLP_TEST_TRANSFER_MODE"] == "complete":
    retained = output.stat().st_size
    (case / "resumed-offset").write_text(str(retained))
    with output.open("ab", buffering=0) as destination:
        destination.write(payload[retained:])
        os.fsync(destination.fileno())
    sys.stdout.write("206\n\n")
    sys.exit(0)

stopping = False
def stop(signum, frame):
    global stopping
    stopping = True
signal.signal(signal.SIGTERM, stop)
with output.open("ab", buffering=0) as destination:
    destination.write(payload[:7])
    os.fsync(destination.fileno())
(case / "child-started").write_text(str(os.getpid()))
while not stopping:
    time.sleep(0.01)
lock = Path(os.environ["FNLP_TEST_TRANSFER_LOCK"])
(case / "child-stopping").write_text("locked" if lock.is_dir() else "unlocked")
while not (case / "allow-exit").exists():
    time.sleep(0.01)
(case / "child-exited").write_text("locked" if lock.is_dir() else "unlocked")
''')
curl.chmod(0o700)
payload = b"pinned conversion source bytes\n"
digest = hashlib.sha256(payload).hexdigest()
catalog = root / "catalog.txt"
catalog.write_text(f"alpha.bin|{len(payload)}|{digest}\n")

def await_record(path, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = path.read_text()
        except FileNotFoundError:
            value = ""
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path.name}")

def child_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True

for signum, signal_name, expected_code in [
    (signal.SIGINT, "INT", 130),
    (signal.SIGTERM, "TERM", 143),
    (signal.SIGHUP, "HUP", 129),
]:
    case = root / signal_name.lower()
    case.mkdir()
    destination = case / "source"
    destination.mkdir(mode=0o700)
    lock = destination / (".fnlp-fetch-lock-" + revision)
    journal = destination / ".fnlp-fetch-journals" / "alpha.bin.journal"
    partial = destination / "alpha.bin.partial"
    environment = dict(os.environ)
    environment.update({
        "PATH": str(shim) + os.pathsep + os.environ["PATH"],
        "FNLP_FETCH_ALLOW_TEST_BASE_URL": "1",
        "FNLP_TEST_TRANSFER_CASE": str(case),
        "FNLP_TEST_TRANSFER_LOCK": str(lock),
        "FNLP_TEST_TRANSFER_MODE": "block",
    })
    command = [str(fetch), "--dest", str(destination), "--catalog", str(catalog),
               "--allow-untrusted-revision", "--test-base-url", "https://fixture.invalid/source"]
    child_pid = None
    process = None
    with (case / "fetch.out").open("wb") as stdout, (case / "fetch.err").open("wb") as stderr:
        try:
            # Popen avoids an asynchronous shell's inherited ignored SIGINT.
            # Signal only the fetch parent; it must stop and join its own child.
            process = subprocess.Popen(command, env=environment, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            child_pid = int(await_record(case / "child-started"))
            original_journal = journal.read_bytes()
            original_partial = partial.read_bytes()
            assert original_partial == payload[:7]
            assert lock.is_dir()
            process.send_signal(signum)
            assert await_record(case / "child-stopping") == "locked"
            assert process.poll() is None, "parent returned before writer drained"
            assert lock.is_dir(), "lock released while writer is still alive"

            # Cleanup ignores subsequent signals until the recorded child
            # exits. The first signal remains the terminal reason.
            process.send_signal(signal.SIGTERM if signum != signal.SIGTERM else signal.SIGINT)
            retry_environment = dict(environment, FNLP_TEST_TRANSFER_MODE="complete")
            contender = subprocess.run(command, env=retry_environment, capture_output=True, timeout=5)
            assert contender.returncode == 4, contender.stderr.decode()
            assert lock.is_dir(), "contender removed the active writer's lock"
            assert process.poll() is None
            assert journal.read_bytes() == original_journal
            assert partial.read_bytes() == original_partial

            (case / "allow-exit").write_text("drain")
            assert process.wait(timeout=8) == expected_code
            assert await_record(case / "child-exited") == "locked"
            assert not child_exists(child_pid), "recorded transfer remains alive after parent exit"
            assert not lock.exists()
            assert journal.read_bytes() == original_journal
            assert partial.read_bytes() == original_partial
            assert not (destination / "alpha.bin").exists()
            stderr.flush()
            diagnostics = (case / "fetch.err").read_text()
            assert f"INTERRUPTED signal={signal_name} code={expected_code}" in diagnostics
            assert "RESUME command=" in diagnostics
            assert "FETCH_MODEL RESULT=PASS" not in diagnostics

            resumed = subprocess.run(command, env=retry_environment, capture_output=True, timeout=8)
            assert resumed.returncode == 0, resumed.stderr.decode()
            assert int(await_record(case / "resumed-offset")) == len(original_partial)
            assert (destination / "alpha.bin").read_bytes() == payload
            assert not partial.exists()
            assert not lock.exists()
            print(f"INTERRUPT_TEST signal={signal_name} RESULT=PASS exit={expected_code} joined=true resumed=true")
        finally:
            # Retain every fixture. On assertion failure, drain only this
            # fixture's exact child and parent rather than killing a process
            # group, another curl, or an unrelated fetch invocation.
            (case / "allow-exit").write_text("drain")
            if child_pid is not None and child_exists(child_pid):
                try:
                    os.kill(child_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
PY
    then
        cat "$WORK/interrupt-tests.out"
        pass_case interruption owned-writer-drain-lock-lifetime-and-resume
    else
        cat "$WORK/interrupt-tests.out"
        cat "$WORK/interrupt-tests.err" >&2
        fail_case interruption owned-writer-drain-lock-lifetime-and-resume
    fi
}

if [ "${1:-}" = --interrupt-only ]; then
    run_interrupt_cases
    exit
fi

if [ "${1:-}" = --disk-only ]; then
    run_disk_cases
    exit
fi

command -v python3 >/dev/null 2>&1 || { log "FETCH_MODEL_TESTS RESULT=FAIL cases=0 failed=missing-python3 retained_work=$WORK"; exit 1; }
command -v openssl >/dev/null 2>&1 || { log "FETCH_MODEL_TESTS RESULT=FAIL cases=0 failed=missing-openssl retained_work=$WORK"; exit 1; }
command -v curl >/dev/null 2>&1 || { log "FETCH_MODEL_TESTS RESULT=FAIL cases=0 failed=missing-curl retained_work=$WORK"; exit 1; }

# 0. Test transport injection stays in this harness, never in the fetch script.
CASES=$((CASES + 1))
if rg -q 'FNLP_FETCH_TEST|--insecure|--connect-to' "$FETCH"; then
    fail_case 0 production-transport-isolation
else
    pass_case 0 production-transport-isolation
fi

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
    FNLP_FETCH_ALLOW_TEST_BASE_URL=1 "$FETCH" --dest "$dest" --catalog "$CATALOG" --allow-untrusted-revision --test-base-url "$BASE" "$@"
}

start_redirect_server() {
    redirect_host=$1
    redirect_location=${2:-}
    REDIRECT_TRACE="$WORK/redirect-$redirect_host-$CASES.trace"
    TLS_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
    cat > "$WORK/redirect-openssl.cnf" <<EOF
[req]
distinguished_name = subject
prompt = no

[subject]
CN = fnlp-fetch-fixture

[v3_leaf]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @names

[names]
DNS.1 = huggingface.co
DNS.2 = $redirect_host
EOF
    if ! openssl req -x509 -newkey rsa:2048 -nodes -keyout "$WORK/redirect-ca.key" -out "$WORK/redirect-ca.crt" -subj /CN=fnlp-fetch-fixture-ca -days 1 > "$WORK/openssl.log" 2>&1 ||
        ! openssl req -new -newkey rsa:2048 -nodes -keyout "$WORK/redirect.key" -out "$WORK/redirect.csr" -config "$WORK/redirect-openssl.cnf" >> "$WORK/openssl.log" 2>&1 ||
        ! openssl x509 -req -in "$WORK/redirect.csr" -CA "$WORK/redirect-ca.crt" -CAkey "$WORK/redirect-ca.key" -CAcreateserial -out "$WORK/redirect.crt" -days 1 -sha256 -extfile "$WORK/redirect-openssl.cnf" -extensions v3_leaf >> "$WORK/openssl.log" 2>&1; then
        return 1
    fi
    FIXTURE_ROOT="$FIXTURE" REDIRECT_HOST="$redirect_host" REDIRECT_LOCATION="$redirect_location" REDIRECT_TRACE="$REDIRECT_TRACE" TLS_PORT="$TLS_PORT" TLS_CERT="$WORK/redirect.crt" TLS_KEY="$WORK/redirect.key" python3 -c '
import http.server
import os
import ssl
from pathlib import Path
from urllib.parse import urlsplit

root = Path(os.environ["FIXTURE_ROOT"])
redirect_host = os.environ["REDIRECT_HOST"]

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        host = self.headers.get("Host", "").split(":", 1)[0].lower()
        with open(os.environ["REDIRECT_TRACE"], "a", encoding="ascii") as trace:
            trace.write("host={} path={} range={} agent={}\n".format(host, self.path, self.headers.get("Range", ""), self.headers.get("User-Agent", "")))
        if host == "huggingface.co" and not self.path.startswith("/redirect-final/"):
            self.send_response(302)
            target = os.environ["REDIRECT_LOCATION"] or "https://{}{}".format(redirect_host, self.path)
            self.send_header("Location", target.replace("{name}", Path(urlsplit(self.path).path).name))
            self.send_header("Content-Length", "15")
            self.end_headers()
            try:
                self.wfile.write(b"REDIRECTPAYLOAD")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if host not in (redirect_host, "huggingface.co"):
            self.send_error(421, "unexpected host")
            return
        candidate = root / Path(urlsplit(self.path).path).name
        if not candidate.is_file():
            self.send_error(404)
            return
        payload = candidate.read_bytes()
        start = 0
        if self.headers.get("Range"):
            start = int(self.headers["Range"].removeprefix("bytes=").removesuffix("-"))
        self.send_response(206 if start else 200)
        if start:
            self.send_header("Content-Range", "bytes {}-{}/{}".format(start, len(payload) - 1, len(payload)))
        self.send_header("Content-Length", str(len(payload) - start))
        self.end_headers()
        self.wfile.write(payload[start:])

    def log_message(self, _format, *_args):
        pass

server = http.server.ThreadingHTTPServer(("127.0.0.1", int(os.environ["TLS_PORT"])), Handler)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(os.environ["TLS_CERT"], os.environ["TLS_KEY"])
server.socket = context.wrap_socket(server.socket, server_side=True)
server.serve_forever()
' > "$WORK/redirect-$redirect_host.log" 2>&1 &
    TLS_SERVER_PID=$!
    sleep 1
    kill -0 "$TLS_SERVER_PID" 2>/dev/null
}

stop_redirect_server() {
    if [ -n "$TLS_SERVER_PID" ]; then
        kill "$TLS_SERVER_PID" 2>/dev/null || true
        wait "$TLS_SERVER_PID" 2>/dev/null || true
        TLS_SERVER_PID=
    fi
}

run_redirect_policy_fetch() {
    redirect_host=$1
    dest=$2
    output=$3
    error=$4
    start_redirect_server "$redirect_host" "${5:-}" || return 1
    wrapper_dir="$WORK/redirect-curl-bin-$redirect_host"
    mkdir -p "$wrapper_dir"
    cat > "$wrapper_dir/curl" <<EOF
#!/bin/sh
exec "$REAL_CURL" --disable --connect-to "huggingface.co:443:127.0.0.1:$TLS_PORT" --connect-to "$redirect_host:443:127.0.0.1:$TLS_PORT" --cacert "$WORK/redirect-ca.crt" "\$@"
EOF
    chmod 700 "$wrapper_dir/curl"
    HTTPS_PROXY= https_proxy= HTTP_PROXY= http_proxy= ALL_PROXY= all_proxy= NO_PROXY= no_proxy= \
        PATH="$wrapper_dir:$PATH" \
        "$FETCH" --dest "$dest" --catalog "$CATALOG" --allow-untrusted-revision > "$output" 2> "$error"
    status=$?
    stop_redirect_server
    return "$status"
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
if run_fetch "$DEST6" > "$WORK/case6-download.out" 2> "$WORK/case6-download.err" && run_fetch "$DEST6" --check-only > "$WORK/case6-pass.out" 2> "$WORK/case6-pass.err"; then
    printf 'X' >> "$DEST6/alpha.bin"
    if run_fetch "$DEST6" --check-only > "$WORK/case6-fail.out" 2> "$WORK/case6-fail.err"; then fail_case 6 check-only-tamper; elif grep -q 'expected_sha256=' "$WORK/case6-fail.err" && grep -q 'observed_sha256=' "$WORK/case6-fail.err"; then pass_case 6 check-only-tamper; else fail_case 6 check-only-tamper; fi
else fail_case 6 check-only-setup; fi

# 7. A non-default revision cannot inherit the default catalog identity.
CASES=$((CASES + 1)); DEST7="$WORK/case7"
if "$FETCH" --dest "$DEST7" --revision deadbeef > "$WORK/case7.out" 2> "$WORK/case7.err"; then fail_case 7 untrusted-revision-refusal; elif [ "$?" -eq 2 ] && grep -q 'UNTRUSTED_REVISION_REFUSED' "$WORK/case7.err"; then pass_case 7 untrusted-revision-refusal; else fail_case 7 untrusted-revision-refusal; fi

# 8. A failed resume prints the exact same --dest invocation and journal path.
CASES=$((CASES + 1)); DEST8="$WORK/case8"; secure_mkdir "$DEST8/.fnlp-fetch-journals"
head -c 5 "$FIXTURE/alpha.bin" > "$DEST8/alpha.bin.partial"
printf 'url=http://127.0.0.1:1/alpha.bin\nrevision=%s\nname=alpha.bin\nbytes=%s\nsha256=%s\n' "$REV" "$ALPHA_BYTES" "$ALPHA_SHA" > "$DEST8/.fnlp-fetch-journals/alpha.bin.journal"
BAD_BASE=http://127.0.0.1:1
if FNLP_FETCH_ALLOW_TEST_BASE_URL=1 "$FETCH" --dest "$DEST8" --catalog "$CATALOG" --allow-untrusted-revision --test-base-url "$BAD_BASE" > "$WORK/case8.out" 2> "$WORK/case8.err"; then fail_case 8 interrupted-resume-guidance; elif grep -Fq "scripts/fetch_model.sh --dest '$DEST8' --allow-untrusted-revision --catalog '$CATALOG'" "$WORK/case8.err" && grep -q 'journal_dir=' "$WORK/case8.err"; then pass_case 8 interrupted-resume-guidance; else fail_case 8 interrupted-resume-guidance; fi

# 9. Check-only refuses an empty closure: exit zero means all catalogued files
# verified, never merely that no model is installed.
CASES=$((CASES + 1)); DEST9="$WORK/case9"
if run_fetch "$DEST9" --check-only > "$WORK/case9.out" 2> "$WORK/case9.err"; then
    fail_case 9 check-only-missing-closure-refusal
elif grep -q 'CHECK_ONLY RESULT=FAIL' "$WORK/case9.err" && grep -q 'VERIFY RESULT=FAIL file=alpha.bin' "$WORK/case9.err" && grep -q 'VERIFY RESULT=FAIL file=beta.bin' "$WORK/case9.err"; then
    pass_case 9 check-only-missing-closure-refusal
else
    fail_case 9 check-only-missing-closure-refusal
fi

# 10. A local TLS redirect to an official regional Xet CDN host passes the real policy.
CASES=$((CASES + 1)); DEST10="$WORK/case10"
if run_redirect_policy_fetch us.aws.cdn.hf.co "$DEST10" "$WORK/case10.out" "$WORK/case10.err" && cmp "$FIXTURE/alpha.bin" "$DEST10/alpha.bin" >/dev/null && cmp "$FIXTURE/beta.bin" "$DEST10/beta.bin" >/dev/null; then pass_case 10 regional-cdn-redirect-accepted; else fail_case 10 regional-cdn-redirect-accepted; fi

# 11. An unlisted redirect is refused before contacting it or writing its body.
CASES=$((CASES + 1)); DEST11="$WORK/case11"
if run_redirect_policy_fetch unlisted.invalid "$DEST11" "$WORK/case11.out" "$WORK/case11.err"; then
    fail_case 11 unlisted-redirect-refusal
elif [ ! -e "$DEST11/alpha.bin" ] && [ ! -s "$DEST11/alpha.bin.partial" ] &&
    ! grep -q 'host=unlisted.invalid ' "$REDIRECT_TRACE" &&
    grep -q 'REDIRECT_HOST_REFUSED phase=pre-contact activation=refused effective_url=https://unlisted.invalid/' "$WORK/case11.err"; then
    pass_case 11 unlisted-redirect-refusal
else
    fail_case 11 unlisted-redirect-refusal
fi

# 12. Failed acquisition cannot release another downloader's empty lock.
CASES=$((CASES + 1)); DEST12="$WORK/case12"
secure_mkdir "$DEST12/.fnlp-fetch-lock-$REV"
run_fetch "$DEST12" --check-only > "$WORK/case12.out" 2> "$WORK/case12.err"
lock_status=$?
run_fetch "$DEST12" --check-only > "$WORK/case12-again.out" 2> "$WORK/case12-again.err"
lock_retry_status=$?
if [ "$lock_status" -eq 4 ] && [ "$lock_retry_status" -eq 4 ] && [ -d "$DEST12/.fnlp-fetch-lock-$REV" ] &&
    grep -q 'revision lock busy or stale' "$WORK/case12.err"; then pass_case 12 contended-lock-preserved
else fail_case 12 contended-lock-preserved; fi

# 13. A custom catalog never inherits the pinned revision's trust implicitly.
CASES=$((CASES + 1)); DEST13="$WORK/case13"
"$FETCH" --dest "$DEST13" --catalog "$CATALOG" --check-only > "$WORK/case13.out" 2> "$WORK/case13.err"
catalog_status=$?
if [ "$catalog_status" -eq 2 ] && [ ! -e "$DEST13" ] && grep -q 'UNTRUSTED_CATALOG_REFUSED' "$WORK/case13.err"; then pass_case 13 custom-catalog-requires-opt-in
else fail_case 13 custom-catalog-requires-opt-in; fi

# 14. An allowlisted-looking path cannot authorize an unlisted hostname.
CASES=$((CASES + 1)); DEST14="$WORK/case14"
if run_redirect_policy_fetch unlisted.invalid "$DEST14" "$WORK/case14.out" "$WORK/case14.err" 'https://unlisted.invalid/example.xethub.hf.co/{name}'; then
    fail_case 14 redirect-path-cannot-authorize-host
elif [ ! -s "$DEST14/alpha.bin.partial" ] && ! grep -q 'host=unlisted.invalid ' "$REDIRECT_TRACE" && grep -q 'phase=pre-contact' "$WORK/case14.err"; then
    pass_case 14 redirect-path-cannot-authorize-host
else fail_case 14 redirect-path-cannot-authorize-host; fi

# 15. Relative redirects retain the original authority and exclude body bytes.
CASES=$((CASES + 1)); DEST15="$WORK/case15"
if run_redirect_policy_fetch us.aws.cdn.hf.co "$DEST15" "$WORK/case15.out" "$WORK/case15.err" '/redirect-final/{name}' &&
    cmp "$FIXTURE/alpha.bin" "$DEST15/alpha.bin" >/dev/null &&
    grep -q 'host=huggingface.co path=/redirect-final/alpha.bin' "$REDIRECT_TRACE" &&
    grep -q 'agent=franken_nlp-agent ' "$REDIRECT_TRACE"; then pass_case 15 relative-redirect-and-agent
else fail_case 15 relative-redirect-and-agent; fi

# 16. A loop is bounded even when every hop uses an authorized hostname.
CASES=$((CASES + 1)); DEST16="$WORK/case16"
if run_redirect_policy_fetch us.aws.cdn.hf.co "$DEST16" "$WORK/case16.out" "$WORK/case16.err" '/loop/{name}'; then
    fail_case 16 redirect-loop-bounded
elif [ "$(wc -l < "$REDIRECT_TRACE" | tr -d '[:space:]')" -eq 9 ] && [ ! -s "$DEST16/alpha.bin.partial" ] && grep -q 'REDIRECT_LIMIT_REFUSED' "$WORK/case16.err"; then
    pass_case 16 redirect-loop-bounded
else fail_case 16 redirect-loop-bounded; fi

# 17. A Range resume survives an accepted redirect without losing its prefix.
CASES=$((CASES + 1)); DEST17="$WORK/case17"; secure_mkdir "$DEST17/.fnlp-fetch-journals"
head -c 5 "$FIXTURE/alpha.bin" > "$DEST17/alpha.bin.partial"
printf 'url=https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve/%s/alpha.bin\nrevision=%s\nname=alpha.bin\nbytes=%s\nsha256=%s\n' "$REV" "$REV" "$ALPHA_BYTES" "$ALPHA_SHA" > "$DEST17/.fnlp-fetch-journals/alpha.bin.journal"
if run_redirect_policy_fetch us.aws.cdn.hf.co "$DEST17" "$WORK/case17.out" "$WORK/case17.err" &&
    cmp "$FIXTURE/alpha.bin" "$DEST17/alpha.bin" >/dev/null &&
    grep -q 'host=us.aws.cdn.hf.co .*range=bytes=5-' "$REDIRECT_TRACE" &&
    ! grep -q 'DECISION=RESTART' "$WORK/case17.err"; then pass_case 17 redirected-range-resume
else fail_case 17 redirected-range-resume; fi

# 18. Authority parsing rejects credentials, nondefault ports and scheme tricks.
CASES=$((CASES + 1))
sed -n '/^effective_host_ok() {/,/^}/p' "$FETCH" > "$WORK/host-policy.sh"
cat >> "$WORK/host-policy.sh" <<'EOF'
for allowed in 'https://huggingface.co/file' 'https://CDN-LFS.HUGGINGFACE.CO:443/file' 'https://us.aws.cdn.hf.co/file' 'https://cas-bridge.xethub.hf.co/file'; do
    effective_host_ok "$allowed" || exit 1
done
for denied in 'https://huggingface.co@unlisted.invalid/file' 'https://unlisted.invalid@huggingface.co/file' 'https://huggingface.co:444/file' 'http://huggingface.co/file' 'https://unlisted.invalid/.cdn.hf.co/file' 'https://a..cdn.hf.co/file' 'https://-a.cdn.hf.co/file' 'https://huggingface.co\@unlisted.invalid/file'; do
    if effective_host_ok "$denied"; then exit 1; fi
done
EOF
if sh "$WORK/host-policy.sh"; then pass_case 18 strict-redirect-authority
else fail_case 18 strict-redirect-authority; fi

# 19. Contention also preserves a lock's retained owner/recovery contents.
CASES=$((CASES + 1)); DEST19="$WORK/case19"
secure_mkdir "$DEST19/.fnlp-fetch-lock-$REV"
printf 'another invocation owns this lock\n' > "$DEST19/.fnlp-fetch-lock-$REV/owner.txt"
run_fetch "$DEST19" --check-only > "$WORK/case19.out" 2> "$WORK/case19.err"
lock_status=$?
if [ "$lock_status" -eq 4 ] && [ -f "$DEST19/.fnlp-fetch-lock-$REV/owner.txt" ] &&
    [ "$(cat "$DEST19/.fnlp-fetch-lock-$REV/owner.txt")" = 'another invocation owns this lock' ] &&
    ! grep -q 'LOCK_RELEASE_DEFERRED' "$WORK/case19.err"; then pass_case 19 nonempty-contended-lock-untouched
else fail_case 19 nonempty-contended-lock-untouched; fi

run_disk_cases
run_interrupt_cases
