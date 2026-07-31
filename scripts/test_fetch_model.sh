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
    FNLP_FETCH_ALLOW_TEST_BASE_URL=1 "$FETCH" --dest "$dest" --catalog "$CATALOG" --test-base-url "$BASE" "$@"
}

start_redirect_server() {
    redirect_host=$1
    TLS_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
    if ! openssl req -x509 -newkey rsa:2048 -nodes -keyout "$WORK/redirect.key" -out "$WORK/redirect.crt" -subj /CN=localhost -days 1 > "$WORK/openssl.log" 2>&1; then
        return 1
    fi
    FIXTURE_ROOT="$FIXTURE" REDIRECT_HOST="$redirect_host" TLS_PORT="$TLS_PORT" TLS_CERT="$WORK/redirect.crt" TLS_KEY="$WORK/redirect.key" python3 -c '
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
        if host == "huggingface.co":
            self.send_response(302)
            self.send_header("Location", "https://{}{}".format(redirect_host, self.path))
            self.end_headers()
            return
        if host != redirect_host:
            self.send_error(421, "unexpected host")
            return
        candidate = root / Path(urlsplit(self.path).path).name
        if not candidate.is_file():
            self.send_error(404)
            return
        payload = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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
    start_redirect_server "$redirect_host" || return 1
    HTTPS_PROXY= https_proxy= HTTP_PROXY= http_proxy= ALL_PROXY= all_proxy= NO_PROXY= no_proxy= \
        FNLP_FETCH_ALLOW_TEST_REDIRECT_POLICY=1 \
        FNLP_FETCH_TEST_CONNECT_TO_1="huggingface.co:443:127.0.0.1:$TLS_PORT" \
        FNLP_FETCH_TEST_CONNECT_TO_2="$redirect_host:443:127.0.0.1:$TLS_PORT" \
        "$FETCH" --dest "$dest" --catalog "$CATALOG" > "$output" 2> "$error"
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

# 10. A local TLS redirect to an official regional Xet CDN host passes the real policy.
CASES=$((CASES + 1)); DEST10="$WORK/case10"
if run_redirect_policy_fetch us.aws.cdn.hf.co "$DEST10" "$WORK/case10.out" "$WORK/case10.err" && cmp "$FIXTURE/alpha.bin" "$DEST10/alpha.bin" >/dev/null && cmp "$FIXTURE/beta.bin" "$DEST10/beta.bin" >/dev/null; then pass_case 10 regional-cdn-redirect-accepted; else fail_case 10 regional-cdn-redirect-accepted; fi

# 11. The same hermetic hop refuses a final redirect outside the explicit allowlist.
CASES=$((CASES + 1)); DEST11="$WORK/case11"
if run_redirect_policy_fetch unlisted.invalid "$DEST11" "$WORK/case11.out" "$WORK/case11.err"; then
    fail_case 11 unlisted-redirect-refusal
elif grep -q 'REDIRECT_HOST_REFUSED effective_url=https://unlisted.invalid/' "$WORK/case11.err"; then
    pass_case 11 unlisted-redirect-refusal
else
    fail_case 11 unlisted-redirect-refusal
fi
