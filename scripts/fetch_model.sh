#!/bin/sh
# Download the pinned Nanbeige4.2-3B conversion source closure.  This is a
# human-run provisioning helper, never an installer or an inference path.

set -u

PROGRAM=${0##*/}
DEFAULT_REVISION=f56ec5a9650268aa098496734743c25ea778bd2d
MODEL=Nanbeige4.2-3B
DEFAULT_DEST="$HOME/.cache/franken_nlp/source/$MODEL/$DEFAULT_REVISION"
DEFAULT_BASE="https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve"
TOTAL_DEFAULT=8360887509
LARGEST_DEFAULT=4973547960
MARGIN_BYTES=67108864
MAX_BYTES=9223372036854775807

DEST=$DEFAULT_DEST
DEST_WAS_SET=0
REVISION=$DEFAULT_REVISION
CATALOG=
CHECK_ONLY=0
ALLOW_UNTRUSTED=0
TEST_BASE=
LOCK_DIR=
LOCK_OWNED=0
JOURNAL_DIR=
VERIFIED_NAMES=' '
USER_AGENT='franken_nlp-agent (+https://github.com/Dicklesworthstone/franken_nlp)'

log() {
    printf '%s FETCH_MODEL %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

usage() {
    cat >&2 <<EOF
Usage: scripts/fetch_model.sh [--dest DIR] [--check-only]
       scripts/fetch_model.sh --revision REV --allow-untrusted-revision --catalog FILE [--dest DIR]

Downloads the pinned Nanbeige4.2-3B source closure for fnlp convert. This is
not an end-user installer. --check-only rehashes every catalogued file.
EOF
}

resume_guidance() {
    printf 'RESUME command=scripts/fetch_model.sh --dest %s' "$(shell_quote "$DEST")" >&2
    if [ "$REVISION" != "$DEFAULT_REVISION" ]; then printf ' --revision %s' "$(shell_quote "$REVISION")" >&2; fi
    if [ "$ALLOW_UNTRUSTED" -eq 1 ]; then printf ' --allow-untrusted-revision' >&2; fi
    if [ -n "$CATALOG" ]; then printf ' --catalog %s' "$(shell_quote "$CATALOG")" >&2; fi
    if [ -n "$TEST_BASE" ]; then printf ' --test-base-url %s' "$(shell_quote "$TEST_BASE")" >&2; fi
    printf ' journal_dir=%s note="retry resumes secure journal-bound partials rather than restarting"\n' "$(shell_quote "$JOURNAL_DIR")" >&2
}

shell_quote() { printf %s "$1" | sed "s/'/'\\\\''/g; s/^/'/; s/$/'/"; }

release_lock() {
    # A failed mkdir never grants ownership of another downloader's lock.
    if [ "$LOCK_OWNED" -eq 1 ] && [ -n "$LOCK_DIR" ]; then
        if rmdir "$LOCK_DIR" 2>/dev/null; then LOCK_OWNED=0
        else log "LOCK_RELEASE_DEFERRED path=$LOCK_DIR"; fi
    fi
}

finish() {
    code=$1
    release_lock
    exit "$code"
}

fail() {
    code=$1
    shift
    log "ERROR code=$code detail=$*"
    resume_guidance
    finish "$code"
}

catalog_default() {
    cat <<'EOF'
model-00001-of-00002.safetensors|4973547960|09d265d5ec837bc64462796b7f8c110be9a135a55ed7a6eb5d07e0e90c976a94
model-00002-of-00002.safetensors|3366076760|31019e7870a044f44bc3f7e981f8c5ecd42d341e5ca6cfdbfd07fb95d95be389
model.safetensors.index.json|16519|30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1
config.json|1019|f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19
tokenizer.json|18450979|1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f
tokenizer.model|2782298|fb41d04798b714520a9b075727b0226538b7330254299062742c50ec8374bc36
tokenizer_config.json|10990|3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518
added_tokens.json|174|9e3b127a27647df2c353cc1e5500826f7cdbe8bd15a458e368bba8422e9719cf
special_tokens_map.json|623|b718fce2b7a8940ffeddc1e67f3b092cc0d13ac885c63a021528786f8c4cf6c0
generation_config.json|187|68c690ce23efb6caae30c006ff3c1efd826297ff1df4338c04f7ac6f685d8746
EOF
}

catalog_lines() {
    if [ -n "$CATALOG" ]; then
        cat "$CATALOG"
    else
        catalog_default
    fi
}

valid_byte_count() {
    case "$1" in ''|*[!0-9]*|0[0-9]*) return 1;; esac
    [ "${#1}" -le 19 ] && [ "$1" -le "$MAX_BYTES" ] 2>/dev/null
}

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        fail 2 "no SHA-256 utility available (need sha256sum or shasum)"
    fi
}

bytes() {
    wc -c < "$1" | tr -d '[:space:]'
}

stat_uid() {
    if stat -c %u "$1" >/dev/null 2>&1; then
        stat -c %u "$1"
    else
        stat -f %u "$1"
    fi
}

stat_mode() {
    if stat -c %a "$1" >/dev/null 2>&1; then
        stat -c %a "$1"
    else
        stat -f %Lp "$1"
    fi
}

is_owner_controlled_dir() {
    path=$1
    [ -d "$path" ] && [ ! -L "$path" ] || return 1
    [ "$(stat_uid "$path")" = "$(id -u)" ] || return 1
    mode=$(stat_mode "$path") || return 1
    group_other=$((mode % 100)) 2>/dev/null || group_other=99
    [ "$group_other" -eq 0 ] 2>/dev/null
}

is_secure_regular() {
    path=$1
    [ -f "$path" ] && [ ! -L "$path" ] || return 1
    [ "$(stat_uid "$path")" = "$(id -u)" ]
}

ensure_destination() {
    old_umask=$(umask)
    umask 077
    mkdir -p "$DEST" || fail 4 "cannot create destination=$DEST"
    is_owner_controlled_dir "$DEST" || fail 4 "destination must be an owner-controlled non-shared directory: $DEST"
    JOURNAL_DIR="$DEST/.fnlp-fetch-journals"
    mkdir -p "$JOURNAL_DIR" || fail 4 "cannot create journal directory=$JOURNAL_DIR"
    umask "$old_umask"
    is_owner_controlled_dir "$JOURNAL_DIR" || fail 4 "journal directory is not owner-controlled: $JOURNAL_DIR"
}

catalog_stats() {
    CATALOG_TOTAL=0
    CATALOG_LARGEST=0
    CATALOG_COUNT=0
    CATALOG_NAMES=' '
    while IFS='|' read -r file length digest; do
        [ -n "$file" ] || continue
        case "$file" in
            *[!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-]*|.*|*"/"*|*".."*) fail 2 "malformed catalog filename=$file";;
        esac
        valid_byte_count "$length" || fail 2 "malformed catalog length file=$file"
        case "$digest" in *[!0123456789abcdef]*) fail 2 "malformed catalog digest file=$file";; esac
        [ "${#digest}" -eq 64 ] || fail 2 "malformed catalog digest file=$file"
        case "$CATALOG_NAMES" in *" $file "*) fail 2 "duplicate catalog name=$file";; esac
        CATALOG_NAMES="$CATALOG_NAMES$file "
        [ "$CATALOG_TOTAL" -le "$((MAX_BYTES - MARGIN_BYTES - length))" ] || fail 2 "catalog byte total exceeds supported disk arithmetic"
        CATALOG_TOTAL=$((CATALOG_TOTAL + length))
        [ "$length" -le "$CATALOG_LARGEST" ] || CATALOG_LARGEST=$length
        CATALOG_COUNT=$((CATALOG_COUNT + 1))
    done <<EOF
$(catalog_lines)
EOF
    [ "$CATALOG_COUNT" -gt 0 ] || fail 2 "catalog has no entries"
    if [ -z "$CATALOG" ]; then
        [ "$CATALOG_COUNT" -eq 10 ] && [ "$CATALOG_TOTAL" -eq "$TOTAL_DEFAULT" ] && [ "$CATALOG_LARGEST" -eq "$LARGEST_DEFAULT" ] || fail 1 "embedded catalog invariant failed"
    fi
}

available_bytes() {
    df -Pk "$DEST" | awk 'NR == 2 { printf "%.0f\n", $4 * 1024 }'
}

verify_existing_closure() {
    # Authenticate once under this invocation's revision lock. Only these
    # exact catalog names may be excluded from subsequent disk admission.
    while IFS='|' read -r inventory_name inventory_length inventory_digest; do
        [ -n "$inventory_name" ] || continue
        if { [ -e "$DEST/$inventory_name" ] || [ -L "$DEST/$inventory_name" ]; } &&
            verify_file "$DEST/$inventory_name" "$inventory_length" "$inventory_digest"; then
            VERIFIED_NAMES="$VERIFIED_NAMES$inventory_name "
        fi
    done <<EOF
$(catalog_lines)
EOF
}

remaining_download_bytes() {
    # Called in a command substitution so journal helpers cannot change the
    # active download's shell variables. A bound partial receives disk credit,
    # never content authority: the complete SHA-256 gate still follows download.
    remaining=0
    while IFS='|' read -r space_name space_length space_digest; do
        [ -n "$space_name" ] || continue
        case "$VERIFIED_NAMES" in *" $space_name "*) continue;; esac
        space_partial="$DEST/$space_name.partial"
        space_journal="$JOURNAL_DIR/$space_name.journal"
        if [ -n "$TEST_BASE" ]; then space_url="${TEST_BASE%/}/$space_name"
        else space_url="$DEFAULT_BASE/$REVISION/$space_name"; fi
        retained=0
        if is_secure_regular "$space_partial" &&
            journal_matches "$space_journal" "$space_url" "$space_name" "$space_length" "$space_digest"; then
            retained=$(bytes "$space_partial") || return 1
            valid_byte_count "$retained" || return 1
            [ "$retained" -le "$space_length" ] || retained=0
        fi
        additional=$((space_length - retained))
        [ "$remaining" -le "$((MAX_BYTES - MARGIN_BYTES - additional))" ] || return 1
        remaining=$((remaining + additional))
    done <<EOF
$(catalog_lines)
EOF
    printf '%s\n' "$remaining"
}

preflight_space() {
    remaining=$(remaining_download_bytes) || fail 4 "cannot determine remaining source bytes"
    available=$(available_bytes)
    valid_byte_count "$available" || fail 4 "cannot determine free space for $DEST"
    required=$((remaining + MARGIN_BYTES))
    # Partials become finals by same-directory rename: they are not a second
    # payload copy. Existing quarantines already reduce measured free bytes.
    if [ "$available" -lt "$required" ]; then
        fail 4 "insufficient disk available=$available needed=$required (remaining=$remaining additional_staging=0 margin=$MARGIN_BYTES)"
    fi
    log "PREFLIGHT RESULT=PASS available=$available needed=$required remaining=$remaining additional_staging=0 margin=$MARGIN_BYTES"
}

acquire_lock() {
    LOCK_DIR="$DEST/.fnlp-fetch-lock-$REVISION"
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        fail 4 "revision lock busy or stale path=$LOCK_DIR; do not bypass concurrent access"
    fi
    LOCK_OWNED=1
    log "LOCK RESULT=PASS path=$LOCK_DIR"
}

quarantine() {
    original=$1
    reason=$2
    [ -e "$original" ] || [ -L "$original" ] || return 0
    qdir="$DEST/quarantine"
    old_umask=$(umask)
    umask 077
    mkdir -p "$qdir" || fail 4 "cannot create quarantine directory=$qdir"
    umask "$old_umask"
    is_owner_controlled_dir "$qdir" || fail 4 "quarantine directory is not owner-controlled: $qdir"
    observed=unreadable
    if is_secure_regular "$original"; then
        observed=$(sha256 "$original")
    elif [ -L "$original" ]; then
        observed=symlink-refused
    fi
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    target="$qdir/${original##*/}.$stamp.$$.${observed}"
    mv "$original" "$target" || fail 4 "cannot quarantine path=$original"
    printf '%s QUARANTINE file=%s target=%s observed_sha256=%s reason=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$original" "$target" "$observed" "$reason" >> "$qdir/quarantine.log"
    log "QUARANTINE file=${original##*/} observed_sha256=$observed reason=$reason"
}

journal_text() {
    printf 'url=%s\nrevision=%s\nname=%s\nbytes=%s\nsha256=%s\n' "$1" "$REVISION" "$2" "$3" "$4"
}

journal_matches() {
    journal=$1
    url=$2
    name=$3
    length=$4
    digest=$5
    is_secure_regular "$journal" || return 1
    actual=$(cat "$journal")
    expected=$(journal_text "$url" "$name" "$length" "$digest")
    [ "$actual" = "$(printf %s "$expected")" ]
}

create_exclusive_regular() {
    path=$1
    old_umask=$(umask)
    umask 077
    ( set -C; : > "$path" ) 2>/dev/null
    created=$?
    umask "$old_umask"
    [ "$created" -eq 0 ] && is_secure_regular "$path"
}

ensure_partial_and_journal() {
    partial=$1
    journal=$2
    url=$3
    name=$4
    length=$5
    digest=$6
    if [ -e "$partial" ] || [ -L "$partial" ]; then
        if [ -L "$partial" ]; then
            fail 1 "refused symlink partial=$partial"
        fi
        if [ -L "$journal" ]; then
            fail 1 "refused symlink journal=$journal"
        fi
        if ! is_secure_regular "$partial" || ! journal_matches "$journal" "$url" "$name" "$length" "$digest"; then
            quarantine "$partial" "unbound-or-mismatched-partial"
            quarantine "$journal" "mismatched-journal"
        fi
    elif [ -e "$journal" ] || [ -L "$journal" ]; then
        quarantine "$journal" "orphaned-journal"
    fi
    if [ ! -e "$partial" ] && ! create_exclusive_regular "$partial"; then
        fail 1 "refused to create secure partial=$partial"
    fi
    if [ ! -e "$journal" ]; then
        old_umask=$(umask)
        umask 077
        ( set -C; journal_text "$url" "$name" "$length" "$digest" > "$journal" ) 2>/dev/null
        created=$?
        umask "$old_umask"
        [ "$created" -eq 0 ] || fail 1 "refused to create journal=$journal"
    fi
    is_secure_regular "$partial" && journal_matches "$journal" "$url" "$name" "$length" "$digest" || fail 1 "partial/journal binding rejected name=$name"
}

effective_host_ok() {
    # Compare the authority, not a glob over the complete URL: a path such as
    # unlisted.invalid/example.cdn.hf.co/file must never satisfy the host rule.
    case "$1" in https://*) ;; *) return 1;; esac
    case "$1" in *[[:space:][:cntrl:]]*|*\\*) return 1;; esac
    policy_authority=${1#https://}
    policy_authority=${policy_authority%%[/?#]*}
    case "$policy_authority" in *:443) policy_authority=${policy_authority%:443};; esac
    policy_host=$(printf %s "$policy_authority" | tr '[:upper:]' '[:lower:]')
    case "$policy_host" in ''|*[!a-z0-9.-]*|.*|*.|*..*) return 1;; esac
    policy_labels=$policy_host
    while [ -n "$policy_labels" ]; do
        policy_label=${policy_labels%%.*}
        case "$policy_label" in -*|*-) return 1;; esac
        case "$policy_labels" in *.*) policy_labels=${policy_labels#*.};; *) break;; esac
    done
    case "$policy_host" in
        huggingface.co|cdn-lfs.huggingface.co|*.xethub.hf.co|*.cdn.hf.co) return 0;;
        *) return 1;;
    esac
}

download_with_progress() {
    transfer_origin=$1
    transfer_partial=$2
    transfer_expected=$3
    curl_log="$JOURNAL_DIR/${transfer_partial##*/}.curl.log"
    curl_meta="$JOURNAL_DIR/${transfer_partial##*/}.curl.meta"
    for transfer_log in "$curl_log" "$curl_meta"; do
        if { [ -e "$transfer_log" ] || [ -L "$transfer_log" ]; } && ! is_secure_regular "$transfer_log"; then
            log "TRANSFER_LOG_REFUSED path=$transfer_log"
            return 90
        fi
    done
    transfer_attempt=1
    while [ "$transfer_attempt" -le 4 ]; do
        transfer_url=$transfer_origin
        transfer_hops=0
        while :; do
            if [ -z "$TEST_BASE" ] && ! effective_host_ok "$transfer_url"; then
                log "REDIRECT_HOST_REFUSED phase=pre-contact activation=refused effective_url=$transfer_url"
                return 90
            fi
            transfer_start=$(bytes "$transfer_partial")
            if [ "$transfer_start" -eq "$transfer_expected" ]; then return 0; fi
            if [ "$transfer_start" -gt "$transfer_expected" ]; then return 63; fi
            # Zero automatic hops lets curl resolve relative Location values,
            # but it returns 47 BEFORE following or writing a redirect body.
            # Only the next iteration may contact the validated destination.
            # Disable user curl config so it cannot add credentials/redirects.
            set -- curl --disable --globoff --silent --show-error --fail --location --max-redirs 0
            if [ -z "$TEST_BASE" ]; then set -- "$@" --proto =https --proto-redir =https; fi
            set -- "$@" --connect-timeout 30 --user-agent "$USER_AGENT"
            "$@" --continue-at - --output "$transfer_partial" --write-out '%{http_code}
%{redirect_url}
' "$transfer_url" > "$curl_meta" 2> "$curl_log" &
            curl_pid=$!
            while kill -0 "$curl_pid" 2>/dev/null; do
                current=$(bytes "$transfer_partial" 2>/dev/null || printf 0)
                log "PROGRESS file=${transfer_partial##*/} bytes=$current expected_bytes=$transfer_expected"
                sleep 1
            done
            wait "$curl_pid"
            curl_status=$?
            transfer_status=$(sed -n '1p' "$curl_meta" | tr -d '\r')
            if [ "$curl_status" -eq 47 ]; then
                case "$transfer_status" in 301|302|303|307|308) ;; *) return 47;; esac
                if [ "$(bytes "$transfer_partial")" != "$transfer_start" ]; then
                    log "REDIRECT_BODY_REFUSED file=${transfer_partial##*/}"
                    return 90
                fi
                if [ "$transfer_hops" -ge 8 ]; then
                    log "REDIRECT_LIMIT_REFUSED maximum=8"
                    return 47
                fi
                transfer_next=$(sed -n '2p' "$curl_meta" | tr -d '\r')
                if [ -z "$transfer_next" ]; then return 47; fi
                transfer_hops=$((transfer_hops + 1))
                transfer_url=$transfer_next
                continue
            fi
            if [ "$curl_status" -eq 0 ]; then
                case "$transfer_status" in 200|206) return 0;; *) return 22;; esac
            fi
            # The caller quarantines a server-refused resume before restarting.
            if [ "$curl_status" -eq 33 ]; then return 33; fi
            break
        done
        if [ "$transfer_attempt" -lt 4 ]; then sleep "$transfer_attempt"; fi
        transfer_attempt=$((transfer_attempt + 1))
    done
    return "$curl_status"
}

verify_file() {
    path=$1
    expected_length=$2
    expected_digest=$3
    if ! is_secure_regular "$path"; then
        if [ -L "$path" ]; then observed_kind=symlink-refused; else observed_kind=missing; fi
        log "VERIFY RESULT=FAIL file=${path##*/} expected_bytes=$expected_length observed_bytes=$observed_kind expected_sha256=$expected_digest observed_sha256=$observed_kind"
        return 1
    fi
    observed_length=$(bytes "$path")
    observed_digest=$(sha256 "$path")
    [ "$observed_length" = "$expected_length" ] && [ "$observed_digest" = "$expected_digest" ] || {
        log "VERIFY RESULT=FAIL file=${path##*/} expected_bytes=$expected_length observed_bytes=$observed_length expected_sha256=$expected_digest observed_sha256=$observed_digest"
        return 1
    }
    log "VERIFY RESULT=PASS file=${path##*/} observed_bytes=$observed_length sha256=$observed_digest"
}

download_one() {
    name=$1
    length=$2
    digest=$3
    if [ -n "$TEST_BASE" ]; then
        url="${TEST_BASE%/}/$name"
    else
        url="$DEFAULT_BASE/$REVISION/$name"
    fi
    final="$DEST/$name"
    partial="$DEST/$name.partial"
    journal="$JOURNAL_DIR/$name.journal"
    log "START file=$name expected_bytes=$length url=$url"
    case "$VERIFIED_NAMES" in *" $name "*) log "DECISION=HIT file=$name"; return 0;; esac
    if [ -e "$final" ] || [ -L "$final" ]; then
        if verify_file "$final" "$length" "$digest"; then
            VERIFIED_NAMES="$VERIFIED_NAMES$name "
            log "DECISION=HIT file=$name"
            return 0
        fi
        quarantine "$final" "existing-file-verification-failed"
    fi
    ensure_partial_and_journal "$partial" "$journal" "$url" "$name" "$length" "$digest"
    start=$(bytes "$partial")
    if [ "$start" -gt "$length" ]; then
        quarantine "$partial" "partial-longer-than-expected"
        quarantine "$journal" "partial-longer-than-expected"
        ensure_partial_and_journal "$partial" "$journal" "$url" "$name" "$length" "$digest"
        start=0
    fi
    preflight_space
    log "DECISION=$([ "$start" -gt 0 ] && printf RESUME || printf MISS) file=$name existing_bytes=$start expected_bytes=$length"
    download_with_progress "$url" "$partial" "$length"
    curl_status=$?
    if [ "$curl_status" -eq 33 ] && [ "$start" -gt 0 ]; then
        log "DECISION=RESTART file=$name reason=server-refused-range-resume"
        quarantine "$partial" "server-refused-range-resume"
        quarantine "$journal" "server-refused-range-resume"
        ensure_partial_and_journal "$partial" "$journal" "$url" "$name" "$length" "$digest"
        preflight_space
        download_with_progress "$url" "$partial" "$length"
        curl_status=$?
    fi
    if [ "$curl_status" -ne 0 ]; then
        fail 3 "network failure url=$url attempts=4 curl_exit=$curl_status log=$JOURNAL_DIR/${partial##*/}.curl.log"
    fi
    if ! verify_file "$partial" "$length" "$digest"; then
        fail 1 "digest/length verification failure file=$name expected_bytes=$length expected_sha256=$digest"
    fi
    sync "$partial" 2>/dev/null || sync
    mv "$partial" "$final" || fail 1 "same-directory activation rename failed file=$name"
    sync "$final" 2>/dev/null || sync
    VERIFIED_NAMES="$VERIFIED_NAMES$name "
    log "COMPLETE file=$name observed_bytes=$length sha256=$digest"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dest) [ "$#" -ge 2 ] || { usage; exit 2; }; DEST=$2; DEST_WAS_SET=1; shift 2;;
        --check-only) CHECK_ONLY=1; shift;;
        --revision) [ "$#" -ge 2 ] || { usage; exit 2; }; REVISION=$2; shift 2;;
        --allow-untrusted-revision) ALLOW_UNTRUSTED=1; shift;;
        --catalog) [ "$#" -ge 2 ] || { usage; exit 2; }; CATALOG=$2; shift 2;;
        --test-base-url) [ "$#" -ge 2 ] || { usage; exit 2; }; TEST_BASE=$2; shift 2;;
        --help|-h) usage; exit 0;;
        *) usage; exit 2;;
    esac
done

case "$REVISION" in *[!0123456789abcdef]*|'') usage; exit 2;; esac
if [ "$REVISION" != "$DEFAULT_REVISION" ]; then
    [ "$ALLOW_UNTRUSTED" -eq 1 ] && [ -n "$CATALOG" ] || { log "UNTRUSTED_REVISION_REFUSED revision=$REVISION requires --allow-untrusted-revision and --catalog"; exit 2; }
    log "UNTRUSTED_REVISION revision=$REVISION catalog=$CATALOG; this download is not the default recipe/catalog identity"
elif [ -n "$CATALOG" ]; then
    [ "$ALLOW_UNTRUSTED" -eq 1 ] || { log "UNTRUSTED_CATALOG_REFUSED revision=$REVISION requires --allow-untrusted-revision with --catalog"; exit 2; }
    log "UNTRUSTED_CATALOG revision=$REVISION catalog=$CATALOG; this download is not the default recipe/catalog identity"
fi
if [ "$DEST_WAS_SET" -eq 0 ]; then
    DEST="$HOME/.cache/franken_nlp/source/$MODEL/$REVISION"
fi
if [ -n "$CATALOG" ] && [ ! -f "$CATALOG" ]; then
    log "ERROR code=2 detail=catalog does not exist: $CATALOG"
    exit 2
fi
if [ -n "$TEST_BASE" ] && [ "${FNLP_FETCH_ALLOW_TEST_BASE_URL:-}" != 1 ]; then
    log "ERROR code=2 detail=--test-base-url requires FNLP_FETCH_ALLOW_TEST_BASE_URL=1"
    exit 2
fi
ensure_destination
catalog_stats
acquire_lock
if [ "$CHECK_ONLY" -eq 0 ]; then verify_existing_closure; fi

status=0
while IFS='|' read -r file length digest; do
    [ -n "$file" ] || continue
    if [ "$CHECK_ONLY" -eq 1 ]; then
        log "CHECK_ONLY file=$file expected_bytes=$length"
        if ! verify_file "$DEST/$file" "$length" "$digest"; then
            status=1
        fi
    else
        download_one "$file" "$length" "$digest"
    fi
done <<EOF
$(catalog_lines)
EOF

if [ "$status" -ne 0 ]; then
    log "CHECK_ONLY RESULT=FAIL files=$CATALOG_COUNT/$CATALOG_COUNT"
    resume_guidance
    finish 1
fi
if [ "$CHECK_ONLY" -eq 1 ]; then
    log "CHECK_ONLY RESULT=PASS files=$CATALOG_COUNT/$CATALOG_COUNT"
else
    log "FETCH_MODEL RESULT=PASS files=$CATALOG_COUNT/$CATALOG_COUNT bytes=$CATALOG_TOTAL"
fi
printf 'NEXT source=%s command="fnlp convert --source %s --source-manifest docs/truth-pack/nanbeige4.2-3b.source.json --recipe nanbeige42-int8-v1 --arch generic -o nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq"\n' "$DEST" "$DEST" >&2
finish 0
