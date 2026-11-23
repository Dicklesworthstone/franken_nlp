# Local-fixture tests for fetch_model.ps1. The retained work directory is
# printed on completion for audit; this harness deliberately never deletes it.
[CmdletBinding()]
param([switch]$DiskOnly, [switch]$LifecycleOnly)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Fetch = Join-Path $PSScriptRoot 'fetch_model.ps1'
$Work = Join-Path ([IO.Path]::GetTempPath()) ('fnlp-fetch-model-test.' + [Guid]::NewGuid().ToString('N'))
$Revision = 'f56ec5a9650268aa098496734743c25ea778bd2d'
$script:Cases = 0
$script:Failed = [System.Collections.Generic.List[string]]::new()
$script:Server = $null
$script:RedirectServer = $null
$script:RetryServer = $null
$script:PwshPath = $null

function Write-Log([string]$Message) { [Console]::Error.WriteLine("{0} FETCH_MODEL_TEST {1}" -f [DateTime]::UtcNow.ToString('o'), $Message) }
function Pass([string]$Case, [string]$Detail) { Write-Log "CASE=$Case RESULT=PASS detail=$Detail" }
function Fail([string]$Case, [string]$Detail) { $script:Failed.Add($Case); Write-Log "CASE=$Case RESULT=FAIL detail=$Detail" }
function Invoke-Fetch([string]$Dest, [string[]]$Extra = @()) {
    $env:FNLP_FETCH_ALLOW_TEST_BASE_URL = '1'
    & $script:PwshPath -NoProfile -File $Fetch -Dest $Dest -Catalog $script:Catalog -TestBaseUrl $script:Base -AllowUntrustedRevision @Extra
    return $LASTEXITCODE
}
function Stop-RedirectServer {
    if ($script:RedirectServer) {
        try { Stop-Process -Id $script:RedirectServer.Id -ErrorAction Stop } catch {}
        $script:RedirectServer = $null
    }
}
function Stop-RetryServer {
    if ($script:RetryServer) {
        try { Stop-Process -Id $script:RetryServer.Id -ErrorAction Stop } catch {}
        $script:RetryServer = $null
    }
}
function Start-InterruptedResumeServer([switch]$MalformedRange, [switch]$OverlongBody, [switch]$SilentHeaders, [switch]$SilentBody, [string]$SourceDirectory = $Fixture) {
    $runId = [Guid]::NewGuid().ToString('N')
    $ready = Join-Path $Work ("retry-$runId.ready")
    $trace = Join-Path $Work ("retry-$runId.trace")
    $serverScript = Join-Path $Work ("retry_server-$runId.py")
    $serverCode = @'
import http.server
import socket
import sys
from pathlib import Path
from urllib.parse import urlsplit

fixture, ready_path, trace_path, mode = sys.argv[1:]
root = Path(fixture)
state = {"first_alpha": True}

class Handler(http.server.BaseHTTPRequestHandler):
    def wait_for_disconnect(self):
        with open(trace_path, "a", encoding="ascii") as trace:
            trace.write("stall-ready={}\n".format(mode))
        self.connection.settimeout(15)
        try:
            closed = self.connection.recv(1) == b""
        except (ConnectionResetError, ConnectionAbortedError):
            closed = True
        except (TimeoutError, OSError):
            closed = False
        with open(trace_path, "a", encoding="ascii") as trace:
            trace.write("stall-closed={}\n".format(closed))
        self.close_connection = True

    def do_GET(self):
        candidate = root / Path(urlsplit(self.path).path).name
        if not candidate.is_file():
            self.send_error(404)
            return
        payload = candidate.read_bytes()
        range_header = self.headers.get("Range", "")
        with open(trace_path, "a", encoding="ascii") as trace:
            trace.write("{} range={}\n".format(candidate.name, range_header))
        if candidate.name == "alpha.bin" and state["first_alpha"]:
            state["first_alpha"] = False
            if mode == "silent-headers":
                self.wait_for_disconnect()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            prefix = 8192 if mode == "silent-body" else 5
            self.wfile.write(payload[:prefix])
            self.wfile.flush()
            if mode == "silent-body":
                self.wait_for_disconnect()
                return
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        if candidate.name == "alpha.bin":
            start = 5
            if mode in ("silent-headers", "silent-body"):
                if not range_header:
                    start = 0
                elif range_header.startswith("bytes=") and range_header.endswith("-"):
                    try:
                        start = int(range_header[6:-1])
                    except ValueError:
                        self.send_error(416)
                        return
                else:
                    self.send_error(416)
                    return
                maximum = 8192 if mode == "silent-body" else 0
                if start < 0 or start > maximum:
                    self.send_error(416)
                    return
            elif range_header != "bytes=5-":
                self.send_error(416, "expected bytes=5-")
                return
            remainder = payload[start:]
            self.send_response(206 if start else 200)
            if mode != "overlong-body":
                self.send_header("Content-Length", str(len(remainder)))
            if start:
                reported_start = 4 if mode == "malformed-range" else start
                self.send_header("Content-Range", "bytes {}-{}/{}".format(reported_start, len(payload) - 1, len(payload)))
            self.end_headers()
            self.wfile.write(remainder)
            if mode == "overlong-body":
                self.wfile.write(b"X")
                self.close_connection = True
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        pass

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(ready_path, "w", encoding="ascii") as ready:
    ready.write(str(server.server_address[1]))
server.serve_forever()
'@
    Set-Content -LiteralPath $serverScript -NoNewline -Encoding utf8 -Value $serverCode
    $mode = if ($SilentHeaders) { 'silent-headers' } elseif ($SilentBody) { 'silent-body' } elseif ($MalformedRange) { 'malformed-range' } elseif ($OverlongBody) { 'overlong-body' } else { 'exact-range' }
    $script:RetryServer = Start-Process -FilePath python3 -ArgumentList @($serverScript, $SourceDirectory, $ready, $trace, $mode) -PassThru -RedirectStandardOutput (Join-Path $Work ("retry-$runId.out")) -RedirectStandardError (Join-Path $Work ("retry-$runId.err"))
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if (Test-Path -LiteralPath $ready) {
            $port = [int](Get-Content -LiteralPath $ready -Raw)
            return [PSCustomObject]@{ Base = "http://127.0.0.1:$port/resolve/$Revision"; Trace = $trace }
        }
        Start-Sleep -Milliseconds 100
    }
    Stop-RetryServer
    return $null
}
function Start-RedirectServer([string]$RedirectHost, [string]$RedirectLocation = '') {
    $runId = [Guid]::NewGuid().ToString('N')
    $cert = Join-Path $Work 'redirect.crt'
    $key = Join-Path $Work 'redirect.key'
    $caCert = Join-Path $Work 'redirect-ca.crt'
    $caKey = Join-Path $Work 'redirect-ca.key'
    $leafCsr = Join-Path $Work 'redirect.csr'
    $leafConfig = Join-Path $Work 'redirect-openssl.cnf'
    $ready = Join-Path $Work ("redirect-$runId.ready")
    $trace = Join-Path $Work ("redirect-$runId.trace")
    $serverScript = Join-Path $Work 'redirect_server.py'
    $serverCode = @'
import http.server
import os
import select
import socket
import socketserver
import ssl
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

fixture, cert, key, redirect_host, ready_path, trace_path, redirect_location = sys.argv[1:]
root = Path(fixture)
if redirect_location == "default":
    redirect_location = ""

class RedirectHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        host = self.headers.get("Host", "").split(":", 1)[0].lower()
        with open(trace_path, "a", encoding="ascii") as trace:
            trace.write("host={} path={} range={} agent={}\n".format(host, self.path, self.headers.get("Range", ""), self.headers.get("User-Agent", "")))
        if host == "huggingface.co" and not self.path.startswith("/redirect-final/"):
            self.send_response(302)
            target = redirect_location or "https://{}{}".format(redirect_host, self.path)
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

tls = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(cert, key)
tls.socket = context.wrap_socket(tls.socket, server_side=True)
tls_port = tls.server_address[1]

class Proxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

class ConnectHandler(socketserver.StreamRequestHandler):
    def handle(self):
        request = self.rfile.readline(65537).decode("ascii", "replace")
        with open(trace_path, "a", encoding="ascii") as trace:
            trace.write("proxy={}\n".format(request.strip()))
        if not request.startswith("CONNECT "):
            self.wfile.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            return
        while True:
            line = self.rfile.readline(65537)
            if line in (b"", b"\r\n", b"\n"):
                break
        upstream = socket.create_connection(("127.0.0.1", tls_port))
        self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        self.connection.setblocking(False)
        upstream.setblocking(False)
        try:
            while True:
                readable, _, _ = select.select([self.connection, upstream], [], [], 10)
                if not readable:
                    continue
                for source in readable:
                    payload = source.recv(65536)
                    if not payload:
                        return
                    (upstream if source is self.connection else self.connection).sendall(payload)
        finally:
            upstream.close()

proxy = Proxy(("127.0.0.1", 0), ConnectHandler)
with open(ready_path, "w", encoding="ascii") as ready:
    ready.write("{} {}\n".format(proxy.server_address[1], tls_port))
threading.Thread(target=tls.serve_forever, daemon=True).start()
proxy.serve_forever()
'@
    Set-Content -LiteralPath $serverScript -NoNewline -Encoding utf8 -Value $serverCode
    @"
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
DNS.2 = $RedirectHost
"@ | Set-Content -LiteralPath $leafConfig -NoNewline -Encoding utf8
    & openssl req -x509 -newkey rsa:2048 -nodes -keyout $caKey -out $caCert -subj '/CN=fnlp-fetch-fixture-ca' -days 1 *> (Join-Path $Work 'openssl.log')
    if ($LASTEXITCODE -ne 0) { return $null }
    & openssl req -new -newkey rsa:2048 -nodes -keyout $key -out $leafCsr -config $leafConfig *>> (Join-Path $Work 'openssl.log')
    if ($LASTEXITCODE -ne 0) { return $null }
    & openssl x509 -req -in $leafCsr -CA $caCert -CAkey $caKey -CAcreateserial -out $cert -days 1 -sha256 -extfile $leafConfig -extensions v3_leaf *>> (Join-Path $Work 'openssl.log')
    if ($LASTEXITCODE -ne 0) { return $null }
    # Start-Process joins ArgumentList, so represent an empty optional target
    # with a sentinel and resolve it in the Python command instead of losing it.
    $locationArgument = if ($RedirectLocation) { $RedirectLocation } else { 'default' }
    $script:RedirectServer = Start-Process -FilePath python3 -ArgumentList @($serverScript, $Fixture, $cert, $key, $RedirectHost, $ready, $trace, $locationArgument) -PassThru -RedirectStandardOutput (Join-Path $Work ("redirect-$runId.out")) -RedirectStandardError (Join-Path $Work ("redirect-$runId.err"))
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if (Test-Path -LiteralPath $ready) {
            $ports = (Get-Content -LiteralPath $ready -Raw).Trim().Split(' ')
            if ($ports.Count -eq 2) { return [PSCustomObject]@{ ProxyPort = [int]$ports[0]; TlsPort = [int]$ports[1]; CaCertificate = $caCert; Trace = $trace } }
        }
        Start-Sleep -Milliseconds 100
    }
    Stop-RedirectServer
    return $null
}
function Invoke-RedirectPolicyFetch([string]$Dest, [int]$ProxyPort, [string]$CaCertificate) {
    $names = @('SSL_CERT_FILE', 'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'NO_PROXY')
    $saved = @{}
    foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    try {
        $env:SSL_CERT_FILE = $CaCertificate
        $env:HTTPS_PROXY = "http://127.0.0.1:$ProxyPort"
        $env:HTTP_PROXY = "http://127.0.0.1:$ProxyPort"
        $env:ALL_PROXY = "http://127.0.0.1:$ProxyPort"
        $env:NO_PROXY = ''
        & $script:PwshPath -NoProfile -File $Fetch -Dest $Dest -Catalog $script:Catalog -AllowUntrustedRevision
        return $LASTEXITCODE
    } finally {
        foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
    }
}
function Write-Journal([string]$Dest, [string]$Name, [int64]$Length, [string]$Digest, [string]$Url) {
    $journalDir = Join-Path $Dest '.fnlp-fetch-journals'
    New-Item -ItemType Directory -Path $journalDir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $journalDir ($Name + '.journal')) -NoNewline -Encoding utf8 -Value "url=$Url`nrevision=$Revision`nname=$Name`nbytes=$Length`nsha256=$Digest`n"
}

function Test-DiskAdmission {
    $script:Cases++
    # Load real functions only, never the source script's main or a transport.
    $tokens = $null; $parseErrors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile($Fetch, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -ne 0) { throw 'fetcher has PowerShell parse errors' }
    foreach ($definition in $ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
        . ([scriptblock]::Create($definition.Extent.Text))
    }
    function Exit-Fetch([int]$Code, [string]$Message = '') { throw "disk-test-exit:$Code $Message" }
    function Get-FreeBytes { $script:SpaceProbes++; return $script:TestFreeBytes }
    function Download-Partial([string]$Url, [string]$Partial, [string]$Journal, $Entry) {
        $script:TestTransfers++
        $retained = Get-Length $Partial
        $payload = [IO.File]::ReadAllBytes((Join-Path $diskFixture $Entry.Name))
        $stream = [IO.File]::Open($Partial, [IO.FileMode]::Append, [IO.FileAccess]::Write)
        try { $stream.Write($payload, [int]$retained, $payload.Length - [int]$retained) }
        finally { $stream.Dispose() }
        $script:TestFreeBytes -= $Entry.Length - $retained
    }
    function Assert-Disk([bool]$Condition, [string]$Label) {
        if (-not $Condition) { throw "disk admission assertion failed: $Label" }
    }
    function Assert-Quarantined([string]$PayloadPath, [string]$JournalText) {
        $qdir = Join-Path $Dest 'quarantine'
        $payloads = @(Get-ChildItem -LiteralPath $qdir -File | Where-Object { $_.Name -like 'alpha.bin.partial.*' })
        $journals = @(Get-ChildItem -LiteralPath $qdir -File | Where-Object { $_.Name -like 'alpha.bin.journal.*' })
        Assert-Disk ($payloads.Count -eq 1 -and $journals.Count -eq 1) 'rejected bytes and one binding retained'
        Assert-Disk ((Get-Sha256 $payloads[0].FullName) -eq (Get-Sha256 $PayloadPath)) 'quarantine payload stays byte-identical'
        Assert-Disk ([IO.File]::ReadAllText($journals[0].FullName) -eq $JournalText) 'quarantine journal stays byte-identical'
    }
    function New-DiskCase([string]$Name) {
        $path = Join-Path $Work ('disk-' + $Name)
        New-Item -ItemType Directory -Path $path | Out-Null
        $script:JournalDir = Join-Path $path '.fnlp-fetch-journals'
        New-Item -ItemType Directory -Path $script:JournalDir | Out-Null
        $script:VerifiedNames = @{}
        $script:SpaceProbes = 0
        $script:TestTransfers = 0
        return $path
    }
    $MarginBytes = [int64]67108864
    $TestBaseUrl = 'https://fixture.invalid/source'
    $DefaultBase = 'https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve'
    $diskFixture = Join-Path $Work 'disk-fixture'
    New-Item -ItemType Directory -Path $diskFixture | Out-Null
    [IO.File]::WriteAllText((Join-Path $diskFixture 'alpha.bin'), "alpha fixture bytes`n", [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText((Join-Path $diskFixture 'beta.bin'), "beta fixture bytes are slightly longer`n", [Text.UTF8Encoding]::new($false))
    $entries = @(foreach ($name in @('alpha.bin', 'beta.bin')) {
        $path = Join-Path $diskFixture $name
        [PSCustomObject]@{ Name = $name; Length = (Get-Length $path); Sha256 = (Get-Sha256 $path) }
    })
    $total = $entries[0].Length + $entries[1].Length

    $Dest = New-DiskCase cached
    foreach ($entry in $entries) { [IO.File]::Copy((Join-Path $diskFixture $entry.Name), (Join-Path $Dest $entry.Name)) }
    $script:TestFreeBytes = [int64]0
    Initialize-VerifiedFiles $entries
    foreach ($entry in $entries) { Download-One $entry $entries }
    Assert-Disk ($script:SpaceProbes -eq 0 -and $script:TestTransfers -eq 0) 'cached closure needs no new disk or network'

    $Dest = New-DiskCase fresh
    $script:TestFreeBytes = $total + $MarginBytes
    Initialize-VerifiedFiles $entries
    Download-One $entries[0] $entries
    Assert-Disk ((Get-RemainingDownloadBytes $entries) -eq $entries[1].Length) 'completed first file not reserved again'
    Download-One $entries[1] $entries
    Assert-Disk ($script:TestFreeBytes -eq $MarginBytes -and (Get-RemainingDownloadBytes $entries) -eq 0) 'exact closure capacity completes'

    $Dest = New-DiskCase later-hit
    [IO.File]::Copy((Join-Path $diskFixture 'beta.bin'), (Join-Path $Dest 'beta.bin'))
    Initialize-VerifiedFiles $entries
    Assert-Disk ((Get-RemainingDownloadBytes $entries) -eq $entries[0].Length) 'later cache hit excluded before first transfer'

    $Dest = New-DiskCase bound-prefixes
    foreach ($entry in $entries) {
        $payload = [IO.File]::ReadAllBytes((Join-Path $diskFixture $entry.Name))
        [IO.File]::WriteAllBytes((Join-Path $Dest ($entry.Name + '.partial')), $payload[0..4])
        Write-NewUtf8File (Join-Path $script:JournalDir ($entry.Name + '.journal')) (Get-JournalText "$TestBaseUrl/$($entry.Name)" $entry)
    }
    Assert-Disk ((Get-RemainingDownloadBytes $entries) -eq ($total - 10)) 'all bound partials receive storage credit'
    $script:TestFreeBytes = $total - 10 + $MarginBytes
    foreach ($entry in $entries) { Download-One $entry $entries }
    Assert-Disk ($script:TestFreeBytes -eq $MarginBytes) 'resume retains exact remaining admission'

    $Dest = New-DiskCase unbound-prefixes
    foreach ($entry in $entries) {
        $payload = [IO.File]::ReadAllBytes((Join-Path $diskFixture $entry.Name))
        [IO.File]::WriteAllBytes((Join-Path $Dest ($entry.Name + '.partial')), $payload[0..4])
    }
    Assert-Disk ((Get-RemainingDownloadBytes $entries) -eq $total) 'unbound partials get no credit'

    $Dest = New-DiskCase corrupt-final
    [IO.File]::WriteAllText((Join-Path $Dest 'alpha.bin'), ('X' * $entries[0].Length))
    [IO.File]::Copy((Join-Path $diskFixture 'beta.bin'), (Join-Path $Dest 'beta.bin'))
    Initialize-VerifiedFiles $entries
    Assert-Disk ((Get-RemainingDownloadBytes $entries) -eq $entries[0].Length) 'same-length corrupt source remains charged'

    $Dest = New-DiskCase complete-verified
    [IO.File]::Copy((Join-Path $diskFixture 'alpha.bin'), (Join-Path $Dest 'alpha.bin.partial'))
    [IO.File]::Copy((Join-Path $diskFixture 'beta.bin'), (Join-Path $Dest 'beta.bin'))
    $binding = Get-JournalText "$TestBaseUrl/alpha.bin" $entries[0]
    Write-NewUtf8File (Join-Path $script:JournalDir 'alpha.bin.journal') $binding
    $script:TestFreeBytes = [int64]0
    Initialize-VerifiedFiles $entries
    Download-One $entries[0] $entries
    Assert-Disk ($script:SpaceProbes -eq 0 -and $script:TestTransfers -eq 0) 'verified completed partial needs no new disk or transport'
    Assert-Disk ((Get-Sha256 (Join-Path $Dest 'alpha.bin')) -eq $entries[0].Sha256) 'verified completed partial publishes exact bytes'
    Assert-Disk (-not (Test-Path -LiteralPath (Join-Path $Dest 'quarantine'))) 'verified completed partial is not quarantined'

    $rejectedAlpha = Join-Path $Work 'rejected-alpha.bin'
    $rejectedBytes = [IO.File]::ReadAllBytes((Join-Path $diskFixture 'alpha.bin'))
    $rejectedBytes[0] = $rejectedBytes[0] -bxor 1
    [IO.File]::WriteAllBytes($rejectedAlpha, $rejectedBytes)

    $Dest = New-DiskCase complete-corrupt
    [IO.File]::Copy($rejectedAlpha, (Join-Path $Dest 'alpha.bin.partial'))
    [IO.File]::Copy((Join-Path $diskFixture 'beta.bin'), (Join-Path $Dest 'beta.bin'))
    Write-NewUtf8File (Join-Path $script:JournalDir 'alpha.bin.journal') $binding
    $script:TestFreeBytes = $entries[0].Length + $MarginBytes
    Initialize-VerifiedFiles $entries
    Download-One $entries[0] $entries
    Assert-Disk ($script:TestTransfers -eq 1 -and $script:TestFreeBytes -eq $MarginBytes) 'corrupt completed partial restarts once with full disk charge'
    Assert-Disk ((Get-Sha256 (Join-Path $Dest 'alpha.bin')) -eq $entries[0].Sha256) 'fresh replacement verifies'
    Assert-Quarantined $rejectedAlpha $binding

    $Dest = New-DiskCase complete-corrupt-no-space
    [IO.File]::Copy($rejectedAlpha, (Join-Path $Dest 'alpha.bin.partial'))
    [IO.File]::Copy((Join-Path $diskFixture 'beta.bin'), (Join-Path $Dest 'beta.bin'))
    Write-NewUtf8File (Join-Path $script:JournalDir 'alpha.bin.journal') $binding
    $script:TestFreeBytes = $MarginBytes
    Initialize-VerifiedFiles $entries
    $refused = $false
    try { Download-One $entries[0] $entries }
    catch { $refused = $_.Exception.Message -match '^disk-test-exit:4 ' }
    Assert-Disk ($refused -and $script:TestTransfers -eq 0) 'lost corrupt-prefix credit must be admitted before replacement'
    Assert-Disk (-not (Test-Path -LiteralPath (Join-Path $Dest 'alpha.bin'))) 'low-space recovery never publishes rejected bytes'
    Assert-Quarantined $rejectedAlpha $binding

    $Dest = New-DiskCase corrupt-resumed-prefix
    [IO.File]::WriteAllBytes((Join-Path $Dest 'alpha.bin.partial'), $rejectedBytes[0..4])
    [IO.File]::Copy((Join-Path $diskFixture 'beta.bin'), (Join-Path $Dest 'beta.bin'))
    Write-NewUtf8File (Join-Path $script:JournalDir 'alpha.bin.journal') $binding
    $script:TestFreeBytes = $entries[0].Length - 5 + $MarginBytes
    Initialize-VerifiedFiles $entries
    $refused = $false
    try { Download-One $entries[0] $entries }
    catch { $refused = $_.Exception.Message -match '^disk-test-exit:1 ' }
    Assert-Disk ($refused -and $script:TestTransfers -eq 1) 'completed corrupt resume refuses without an unbounded retry'
    Assert-Disk (-not (Test-Path -LiteralPath (Join-Path $Dest 'alpha.bin'))) 'corrupt resumed result never publishes'
    Assert-Quarantined $rejectedAlpha $binding
    $script:TestFreeBytes = $entries[0].Length + $MarginBytes
    Download-One $entries[0] $entries
    Assert-Disk ($script:TestTransfers -eq 2 -and $script:TestFreeBytes -eq $MarginBytes) 'later retry starts clean with a new admission'
    Assert-Disk ((Get-Sha256 (Join-Path $Dest 'alpha.bin')) -eq $entries[0].Sha256) 'later retry escapes corrupt completed-prefix trap'

    $Dest = New-DiskCase insufficient
    $script:TestFreeBytes = $total + $MarginBytes - 1
    $refused = $false
    try { Download-One $entries[0] $entries }
    catch { $refused = $_.Exception.Message -match '^disk-test-exit:4 ' }
    Assert-Disk ($refused -and $script:TestTransfers -eq 0) 'one-byte short refuses before transport'

    $Dest = New-DiskCase overflow
    $overflow = @([PSCustomObject]@{ Name = 'large.bin'; Length = [int64]::MaxValue; Sha256 = ('0' * 64) })
    $refused = $false
    try { Get-RemainingDownloadBytes $overflow | Out-Null }
    catch { $refused = $_.Exception.Message -match '^disk-test-exit:4 ' }
    Assert-Disk $refused 'remaining plus margin overflow refuses'
    $Catalog = Join-Path $Work 'overflow-catalog.txt'
    [IO.File]::WriteAllText($Catalog, ('large.bin|9223372036854775807|' + ('0' * 64) + "`n"))
    $refused = $false
    try { Get-Catalog | Out-Null }
    catch { $refused = $_.Exception.Message -match '^disk-test-exit:2 ' }
    Assert-Disk $refused 'catalog plus margin overflow refuses'
    Pass 'disk-admission' 'twelve cache-recovery-resume-capacity-and-overflow scenarios without network'
}

function Test-InterruptedTransfer([string]$Stage) {
    $script:Cases++
    $destination = Join-Path $Work ("interrupted-" + $Stage)
    $fixtureRoot = Join-Path $Work ("interrupt-fixture-" + $Stage)
    New-Item -ItemType Directory -Path $fixtureRoot | Out-Null
    $payload = [Text.Encoding]::UTF8.GetBytes('bounded retained source bytes ' * 2048)
    $source = Join-Path $fixtureRoot 'alpha.bin'
    [IO.File]::WriteAllBytes($source, $payload)
    $digest = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    $catalogPath = Join-Path $Work ("interrupt-" + $Stage + ".catalog")
    [IO.File]::WriteAllText($catalogPath, ('alpha.bin|' + $payload.Length + '|' + $digest + [char]10))
    $server = if ($Stage -eq 'headers') {
        Start-InterruptedResumeServer -SilentHeaders -SourceDirectory $fixtureRoot
    } else {
        Start-InterruptedResumeServer -SilentBody -SourceDirectory $fixtureRoot
    }
    if (-not $server) { throw "could not start silent-$Stage fixture" }
    $runner = [Management.Automation.PowerShell]::Create()
    $invocation = $null
    $stop = $null
    try {
        $env:FNLP_FETCH_ALLOW_TEST_BASE_URL = '1'
        [void]$runner.AddCommand($Fetch).AddParameter('Dest', $destination).AddParameter('Catalog', $catalogPath).AddParameter('TestBaseUrl', $server.Base).AddParameter('AllowUntrustedRevision')
        $invocation = $runner.BeginInvoke()
        $partial = Join-Path $destination 'alpha.bin.partial'
        $journal = Join-Path $destination '.fnlp-fetch-journals/alpha.bin.journal'
        $lock = Join-Path $destination ('.fnlp-fetch-lock-' + $Revision)
        $waiting = $false
        $deadline = [DateTime]::UtcNow.AddSeconds(10)
        while ([DateTime]::UtcNow -lt $deadline -and -not $invocation.IsCompleted) {
            $trace = if (Test-Path -LiteralPath $server.Trace) { [IO.File]::ReadAllText($server.Trace) } else { '' }
            $retained = if (Test-Path -LiteralPath $partial) { (Get-Item -LiteralPath $partial).Length } else { -1 }
            if ($trace -match "stall-ready=silent-$Stage" -and
                (($Stage -eq 'headers' -and $retained -eq 0) -or ($Stage -eq 'body' -and $retained -gt 0))) {
                $waiting = $true
                break
            }
            Start-Sleep -Milliseconds 50
        }
        if (-not $waiting -or -not (Test-Path -LiteralPath $lock)) { throw "fetch never reached the silent-$Stage wait under its lock" }
        $journalBytes = [IO.File]::ReadAllText($journal)

        # A real competing invocation must not enter or remove this live lock.
        & $script:PwshPath -NoProfile -File $Fetch -Dest $destination -Catalog $catalogPath -TestBaseUrl $server.Base -AllowUntrustedRevision 2> (Join-Path $Work ("interrupt-" + $Stage + "-contender.err"))
        if ($LASTEXITCODE -ne 4 -or -not (Test-Path -LiteralPath $lock)) { throw 'concurrent acquisition changed the owned lock' }

        # Exercise the same PowerShell stop mechanism as Ctrl+C, without a
        # platform-specific process signal or an unbounded synchronous Stop.
        $stop = $runner.BeginStop($null, $null)
        $deadline = [DateTime]::UtcNow.AddSeconds(5)
        while (-not $stop.IsCompleted -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 50 }
        if (-not $stop.IsCompleted) { throw 'pipeline stop did not finish within five seconds' }
        $runner.EndStop($stop)
        try { [void]$runner.EndInvoke($invocation) }
        catch {
            if ($_.Exception.GetBaseException() -isnot [Management.Automation.PipelineStoppedException]) { throw }
        }
        if ($runner.InvocationStateInfo.State -ne [Management.Automation.PSInvocationState]::Stopped) { throw 'stopped acquisition became ordinary completion' }
        if ((Test-Path -LiteralPath $lock) -or (Test-Path -LiteralPath (Join-Path $destination 'alpha.bin'))) { throw 'stop retained its lock or published incomplete bytes' }
        if ([IO.File]::ReadAllText($journal) -cne $journalBytes) { throw 'stop changed resume authority' }
        $exclusive = [IO.File]::Open($partial, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        $exclusive.Dispose()
        $retainedBytes = [IO.File]::ReadAllBytes($partial)
        $retained = $retainedBytes.Length
        if (($Stage -eq 'headers' -and $retained -ne 0) -or
            ($Stage -eq 'body' -and ($retained -le 0 -or $retained -gt 8192))) { throw 'unexpected retained prefix length' }
        for ($index = 0; $index -lt $retained; $index++) {
            if ($retainedBytes[$index] -ne $payload[$index]) { throw 'stopped transfer changed its retained source prefix' }
        }
        $prefixDigest = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
        Start-Sleep -Milliseconds 200
        if ((Get-Item -LiteralPath $partial).Length -ne $retained -or
            (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $prefixDigest) { throw 'a writer outlived pipeline stop' }
        $deadline = [DateTime]::UtcNow.AddSeconds(3)
        while ([IO.File]::ReadAllText($server.Trace) -notmatch 'stall-closed=True' -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 50 }
        $trace = [IO.File]::ReadAllText($server.Trace)
        if ($trace -notmatch 'stall-closed=True' -or @($trace -split [char]10 | Where-Object { $_ -like 'alpha.bin range=*' }).Count -ne 1) {
            throw 'stop did not close its connection or retried after cancellation'
        }

        & $script:PwshPath -NoProfile -File $Fetch -Dest $destination -Catalog $catalogPath -TestBaseUrl $server.Base -AllowUntrustedRevision 2> (Join-Path $Work ("interrupt-" + $Stage + "-resumed.err"))
        if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath (Join-Path $destination 'alpha.bin') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
            throw 'stopped acquisition could not resume and verify the exact source'
        }
        $range = if ($retained -gt 0) { "bytes=$retained-" } else { '' }
        $requests = @(Get-Content -LiteralPath $server.Trace | Where-Object { $_ -like 'alpha.bin range=*' })
        if ($requests.Count -ne 2 -or $requests[1] -cne "alpha.bin range=$range") { throw 'retry did not bind the actual retained byte count' }
        Pass ("interrupt-" + $Stage) 'stops-and-drains-before-owned-lock-release-then-resumes-exact-prefix'
    } catch {
        Fail ("interrupt-" + $Stage) $_.Exception.Message
        throw
    } finally {
        # A failing old implementation must not hang the test host. Closing
        # this exact fixture server unblocks its pending socket before cleanup.
        Stop-RetryServer
        if ($null -ne $invocation -and -not $invocation.IsCompleted) {
            if ($null -eq $stop) { $stop = $runner.BeginStop($null, $null) }
            $deadline = [DateTime]::UtcNow.AddSeconds(5)
            while (-not $invocation.IsCompleted -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 50 }
        }
        if ($null -eq $invocation -or $invocation.IsCompleted) { $runner.Dispose() }
        else { throw 'fixture pipeline did not drain after its exact server stopped' }
    }
}

function Test-UnexpectedFailureCleanup {
    $script:Cases++
    $destination = Join-Path $Work 'unexpected-failure'
    New-Item -ItemType Directory -Path $destination | Out-Null
    $source = Join-Path $destination 'ready.bin'
    [IO.File]::WriteAllText($source, 'retained verified source', [Text.UTF8Encoding]::new($false))
    $digest = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    $catalogPath = Join-Path $Work 'unexpected-failure-catalog.txt'
    $catalogLine = 'ready.bin|' + (Get-Item -LiteralPath $source).Length + '|' + $digest
    [IO.File]::WriteAllText($catalogPath, $catalogLine + [char]10)
    $wrapper = Join-Path $Work 'unexpected-failure-wrapper.ps1'
    $wrapperBody = @'
param([string]$Fetcher, [string]$Destination, [string]$CatalogPath)
# Inject a failure in the existing command dependency, after the real fetcher
# has acquired its revision lock. No production test hooks are introduced.
function Get-FileHash {
    param([string]$LiteralPath, [string]$Algorithm)
    throw [IO.IOException]::new('fixture verification failure')
}
& $Fetcher -Dest $Destination -Catalog $CatalogPath -AllowUntrustedRevision -CheckOnly
exit $LASTEXITCODE
'@
    [IO.File]::WriteAllText($wrapper, $wrapperBody, [Text.UTF8Encoding]::new($false))
    $errorPath = Join-Path $Work 'unexpected-failure.err'
    & $script:PwshPath -NoProfile -File $wrapper -Fetcher $Fetch -Destination $destination -CatalogPath $catalogPath 2> $errorPath
    $status = $LASTEXITCODE
    $lock = Join-Path $destination ('.fnlp-fetch-lock-' + $Revision)
    if ($status -ne 1 -or (Test-Path -LiteralPath $lock) -or
        (Get-Content -LiteralPath $errorPath -Raw) -notmatch 'fixture verification failure' -or
        (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
        Fail 'unexpected-failure' "owned-lock-release exit=$status"
        return
    }
    & $script:PwshPath -NoProfile -File $Fetch -Dest $destination -Catalog $catalogPath -AllowUntrustedRevision -CheckOnly
    if ($LASTEXITCODE -eq 0 -and -not (Test-Path -LiteralPath $lock)) {
        Pass 'unexpected-failure' 'verification-error-releases-only-owned-lock-and-allows-clean-retry'
    } else { Fail 'unexpected-failure' 'retry-after-unexpected-failure' }
}

try {
    if ($DiskOnly -and $LifecycleOnly) { throw 'choose either -DiskOnly or -LifecycleOnly' }
    $script:PwshPath = (Get-Command pwsh -ErrorAction Stop).Source
    if ($DiskOnly) {
        New-Item -ItemType Directory -Path $Work | Out-Null
        Test-DiskAdmission
        Test-UnexpectedFailureCleanup
        return
    }
    if (-not (Get-Command python3 -ErrorAction SilentlyContinue)) { throw 'python3 is required for the local fixture server' }
    if (-not $LifecycleOnly -and -not (Get-Command openssl -ErrorAction SilentlyContinue)) { throw 'openssl is required for the local TLS redirect server' }
    $script:PwshPath = (Get-Command pwsh -ErrorAction Stop).Source

    # Keep the redirect fixture honest: it supplies a private CA through the
    # process trust store rather than introducing a production escape hatch for
    # certificate or redirect policy. This mirrors case 0 in the shell harness.
    $script:Cases++
    $fetchSource = Get-Content -LiteralPath $Fetch -Raw
    if ($fetchSource -match 'FNLP_FETCH_TEST|FNLP_FETCH_TEST_ALLOW_INSECURE_TLS|ServerCertificateCustomValidationCallback|DangerousAcceptAnyServerCertificateValidator|SkipCertificateCheck') {
        Fail '0' 'production-transport-isolation'
    } else {
        Pass '0' 'production-transport-isolation'
    }

    $script:Cases++
    $pwshVersion = [Version](& $script:PwshPath -NoProfile -Command '$PSVersionTable.PSVersion.ToString()')
    if ($pwshVersion.Major -ge 7) {
        Pass 'runtime-host' "pwsh-$pwshVersion"
    } else {
        Fail 'runtime-host' "requires-pwsh-7-or-newer actual=$pwshVersion"
    }
    New-Item -ItemType Directory -Path $Work | Out-Null
    if ($LifecycleOnly) {
        Test-UnexpectedFailureCleanup
        Test-InterruptedTransfer 'headers'
        Test-InterruptedTransfer 'body'
        return
    }
    $Fixture = Join-Path $Work ('fixture/resolve/' + $Revision)
    New-Item -ItemType Directory -Path $Fixture -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $Fixture 'alpha.bin'), "alpha fixture bytes`n", [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText((Join-Path $Fixture 'beta.bin'), "beta fixture bytes are slightly longer`n", [Text.UTF8Encoding]::new($false))
    $alpha = Join-Path $Fixture 'alpha.bin'; $beta = Join-Path $Fixture 'beta.bin'
    $alphaLength = (Get-Item $alpha).Length; $betaLength = (Get-Item $beta).Length
    $alphaSha = (Get-FileHash $alpha -Algorithm SHA256).Hash.ToLowerInvariant(); $betaSha = (Get-FileHash $beta -Algorithm SHA256).Hash.ToLowerInvariant()
    $script:Catalog = Join-Path $Work 'catalog.txt'
    Set-Content -LiteralPath $script:Catalog -NoNewline -Encoding ascii -Value "alpha.bin|$alphaLength|$alphaSha`nbeta.bin|$betaLength|$betaSha`n"
    $port = [int](& python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
    $script:Server = Start-Process -FilePath python3 -ArgumentList @('-m','http.server',"$port",'--bind','127.0.0.1','--directory',(Join-Path $Work 'fixture')) -PassThru -RedirectStandardOutput (Join-Path $Work 'server.out') -RedirectStandardError (Join-Path $Work 'server.err')
    Start-Sleep -Milliseconds 300
    $script:Base = "http://127.0.0.1:$port/resolve/$Revision"

    $script:Cases++; $d1 = Join-Path $Work 'case1'
    if ((Invoke-Fetch $d1) -eq 0 -and (Compare-Object (Get-Content $alpha -Raw) (Get-Content (Join-Path $d1 'alpha.bin') -Raw)).Count -eq 0) { Pass '1' 'fresh-download' } else { Fail '1' 'fresh-download' }

    $script:Cases++; $d2 = Join-Path $Work 'case2'; New-Item -ItemType Directory -Path $d2 -Force | Out-Null
    [IO.File]::WriteAllBytes((Join-Path $d2 'alpha.bin.partial'), [IO.File]::ReadAllBytes($alpha)[0..4]); Write-Journal $d2 'alpha.bin' $alphaLength $alphaSha ($script:Base + '/alpha.bin')
    if ((Invoke-Fetch $d2) -eq 0 -and (Get-FileHash (Join-Path $d2 'alpha.bin') -Algorithm SHA256).Hash.ToLowerInvariant() -eq $alphaSha) { Pass '2' 'journal-resume' } else { Fail '2' 'journal-resume' }

    $script:Cases++; $d3 = Join-Path $Work 'case3'; New-Item -ItemType Directory -Path $d3 -Force | Out-Null; Set-Content -LiteralPath (Join-Path $d3 'alpha.bin.partial') -NoNewline -Value 'unbound bytes'
    if ((Invoke-Fetch $d3) -eq 0 -and (Get-Content (Join-Path $d3 'quarantine/quarantine.log') -Raw) -match 'unbound-or-mismatched-partial') { Pass '3' 'unbound-partial-quarantine' } else { Fail '3' 'unbound-partial-quarantine' }

    $script:Cases++; $d4 = Join-Path $Work 'case4'; New-Item -ItemType Directory -Path $d4 -Force | Out-Null
    try { New-Item -ItemType SymbolicLink -Path (Join-Path $d4 'alpha.bin.partial') -Target $alpha -ErrorAction Stop | Out-Null; $status = Invoke-Fetch $d4; if ($status -eq 1) { Pass '4' 'reparse-point-refusal' } else { Fail '4' "reparse-point-refusal exit=$status" } } catch { Fail '4' "could-not-create-symlink detail=$($_.Exception.Message)" }

    $script:Cases++; $d5 = Join-Path $Work 'case5'; New-Item -ItemType Directory -Path $d5 -Force | Out-Null; Set-Content -LiteralPath (Join-Path $d5 'alpha.bin') -NoNewline -Value 'corrupt old final'
    if ((Invoke-Fetch $d5) -eq 0 -and (Get-Content (Join-Path $d5 'quarantine/quarantine.log') -Raw) -match 'existing-file-verification-failed') { Pass '5' 'corrupt-final-quarantine' } else { Fail '5' 'corrupt-final-quarantine' }

    $script:Cases++; $d6 = Join-Path $Work 'case6'
    if ((Invoke-Fetch $d6) -eq 0 -and (Invoke-Fetch $d6 @('-CheckOnly')) -eq 0) {
        Add-Content -LiteralPath (Join-Path $d6 'alpha.bin') -NoNewline -Value 'X'
        if ((Invoke-Fetch $d6 @('-CheckOnly')) -eq 1) { Pass '6' 'check-only-tamper' } else { Fail '6' 'check-only-tamper' }
    } else { Fail '6' 'check-only-setup' }

    $script:Cases++; $d7 = Join-Path $Work 'case7'
    $env:FNLP_FETCH_ALLOW_TEST_BASE_URL = '1'; & pwsh -NoProfile -File $Fetch -Dest $d7 -Revision deadbeef 2> (Join-Path $Work 'case7.err'); $status = $LASTEXITCODE
    if ($status -eq 2 -and (Get-Content (Join-Path $Work 'case7.err') -Raw) -match 'UNTRUSTED_REVISION_REFUSED') { Pass '7' 'untrusted-revision-refusal' } else { Fail '7' "untrusted-revision-refusal exit=$status" }

    $script:Cases++; $d7catalog = Join-Path $Work 'case7catalog'
    $env:FNLP_FETCH_ALLOW_TEST_BASE_URL = '1'; & pwsh -NoProfile -File $Fetch -Dest $d7catalog -Catalog $script:Catalog -TestBaseUrl $script:Base 2> (Join-Path $Work 'case7catalog.err'); $status = $LASTEXITCODE
    if ($status -eq 2 -and (Get-Content (Join-Path $Work 'case7catalog.err') -Raw) -match 'UNTRUSTED_CATALOG_REFUSED') { Pass '7catalog' 'default-revision-custom-catalog-refusal' } else { Fail '7catalog' "default-revision-custom-catalog-refusal exit=$status" }

    $script:Cases++; $d8 = Join-Path $Work 'case8'; New-Item -ItemType Directory -Path $d8 -Force | Out-Null; [IO.File]::WriteAllBytes((Join-Path $d8 'alpha.bin.partial'), [IO.File]::ReadAllBytes($alpha)[0..4]); Write-Journal $d8 'alpha.bin' $alphaLength $alphaSha 'http://127.0.0.1:1/alpha.bin'
    $env:FNLP_FETCH_ALLOW_TEST_BASE_URL = '1'; & pwsh -NoProfile -File $Fetch -Dest $d8 -Catalog $script:Catalog -TestBaseUrl 'http://127.0.0.1:1' -AllowUntrustedRevision 2> (Join-Path $Work 'case8.err'); $status = $LASTEXITCODE
    if ($status -eq 3 -and (Get-Content (Join-Path $Work 'case8.err') -Raw) -match ([regex]::Escape("-Dest `"$d8`""))) { Pass '8' 'interrupted-resume-guidance' } else { Fail '8' "interrupted-resume-guidance exit=$status" }

    $script:Cases++; $d8retry = Join-Path $Work 'case8retry'; $retry = Start-InterruptedResumeServer
    if ($retry) {
        $env:FNLP_FETCH_ALLOW_TEST_BASE_URL = '1'; & pwsh -NoProfile -File $Fetch -Dest $d8retry -Catalog $script:Catalog -TestBaseUrl $retry.Base -AllowUntrustedRevision 2> (Join-Path $Work 'case8retry.err'); $status = $LASTEXITCODE
        if ($status -eq 0 -and (Get-FileHash (Join-Path $d8retry 'alpha.bin') -Algorithm SHA256).Hash.ToLowerInvariant() -eq $alphaSha -and (Get-Content -LiteralPath $retry.Trace -Raw) -match 'alpha.bin range=bytes=5-') { Pass '8retry' 'retry-rebinds-range-after-partial-body' } else { Fail '8retry' "retry-rebinds-range-after-partial-body exit=$status" }
    } else {
        Fail '8retry' 'retry-server'
    }
    Stop-RetryServer

    $script:Cases++; $d8malformed = Join-Path $Work 'case8malformed'; $malformed = Start-InterruptedResumeServer -MalformedRange
    if ($malformed) {
        $env:FNLP_FETCH_ALLOW_TEST_BASE_URL = '1'; & pwsh -NoProfile -File $Fetch -Dest $d8malformed -Catalog $script:Catalog -TestBaseUrl $malformed.Base -AllowUntrustedRevision 2> (Join-Path $Work 'case8malformed.err'); $status = $LASTEXITCODE
        $partial = Join-Path $d8malformed 'alpha.bin.partial'
        if ($status -eq 3 -and -not (Test-Path -LiteralPath (Join-Path $d8malformed 'alpha.bin')) -and (Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial).Length -eq 5 -and (Get-Content -LiteralPath (Join-Path $Work 'case8malformed.err') -Raw) -match 'mismatched Content-Range') { Pass '8malformed' 'mismatched-content-range-refusal' } else { Fail '8malformed' "mismatched-content-range-refusal exit=$status" }
    } else {
        Fail '8malformed' 'malformed-range-server'
    }
    Stop-RetryServer

    $script:Cases++; $d8overlong = Join-Path $Work 'case8overlong'; $overlong = Start-InterruptedResumeServer -OverlongBody
    if ($overlong) {
        $env:FNLP_FETCH_ALLOW_TEST_BASE_URL = '1'; & pwsh -NoProfile -File $Fetch -Dest $d8overlong -Catalog $script:Catalog -TestBaseUrl $overlong.Base -AllowUntrustedRevision 2> (Join-Path $Work 'case8overlong.err'); $status = $LASTEXITCODE
        $partial = Join-Path $d8overlong 'alpha.bin.partial'
        if ($status -eq 0 -and -not (Test-Path -LiteralPath $partial) -and (Get-FileHash (Join-Path $d8overlong 'alpha.bin') -Algorithm SHA256).Hash.ToLowerInvariant() -eq $alphaSha -and (Get-Content -LiteralPath $overlong.Trace -Raw) -match 'alpha.bin range=bytes=5-') { Pass '8overlong' 'overlong-body-never-appended' } else { Fail '8overlong' "overlong-body-never-appended exit=$status" }
    } else {
        Fail '8overlong' 'overlong-body-server'
    }
    Stop-RetryServer

    $script:Cases++; $d9 = Join-Path $Work 'case9'; $status = Invoke-Fetch $d9 @('-CheckOnly')
    if ($status -eq 1) { Pass '9' 'check-only-missing-closure-refusal' } else { Fail '9' "check-only-missing-closure-refusal exit=$status" }

    $script:Cases++; $d10 = Join-Path $Work 'case10'; $accepted = Start-RedirectServer 'us.aws.cdn.hf.co'
    if ($accepted -and (Invoke-RedirectPolicyFetch $d10 $accepted.ProxyPort $accepted.CaCertificate) -eq 0 -and (Get-FileHash (Join-Path $d10 'alpha.bin') -Algorithm SHA256).Hash.ToLowerInvariant() -eq $alphaSha) {
        Pass '10' 'regional-cdn-redirect-accepted'
    } else {
        Fail '10' 'regional-cdn-redirect-accepted'
    }
    Stop-RedirectServer

    $script:Cases++; $d11 = Join-Path $Work 'case11'; $rejected = Start-RedirectServer 'unlisted.invalid'; $case11Err = Join-Path $Work 'case11.err'
    if ($rejected) {
        $status = Invoke-RedirectPolicyFetch $d11 $rejected.ProxyPort $rejected.CaCertificate 2> $case11Err
        $contacts = Get-Content -LiteralPath $rejected.Trace -Raw
        if ($status -eq 3 -and -not (Test-Path -LiteralPath (Join-Path $d11 'alpha.bin')) -and (Get-Item (Join-Path $d11 'alpha.bin.partial')).Length -eq 0 -and $contacts -notmatch 'host=unlisted.invalid |CONNECT unlisted.invalid:' -and (Get-Content -LiteralPath $case11Err -Raw) -match 'REDIRECT_HOST_REFUSED phase=pre-contact activation=refused effective_url=https://unlisted.invalid/') { Pass '11' 'unlisted-redirect-refused-before-contact' } else { Fail '11' "unlisted-redirect-refusal exit=$status" }
    } else {
        Fail '11' 'unlisted-redirect-server'
    }
    Stop-RedirectServer

    # A failed acquisition must preserve a lock owned by another invocation.
    $script:Cases++; $d12 = Join-Path $Work 'case12'; $lock = Join-Path $d12 ('.fnlp-fetch-lock-' + $Revision)
    New-Item -ItemType Directory -Path $lock -Force | Out-Null
    $firstStatus = Invoke-Fetch $d12 @('-CheckOnly')
    $secondStatus = Invoke-Fetch $d12 @('-CheckOnly')
    if ($firstStatus -eq 4 -and $secondStatus -eq 4 -and (Test-Path -LiteralPath $lock -PathType Container)) { Pass '12' 'contended-lock-preserved' }
    else { Fail '12' "contended-lock-preserved first=$firstStatus second=$secondStatus" }

    $script:Cases++; $d13 = Join-Path $Work 'case13'
    & $script:PwshPath -NoProfile -File $Fetch -Dest $d13 -Catalog $script:Catalog -CheckOnly 2> (Join-Path $Work 'case13.err')
    $status = $LASTEXITCODE
    if ($status -eq 2 -and -not (Test-Path -LiteralPath $d13) -and (Get-Content (Join-Path $Work 'case13.err') -Raw) -match 'UNTRUSTED_CATALOG_REFUSED') { Pass '13' 'custom-catalog-requires-opt-in' }
    else { Fail '13' "custom-catalog-requires-opt-in exit=$status" }

    $script:Cases++; $d14 = Join-Path $Work 'case14'
    $confused = Start-RedirectServer 'unlisted.invalid' 'https://unlisted.invalid/example.xethub.hf.co/{name}'
    if ($confused) {
        $status = Invoke-RedirectPolicyFetch $d14 $confused.ProxyPort $confused.CaCertificate 2> (Join-Path $Work 'case14.err')
        if ($status -eq 3 -and (Get-Item (Join-Path $d14 'alpha.bin.partial')).Length -eq 0 -and (Get-Content $confused.Trace -Raw) -notmatch 'host=unlisted.invalid |CONNECT unlisted.invalid:') { Pass '14' 'redirect-path-cannot-authorize-host' }
        else { Fail '14' "redirect-path-cannot-authorize-host exit=$status" }
    } else { Fail '14' 'confused-host-server' }
    Stop-RedirectServer

    $script:Cases++; $d15 = Join-Path $Work 'case15'; $relative = Start-RedirectServer 'us.aws.cdn.hf.co' '/redirect-final/{name}'
    if ($relative -and (Invoke-RedirectPolicyFetch $d15 $relative.ProxyPort $relative.CaCertificate) -eq 0 -and (Get-FileHash (Join-Path $d15 'alpha.bin') -Algorithm SHA256).Hash.ToLowerInvariant() -eq $alphaSha -and (Get-Content $relative.Trace -Raw) -match 'agent=franken_nlp-agent ') { Pass '15' 'relative-redirect-and-agent' }
    else { Fail '15' 'relative-redirect-and-agent' }
    Stop-RedirectServer

    $script:Cases++; $d16 = Join-Path $Work 'case16'; $loop = Start-RedirectServer 'us.aws.cdn.hf.co' '/loop/{name}'
    if ($loop) {
        $status = Invoke-RedirectPolicyFetch $d16 $loop.ProxyPort $loop.CaCertificate 2> (Join-Path $Work 'case16.err')
        $requests = @(Get-Content $loop.Trace | Where-Object { $_ -match '^host=' }).Count
        if ($status -eq 3 -and $requests -eq 36 -and (Get-Item (Join-Path $d16 'alpha.bin.partial')).Length -eq 0 -and (Get-Content (Join-Path $Work 'case16.err') -Raw) -match 'redirect limit exceeded') { Pass '16' 'redirect-loop-bounded' }
        else { Fail '16' "redirect-loop-bounded exit=$status requests=$requests" }
    } else { Fail '16' 'redirect-loop-server' }
    Stop-RedirectServer

    $script:Cases++; $d17 = Join-Path $Work 'case17'
    Write-Journal $d17 'alpha.bin' $alphaLength $alphaSha "https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve/$Revision/alpha.bin"
    [IO.File]::WriteAllBytes((Join-Path $d17 'alpha.bin.partial'), [IO.File]::ReadAllBytes($alpha)[0..4])
    $resumed = Start-RedirectServer 'us.aws.cdn.hf.co'
    if ($resumed -and (Invoke-RedirectPolicyFetch $d17 $resumed.ProxyPort $resumed.CaCertificate) -eq 0 -and (Get-FileHash (Join-Path $d17 'alpha.bin') -Algorithm SHA256).Hash.ToLowerInvariant() -eq $alphaSha -and (Get-Content $resumed.Trace -Raw) -match 'host=us.aws.cdn.hf.co .*range=bytes=5-') { Pass '17' 'redirected-range-resume' }
    else { Fail '17' 'redirected-range-resume' }
    Stop-RedirectServer

    # Execute only the parsed policy function, without starting the fetcher.
    $script:Cases++
    $tokens = $null; $parseErrors = $null
    $fetchAst = [Management.Automation.Language.Parser]::ParseFile($Fetch, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -ne 0) { throw 'fetcher has PowerShell parse errors' }
    $hostFunction = $fetchAst.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-EffectiveHost' }, $true)
    . ([scriptblock]::Create($hostFunction.Extent.Text))
    $policyPassed = $true
    foreach ($allowed in @('https://huggingface.co/file', 'https://CDN-LFS.HUGGINGFACE.CO:443/file', 'https://us.aws.cdn.hf.co/file', 'https://cas-bridge.xethub.hf.co/file')) {
        if (-not (Test-EffectiveHost ([Uri]$allowed))) { $policyPassed = $false }
    }
    foreach ($denied in @('https://huggingface.co@unlisted.invalid/file', 'https://unlisted.invalid@huggingface.co/file', 'https://huggingface.co:444/file', 'http://huggingface.co/file', 'https://unlisted.invalid/.cdn.hf.co/file', 'https://a..cdn.hf.co/file', 'https://-a.cdn.hf.co/file', 'https://huggingface.co\@unlisted.invalid/file')) {
        $deniedUri = $null
        if ([Uri]::TryCreate($denied, [UriKind]::Absolute, [ref]$deniedUri) -and (Test-EffectiveHost $deniedUri)) { $policyPassed = $false }
    }
    if ($policyPassed) { Pass '18' 'strict-redirect-authority' } else { Fail '18' 'strict-redirect-authority' }

    $script:Cases++; $d19 = Join-Path $Work 'case19'; $nonemptyLock = Join-Path $d19 ('.fnlp-fetch-lock-' + $Revision)
    New-Item -ItemType Directory -Path $nonemptyLock -Force | Out-Null
    $ownerMarker = Join-Path $nonemptyLock 'owner.txt'
    [IO.File]::WriteAllText($ownerMarker, 'another invocation owns this lock')
    $status = Invoke-Fetch $d19 @('-CheckOnly') 2> (Join-Path $Work 'case19.err')
    if ($status -eq 4 -and [IO.File]::ReadAllText($ownerMarker) -eq 'another invocation owns this lock' -and (Get-Content (Join-Path $Work 'case19.err') -Raw) -notmatch 'LOCK_RELEASE_DEFERRED') { Pass '19' 'nonempty-contended-lock-untouched' }
    else { Fail '19' "nonempty-contended-lock-untouched exit=$status" }
    Test-DiskAdmission
    Test-UnexpectedFailureCleanup
    Test-InterruptedTransfer 'headers'
    Test-InterruptedTransfer 'body'
} catch {
    Fail 'harness' $_.Exception.Message
} finally {
    if ($script:Server) { try { Stop-Process -Id $script:Server.Id -ErrorAction Stop } catch {} }
    Stop-RedirectServer
    Stop-RetryServer
    if ($script:Failed.Count -eq 0) { Write-Log "FETCH_MODEL_TESTS RESULT=PASS cases=$script:Cases failed=none retained_work=$Work"; exit 0 }
    Write-Log "FETCH_MODEL_TESTS RESULT=FAIL cases=$script:Cases failed=$($script:Failed -join ',') retained_work=$Work"; exit 1
}
