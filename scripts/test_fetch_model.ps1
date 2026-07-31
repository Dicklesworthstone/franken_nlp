# Local-fixture tests for fetch_model.ps1. The retained work directory is
# printed on completion for audit; this harness deliberately never deletes it.
[CmdletBinding()]
param()

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
$script:PwshPath = $null

function Write-Log([string]$Message) { [Console]::Error.WriteLine("{0} FETCH_MODEL_TEST {1}" -f [DateTime]::UtcNow.ToString('o'), $Message) }
function Pass([string]$Case, [string]$Detail) { Write-Log "CASE=$Case RESULT=PASS detail=$Detail" }
function Fail([string]$Case, [string]$Detail) { $script:Failed.Add($Case); Write-Log "CASE=$Case RESULT=FAIL detail=$Detail" }
function Invoke-Fetch([string]$Dest, [string[]]$Extra = @()) {
    $env:FNLP_FETCH_ALLOW_TEST_BASE_URL = '1'
    & $script:PwshPath -NoProfile -File $Fetch -Dest $Dest -Catalog $script:Catalog -TestBaseUrl $script:Base @Extra
    return $LASTEXITCODE
}
function Stop-RedirectServer {
    if ($script:RedirectServer) {
        try { Stop-Process -Id $script:RedirectServer.Id -ErrorAction Stop } catch {}
        $script:RedirectServer = $null
    }
}
function Start-RedirectServer([string]$RedirectHost) {
    $cert = Join-Path $Work 'redirect.crt'
    $key = Join-Path $Work 'redirect.key'
    $ready = Join-Path $Work 'redirect.ready'
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

fixture, cert, key, redirect_host, ready_path = sys.argv[1:]
root = Path(fixture)

class RedirectHandler(http.server.BaseHTTPRequestHandler):
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
    & openssl req -x509 -newkey rsa:2048 -nodes -keyout $key -out $cert -subj '/CN=localhost' -days 1 *> (Join-Path $Work 'openssl.log')
    if ($LASTEXITCODE -ne 0) { return $null }
    $script:RedirectServer = Start-Process -FilePath python3 -ArgumentList @($serverScript, $Fixture, $cert, $key, $RedirectHost, $ready) -PassThru -RedirectStandardOutput (Join-Path $Work 'redirect.out') -RedirectStandardError (Join-Path $Work 'redirect.err')
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if (Test-Path -LiteralPath $ready) {
            $ports = (Get-Content -LiteralPath $ready -Raw).Trim().Split(' ')
            if ($ports.Count -eq 2) { return [PSCustomObject]@{ ProxyPort = [int]$ports[0]; TlsPort = [int]$ports[1] } }
        }
        Start-Sleep -Milliseconds 100
    }
    Stop-RedirectServer
    return $null
}
function Invoke-RedirectPolicyFetch([string]$Dest, [int]$ProxyPort) {
    $names = @('FNLP_FETCH_ALLOW_TEST_REDIRECT_POLICY', 'FNLP_FETCH_TEST_ALLOW_INSECURE_TLS', 'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'NO_PROXY')
    $saved = @{}
    foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    try {
        $env:FNLP_FETCH_ALLOW_TEST_REDIRECT_POLICY = '1'
        $env:FNLP_FETCH_TEST_ALLOW_INSECURE_TLS = '1'
        $env:HTTPS_PROXY = "http://127.0.0.1:$ProxyPort"
        $env:HTTP_PROXY = "http://127.0.0.1:$ProxyPort"
        $env:ALL_PROXY = "http://127.0.0.1:$ProxyPort"
        $env:NO_PROXY = ''
        & $script:PwshPath -NoProfile -File $Fetch -Dest $Dest -Catalog $script:Catalog
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

try {
    if (-not (Get-Command python3 -ErrorAction SilentlyContinue)) { throw 'python3 is required for the local fixture server' }
    if (-not (Get-Command openssl -ErrorAction SilentlyContinue)) { throw 'openssl is required for the local TLS redirect server' }
    $script:PwshPath = (Get-Command pwsh -ErrorAction Stop).Source
    $script:Cases++
    $pwshVersion = [Version](& $script:PwshPath -NoProfile -Command '$PSVersionTable.PSVersion.ToString()')
    if ($pwshVersion.Major -ge 7) {
        Pass 'runtime-host' "pwsh-$pwshVersion"
    } else {
        Fail 'runtime-host' "requires-pwsh-7-or-newer actual=$pwshVersion"
    }
    New-Item -ItemType Directory -Path $Work | Out-Null
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

    $script:Cases++; $d8 = Join-Path $Work 'case8'; New-Item -ItemType Directory -Path $d8 -Force | Out-Null; [IO.File]::WriteAllBytes((Join-Path $d8 'alpha.bin.partial'), [IO.File]::ReadAllBytes($alpha)[0..4]); Write-Journal $d8 'alpha.bin' $alphaLength $alphaSha 'http://127.0.0.1:1/alpha.bin'
    $env:FNLP_FETCH_ALLOW_TEST_BASE_URL = '1'; & pwsh -NoProfile -File $Fetch -Dest $d8 -Catalog $script:Catalog -TestBaseUrl 'http://127.0.0.1:1' 2> (Join-Path $Work 'case8.err'); $status = $LASTEXITCODE
    if ($status -eq 3 -and (Get-Content (Join-Path $Work 'case8.err') -Raw) -match ([regex]::Escape("-Dest `"$d8`""))) { Pass '8' 'interrupted-resume-guidance' } else { Fail '8' "interrupted-resume-guidance exit=$status" }

    $script:Cases++; $d9 = Join-Path $Work 'case9'; $status = Invoke-Fetch $d9 @('-CheckOnly')
    if ($status -eq 0) { Pass '9' 'check-only-no-model-skip' } else { Fail '9' "check-only-no-model-skip exit=$status" }

    $script:Cases++; $d10 = Join-Path $Work 'case10'; $accepted = Start-RedirectServer 'us.aws.cdn.hf.co'
    if ($accepted -and (Invoke-RedirectPolicyFetch $d10 $accepted.ProxyPort) -eq 0 -and (Get-FileHash (Join-Path $d10 'alpha.bin') -Algorithm SHA256).Hash.ToLowerInvariant() -eq $alphaSha) {
        Pass '10' 'regional-cdn-redirect-accepted'
    } else {
        Fail '10' 'regional-cdn-redirect-accepted'
    }
    Stop-RedirectServer

    $script:Cases++; $d11 = Join-Path $Work 'case11'; $rejected = Start-RedirectServer 'unlisted.invalid'; $case11Err = Join-Path $Work 'case11.err'
    if ($rejected) {
        $status = Invoke-RedirectPolicyFetch $d11 $rejected.ProxyPort 2> $case11Err
        if ($status -ne 0 -and (Get-Content -LiteralPath $case11Err -Raw) -match 'REDIRECT_HOST_REFUSED effective_url=https://unlisted.invalid/') { Pass '11' 'unlisted-redirect-refusal' } else { Fail '11' "unlisted-redirect-refusal exit=$status" }
    } else {
        Fail '11' 'unlisted-redirect-server'
    }
    Stop-RedirectServer
} catch {
    Fail 'harness' $_.Exception.Message
} finally {
    if ($script:Server) { try { Stop-Process -Id $script:Server.Id -ErrorAction Stop } catch {} }
    Stop-RedirectServer
    if ($script:Failed.Count -eq 0) { Write-Log "FETCH_MODEL_TESTS RESULT=PASS cases=$script:Cases failed=none retained_work=$Work"; exit 0 }
    Write-Log "FETCH_MODEL_TESTS RESULT=FAIL cases=$script:Cases failed=$($script:Failed -join ',') retained_work=$Work"; exit 1
}
