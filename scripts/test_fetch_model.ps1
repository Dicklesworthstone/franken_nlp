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

function Write-Log([string]$Message) { [Console]::Error.WriteLine("{0} FETCH_MODEL_TEST {1}" -f [DateTime]::UtcNow.ToString('o'), $Message) }
function Pass([string]$Case, [string]$Detail) { Write-Log "CASE=$Case RESULT=PASS detail=$Detail" }
function Fail([string]$Case, [string]$Detail) { $script:Failed.Add($Case); Write-Log "CASE=$Case RESULT=FAIL detail=$Detail" }
function Invoke-Fetch([string]$Dest, [string[]]$Extra = @()) {
    $env:FNLP_FETCH_ALLOW_TEST_BASE_URL = '1'
    & pwsh -NoProfile -File $Fetch -Dest $Dest -Catalog $script:Catalog -TestBaseUrl $script:Base @Extra
    return $LASTEXITCODE
}
function Write-Journal([string]$Dest, [string]$Name, [int64]$Length, [string]$Digest, [string]$Url) {
    $journalDir = Join-Path $Dest '.fnlp-fetch-journals'
    New-Item -ItemType Directory -Path $journalDir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $journalDir ($Name + '.journal')) -NoNewline -Encoding utf8 -Value "url=$Url`nrevision=$Revision`nname=$Name`nbytes=$Length`nsha256=$Digest`n"
}

try {
    if (-not (Get-Command python3 -ErrorAction SilentlyContinue)) { throw 'python3 is required for the local fixture server' }
    $script:Cases++
    $effectiveHostPolicy = (Select-String -LiteralPath $Fetch -Pattern '^\s*return \$host ' | Select-Object -Last 1).Line
    if ($effectiveHostPolicy -match "\$host -like '\*\.xethub\.hf\.co'" -and $effectiveHostPolicy -match "\$host -like '\*\.cdn\.hf\.co'") {
        Pass 'redirect-policy' 'official-xet-cdn-hosts-accepted'
    } else {
        Fail 'redirect-policy' 'official-xet-cdn-hosts-accepted'
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
} catch {
    Fail 'harness' $_.Exception.Message
} finally {
    if ($script:Server) { try { Stop-Process -Id $script:Server.Id -ErrorAction Stop } catch {} }
    if ($script:Failed.Count -eq 0) { Write-Log "FETCH_MODEL_TESTS RESULT=PASS cases=$script:Cases failed=none retained_work=$Work"; exit 0 }
    Write-Log "FETCH_MODEL_TESTS RESULT=FAIL cases=$script:Cases failed=$($script:Failed -join ',') retained_work=$Work"; exit 1
}
